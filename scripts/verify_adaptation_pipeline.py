#!/usr/bin/env python3
"""Verify that training updates actively alter the deployed policy and worker actions.

This tests the exact deployment and synchronization path:
1. Instantiates the Dreamer agent with foundation_only=False.
2. Verifies initial actions and checkpoint hash.
3. Performs an optimizer update step on the policy parameters.
4. Broadcasts/saves the updated model.
5. Loads the broadcast into an independent worker actor instance.
6. Asserts that worker parameters match the update and worker actions change (Delta a != 0).
7. Asserts latency is within the 50 ms deadline.
"""

from collections import deque
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from gymnasium import spaces

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tmrl.config.config_constants as cfg
from tmrl.custom.torch.dreamer import TorchDreamerActor, TorchDreamerAgent


def tensor_state_digest(state_dict, prefixes=("policy.", "mu_head.", "log_std_head.")):
    digest = hashlib.sha256()
    for key in sorted(key for key in state_dict if any(key.startswith(p) for p in prefixes)):
        value = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def make_dummy_observation():
    speed = torch.tensor([[50.0 / 10.0]], dtype=torch.float32)
    gear = torch.tensor([[3.0 / 10.0]], dtype=torch.float32)
    rpm = torch.tensor([[5000.0]], dtype=torch.float32)
    img = torch.rand(1, cfg.IMG_HIST_LEN, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, dtype=torch.float32)
    prev1 = torch.tensor([[1.0, -1.0, 0.0]], dtype=torch.float32)
    prev2 = torch.tensor([[1.0, -1.0, 0.0]], dtype=torch.float32)
    return (speed, gear, rpm, img, prev1, prev2)


def main():
    print("[+] Verifying Adaptation Training & Deployment Pipeline...", flush=True)
    device = "cpu"
    foundation_path = ROOT / "weights" / "car_brain_1m_curriculum" / "car_brain_multimodal.pt"
    if not foundation_path.is_file():
        raise FileNotFoundError(f"Foundation checkpoint not found: {foundation_path}")

    obs_space = spaces.Tuple((
        spaces.Box(0.0, 1000.0, shape=(1,), dtype=np.float32),
        spaces.Box(0.0, 6.0, shape=(1,), dtype=np.float32),
        spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float32),
        spaces.Box(0.0, 255.0, shape=(cfg.IMG_HIST_LEN, cfg.IMG_HEIGHT, cfg.IMG_WIDTH), dtype=np.float32),
        spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
        spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
    ))
    act_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

    # 1. Instantiate Agent with FOUNDATION_ONLY=False
    print("[1] Instantiating TorchDreamerAgent with foundation_only=False...", flush=True)
    agent = TorchDreamerAgent(
        observation_space=obs_space,
        action_space=act_space,
        device=device,
        foundation_only=False,
        use_foundation_encoder=True,
        foundation_weights_path=str(foundation_path),
        freeze_foundation=True,
        lr_foundation=0.0,
        lr_actor=1e-3,
        residual_scale=0.5,
        reload_foundation_on_actor_load=True,
    )

    # 2. Get baseline actions on fixed observation
    obs = make_dummy_observation()
    agent.actor.eval()
    started = time.perf_counter()
    init_action = agent.actor.act(obs, test=True)
    latency_ms = (time.perf_counter() - started) * 1000.0
    print(f"    Initial action: {init_action.tolist()}, latency: {latency_ms:.2f} ms", flush=True)

    initial_digest = tensor_state_digest(agent.actor.state_dict())

    # 3. Simulate training updates on actor policy parameters
    print("[2] Executing synthetic gradient update on actor policy parameters...", flush=True)
    agent.actor.train()
    # Dummy loss to update policy heads
    feat = torch.randn(2, agent.hidden_dim + agent.latent_dim, device=device)
    action_pred, _ = agent.actor.forward_features(feat, test=False, with_logprob=True)
    loss = (action_pred ** 2).sum()
    agent.actor_optimizer.zero_grad()
    loss.backward()
    agent.actor_optimizer.step()

    updated_digest = tensor_state_digest(agent.actor.state_dict())
    assert initial_digest != updated_digest, "FATAL: Policy parameters did not change after optimizer step!"
    print(f"    Policy weights changed: digest before={initial_digest[:8]}..., after={updated_digest[:8]}...", flush=True)

    # 4. Save broadcast checkpoint
    out_dir = ROOT / "output" / "adaptation_benchmark" / "verification"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = out_dir / "test_broadcast_actor.pt"
    torch.save(agent.actor.state_dict(), checkpoint_file)
    ckpt_sha256 = hashlib.sha256(checkpoint_file.read_bytes()).hexdigest()
    print(f"[3] Saved broadcast checkpoint: {checkpoint_file.name} (SHA-256: {ckpt_sha256[:12]}...)", flush=True)

    # 5. Instantiate independent Worker Actor
    print("[4] Instantiating independent Worker Actor and loading broadcast...", flush=True)
    worker_actor = TorchDreamerActor(
        observation_space=obs_space,
        action_space=act_space,
        device=device,
        latent_dim=agent.latent_dim,
        hidden_dim=agent.hidden_dim,
        policy_hidden_dim=agent.policy_hidden_dim,
        img_channels=cfg.IMG_HIST_LEN,
        img_height=cfg.IMG_HEIGHT,
        img_width=cfg.IMG_WIDTH,
        use_foundation_encoder=True,
        foundation_weights_path=str(foundation_path),
        freeze_foundation=True,
        foundation_only=False,
        residual_scale=0.5,
        reload_foundation_on_actor_load=True,
    ).to(device)

    # Load broadcast into worker
    worker_actor.load(str(checkpoint_file), device=device)
    assert worker_actor.last_load_succeeded, f"FATAL: Worker load failed: {worker_actor.last_load_error}"

    worker_digest = tensor_state_digest(worker_actor.state_dict())
    assert worker_digest == updated_digest, "FATAL: Worker policy weights do not match trainer updated weights!"
    print("    Worker loaded updated weights successfully and digests match!", flush=True)

    # 6. Evaluate worker action on fixed observation
    print("[5] Evaluating worker actor on fixed observation...", flush=True)
    worker_actor.eval()
    started = time.perf_counter()
    new_action = worker_actor.act(obs, test=True)
    eval_latency_ms = (time.perf_counter() - started) * 1000.0

    action_diff = float(np.max(np.abs(new_action - init_action)))
    print(f"    Worker action after update: {new_action.tolist()}", flush=True)
    print(f"    Max absolute action shift: {action_diff:.6f}", flush=True)
    print(f"    Worker inference latency: {eval_latency_ms:.2f} ms", flush=True)

    assert action_diff > 1e-4, f"FATAL: Worker action did not change! Diff = {action_diff}"
    assert eval_latency_ms < 50.0, f"FATAL: Inference latency {eval_latency_ms:.2f} ms exceeded 50 ms deadline!"

    report = {
        "status": "PASS",
        "foundation_only_disabled": True,
        "trainer_optimizer_updates_policy": True,
        "checkpoint_sha256": ckpt_sha256,
        "worker_load_succeeded": True,
        "action_shift_magnitude": action_diff,
        "inference_latency_ms": eval_latency_ms,
        "deadline_passed": eval_latency_ms < 50.0,
        "initial_action": init_action.tolist(),
        "updated_worker_action": new_action.tolist(),
    }
    report_path = out_dir / "pipeline_verification_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[+] All verification checks PASSED! Report saved to {report_path}", flush=True)


if __name__ == "__main__":
    main()
