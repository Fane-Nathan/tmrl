#!/usr/bin/env python3
"""Offline safety gate for the policy that will control the TMRL worker.

This replays recorded camera/telemetry observations through the exact
``TorchDreamerActor`` deployment path.  When a broadcast ``.tmod`` is supplied,
the audit also verifies that loading it cannot replace the canonical foundation
weights with a stale or Dreamer-modified copy.
"""

import argparse
from collections import deque
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tmrl.custom.torch.dreamer import TorchDreamerActor
from scripts.train_multitrack_vision_curriculum import select_episode_split


def tensor_state_digest(state_dict, prefix="encoder.brain."):
    digest = hashlib.sha256()
    for key in sorted(key for key in state_dict if key.startswith(prefix)):
        value = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def worker_observation(state, image, action_history):
    """Convert a recorded demonstration step to the worker's preprocessed form."""
    state = torch.as_tensor(state, dtype=torch.float32)
    image = torch.as_tensor(image)
    if image.dtype == torch.uint8 or float(image.max()) > 1.5:
        image = image.float() / 256.0
    else:
        # Demonstrations were recorded using /255 while the actual worker uses
        # /256.  Reproduce the deployed input exactly.
        image = image.float() * (255.0 / 256.0)
    action_history = tuple(
        torch.as_tensor(action, dtype=torch.float32).reshape(1, 3)
        for action in action_history
    )
    return (
        (state[0] / 10.0).reshape(1, 1),
        (state[3] / 10.0).reshape(1, 1),
        state[4].reshape(1, 1),
        image.unsqueeze(0),
        *action_history,
    )


def evaluate(args):
    checkpoint = Path(args.foundation).resolve()
    demo_path = Path(args.demos).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Foundation checkpoint not found: {checkpoint}")
    if not demo_path.is_file():
        raise FileNotFoundError(f"Demonstration dataset not found: {demo_path}")

    payload = torch.load(demo_path, map_location="cpu", weights_only=False)
    episodes = payload.get("episodes", [])
    if not episodes:
        raise ValueError(f"No episodes found in {demo_path}")
    if args.split != "all":
        episodes = select_episode_split(episodes, args.split)

    active_map_matches_source = None
    active_map_sha256 = None
    expected_map_sha256 = args.expected_map_sha256
    sidecar = demo_path.with_name(f"{demo_path.stem}.metadata.json")
    if expected_map_sha256 is None and sidecar.is_file():
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        expected_map_sha256 = metadata.get("map_sha256") or metadata.get("source_track", {}).get("map_sha256")
    if args.active_map:
        active_map = Path(args.active_map).resolve()
        if not active_map.is_file():
            raise FileNotFoundError(f"Active map not found: {active_map}")
        active_map_sha256 = file_digest(active_map)
        active_map_matches_source = bool(
            expected_map_sha256
            and active_map_sha256.lower() == expected_map_sha256.lower()
        )

    observation_space = spaces.Tuple(
        (
            spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            spaces.Box(0.0, 1.0, shape=(4, 96, 96), dtype=np.float32),
            spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
            spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
        )
    )
    action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    actor = TorchDreamerActor(
        observation_space=observation_space,
        action_space=action_space,
        device=args.device,
        use_foundation_encoder=True,
        foundation_weights_path=str(checkpoint),
        freeze_foundation=True,
        foundation_only=True,
        foundation_discrete_actions=args.discrete_actions,
        foundation_steer_threshold=args.steer_threshold,
        reload_foundation_on_actor_load=True,
    ).to(args.device)

    digest_before = tensor_state_digest(actor.state_dict())
    broadcast_loaded = None
    canonical_preserved = True
    if args.broadcast:
        broadcast = Path(args.broadcast).resolve()
        if not broadcast.is_file():
            raise FileNotFoundError(f"Dreamer broadcast not found: {broadcast}")
        actor.load(str(broadcast), device=args.device)
        broadcast_loaded = bool(actor.last_load_succeeded)
        canonical_preserved = digest_before == tensor_state_digest(actor.state_dict())

    predictions = []
    targets = []
    action_latencies_ms = []
    episode_counts = []
    for episode in episodes:
        images = episode["imgs"]
        states = episode["states"]
        actions = episode["actions"]
        available = min(len(images), len(states), len(actions))
        stop = available
        if args.max_steps_per_episode > 0:
            stop = min(stop, args.max_steps_per_episode)
        actor.reset()
        action_history = deque(
            (np.zeros(3, dtype=np.float32) for _ in range(2)), maxlen=2
        )
        count = 0
        with torch.no_grad():
            for index in range(0, stop, args.stride):
                obs = worker_observation(states[index], images[index], action_history)
                started = time.perf_counter()
                prediction = actor.act(obs, test=True)
                action_latencies_ms.append((time.perf_counter() - started) * 1000.0)
                target = np.asarray(actions[index], dtype=np.float32)
                predictions.append(np.asarray(prediction, dtype=np.float32))
                targets.append(target)
                # Deployment feeds the policy's own prior action back through
                # RTGym.  Teacher-forcing the expert target here hides
                # autoregressive drift and can pass a controller that crashes.
                action_history.append(
                    target
                    if args.teacher_forced_actions
                    else np.asarray(prediction, dtype=np.float32)
                )
                count += 1
        episode_counts.append(count)

    prediction = np.stack(predictions)
    target = np.stack(targets)
    steer_abs_error = np.abs(prediction[:, 2] - target[:, 2])
    turning = np.abs(target[:, 2]) >= args.turn_threshold
    straight = np.abs(target[:, 2]) <= args.straight_threshold
    turn_sign_accuracy = (
        float(np.mean(np.sign(prediction[turning, 2]) == np.sign(target[turning, 2])))
        if np.any(turning)
        else 0.0
    )
    straight_abs_steer = (
        float(np.mean(np.abs(prediction[straight, 2])))
        if np.any(straight)
        else float("inf")
    )
    metrics = {
        "passed": False,
        "foundation": str(checkpoint),
        "demos": str(demo_path),
        "foundation_sha256": file_digest(checkpoint),
        "demos_sha256": file_digest(demo_path),
        "split": args.split,
        "episodes": len(episodes),
        "evaluated_steps": int(len(target)),
        "steps_per_episode": episode_counts,
        "action_feedback": (
            "expert_teacher_forced"
            if args.teacher_forced_actions
            else "policy_autoregressive_fixed_observations"
        ),
        "action_history_order": "oldest_to_newest",
        "validates_closed_loop_driving": False,
        "discrete_actions": bool(args.discrete_actions),
        "steer_threshold": float(args.steer_threshold),
        "broadcast_loaded": broadcast_loaded,
        "canonical_foundation_preserved": canonical_preserved,
        "active_map_sha256": active_map_sha256,
        "expected_map_sha256": expected_map_sha256,
        "active_map_matches_source": active_map_matches_source,
        "steer_mae": float(np.mean(steer_abs_error)),
        "steer_p95_error": float(np.quantile(steer_abs_error, 0.95)),
        "turn_sign_accuracy": turn_sign_accuracy,
        "straight_abs_steer": straight_abs_steer,
        "gas_mae": float(np.mean(np.abs(prediction[:, 0] - target[:, 0]))),
        "brake_mae": float(np.mean(np.abs(prediction[:, 1] - target[:, 1]))),
        "mean_action_latency_ms": float(np.mean(action_latencies_ms)),
        "p95_action_latency_ms": float(np.quantile(action_latencies_ms, 0.95)),
        "control_deadline_miss_rate": float(
            np.mean(np.asarray(action_latencies_ms) > args.control_deadline_ms)
        ),
    }
    metrics["passed"] = bool(
        canonical_preserved
        and broadcast_loaded is not False
        and active_map_matches_source is not False
        and metrics["steer_mae"] <= args.max_steer_mae
        and metrics["gas_mae"] <= args.max_gas_mae
        and metrics["brake_mae"] <= args.max_brake_mae
        and metrics["turn_sign_accuracy"] >= args.min_turn_sign_accuracy
        and metrics["straight_abs_steer"] <= args.max_straight_abs_steer
        and metrics["p95_action_latency_ms"] <= args.max_p95_action_latency_ms
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Audit the exact foundation-only Dreamer deployment path"
    )
    parser.add_argument(
        "--foundation",
        default="weights/car_brain_1m_curriculum/car_brain_multimodal.pt",
    )
    parser.add_argument("--demos", default="data/target_track_demos.pt")
    parser.add_argument("--split", choices=("all", "train", "validation"), default="all",
                        help="Use validation for a candidate audit; never mix trained and held-out laps")
    parser.add_argument("--broadcast", default=None)
    parser.add_argument("--active-map", default=None)
    parser.add_argument("--expected-map-sha256", default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-steps-per-episode", type=int, default=0)
    parser.add_argument(
        "--teacher-forced-actions",
        action="store_true",
        help="diagnostic only: feed expert prior actions instead of policy outputs",
    )
    parser.add_argument(
        "--discrete-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="decode the keyboard-trained policy to binary/ternary controls",
    )
    parser.add_argument("--steer-threshold", type=float, default=0.05)
    parser.add_argument("--turn-threshold", type=float, default=0.10)
    parser.add_argument("--straight-threshold", type=float, default=0.05)
    parser.add_argument("--max-steer-mae", type=float, default=0.10)
    parser.add_argument("--max-gas-mae", type=float, default=0.30)
    parser.add_argument("--max-brake-mae", type=float, default=0.10)
    parser.add_argument("--min-turn-sign-accuracy", type=float, default=0.90)
    parser.add_argument("--max-straight-abs-steer", type=float, default=0.10)
    parser.add_argument("--control-deadline-ms", type=float, default=50.0)
    parser.add_argument("--max-p95-action-latency-ms", type=float, default=40.0)
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be at least 1")

    metrics = evaluate(args)
    print(json.dumps(metrics, indent=2))
    if args.json_path:
        output = Path(args.json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if metrics["passed"] else 2)


if __name__ == "__main__":
    main()
