#!/usr/bin/env python3
"""
Independent Reactive Robust Baseline (MLP REDQ-SAC without Transformer/History).

Purpose:
Answers the question:
"Could a separately trained reactive robust policy achieve the same performance as the adaptive sequence agent?"

Specifications:
- No transformer, no sequence replay, no context deque across time.
- Input is the same single augmented token used by the sequence agent:
  [s_t, a_{t-1}, r_{t-1}, d_{t-1}] (25-dim in HalfCheetah-v5).
  It has no multi-step context.
- REDQ critic ensemble (N=10, M=2) with LayerNorm to match capacity and stability.
- Trained on the exact same randomized fault distribution (action_scale + action_noise).
- Clean validation-selected checkpointing (saving best validation checkpoint).
- Evaluated on identical held-out test splits (action_latency, dead_actuator, sign_flip).
- Exports raw episode returns to JSON for statistical comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark_fault_env import FaultInjectionWrapper, make_fault_env
from scripts.confirmatory_logging import (
    compute_file_sha256,
    setup_run_directory,
    update_manifest,
)


class MLPGaussianActor(nn.Module):
    def __init__(self, obs_dim: int = 25, act_dim: int = 6, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)

    def forward(self, x: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        feat = self.net(x)
        mu = self.mu(feat)
        log_std = torch.clamp(self.log_std(feat), -20, 2)
        std = torch.exp(log_std)
        dist = Normal(mu, std)

        if deterministic:
            raw_act = mu
            log_prob = None
        else:
            raw_act = dist.rsample()
            log_prob = dist.log_prob(raw_act).sum(dim=-1, keepdim=True)
            log_prob -= (2.0 * (math.log(2.0) - raw_act - F.softplus(-2.0 * raw_act))).sum(dim=-1, keepdim=True)

        action = torch.tanh(raw_act)
        return action, log_prob


class MLPCritic(nn.Module):
    def __init__(self, obs_dim: int = 25, act_dim: int = 6, hidden_dim: int = 256, use_layernorm: bool = True):
        super().__init__()
        layers = []
        layers.append(nn.Linear(obs_dim + act_dim, hidden_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1))


class ReactiveREDQAgent(nn.Module):
    def __init__(
        self,
        obs_dim: int = 25,
        act_dim: int = 6,
        num_critics: int = 10,
        subset_m: int = 2,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_critics = num_critics
        self.subset_m = subset_m

        self.actor = MLPGaussianActor(obs_dim, act_dim)
        self.critics = nn.ModuleList([
            MLPCritic(obs_dim, act_dim, use_layernorm=use_layernorm)
            for _ in range(num_critics)
        ])
        self.target_critics = nn.ModuleList([
            MLPCritic(obs_dim, act_dim, use_layernorm=use_layernorm)
            for _ in range(num_critics)
        ])
        self.target_critics.load_state_dict(self.critics.state_dict())
        for p in self.target_critics.parameters():
            p.requires_grad = False

    def act(self, obs: np.ndarray, deterministic: bool = True, device: str = "cpu") -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            act, _ = self.actor(obs_t, deterministic=deterministic)
            return act[0].cpu().numpy()


class TransitionReplayBuffer:
    def __init__(self, capacity: int = 1_000_000, obs_dim: int = 25, act_dim: int = 6):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rews = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs: np.ndarray, act: np.ndarray, rew: float, next_obs: np.ndarray, done: bool):
        self.obs[self.ptr] = obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: str = "cpu") -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx], dtype=torch.float32, device=device),
            "acts": torch.as_tensor(self.acts[idx], dtype=torch.float32, device=device),
            "rews": torch.as_tensor(self.rews[idx], dtype=torch.float32, device=device),
            "next_obs": torch.as_tensor(self.next_obs[idx], dtype=torch.float32, device=device),
            "dones": torch.as_tensor(self.dones[idx], dtype=torch.float32, device=device),
        }


def evaluate_reactive_policy(
    agent: ReactiveREDQAgent,
    split: str,
    episodes: int = 25,
    device: str = "cpu",
    base_seed: int = 50000,
) -> Tuple[float, List[float], List[int]]:
    agent.eval()
    env = make_fault_env("HalfCheetah-v5", split=split, seed=base_seed)
    returns: List[float] = []
    lengths: List[int] = []

    for ep in range(episodes):
        ep_seed = base_seed + ep * 100
        obs, _ = env.reset(seed=ep_seed)
        ep_ret = 0.0
        ep_len = 0
        term, trunc = False, False

        while not (term or trunc):
            act = agent.act(obs, deterministic=True, device=device)
            obs, rew, term, trunc, _ = env.step(act)
            ep_ret += rew
            ep_len += 1
        returns.append(float(ep_ret))
        lengths.append(int(ep_len))

    env.close()
    return float(np.mean(returns)), returns, lengths


def train_reactive_baseline(
    seed: int = 42,
    iterations: int = 100,
    collection_episodes_per_iter: int = 4,
    gradient_steps_per_iter: int = 100,
    utd: int = 10,
    batch_size: int = 64,
    use_layernorm: bool = True,
    val_interval: int = 10,
    val_episodes: int = 10,
    test_episodes: int = 25,
    base_eval_seed: int = 50000,
    num_threads: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "./fault_benchmark_results",
    config_path: Optional[str] = None,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    manifest_path = Path(output_dir) / "confirmatory_manifest.csv"
    os.makedirs(output_dir, exist_ok=True)

    config_dict: Dict[str, Any] = {}
    protocol_version = "unversioned"
    protocol_sha256 = ""
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        protocol_version = str(config_dict.get("protocol_version", "unversioned"))
        protocol_sha256 = compute_file_sha256(config_path)
        budget = config_dict.get("training_budget", {})
        iterations = budget.get("iterations", iterations)
        collection_episodes_per_iter = budget.get("collection_episodes_per_iter", collection_episodes_per_iter)
        gradient_steps_per_iter = budget.get("gradient_steps_per_iter", gradient_steps_per_iter)
        utd = budget.get("utd", utd)
        val_interval = budget.get("val_interval", val_interval)
        val_episodes = budget.get("val_episodes", val_episodes)
        eval_proto = config_dict.get("evaluation_protocol", {})
        test_episodes = eval_proto.get("episodes_per_condition", test_episodes)
        base_eval_seed = eval_proto.get("base_eval_seed", base_eval_seed)
        opt_cfg = config_dict.get("optimization", {})
        batch_size = opt_cfg.get("batch_size", batch_size)
        base_cfg = config_dict.get("reactive_baseline_architecture", {})
        use_layernorm = base_cfg.get("use_layernorm", use_layernorm)

    if smoke_test:
        iterations = 2
        collection_episodes_per_iter = 1
        gradient_steps_per_iter = 5
        utd = 2
        val_episodes = 2
        test_episodes = 2
        val_interval = 2

    # Reproducibility run directory
    run_name = f"reactive_seed_{seed}"
    run_config = {
        "algorithm": "reactive_robust_baseline",
        "seed": seed,
        "iterations": iterations,
        "collection_episodes_per_iter": collection_episodes_per_iter,
        "gradient_steps_per_iter": gradient_steps_per_iter,
        "utd": utd,
        "batch_size": batch_size,
        "use_layernorm": use_layernorm,
        "val_interval": val_interval,
        "val_episodes": val_episodes,
        "test_episodes": test_episodes,
        "base_eval_seed": base_eval_seed,
        "device": device,
        "protocol_version": protocol_version,
        "protocol_sha256": protocol_sha256,
    }
    run_dir = setup_run_directory(output_dir, run_name, run_config, seed, REPO_ROOT)

    train_start_str = time.strftime("%Y-%m-%d %H:%M:%S")
    update_manifest(manifest_path, {
        "algorithm": "reactive_robust_baseline",
        "seed": seed,
        "status": "TRAINING",
        "train_start": train_start_str,
        "train_end": "",
        "training_iterations": iterations,
        "environment_steps": 0,
        "validation_checkpoint": "",
        "checkpoint_sha256": "",
        "validation_score": "",
        "test_completed": "False",
        "raw_results_file": "",
        "notes": "Reactive MLP REDQ baseline (memoryless)",
    })

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_env = make_fault_env("HalfCheetah-v5", split="train", seed=seed)
    obs_dim = train_env.observation_space.shape[0]
    act_dim = train_env.action_space.shape[0]

    agent = ReactiveREDQAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        num_critics=10,
        subset_m=2,
        use_layernorm=use_layernorm,
    ).to(device)

    actor_opt = torch.optim.Adam(agent.actor.parameters(), lr=3e-4)
    critic_opt = torch.optim.Adam(agent.critics.parameters(), lr=1.5e-4)

    target_entropy = -float(act_dim)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_opt = torch.optim.Adam([log_alpha], lr=3e-4)

    replay = TransitionReplayBuffer(capacity=1_000_000, obs_dim=obs_dim, act_dim=act_dim)

    best_val_score = -float("inf")
    best_val_iter = 0
    best_checkpoint_path = run_dir / "best_validation_checkpoint.pt"
    top_checkpoint_path = Path(output_dir) / f"reactive_best_val_seed_{seed}.pt"
    val_history = []
    total_env_steps = 0

    val_metrics_file = run_dir / "validation_metrics.csv"
    with open(val_metrics_file, "w", newline="", encoding="utf-8") as f:
        f.write("iteration,val_return,best_so_far\n")

    print(f"[{time.strftime('%X')}] Starting Reactive Seed {seed} | Device: {device} | LayerNorm: {use_layernorm}", flush=True)

    first_training_reset = True
    for it in range(1, iterations + 1):
        t_it_start = time.time()
        agent.train()
        for _ in range(collection_episodes_per_iter):
            if first_training_reset:
                obs, _ = train_env.reset(seed=seed)
                first_training_reset = False
            else:
                obs, _ = train_env.reset()
            term, trunc = False, False

            while not (term or trunc):
                act = agent.act(obs, deterministic=False, device=device)
                next_obs, rew, term, trunc, _ = train_env.step(act)
                replay.add(obs, act, rew, next_obs, term)
                obs = next_obs
                total_env_steps += 1

        if replay.size >= batch_size:
            for _ in range(gradient_steps_per_iter):
                for _ in range(utd):
                    batch = replay.sample(batch_size, device=device)
                    obs_b = batch["obs"]
                    act_b = batch["acts"]
                    rew_b = batch["rews"]
                    next_obs_b = batch["next_obs"]
                    done_b = batch["dones"]

                    with torch.no_grad():
                        next_act_b, next_log_pi = agent.actor(next_obs_b, deterministic=False)
                        sampled_critics = random.sample(range(agent.num_critics), agent.subset_m)
                        q_targets = [agent.target_critics[idx](next_obs_b, next_act_b) for idx in sampled_critics]
                        q_min = torch.min(torch.stack(q_targets, dim=0), dim=0).values
                        alpha = log_alpha.exp().detach()
                        target_y = rew_b + (1.0 - done_b) * 0.99 * (q_min - alpha * next_log_pi)

                    critic_losses = [F.mse_loss(critic(obs_b, act_b), target_y) for critic in agent.critics]
                    total_critic_loss = sum(critic_losses)

                    critic_opt.zero_grad(set_to_none=True)
                    total_critic_loss.backward()
                    nn.utils.clip_grad_norm_(agent.critics.parameters(), 10.0)
                    critic_opt.step()

                    with torch.no_grad():
                        for param, target_param in zip(agent.critics.parameters(), agent.target_critics.parameters()):
                            target_param.data.mul_(0.995).add_(param.data, alpha=0.005)

                # Actor & Alpha Update
                pred_act, log_pi = agent.actor(obs_b, deterministic=False)
                q1 = agent.critics[0](obs_b, pred_act)
                q2 = agent.critics[1](obs_b, pred_act)
                q_min = torch.min(q1, q2)

                alpha = log_alpha.exp()
                actor_loss = (alpha.detach() * log_pi - q_min).mean()

                actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(agent.actor.parameters(), 10.0)
                actor_opt.step()

                alpha_loss = -(log_alpha * (log_pi.detach() + target_entropy)).mean()
                alpha_opt.zero_grad(set_to_none=True)
                alpha_loss.backward()
                alpha_opt.step()

        # Validation on VALIDATION SPLIT ONLY
        if it % val_interval == 0 or it == iterations:
            base_validation_seed = int(
                config_dict.get("evaluation_protocol", {}).get("base_validation_seed", 40000)
            )
            val_score, _, _ = evaluate_reactive_policy(
                agent,
                split="val",
                episodes=val_episodes,
                device=device,
                base_seed=base_validation_seed,
            )
            val_history.append({"iter": it, "val_return": val_score})
            improved = val_score > best_val_score
            if improved:
                best_val_score = val_score
                best_val_iter = it
                torch.save(agent.state_dict(), str(best_checkpoint_path))
                torch.save(agent.state_dict(), str(top_checkpoint_path))

            with open(val_metrics_file, "a", newline="", encoding="utf-8") as f:
                f.write(f"{it},{val_score:.4f},{best_val_score:.4f}\n")

            print(f"  Iter {it:3d}/{iterations} ({time.time()-t_it_start:.1f}s) | Reactive Val Return: {val_score:+.2f} {'[BEST]' if improved else ''}", flush=True)
        else:
            print(f"  Iter {it:3d}/{iterations} ({time.time()-t_it_start:.1f}s) | Reactive step complete", flush=True)

    train_end_str = time.strftime("%Y-%m-%d %H:%M:%S")

    # FINAL TEST EVALUATION on held-out test splits
    print(f"\n[{time.strftime('%X')}] Finalizing Reactive Seed {seed}: Evaluating on held-out test splits...", flush=True)
    if os.path.exists(best_checkpoint_path):
        agent.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
    elif os.path.exists(top_checkpoint_path):
        agent.load_state_dict(torch.load(top_checkpoint_path, map_location=device))

    checkpoint_hash = compute_file_sha256(best_checkpoint_path) if os.path.exists(best_checkpoint_path) else "NO_CHECKPOINT_HASH"

    checkpoint_metadata = {
        "seed": seed,
        "best_iteration": best_val_iter,
        "best_val_score": best_val_score,
        "checkpoint_sha256": checkpoint_hash,
        "frozen_timestamp": train_end_str,
        "selection_rule": "argmax_validation_score",
        "validation_split": "unseen parameters of action_scale & action_noise",
        "protocol_version": protocol_version,
        "protocol_sha256": protocol_sha256,
    }
    with open(run_dir / "checkpoint_metadata.json", "w") as f:
        json.dump(checkpoint_metadata, f, indent=2)

    raw_episodes_csv = run_dir / "final_test_raw_episodes.csv"
    with open(raw_episodes_csv, "w", newline="", encoding="utf-8") as f:
        f.write("algorithm,seed,fault_condition,mode,episode_id,environment_seed,return,episode_length,checkpoint_hash\n")

    test_results: Dict[str, Any] = {}
    for condition in ["test_latency", "test_dead", "test_sign_flip"]:
        score, raw, lens = evaluate_reactive_policy(
            agent, split=condition, episodes=test_episodes, device=device, base_seed=base_eval_seed
        )
        test_results[condition] = {
            "mean_return": score,
            "raw_returns": raw,
            "raw_lengths": lens,
        }

        with open(raw_episodes_csv, "a", newline="", encoding="utf-8") as f:
            for ep_idx in range(len(raw)):
                ep_s = base_eval_seed + ep_idx * 100
                f.write(f"reactive_robust_baseline,{seed},{condition},reactive,{ep_idx},{ep_s},{raw[ep_idx]:.4f},{lens[ep_idx]},{checkpoint_hash}\n")

        endpoint_tag = "PRIMARY ENDPOINT" if condition == "test_latency" else "SECONDARY ENDPOINT"
        print(f"  [{endpoint_tag}] Reactive Condition '{condition:15s}' | Return: {score:+.2f}", flush=True)

    summary = {
        "algorithm": "reactive_robust_baseline",
        "seed": seed,
        "protocol_version": protocol_version,
        "protocol_sha256": protocol_sha256,
        "observation_definition": "single_augmented_token_s_a_prev_r_prev_d_prev",
        "use_layernorm": use_layernorm,
        "best_val_score": best_val_score,
        "best_val_iter": best_val_iter,
        "checkpoint_sha256": checkpoint_hash,
        "val_history": val_history,
        "test_results": test_results,
    }

    with open(run_dir / "final_test_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    report_path = Path(output_dir) / f"reactive_report_seed_{seed}.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    update_manifest(manifest_path, {
        "algorithm": "reactive_robust_baseline",
        "seed": seed,
        "status": "COMPLETE",
        "train_start": train_start_str,
        "train_end": train_end_str,
        "training_iterations": iterations,
        "environment_steps": total_env_steps,
        "validation_checkpoint": str(best_checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "validation_score": round(best_val_score, 4),
        "test_completed": "True",
        "raw_results_file": str(raw_episodes_csv),
        "notes": f"Primary latency return = {test_results['test_latency']['mean_return']:+.4f}",
    })

    print(f"Saved reactive baseline report to {report_path}\n", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Reactive Robust REDQ Baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default=None, help="Path to confirmatory_protocol.json")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--collection_episodes_per_iter", type=int, default=4)
    parser.add_argument("--gradient_steps_per_iter", type=int, default=100)
    parser.add_argument("--utd", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_interval", type=int, default=10)
    parser.add_argument("--val_episodes", type=int, default=10)
    parser.add_argument("--test_episodes", type=int, default=25)
    parser.add_argument("--num_threads", type=int, default=4, help="Number of CPU threads per process")
    parser.add_argument("--fast", action="store_true", help="Fast training for development/benchmark verification")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./fault_benchmark_results")
    args = parser.parse_args()

    if args.fast:
        iters = 20
        coll = 2
        grads = 25
        utd = 4
        val_int = 5
        val_ep = 5
        test_ep = 25
    else:
        iters = args.iterations
        coll = args.collection_episodes_per_iter
        grads = args.gradient_steps_per_iter
        utd = args.utd
        val_int = args.val_interval
        val_ep = args.val_episodes
        test_ep = args.test_episodes

    train_reactive_baseline(
        seed=args.seed,
        iterations=iters,
        collection_episodes_per_iter=coll,
        gradient_steps_per_iter=grads,
        utd=utd,
        batch_size=args.batch_size,
        val_interval=val_int,
        val_episodes=val_ep,
        test_episodes=test_ep,
        num_threads=args.num_threads,
        config_path=args.config,
        smoke_test=args.smoke_test,
        output_dir=args.output_dir,
    )
