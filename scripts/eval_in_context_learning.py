#!/usr/bin/env python3
"""
Evaluation Harness for In-Context Learning (ICL) in TrackMania.

Replicates the paper's No-History Ablation:
- Compares Adaptive (full context history) vs. No-History (cleared deque every step).
- Computes the Adaptation Gap: G = R_adaptive - R_no_history.
- Calculates 95% Confidence Intervals across evaluation runs/seeds.
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure local repository root takes precedence
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from tmrl.custom.torch.car_brain import CarBrain


def compute_95_ci(data: List[float]) -> Tuple[float, float]:
    """Computes mean and half-width of 95% confidence interval using Student's t distribution."""
    n = len(data)
    if n < 2:
        return float(np.mean(data)), 0.0
    mean = float(np.mean(data))
    std_err = float(np.std(data, ddof=1) / math.sqrt(n))
    # Approximate t-critical value for 95% CI (1.96 for large n, ~2.78 for n=3)
    t_crit = 2.776 if n == 3 else (2.571 if n == 4 else 1.96)
    return mean, t_crit * std_err


def run_synthetic_eval(
    model: CarBrain,
    num_episodes: int = 10,
    episode_length: int = 150,
    state_dim: int = 15,
    action_dim: int = 3,
    device: str = "cpu",
    inject_latency: int = 0,
) -> Dict[str, float]:
    """
    Simulates evaluation episodes to compute the adaptation gap.
    Can inject latency to test cross-mechanism fault adaptation.
    """
    model.to(device)
    model.eval()

    def run_policy(adaptive: bool) -> List[float]:
        model.adaptive = adaptive
        returns = []

        for ep in range(num_episodes):
            model.reset_context()
            ep_return = 0.0
            # Latency buffer for action latency fault injection
            latency_buffer = [np.zeros(action_dim, dtype=np.float32) for _ in range(inject_latency + 1)]

            # Generate synthetic track state trajectory
            curr_state = np.random.randn(state_dim).astype(np.float32) * 0.5
            prev_action = np.zeros(action_dim, dtype=np.float32)

            for t in range(episode_length):
                # Predict action
                action = model.step_in_context(
                    state=curr_state,
                    prev_action=prev_action,
                    prev_reward=0.0,
                    prev_done=False,
                    device=torch.device(device),
                )

                # Action execution (with optional latency fault)
                latency_buffer.append(action.copy())
                executed_action = latency_buffer.pop(0)

                # Simulated track dynamics: progress reward with penalty for jerky steering
                gas, brake, steer = executed_action[0], executed_action[1], executed_action[2]
                speed = max(0.0, float(curr_state[0] + gas * 0.5 - brake * 0.8))
                steer_penalty = abs(steer) * 0.1
                step_reward = speed - steer_penalty
                ep_return += step_reward

                # Next state simulation
                curr_state = curr_state * 0.95 + np.random.randn(state_dim).astype(np.float32) * 0.1
                curr_state[0] = speed
                prev_action = executed_action

            returns.append(ep_return)
        return returns

    # Run Adaptive
    adaptive_returns = run_policy(adaptive=True)
    # Run No-History
    no_history_returns = run_policy(adaptive=False)

    ad_mean, ad_ci = compute_95_ci(adaptive_returns)
    nh_mean, nh_ci = compute_95_ci(no_history_returns)

    gaps = [a - nh for a, nh in zip(adaptive_returns, no_history_returns)]
    gap_mean, gap_ci = compute_95_ci(gaps)

    return {
        "adaptive_mean": ad_mean,
        "adaptive_ci": ad_ci,
        "no_history_mean": nh_mean,
        "no_history_ci": nh_ci,
        "gap_mean": gap_mean,
        "gap_ci": gap_ci,
        "raw_adaptive": adaptive_returns,
        "raw_no_history": no_history_returns,
        "raw_gaps": gaps,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate In-Context Learning on TrackMania Car Brain")
    parser.add_argument("--model_path", type=str, default=None, help="Path to car_brain.safetensors or car_brain.pt")
    parser.add_argument("--episodes", type=int, default=15, help="Number of evaluation episodes per condition")
    parser.add_argument("--episode_len", type=int, default=120, help="Timesteps per episode")
    parser.add_argument("--latency_fault", type=int, default=0, help="Action delay steps (0=no fault, 2=latency fault)")
    parser.add_argument("--output_file", type=str, default="./icl_eval_summary.json", help="Summary JSON output")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating In-Context Learning on {device}...")

    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading pretrained Car Brain from {args.model_path}...")
        model = CarBrain.from_pretrained(args.model_path, device=device)
    else:
        print("No pretrained path provided or file not found. Evaluating randomly initialized Car Brain...")
        model = CarBrain().to(device)

    print(f"Running {args.episodes} episodes per condition (Adaptive vs. No-History)...")
    if args.latency_fault > 0:
        print(f"Injecting cross-mechanism Action Latency fault: {args.latency_fault} steps delay.")

    results = run_synthetic_eval(
        model=model,
        num_episodes=args.episodes,
        episode_length=args.episode_len,
        device=device,
        inject_latency=args.latency_fault,
    )

    print("\n=======================================================")
    print("           IN-CONTEXT LEARNING EVALUATION              ")
    print("=======================================================")
    print(f"Adaptive Return:    {results['adaptive_mean']:.2f} ± {results['adaptive_ci']:.2f} (95% CI)")
    print(f"No-History Return:  {results['no_history_mean']:.2f} ± {results['no_history_ci']:.2f} (95% CI)")
    print("-------------------------------------------------------")
    print(f"Adaptation Gap (G): {results['gap_mean']:+.2f} ± {results['gap_ci']:.2f} (95% CI)")
    if results['gap_mean'] > 0:
        print(">> In-Context Adaptation Advantage Observed! Memory buys performance.")
    else:
        print(">> Memoryless baseline matches or exceeds adaptive (untrained/converging).")
    print("=======================================================\n")

    def to_py(val):
        if isinstance(val, (np.floating, float)):
            return float(val)
        if isinstance(val, (np.integer, int)):
            return int(val)
        if isinstance(val, list):
            return [to_py(v) for v in val]
        if isinstance(val, dict):
            return {k: to_py(v) for k, v in val.items()}
        return val

    with open(args.output_file, "w") as f:
        json.dump(to_py(results), f, indent=2)
    print(f"Saved evaluation report to {args.output_file}")


if __name__ == "__main__":
    main()
