"""Per-action inference latency of the pilot controllers (reviewer request: 100-500 Hz control).

Measures the deployment call used in every rollout: TransformerRL2Agent.step_in_context (rolling
context of up to 32 tokens, full recomputation each step, no KV cache) and ReactiveREDQAgent.act.
Inputs are real model-input tokens replayed from saved evaluation trajectories; the simulator step
is excluded. Run on an otherwise idle machine.

  python benchmark_latency.py --trajectories PILOT_DIR --out latency.json
"""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from learner import load_agent

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
BUDGETS_MS = {"500 Hz": 2.0, "100 Hz": 10.0}


def cpu_name():
    try:
        out = subprocess.check_output(["powershell", "-NoProfile", "-Command",
                                       "(Get-CimInstance Win32_Processor).Name"], text=True, timeout=20)
        return out.strip() or platform.processor()
    except Exception:
        return platform.processor()


def tokens_for(trajectories, algorithm, seed):
    path = next(Path(trajectories).glob(f"{algorithm}_seed{seed}_sensor_delay_replay0/trajectory_0_clean_0.npz"))
    return np.load(path)["observations"].astype(np.float32)


def measure(model, algorithm, tokens, device, steps, warmup, episode_length):
    model = model.to(device)
    timings = []
    for i in range(warmup + steps):
        if algorithm == "transformer" and i % episode_length == 0:
            model.reset_context()
        token = tokens[i % len(tokens)]
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter_ns()
        if algorithm == "transformer":
            model.step_in_context(token, adaptive=True, deterministic=True, device=device)
        else:
            model.act(token, deterministic=True, device=device)
        elapsed = (time.perf_counter_ns() - start) / 1e6  # .cpu().numpy() in the call synchronizes
        if i >= warmup:
            timings.append(elapsed)
    return np.asarray(timings)


def summarize(ms, context_full_fraction):
    return {"steps": int(ms.size), "mean_ms": float(ms.mean()), "p50_ms": float(np.percentile(ms, 50)),
            "p95_ms": float(np.percentile(ms, 95)), "p99_ms": float(np.percentile(ms, 99)),
            "max_ms": float(ms.max()),
            **{f"fraction_over_{name.replace(' ', '')}": float((ms > limit).mean()) for name, limit in BUDGETS_MS.items()},
            "full_context_fraction": context_full_fraction}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True, help="Pilot output directory")
    parser.add_argument("--sources", type=Path, default=WORKSPACE / "fault_benchmark_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    episode_length = 1000
    devices = [("cpu", t) for t in args.threads] + ([("cuda", 1)] if torch.cuda.is_available() else [])
    results = []
    for algorithm in ["transformer", "reactive"]:
        checkpoint = args.sources / f"{algorithm}_seed_{args.seed}" / "best_validation_checkpoint.pt"
        tokens = tokens_for(args.trajectories, algorithm, args.seed)
        parameters = None
        for device, threads in devices:
            torch.set_num_threads(threads)
            model = load_agent(algorithm, checkpoint, context=32)
            if parameters is None:
                deploy = [p for n, p in model.named_parameters()
                          if not n.startswith(("critics", "target_", "task_encoder.classifier"))]
                parameters = {"total": sum(p.numel() for p in model.parameters()),
                              "inference_path_upper_bound": sum(p.numel() for p in deploy)}
            ms = measure(model, algorithm, tokens, device, args.steps, args.warmup, episode_length)
            full = 1.0 if algorithm == "reactive" else float(np.mean(
                [((args.warmup + i) % episode_length) >= 31 for i in range(args.steps)]))
            row = {"algorithm": algorithm, "device": device, "cpu_threads": threads,
                   "context": 32 if algorithm == "transformer" else 1, **summarize(ms, full),
                   "parameters": parameters}
            results.append(row)
            print(f"{algorithm:11s} {device:4s} threads={threads} mean={row['mean_ms']:.3f} "
                  f"p95={row['p95_ms']:.3f} p99={row['p99_ms']:.3f} max={row['max_ms']:.2f} ms", flush=True)
    report = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "what": "Per-action policy inference latency, batch 1, deterministic action; simulator excluded",
              "source_checkpoints_seed": args.seed, "steps": args.steps, "warmup": args.warmup,
              "budgets_ms": BUDGETS_MS, "cpu": cpu_name(),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "torch": torch.__version__, "python": platform.python_version(), "platform": platform.platform(),
              "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
