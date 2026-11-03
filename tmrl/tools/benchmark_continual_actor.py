"""Measure warmed single-observation JAX Dreamer rollout latency."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from gymnasium import spaces

from tmrl.custom.jax.continual_dreamer import JAXDreamerActor
from tmrl.custom.torch.jax_dreamer_actor import TorchJAXDreamerActor


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def benchmark_actor(
    image_size=96,
    image_history=4,
    latent_dim=128,
    hidden_dim=256,
    policy_hidden_dim=256,
    encoder_channels=(16, 32, 64, 64),
    iterations=200,
    runtime="torch",
    device=None,
    stochastic=False,
    realtime_cpu_tuning=False,
):
    observation_space = spaces.Tuple(
        (
            spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            spaces.Box(
                0.0,
                1.0,
                (image_history, image_size, image_size),
                dtype=np.float32,
            ),
            spaces.Box(-1.0, 1.0, (3,), dtype=np.float32),
            spaces.Box(-1.0, 1.0, (3,), dtype=np.float32),
        )
    )
    action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
    actor_cls = TorchJAXDreamerActor if runtime == "torch" else JAXDreamerActor
    if device is None:
        device = "cuda" if runtime == "torch" and torch.cuda.is_available() else "cpu"
    actor_kwargs = dict(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        policy_hidden_dim=policy_hidden_dim,
        encoder_channels=encoder_channels,
        img_channels=image_history,
        img_height=image_size,
        img_width=image_size,
    )
    if runtime == "torch":
        actor_kwargs["realtime_cpu_tuning"] = realtime_cpu_tuning
    actor = actor_cls(observation_space, action_space, **actor_kwargs).to_device(device)
    observation = tuple(
        np.zeros(space.shape, dtype=space.dtype) for space in observation_space
    )
    compile_started = time.perf_counter()
    actor.warmup()
    compile_seconds = time.perf_counter() - compile_started

    durations = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        actor.act_(observation, test=not stochastic)
        durations.append((time.perf_counter_ns() - started) / 1e6)
    values = np.asarray(durations)
    return {
        "iterations": int(iterations),
        "compile_seconds": compile_seconds,
        "architecture": {
            "latent_dim": int(latent_dim),
            "hidden_dim": int(hidden_dim),
            "policy_hidden_dim": int(policy_hidden_dim),
            "encoder_channels": [int(value) for value in encoder_channels],
        },
        "stochastic": bool(stochastic),
        "realtime_cpu_tuning": bool(realtime_cpu_tuning),
        "latency_ms": {
            "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "max": float(values.max()),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--runtime", choices=("torch", "jax"), default="torch")
    parser.add_argument("--device")
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--policy-hidden-dim", type=int, default=256)
    parser.add_argument(
        "--encoder-channels",
        default="16,32,64,64",
        help="Four comma-separated visual encoder widths.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Benchmark the exploratory policy path used by train episodes.",
    )
    parser.add_argument(
        "--realtime-cpu-tuning",
        action="store_true",
        help="Apply the same CPU threads, affinity, and priority tuning as the worker.",
    )
    parser.add_argument("--max-p95-ms", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    encoder_channels = tuple(int(value) for value in args.encoder_channels.split(","))
    if len(encoder_channels) != 4 or any(value < 1 for value in encoder_channels):
        parser.error("--encoder-channels must contain four positive integers")
    result = benchmark_actor(
        iterations=args.iterations,
        runtime=args.runtime,
        device=args.device,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        policy_hidden_dim=args.policy_hidden_dim,
        encoder_channels=encoder_channels,
        stochastic=args.stochastic,
        realtime_cpu_tuning=args.realtime_cpu_tuning,
    )
    result["runtime"] = args.runtime
    if args.max_p95_ms is not None:
        result["max_p95_ms"] = args.max_p95_ms
        result["passed"] = result["latency_ms"]["p95"] <= args.max_p95_ms
    else:
        result["passed"] = True
    if args.output is not None:
        _write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
