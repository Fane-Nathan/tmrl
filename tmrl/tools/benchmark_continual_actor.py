"""Measure warmed single-observation JAX Dreamer rollout latency."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any


def _write_json_atomic(path: Path | str, payload: dict[str, Any]) -> None:
    """Atomically write ``payload`` as formatted JSON to ``path``."""
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


def _load_runtime_deps(runtime: str) -> tuple[Any, Any, type[Any], Any | None]:
    """Lazily load only the dependencies needed by ``runtime`` ('torch' or 'jax')."""
    if runtime not in {"torch", "jax"}:
        raise ValueError(f"Unsupported runtime {runtime!r}; expected 'torch' or 'jax'")

    try:
        import numpy as np
        from gymnasium import spaces
    except ImportError as exc:
        raise RuntimeError(
            f"Actor benchmarking requires base RL dependencies (missing: {exc.name}). "
            "Install with `pip install tmrl`."
        ) from exc

    if runtime == "torch":
        try:
            import torch
            from tmrl.custom.torch.jax_dreamer_actor import TorchJAXDreamerActor
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch actor benchmarking requires PyTorch dependencies "
                f"(missing: {exc.name}). Install with `pip install \"tmrl[torch]\"`."
            ) from exc
        return np, spaces, TorchJAXDreamerActor, torch

    try:
        import jax  # noqa: F401
        from tmrl.custom.jax.continual_dreamer import JAXDreamerActor
    except ImportError as exc:
        raise RuntimeError(
            "JAX actor benchmarking requires JAX/Flax/Optax dependencies "
            f"(missing: {exc.name}). Install with `pip install \"tmrl[jax]\"` "
            "or `requirements/continual-jax-wsl2.txt`."
        ) from exc
    return np, spaces, JAXDreamerActor, None


def benchmark_actor(
    image_size: int = 96,
    image_history: int = 4,
    latent_dim: int = 128,
    hidden_dim: int = 256,
    policy_hidden_dim: int = 256,
    encoder_channels: tuple[int, ...] = (16, 32, 64, 64),
    iterations: int = 200,
    runtime: str = "torch",
    device: str | None = None,
    stochastic: bool = False,
    realtime_cpu_tuning: bool = False,
) -> dict[str, Any]:
    """Warm up and benchmark single-observation inference latency."""
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if latent_dim < 1 or hidden_dim < 1 or policy_hidden_dim < 1:
        raise ValueError("latent_dim, hidden_dim, and policy_hidden_dim must be >= 1")
    if len(encoder_channels) != 4 or any(value < 1 for value in encoder_channels):
        raise ValueError("encoder_channels must contain four positive integers")

    np, spaces, actor_cls, torch_mod = _load_runtime_deps(runtime)

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

    if device is None:
        if runtime == "torch":
            device = "cuda" if torch_mod.cuda.is_available() else "cpu"
        else:
            device = None

    actor_kwargs: dict[str, Any] = dict(
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
        "realtime_cpu_tuning": bool(realtime_cpu_tuning and runtime == "torch"),
        "latency_ms": {
            "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "max": float(values.max()),
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, run the actor latency benchmark, and report JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Number of timed inference steps after warmup (>= 1).",
    )
    parser.add_argument(
        "--runtime",
        choices=("torch", "jax"),
        default="torch",
        help="Actor runtime implementation to benchmark.",
    )
    parser.add_argument(
        "--device",
        help="Target device override (e.g., cpu or cuda).",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=128,
        help="Latent policy input width (>= 1).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="World-model recurrent hidden width (>= 1).",
    )
    parser.add_argument(
        "--policy-hidden-dim",
        type=int,
        default=256,
        help="Actor MLP hidden width (>= 1).",
    )
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
    parser.add_argument(
        "--max-p95-ms",
        type=float,
        help="Maximum allowed p95 latency in milliseconds (> 0).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path.",
    )
    args = parser.parse_args(argv)

    if args.iterations < 1:
        parser.error("--iterations must be >= 1")
    if args.latent_dim < 1:
        parser.error("--latent-dim must be >= 1")
    if args.hidden_dim < 1:
        parser.error("--hidden-dim must be >= 1")
    if args.policy_hidden_dim < 1:
        parser.error("--policy-hidden-dim must be >= 1")
    if args.max_p95_ms is not None and (
        not math.isfinite(args.max_p95_ms) or args.max_p95_ms <= 0
    ):
        parser.error("--max-p95-ms must be a finite number > 0")

    try:
        encoder_channels = tuple(
            int(value.strip()) for value in args.encoder_channels.split(",")
        )
    except ValueError:
        parser.error("--encoder-channels must contain four positive integers")
    if len(encoder_channels) != 4 or any(value < 1 for value in encoder_channels):
        parser.error("--encoder-channels must contain four positive integers")

    try:
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
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        print(f"Actor benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
