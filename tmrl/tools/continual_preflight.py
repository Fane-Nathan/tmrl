"""Standalone environment preflight for the continual JAX trainer.

Run this file directly so the check does not import ``tmrl.__init__`` and does
not require a TrackMania ``TmrlData`` directory inside WSL2.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


PACKAGE_NAMES = ("jax", "jaxlib", "flax", "optax", "numpy")

# This tool is intentionally safe to run alongside a baseline process. Full
# trainer runs can choose their own allocator policy explicitly.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _run_command(command: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "returncode": None, "output": str(exc)}
    output = (completed.stdout or completed.stderr).strip()
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "output": output,
    }


def _git_state(repo_root: Path) -> dict[str, Any]:
    git = ["git", "-c", "core.autocrlf=true"]
    commit = _run_command([*git, "rev-parse", "HEAD"], cwd=repo_root)
    status = _run_command([*git, "status", "--short"], cwd=repo_root)
    return {
        "commit": commit["output"] if commit["ok"] else None,
        "dirty": bool(status["output"]) if status["ok"] else None,
        "status": status["output"].splitlines() if status["ok"] else [],
    }


def _is_wsl2() -> bool:
    try:
        version = Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False
    return "microsoft" in version and "wsl2" in version


def _tiny_nnx_train_step() -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx

    class TinyRegressor(nnx.Module):
        def __init__(self, rngs: nnx.Rngs):
            self.hidden = nnx.Linear(4, 8, rngs=rngs)
            self.output = nnx.Linear(8, 1, rngs=rngs)

        def __call__(self, x):
            return self.output(jax.nn.silu(self.hidden(x)))

    model = TinyRegressor(nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)
    x = jnp.arange(64, dtype=jnp.float32).reshape(16, 4) / 64.0
    y = jnp.sum(x, axis=-1, keepdims=True)

    def loss_value(module):
        return jnp.mean(jnp.square(module(x) - y))

    @nnx.jit
    def train_step(module, opt):
        loss, gradients = nnx.value_and_grad(loss_value)(module)
        opt.update(module, gradients)
        return loss

    initial_loss = float(loss_value(model))
    started = time.perf_counter()
    losses = []
    for _ in range(8):
        loss = train_step(model, optimizer)
        loss.block_until_ready()
        losses.append(float(loss))
    elapsed = time.perf_counter() - started
    final_loss = float(loss_value(model))
    return {
        "ok": final_loss < initial_loss,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "compiled_steps": len(losses),
        "compile_and_train_seconds": elapsed,
    }


def collect_preflight(repo_root: Path | None = None) -> dict[str, Any]:
    import jax

    repo_root = repo_root or Path(__file__).resolve().parents[2]
    devices = [
        {
            "id": int(device.id),
            "platform": device.platform,
            "device_kind": device.device_kind,
        }
        for device in jax.devices()
    ]
    backend = jax.default_backend()
    nvidia = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    nnx_step = _tiny_nnx_train_step()
    return {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "executable": sys.executable,
            "is_wsl2": _is_wsl2(),
        },
        "packages": _package_versions(),
        "jax": {
            "backend": backend,
            "devices": devices,
            "gpu_available": any(device["platform"] == "gpu" for device in devices),
        },
        "nvidia_smi": nvidia,
        "nnx_train_step": nnx_step,
        "git": _git_state(repo_root),
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    tmp_path = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON artifact path.")
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail unless JAX selects a GPU backend.",
    )
    args = parser.parse_args(argv)

    try:
        payload = collect_preflight()
    except Exception as exc:
        print(f"Preflight failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    failures = []
    if not payload["nnx_train_step"]["ok"]:
        failures.append("NNX optimizer step did not reduce the tiny regression loss")
    if args.require_gpu and not payload["jax"]["gpu_available"]:
        failures.append("JAX did not report a GPU device")
    payload["passed"] = not failures
    payload["failures"] = failures

    if args.output is not None:
        _atomic_write_json(args.output.resolve(), payload)

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
