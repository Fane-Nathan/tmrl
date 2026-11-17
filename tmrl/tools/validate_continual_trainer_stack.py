"""Instantiate and serialize the configured JAX trainer without networking."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _sha256(path: Path | str) -> str:
    """Compute the lowercase SHA-256 hex digest of ``path``."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def validate(output_directory: Path | str) -> dict[str, Any]:
    """Instantiate the configured JAX trainer, verify checkpointing, and emit metadata."""
    try:
        import jax
    except ImportError as exc:
        raise RuntimeError(
            "CONTINUAL_DREAMER_JAX stack validation requires JAX/Flax/Optax "
            f"(missing: {exc.name}). Install with `pip install \"tmrl[jax]\"` "
            "or `requirements/continual-jax-wsl2.txt`."
        ) from exc

    import tmrl.config.config_constants as cfg
    import tmrl.config.config_objects as cfg_obj

    if not cfg_obj.IS_JAX_DREAMER:
        if not cfg.CONFIG_LOADED:
            raise RuntimeError(
                f"No TMRL config file was loaded from {cfg.CONFIG_FILE}; default "
                f"algorithm is {cfg_obj.ALG_NAME!r}, not CONTINUAL_DREAMER_JAX. "
                "Generate one with `python -m tmrl.tools.prepare_continual_jax_config` "
                "or set TMRL_CONFIG_PATH."
            )
        raise RuntimeError(
            f"Configured algorithm in {cfg.CONFIG_FILE} is {cfg_obj.ALG_NAME!r}, "
            "not CONTINUAL_DREAMER_JAX."
        )

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / "trainer_init.tcpt"
    actor_path = output_directory / "actor_init.tmod"
    manifest_path = output_directory / "trainer_stack.json"

    try:
        started = time.perf_counter()
        trainer = cfg_obj.TRAINER()
        initialization_seconds = time.perf_counter() - started

        checkpoint_started = time.perf_counter()
        cfg_obj.DUMP_RUN_INSTANCE_FN(trainer, checkpoint_path)
        restored = cfg_obj.LOAD_RUN_INSTANCE_FN(checkpoint_path)
        cfg_obj.UPDATER_FN(restored, cfg_obj.TRAINER)
        checkpoint_seconds = time.perf_counter() - checkpoint_started
        restored.agent.get_actor().save(actor_path)
    except ImportError as exc:
        extra_note = (
            " (CPU PyTorch is also required until continual replay is decoupled from torch)"
            if exc.name == "torch"
            else ""
        )
        raise RuntimeError(
            "CONTINUAL_DREAMER_JAX stack validation requires optional runtime "
            f"dependencies (missing: {exc.name}){extra_note}. Install with "
            "`pip install \"tmrl[jax]\"` or `requirements/continual-jax-wsl2.txt`."
        ) from exc

    payload = {
        "algorithm": cfg_obj.ALG_NAME,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "checkpoint_mode": restored.agent.checkpoint_mode,
        "trainer_class": type(restored).__name__,
        "memory_class": type(restored.memory).__name__,
        "agent_class": type(restored.agent).__name__,
        "initialization_seconds": initialization_seconds,
        "checkpoint_round_trip_seconds": checkpoint_seconds,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": _sha256(checkpoint_path),
        },
        "actor_bundle": {
            "path": str(actor_path.resolve()),
            "bytes": actor_path.stat().st_size,
            "sha256": _sha256(actor_path),
            "architecture": restored.agent.get_actor().architecture_manifest(),
        },
    }
    _write_json_atomic(manifest_path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, validate the continual JAX trainer stack, and print JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="Directory where trainer_init.tcpt, actor_init.tmod, and trainer_stack.json are written.",
    )
    args = parser.parse_args(argv)
    try:
        result = validate(args.output_directory)
    except Exception as exc:
        print(
            f"Trainer stack validation failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
