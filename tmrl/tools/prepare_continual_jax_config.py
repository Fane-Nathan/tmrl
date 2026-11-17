"""Create a separate CONTINUAL_DREAMER_JAX config without touching a live run."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


DEFAULT_RUN_NAME = "Continual_Dreamer_JAX_M1_Stable_v2"


def build_config(
    source: dict[str, Any],
    run_name: str = DEFAULT_RUN_NAME,
    scrub_wandb_key: bool = False,
    trainer_server_ip: str | None = None,
) -> dict[str, Any]:
    """Return a derived ``CONTINUAL_DREAMER_JAX`` config dictionary from ``source``."""
    if not isinstance(source, dict):
        raise ValueError("source config must be a JSON object")
    if run_name is None or not str(run_name).strip():
        raise ValueError("run_name must be a non-empty string")
    if trainer_server_ip is not None and not str(trainer_server_ip).strip():
        raise ValueError("trainer_server_ip must be a non-empty string when provided")

    result = deepcopy(source)
    if not isinstance(result.get("ALG"), dict):
        raise ValueError("source config is missing the ALG object")
    if not isinstance(result.get("ENV"), dict):
        raise ValueError("source config is missing the ENV object")
    interface = str(result["ENV"].get("RTGYM_INTERFACE", "")).upper()
    if "LIDAR" in interface:
        raise ValueError("CONTINUAL_DREAMER_JAX M1 requires image observations")
    if not result["ENV"].get("IMG_GRAYSCALE", False):
        raise ValueError("CONTINUAL_DREAMER_JAX M1 requires grayscale images")

    result["RUN_NAME"] = str(run_name)
    if trainer_server_ip is not None:
        result["PUBLIC_IP_SERVER"] = str(trainer_server_ip)
        result["LOCALHOST_TRAINER"] = False
    result["ALG"]["ALGORITHM"] = "CONTINUAL_DREAMER_JAX"
    result["ALG"]["AZR_IMAGINATION"] = {"ENABLED": False}
    result["ALG"]["CONTINUAL_DREAMER"] = {
        "BATCH_SIZE": 8,
        "SEQUENCE_LENGTH": 16,
        "LR_WORLD_MODEL": 0.0001,
        "LR_ACTOR": 0.00003,
        "LR_CRITIC": 0.00003,
        "LAMBDA": 0.95,
        "FREE_NATS": 1.0,
        "HORIZON": 8,
        "IMAGINATION_BATCH_SIZE": 32,
        "BURN_IN": 5,
        "WARMUP_STEPS": 2000,
        "ENTROPY_SCALE": 0.0003,
        "TARGET_POLYAK": 0.99,
        "GRAD_CLIP": 10.0,
        "SEED": 0,
        "REPLAY_MEMORY": {
            "MODE": "uniform",
            "CANDIDATES": 128,
            "UNIFORM_FRACTION": 0.25,
            "EMBED_DIM": 16,
            "CURVATURE": 1.0,
            "TANGENT_SCALE": 0.75,
            "MAX_RADIUS": 0.95,
            "SEED": 0,
        },
        "FOUNDATION": {
            "LATENT_DIM": 128,
            "HIDDEN_DIM": 256,
            "POLICY_HIDDEN_DIM": 256,
            "ENCODER_CHANNELS": [16, 32, 64, 64],
            "WORKER_DEVICE": "cpu",
            "WORKER_REALTIME_CPU_TUNING": True,
            "WORKER_CPU_THREADS": 8,
            "WORKER_CPU_AFFINITY_COUNT": 8,
            "WORKER_HIGH_PRIORITY": True,
        },
        "EXPERTS": {"ENABLED": False},
    }
    if scrub_wandb_key:
        result["WANDB_KEY"] = ""
    return result


def write_json_atomic(
    path: Path | str,
    payload: dict[str, Any],
    overwrite: bool = False,
) -> None:
    """Atomically write ``payload`` as formatted JSON to ``path``."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to replace existing config without --overwrite: {path}"
        )
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
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, build the JAX Dreamer config, and write it atomically."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the base TMRL config.json file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination path for the generated CONTINUAL_DREAMER_JAX config.",
    )
    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help="Run identifier written to RUN_NAME in the output config.",
    )
    parser.add_argument(
        "--scrub-wandb-key",
        action="store_true",
        help="Clear WANDB_KEY in the generated config.",
    )
    parser.add_argument(
        "--trainer-server-ip",
        help="Windows relay address reachable from WSL2; disables LOCALHOST_TRAINER.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing file at --output.",
    )
    args = parser.parse_args(argv)

    try:
        with args.source.open("r", encoding="utf-8") as handle:
            source = json.load(handle)
        output = build_config(
            source,
            run_name=args.run_name,
            scrub_wandb_key=args.scrub_wandb_key,
            trainer_server_ip=args.trainer_server_ip,
        )
        write_json_atomic(args.output, output, overwrite=args.overwrite)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, FileExistsError) as exc:
        print(f"Config preparation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "run_name": output["RUN_NAME"],
                "algorithm": output["ALG"]["ALGORITHM"],
                "azr_enabled": output["ALG"]["AZR_IMAGINATION"]["ENABLED"],
                "experts_enabled": output["ALG"]["CONTINUAL_DREAMER"][
                    "EXPERTS"
                ]["ENABLED"],
                "wandb_key_scrubbed": bool(args.scrub_wandb_key),
                "trainer_server_ip": (
                    output["PUBLIC_IP_SERVER"]
                    if not output.get("LOCALHOST_TRAINER", True)
                    else "127.0.0.1"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
