"""Create a separate CONTINUAL_DREAMER_JAX config without touching a live run."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile


DEFAULT_RUN_NAME = "Continual_Dreamer_JAX_M1_Stable_v2"


def build_config(
    source,
    run_name=DEFAULT_RUN_NAME,
    scrub_wandb_key=False,
    trainer_server_ip=None,
):
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


def write_json_atomic(path, payload, overwrite=False):
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--scrub-wandb-key", action="store_true")
    parser.add_argument(
        "--trainer-server-ip",
        help="Windows relay address reachable from WSL2; disables LOCALHOST_TRAINER.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    with args.source.open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    output = build_config(
        source,
        run_name=args.run_name,
        scrub_wandb_key=args.scrub_wandb_key,
        trainer_server_ip=args.trainer_server_ip,
    )
    write_json_atomic(args.output, output, overwrite=args.overwrite)
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
