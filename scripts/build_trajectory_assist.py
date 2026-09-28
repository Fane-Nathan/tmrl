#!/usr/bin/env python3
"""Build a compact deployment trajectory from a recorded demonstration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", default="data/target_track_demos.pt")
    parser.add_argument(
        "--episode",
        type=int,
        default=-1,
        help="Episode index; negative values count from the end (default: last episode)",
    )
    parser.add_argument(
        "--output", default="data/target_track_trajectory_assist.npz"
    )
    args = parser.parse_args()

    source = Path(args.demos).resolve()
    output = Path(args.output).resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    episodes = payload.get("episodes", [])
    episode_index = args.episode if args.episode >= 0 else len(episodes) + args.episode
    if not 0 <= episode_index < len(episodes):
        parser.error(
            f"--episode must select one of {len(episodes)} episodes"
        )
    episode = episodes[episode_index]
    states = np.asarray(episode["states"], dtype=np.float32)
    actions = np.asarray(episode["actions"], dtype=np.float32)
    if states.ndim != 2 or states.shape[1] < 9:
        raise ValueError(f"Expected states shaped [T, >=9], got {states.shape}")
    if actions.shape != (len(states), 3):
        raise ValueError(
            f"Expected actions shaped {(len(states), 3)}, got {actions.shape}"
        )
    positions = states[:, 6:9]
    speeds_mps = states[:, 0] * 100.0
    if not all(np.isfinite(array).all() for array in (positions, speeds_mps, actions)):
        raise ValueError("Trajectory contains non-finite values")

    metadata = {
        "schema_version": 2,
        "source": str(source),
        "source_sha256": sha256(source),
        "episode_index": episode_index,
        "steps": int(len(states)),
        "control_hz": 20.0,
        "action_format": "gas_brake_steer_ternary",
        "speed_unit": "m/s",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        positions=positions,
        actions=actions,
        speeds_mps=speeds_mps.astype(np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps({**metadata, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
