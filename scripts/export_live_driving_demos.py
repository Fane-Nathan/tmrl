#!/usr/bin/env python3
"""Export verified completed live runs as explicitly assisted BC demonstrations.

Images precede their target command. Entire runs, never overlapping windows,
are held out for validation. This data is known-track teacher data, not proof
of learned driving or cross-track one-shot generalization.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified_run(result_path):
    result_path = Path(result_path).resolve()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema_version", 0) < 2:
        raise ValueError("Run lacks timing/camera/track provenance; record a schema-v2 run")
    if not (result.get("finished") is True and result.get("finish_signal_observed") is True
            and result.get("reason") == "finish_signal" and result.get("controls_released") is True):
        raise ValueError("Only finish-confirmed, controls-released runs may become demonstrations")
    if result.get("action_alignment") != "observation_t_then_command_t":
        raise ValueError("Unsupported observation/action alignment")
    if result.get("telemetry_speed_unit") != "m/s":
        raise ValueError("Unknown telemetry speed unit")
    if not result.get("map_sha256") or result.get("camera_label") in (None, "", "unknown"):
        raise ValueError("Map hash and user-declared camera are required")
    archive_path = result_path.with_suffix(".npz")
    with np.load(archive_path, allow_pickle=False) as archive:
        telemetry = archive["telemetry"]
        actions = archive["actions"]
        frames = archive["frames"]
        obs_time = archive["observation_seconds"]
        act_time = archive["command_seconds"]
    n = len(actions)
    if n < 32 or telemetry.shape != (n + 1, 11) or actions.shape != (n, 3):
        raise ValueError("Expected at least 32 commands and one additional terminal observation")
    if result.get("steps") != n or result.get("observations") != n + 1:
        raise ValueError("Manifest row counts disagree with the archive")
    if frames.shape != (n + 1, 96, 96) or frames.dtype != np.uint8:
        raise ValueError("Expected one 96x96 uint8 grayscale frame per observation")
    if obs_time.shape != (n + 1,) or act_time.shape != (n,):
        raise ValueError("Missing observation/command timestamps")
    if not all(np.isfinite(a).all() for a in (telemetry, actions, obs_time, act_time)):
        raise ValueError("Run contains non-finite values")
    if np.any(np.abs(actions) > 1.0):
        raise ValueError("Commands must be in [-1, 1]")
    if not bool(telemetry[-1, 8]) or np.any(telemetry[:-1, 8]):
        raise ValueError("Finish must occur only in the final observation")
    intervals = np.diff(obs_time)
    if (np.any(intervals <= 0) or np.any(intervals > 3 * result["period_seconds"])
            or np.any(act_time < obs_time[:-1]) or np.any(act_time >= obs_time[1:])):
        raise ValueError("Non-causal command timing or a sampling gap exceeds three control periods")

    # Repeat the initial frame to fill history, without crossing episode bounds.
    history_index = np.maximum(0, np.arange(n)[:, None] + np.arange(-3, 1)[None, :])
    states = np.zeros((n, 15), dtype=np.float32)
    states[:, 0] = telemetry[:-1, 0] / 100.0
    states[:, 1:3] = telemetry[:-1, 5:7]
    states[:, 3] = telemetry[:-1, 9]
    states[:, 4] = telemetry[:-1, 10] / 10000.0
    states[:, 6:9] = telemetry[:-1, 2:5]
    states[:, 9:13] = 1.0
    previous = np.zeros_like(actions)
    previous[1:] = actions[:-1]
    done = np.zeros(n, dtype=bool)
    done[-1] = True
    provenance = {
        "run_id": result["run_id"], "result_path": str(result_path),
        "result_sha256": sha256(result_path), "archive_sha256": sha256(archive_path),
        "track_id": result.get("track_id", "unknown"), "map_sha256": result["map_sha256"],
        "camera_label": result["camera_label"],
        "camera_label_source": result.get("camera_label_source"),
        "track_identity_check": result.get("track_identity_check"),
        "teacher_mode": result["mode"], "assisted": result["assisted"],
        "controller_config": result["controller_config"], "finished": True,
        "steps": n, "sampling_interval_p95_ms": float(np.percentile(intervals, 95) * 1000),
        "sampling_interval_max_ms": float(intervals.max() * 1000),
        "max_deviation_m": result["max_deviation_m"],
    }
    episode = {"imgs": frames[history_index], "states": states,
               "actions": actions.astype(np.float32), "previous_actions": previous,
               "terminated": done, "observation_seconds": obs_time[:-1],
               "command_seconds": act_time, "metadata": provenance}
    return episode, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.runs) < 2:
        parser.error("Provide at least two completed runs; the last is held out for validation")
    manifest_path = args.output.with_suffix(".metadata.json")
    if args.output.exists() or manifest_path.exists():
        parser.error("Output already exists; choose a new dataset name")
    episodes, sources = zip(*(load_verified_run(path) for path in args.runs))
    if len({source["archive_sha256"] for source in sources}) != len(sources):
        parser.error("Duplicate runs would leak into the validation split")
    if len({(source["map_sha256"], source["camera_label"]) for source in sources}) != 1:
        parser.error("This known-track dataset requires one map and one camera convention")
    for index, episode in enumerate(episodes):
        episode["split"] = "validation" if index == len(episodes) - 1 else "train"
    metadata = {
        "schema_version": 3, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "track_id": sources[0]["track_id"], "map_sha256": sources[0]["map_sha256"],
        "camera_label": sources[0]["camera_label"], "episode_count": len(episodes),
        "transition_count": sum(len(ep["actions"]) for ep in episodes),
        "sampling_hz": 20, "action_alignment": "observation_t_then_command_t",
        "image_format": "4x96x96 grayscale uint8", "speed_unit": "m/s",
        "state_format": "15D; speed/100, gear, rpm/10000, diagnostic xyz at 6:9",
        "action_format": "gas/brake signed triggers; continuous steering; do not ternary-decode steering",
        "split_rule": "last entire run held out; training must honor episode.split",
        "use_limitations": ["Known-track teacher trajectories, not independent human demonstrations",
                            "No cross-track generalization evidence", "No recovery examples or collision sensor",
                            "Camera label is user-declared; map check is file hash plus live start, not live UID",
                            "Consumer must mask diagnostic xyz/steering/gas from model inputs"],
        "sources": list(sources),
    }
    import torch
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"episodes": list(episodes), "has_vision": True, "metadata": metadata}, args.output)
    metadata["dataset_sha256"] = sha256(args.output)
    manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "episodes": len(episodes),
                      "steps": metadata["transition_count"], "sha256": metadata["dataset_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
