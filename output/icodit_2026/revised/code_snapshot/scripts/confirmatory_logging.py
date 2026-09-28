#!/usr/bin/env python3
"""
Confirmatory Experiment Reproducibility & Logging Infrastructure.

Provides standardized provenance, environment capture, SHA-256 checkpoint hashing,
run directory isolation, and experiment manifest management.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import psutil
import torch


def get_git_commit(repo_root: Path) -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_COMMIT"


def get_git_diff(repo_root: Path) -> str:
    try:
        res = subprocess.run(
            ["git", "diff"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )
        return res.stdout
    except Exception:
        return ""


def compute_file_sha256(filepath: str | Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_hardware_info() -> Dict[str, Any]:
    mem = psutil.virtual_memory()
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "None (CPU Execution)"
    return {
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_ram_gb": round(mem.total / (1024 ** 3), 2),
        "available_ram_gb": round(mem.available / (1024 ** 3), 2),
        "cuda_available": cuda_avail,
        "cuda_device_count": torch.cuda.device_count() if cuda_avail else 0,
        "device_name": gpu_name,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }


def get_environment_info() -> Dict[str, Any]:
    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "executable": sys.executable,
        "os_name": os.name,
        "platform": platform.platform(),
    }


def setup_run_directory(
    output_dir: str | Path,
    run_name: str,
    config: Dict[str, Any],
    seed: int,
    repo_root: Path,
) -> Path:
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(run_dir / "protocol_version.txt", "w") as f:
        f.write(str(config.get("protocol_version", "1.0.0")) + "\n")

    with open(run_dir / "seed.txt", "w") as f:
        f.write(f"{seed}\n")

    git_commit = get_git_commit(repo_root)
    with open(run_dir / "git_commit.txt", "w") as f:
        f.write(f"{git_commit}\n")

    git_diff = get_git_diff(repo_root)
    with open(run_dir / "git_diff.patch", "w", encoding="utf-8") as f:
        f.write(git_diff)

    hw = get_hardware_info()
    with open(run_dir / "hardware.txt", "w") as f:
        for k, v in hw.items():
            f.write(f"{k}: {v}\n")

    env = get_environment_info()
    with open(run_dir / "environment.json", "w") as f:
        json.dump(env, f, indent=2)

    return run_dir


def update_manifest(
    manifest_path: str | Path,
    entry: Dict[str, Any],
) -> None:
    manifest_path = Path(manifest_path)
    fields = [
        "algorithm",
        "seed",
        "status",
        "train_start",
        "train_end",
        "training_iterations",
        "environment_steps",
        "validation_checkpoint",
        "checkpoint_sha256",
        "validation_score",
        "test_completed",
        "raw_results_file",
        "notes",
    ]

    existing_rows = []
    if manifest_path.exists():
        with open(manifest_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows.append(row)

    # Check if row with same algorithm and seed exists
    updated = False
    for i, r in enumerate(existing_rows):
        if r.get("algorithm") == str(entry.get("algorithm")) and str(r.get("seed")) == str(entry.get("seed")):
            existing_rows[i] = {k: str(entry.get(k, r.get(k, ""))) for k in fields}
            updated = True
            break

    if not updated:
        existing_rows.append({k: str(entry.get(k, "")) for k in fields})

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing_rows)
