"""Create an immutable, checksummed baseline snapshot for continual learning."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, BinaryIO


COPY_CHUNK_BYTES = 16 * 1024 * 1024


def _copy_and_hash(source: BinaryIO, destination: BinaryIO) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = source.read(COPY_CHUNK_BYTES)
        if not chunk:
            break
        destination.write(chunk)
        digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
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


def freeze_sources(
    sources: list[Path],
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    sources = [source.resolve(strict=True) for source in sources]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Baseline directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = []
    for index, source in enumerate(sources):
        if not source.is_file():
            raise ValueError(f"Baseline source is not a file: {source}")
        if output_dir == source.parent or output_dir in source.parents:
            raise ValueError("Baseline output cannot contain one of its sources")

        before = source.stat()
        destination = output_dir / f"{index:02d}_{source.name}"
        partial = output_dir / f".{destination.name}.partial"
        try:
            with source.open("rb") as source_handle, partial.open("xb") as dest_handle:
                sha256 = _copy_and_hash(source_handle, dest_handle)
                dest_handle.flush()
                os.fsync(dest_handle.fileno())
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                raise RuntimeError(
                    f"Source changed while it was being copied: {source}"
                )
            os.replace(partial, destination)
            shutil.copystat(source, destination)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        artifacts.append(
            {
                "source": str(source),
                "snapshot": destination.name,
                "size_bytes": before.st_size,
                "source_mtime_ns": before.st_mtime_ns,
                "sha256": sha256,
            }
        )

    manifest = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(repo_root.resolve()),
        "artifacts": artifacts,
    }
    _atomic_json(output_dir / "baseline_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        required=True,
        help="Source artifact to freeze. Repeat for checkpoint and actor files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args(argv)
    try:
        manifest = freeze_sources(args.source, args.output_dir, args.repo_root)
    except Exception as exc:
        print(f"Baseline freeze failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

