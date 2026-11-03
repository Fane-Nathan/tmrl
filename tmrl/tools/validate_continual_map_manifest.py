"""Validate and checksum the immutable continual-learning map split."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


SPLITS = {
    "foundation_train",
    "foundation_validation",
    "continual_stream",
    "held_out_zero_shot",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def canonical_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def validate_manifest(payload: dict[str, Any], require_frozen: bool = False) -> list[str]:
    errors = []
    if payload.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    if require_frozen and payload.get("frozen") is not True:
        errors.append("manifest must set frozen=true")
    maps = payload.get("maps")
    if not isinstance(maps, list) or not maps:
        return errors + ["maps must be a non-empty list"]

    seen_ids = set()
    seen_splits = set()
    for index, entry in enumerate(maps):
        prefix = f"maps[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{prefix} must be an object")
            continue
        map_id = entry.get("map_id")
        if not isinstance(map_id, str) or not map_id.strip():
            errors.append(f"{prefix}.map_id must be a non-empty string")
        elif map_id in seen_ids:
            errors.append(f"duplicate map_id: {map_id}")
        else:
            seen_ids.add(map_id)
        split = entry.get("split")
        if split not in SPLITS:
            errors.append(f"{prefix}.split must be one of {sorted(SPLITS)}")
        else:
            seen_splits.add(split)
        for hash_name in ("track_sha256", "reward_sha256"):
            value = entry.get(hash_name)
            if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
                errors.append(f"{prefix}.{hash_name} must be a lowercase SHA-256")
        ghost_hash = entry.get("ghost_sha256")
        if ghost_hash is not None and (
            not isinstance(ghost_hash, str)
            or not SHA256_PATTERN.fullmatch(ghost_hash)
        ):
            errors.append(f"{prefix}.ghost_sha256 must be null or a lowercase SHA-256")

    missing_splits = SPLITS - seen_splits
    if missing_splits:
        errors.append(f"missing map splits: {sorted(missing_splits)}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--require-frozen", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Manifest read failed: {exc}", file=sys.stderr)
        return 2
    errors = validate_manifest(payload, require_frozen=args.require_frozen)
    result = {
        "valid": not errors,
        "errors": errors,
        "canonical_sha256": canonical_sha256(payload),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())

