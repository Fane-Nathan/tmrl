"""Stage the camera-ready code and data release locally. Nothing is uploaded or published.

  python scripts/prepare_camera_ready_release.py --list    # write FILE_LIST.md / MANIFEST.csv plan only
  python scripts/prepare_camera_ready_release.py --stage   # copy the repository tree and build archives

Repository tree: code, protocols, analysis outputs and small per-run records (JSON/CSV/JSONL/logs).
Archives (for a GitHub release or Zenodo): source checkpoints and, per study shard, branch checkpoints,
evaluation trajectories, frozen inputs and code snapshots. Excluded on purpose: the anonymous-review and
W&B helper scripts, git_diff.patch files (local diffs of unrelated TrackMania code), failed-attempt
outputs (listed in FAILED_ATTEMPTS.md instead) and the paper text (governed by the publisher agreement).
Every staged text file is scanned for credential-like strings; any hit stops staging.
"""
import argparse
import csv
import hashlib
import re
import shutil
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/sensor_adaptation/release_camera_ready_v1"
EXP = "experiments/sensor_adaptation_v1"
CODE = [f"{EXP}/{name}" for name in (
    "README.md", "analyze_extension.py", "analyze_replay_bias.py", "analyze_results.py", "benchmark_latency.py",
    "build_paper_assets.py", "frozen_models.py", "learner.py", "model_provenance.json", "protocol.json",
    "protocol_ant.json", "protocol_extension.json", "replay.py", "run_study.py", "sensor_env.py", "test_contracts.py")]
CODE += [f"scripts/{name}" for name in (
    "train_transformer_source_recipe_v1.py", "train_reactive_source_recipe_v1.py", "benchmark_fault_env.py",
    "confirmatory_logging.py", "build_camera_ready_assets.py", "run_camera_ready_pipeline.py",
    "prepare_camera_ready_release.py")]
STUDIES = {"halfcheetah": "output/sensor_adaptation/extension_20261031_v1",
           "ant": "output/sensor_adaptation/ant_20261031_v1",
           "halfcheetah_pilot": "output/sensor_adaptation/pilot_20260916_v1"}
SOURCES = {"halfcheetah": "fault_benchmark_results", "ant": "fault_benchmark_results_ant"}
RECORDS = ["output/sensor_adaptation/latency_20261031.json", "output/sensor_adaptation/budget_verification.json",
           "output/sensor_adaptation/rerun_reproducibility_hc_reactive.json"]
SMALL = {".json", ".jsonl", ".csv", ".md", ".txt", ".log", ".tex"}
SKIP_NAMES = {"git_diff.patch"}
SECRET = re.compile(r"(api[_-]?key|secret|passw(or)?d|token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}|ghp_[A-Za-z0-9]{20,}|"
                    r"AKIA[0-9A-Z]{16}|wandb_[A-Za-z0-9]{20,}|\b[0-9a-f]{40}\b(?=.*(wandb|key))", re.I)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def plan():
    tree, archives = [], {}
    tree += [(p, p) for p in CODE if (ROOT / p).exists()]
    tree += [(p, f"records/{Path(p).name}") for p in RECORDS if (ROOT / p).exists()]
    for task, rel in STUDIES.items():
        root = ROOT / rel
        if not root.exists():
            continue
        sharded = any(root.glob("*/status.json"))
        for path in sorted(root.rglob("*")):
            if not path.is_file() or "failed_attempts" in path.parts or path.name in SKIP_NAMES:
                continue
            inner = path.relative_to(root)
            if path.suffix in SMALL and inner.parts[0] != "source" and "inputs" not in inner.parts:
                tree.append((str(path.relative_to(ROOT)), f"results/{task}/{inner.as_posix()}"))
            else:  # checkpoints, trajectories, replay pools, frozen inputs and code snapshots
                name = f"{task}__{inner.parts[0]}.zip" if sharded and len(inner.parts) > 1 else f"{task}.zip"
                archives.setdefault(name, []).append((path, inner.as_posix()))
    for task, rel in SOURCES.items():
        root = ROOT / rel
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or "failed_attempts" in path.parts or path.name in SKIP_NAMES:
                continue
            inner = path.relative_to(root)
            if path.suffix in SMALL:
                tree.append((str(path.relative_to(ROOT)), f"sources/{task}/{inner.as_posix()}"))
            elif path.name == "best_validation_checkpoint.pt":
                archives.setdefault(f"sources_{task}.zip", []).append((path, inner.as_posix()))
    return tree, archives


def failed_attempts_note():
    lines = ["# Failed attempts", "",
             "Runs that crashed (Windows commit-limit or GPU-memory failures) were moved, never deleted, and rerun",
             "from scratch with identical seeds. They are not part of any analysis. Locations on the original machine:", ""]
    for rel in list(STUDIES.values()) + [f"{s}/logs" for s in SOURCES.values()]:
        folder = ROOT / rel / "failed_attempts"
        if folder.exists():
            lines += [f"- `{rel}/failed_attempts/{p.name}`" for p in sorted(folder.iterdir())]
    lines += ["", "The nine HalfCheetah reactive branches completed before one failure were rerun; all nine reproduce",
              "exactly (`records/rerun_reproducibility_hc_reactive.json`)."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--stage", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    tree, archives = plan()
    tree_bytes = sum((ROOT / src).stat().st_size for src, _ in tree)
    archive_bytes = {name: sum(p.stat().st_size for p, _ in files) for name, files in archives.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    summary = [f"# Camera-ready release plan ({time.strftime('%Y-%m-%d %H:%M')})", "",
               f"Repository tree: {len(tree)} files, {tree_bytes / 2**20:.1f} MiB.", "",
               "| Top-level folder | Files | MiB |", "|---|---|---|"]
    groups = {}
    for src, dst in tree:
        key = "/".join(dst.split("/")[:2]) if dst.startswith(("results/", "sources/")) else dst.split("/")[0]
        n, b = groups.get(key, (0, 0))
        groups[key] = (n + 1, b + (ROOT / src).stat().st_size)
    summary += [f"| `{k}` | {n} | {b / 2**20:.1f} |" for k, (n, b) in sorted(groups.items())]
    summary += ["", f"Archives: {len(archives)} files, {sum(archive_bytes.values()) / 2**30:.2f} GiB.", "",
                "| Archive | Files | MiB |", "|---|---|---|"]
    summary += [f"| `{name}` | {len(archives[name])} | {archive_bytes[name] / 2**20:.0f} |" for name in sorted(archives)]
    summary += ["", "Code files:", ""] + [f"- `{dst}`" for src, dst in tree if src in CODE]
    (args.out / "FILE_LIST.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    with open(args.out / "PLANNED_TREE.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "release_path", "bytes"])
        writer.writerows((src, dst, (ROOT / src).stat().st_size) for src, dst in tree)
    print(f"tree {len(tree)} files {tree_bytes / 2**20:.1f} MiB; archives {len(archives)} "
          f"{sum(archive_bytes.values()) / 2**30:.2f} GiB -> {args.out / 'FILE_LIST.md'}")
    if not args.stage:
        return
    repo = args.out / "repository"
    if repo.exists():
        raise SystemExit(f"{repo} exists; use a new --out")
    hits = []
    for src, dst in tree:
        target = repo / dst
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / src, target)
        if target.suffix in SMALL | {".py"}:
            for i, line in enumerate(target.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if SECRET.search(line):
                    hits.append(f"{dst}:{i}: {line.strip()[:120]}")
    (repo / "FAILED_ATTEMPTS.md").write_text(failed_attempts_note(), encoding="utf-8")
    if hits:
        (args.out / "SECRET_SCAN_HITS.txt").write_text("\n".join(hits) + "\n", encoding="utf-8")
        raise SystemExit(f"{len(hits)} credential-like lines; see SECRET_SCAN_HITS.txt. Nothing archived.")
    dist = args.out / "archives"
    dist.mkdir()
    rows = []
    for name, files in sorted(archives.items()):
        with zipfile.ZipFile(dist / name, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for path, inner in files:
                archive.write(path, inner)
        rows.append((name, (dist / name).stat().st_size, sha256(dist / name)))
        print(f"archived {name}", flush=True)
    with open(dist / "SHA256SUMS.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["archive", "bytes", "sha256"])
        writer.writerows(rows)
    print(f"staged {repo} and {len(rows)} archives in {dist}")


if __name__ == "__main__":
    main()
