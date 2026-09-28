#!/usr/bin/env python3
"""
Confirmatory Study Parallel Batch Orchestrator.

Manages end-to-end execution of:
1. Sequence Transformer RL^2 agent (8 seeds: 42-49).
2. Reactive Robust Baseline (8 seeds: 42-49).
3. Concurrency control: Runs up to N (default 4) parallel worker processes.
4. Thread pinning: Limits each worker to T (default 4) CPU threads via OMP/MKL and torch.
5. Live logging: Streams logs to individual log files and displays status summaries.
6. Auto-synthesis: Invokes scripts/generate_confirmatory_results.py upon completion.
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_EXE = sys.executable


def get_job_list(
    algorithms: List[str],
    seeds: List[int],
    output_dir: Path,
    expected_protocol_sha256: str,
    skip_completed: bool = True,
) -> List[Dict[str, Any]]:
    jobs = []
    for alg in algorithms:
        for seed in seeds:
            # Check if report already exists and is complete
            if alg == "transformer":
                script_path = REPO_ROOT / "scripts" / "train_transformer_fault_adaptation.py"
                report_file = output_dir / f"report_seed_{seed}_lambda_0.05.json"
                job_name = f"transformer_seed_{seed}"
            elif alg == "reactive":
                script_path = REPO_ROOT / "scripts" / "train_reactive_robust_baseline.py"
                report_file = output_dir / f"reactive_report_seed_{seed}.json"
                job_name = f"reactive_seed_{seed}"
            else:
                raise ValueError(f"Unknown algorithm: {alg}")

            already_done = False
            if skip_completed and report_file.exists():
                try:
                    import json
                    with open(report_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if "test_results" in data and "test_latency" in data["test_results"]:
                        if (
                            data.get("best_val_score") is not None
                            and data.get("protocol_sha256") == expected_protocol_sha256
                        ):
                            already_done = True
                except Exception:
                    already_done = False

            jobs.append({
                "algorithm": alg,
                "seed": seed,
                "script": script_path,
                "report_file": report_file,
                "job_name": job_name,
                "already_done": already_done,
            })
    return jobs


def run_batch(
    jobs: List[Dict[str, Any]],
    max_parallel: int,
    num_threads: int,
    iterations: int,
    config_path: Path,
    output_dir: Path,
    smoke_test: bool = False,
) -> bool:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    pending_jobs = [j for j in jobs if not j["already_done"]]
    skipped_jobs = [j for j in jobs if j["already_done"]]

    print("=" * 80)
    print("CONFIRMATORY EXPERIMENT BATCH ORCHESTRATOR")
    print("=" * 80)
    print(f"Total jobs requested: {len(jobs)}")
    print(f"Already completed (skipping): {len(skipped_jobs)}")
    print(f"Jobs to run: {len(pending_jobs)}")
    print(f"Concurrency limit: {max_parallel} parallel processes")
    print(f"Threads per process: {num_threads} (Total max threads: {max_parallel * num_threads})")
    print(f"Training iterations per job: {iterations}")
    print(f"Smoke test mode: {smoke_test}")
    print(f"Config path: {config_path}")
    print(f"Output directory: {output_dir}")
    print("=" * 80, flush=True)

    if not pending_jobs:
        print("All requested jobs have already been completed!")
        return True

    active: Dict[str, Dict[str, Any]] = {}
    failed_jobs: List[str] = []
    completed_jobs: List[str] = []

    def start_job(job: Dict[str, Any]) -> None:
        name = job["job_name"]
        log_path = log_dir / f"{name}.log"
        log_f = open(log_path, "w", encoding="utf-8")

        cmd = [
            PYTHON_EXE,
            str(job["script"]),
            "--seed", str(job["seed"]),
            "--config", str(config_path),
            "--iterations", str(iterations),
            "--num_threads", str(num_threads),
            "--output_dir", str(output_dir),
        ]
        if smoke_test:
            cmd.append("--smoke_test")

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(num_threads)
        env["MKL_NUM_THREADS"] = str(num_threads)
        env["OPENBLAS_NUM_THREADS"] = str(num_threads)
        env["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
        env["NUMEXPR_NUM_THREADS"] = str(num_threads)

        t_start = time.time()
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] STARTING: {name} (logging to {log_path.name})", flush=True)

        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
            env=env,
        )
        active[name] = {
            "proc": proc,
            "job": job,
            "log_f": log_f,
            "log_path": log_path,
            "start_time": t_start,
        }

    job_queue = list(pending_jobs)
    total_to_run = len(job_queue)

    while job_queue or active:
        # Start new jobs if capacity allows
        while job_queue and len(active) < max_parallel:
            job = job_queue.pop(0)
            start_job(job)

        time.sleep(5)

        # Check for finished jobs
        finished_names = []
        for name, info in active.items():
            ret = info["proc"].poll()
            if ret is not None:
                info["log_f"].close()
                elapsed = time.time() - info["start_time"]
                elapsed_min = elapsed / 60.0
                finished_names.append(name)

                if ret == 0:
                    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] COMPLETED: {name} in {elapsed_min:.1f}m ({len(completed_jobs)+1}/{total_to_run})", flush=True)
                    completed_jobs.append(name)
                else:
                    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] FAILED: {name} with returncode {ret} after {elapsed_min:.1f}m. Check log: {info['log_path']}", flush=True)
                    failed_jobs.append(name)

        for name in finished_names:
            del active[name]

        # Status heartbeat every 60 seconds
        now = time.time()
        if active and int(now) % 60 < 5:
            active_str = ", ".join([f"{n} ({int((now - info['start_time'])/60)}m)" for n, info in active.items()])
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Active ({len(active)}): {active_str} | Remaining in queue: {len(job_queue)}", flush=True)

    print("\n" + "=" * 80)
    print("BATCH EXECUTION FINISHED")
    print(f"Successfully completed: {len(completed_jobs)}/{total_to_run}")
    if failed_jobs:
        print(f"FAILED jobs ({len(failed_jobs)}): {', '.join(failed_jobs)}")
    print("=" * 80, flush=True)

    return len(failed_jobs) == 0


def main():
    parser = argparse.ArgumentParser(description="Confirmatory Experiment Parallel Batch Orchestrator")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["transformer", "reactive"],
        choices=["transformer", "reactive", "all"],
        help="Algorithms to run",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds to execute. Defaults to seed_list from the protocol.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Maximum concurrent processes (default: 4)",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=4,
        help="Threads per worker process (default: 4)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Training iterations per seed (default: 100)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "confirmatory_protocol_v2.json"),
        help="Path to protocol configuration JSON",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(REPO_ROOT / "fault_benchmark_results_v2"),
        help="Output directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if report already exists",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run 1 iteration smoke test across requested seeds",
    )
    parser.add_argument(
        "--no_auto_analyze",
        action="store_true",
        help="Disable automatic statistical analysis after completion",
    )
    args = parser.parse_args()

    algs = args.algorithms
    if "all" in algs:
        algs = ["transformer", "reactive"]

    output_dir = Path(args.output_dir).resolve()
    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Confirmatory protocol not found: {config_path}")

    import hashlib
    import json
    protocol_bytes = config_path.read_bytes()
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    seeds = args.seeds if args.seeds is not None else list(protocol["seed_list"])

    jobs = get_job_list(
        algorithms=algs,
        seeds=seeds,
        output_dir=output_dir,
        expected_protocol_sha256=protocol_sha256,
        skip_completed=not args.force,
    )

    success = run_batch(
        jobs=jobs,
        max_parallel=args.parallel,
        num_threads=args.num_threads,
        iterations=args.iterations,
        config_path=config_path,
        output_dir=output_dir,
        smoke_test=args.smoke_test,
    )

    if success and not args.no_auto_analyze and not args.smoke_test:
        print("\nLaunching statistical analysis and artifact generation...", flush=True)
        res = subprocess.run(
            [
                PYTHON_EXE,
                str(REPO_ROOT / "scripts" / "generate_confirmatory_results.py"),
                "--results_dir", str(output_dir),
                "--protocol", str(config_path),
            ],
            cwd=str(REPO_ROOT),
        )
        if res.returncode == 0:
            print("Statistical analysis completed successfully.")
        else:
            print(f"Statistical analysis failed with code {res.returncode}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
