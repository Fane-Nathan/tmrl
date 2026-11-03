"""Train Ant-v5 source policies with the HalfCheetah source recipe, gated on free RAM.

Each job is one controller/seed (~1.8 GB resident). A job starts only when fewer than
--max-parallel jobs run and at least --min-free-gb physical memory is free, so concurrent
experiments are not pushed into allocation failures.

  python scripts/run_ant_sources.py --seeds 42 43 44 45 46 47 48 49
"""
import argparse
import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPES = {"transformer": "scripts/train_transformer_source_recipe_v1.py",
           "reactive": "scripts/train_reactive_source_recipe_v1.py"}


class MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def free_gb():
    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.ullAvailPhys / 2**30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[])
    parser.add_argument("--output-dir", default="fault_benchmark_results_ant")
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--min-free-gb", type=float, default=3.5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--algorithms", nargs="+", default=["transformer", "reactive"], choices=list(RECIPES))
    parser.add_argument("--jobs", nargs="+", help="Explicit algorithm:seed jobs, overriding --algorithms/--seeds")
    args = parser.parse_args()
    out = ROOT / args.output_dir
    (out / "logs").mkdir(parents=True, exist_ok=True)
    # Longest jobs first.
    if args.jobs:
        queue = [(job.split(":")[0], int(job.split(":")[1])) for job in args.jobs]
        if any(a not in RECIPES for a, _ in queue):
            raise ValueError(f"Unknown algorithm in {args.jobs}")
    else:
        queue = [(a, s) for a in args.algorithms for s in args.seeds]
    running, failed = {}, []
    env = {**os.environ, **{k: str(args.threads) for k in
                            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    while queue or running:
        for key, (proc, log) in list(running.items()):
            if proc.poll() is not None:
                log.close()
                print(f"{time.strftime('%H:%M:%S')} done {key} exit={proc.returncode}", flush=True)
                if proc.returncode != 0:
                    failed.append(key)
                del running[key]
        if queue and len(running) < args.max_parallel and free_gb() >= args.min_free_gb:
            algorithm, seed = queue.pop(0)
            log = open(out / "logs" / f"{algorithm}_seed_{seed}.log", "w", encoding="utf-8")
            cmd = [sys.executable, "-u", RECIPES[algorithm], "--env", "Ant-v5", "--seed", str(seed),
                   "--num_threads", str(args.threads), "--output_dir", args.output_dir]
            running[(algorithm, seed)] = (subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env), log)
            print(f"{time.strftime('%H:%M:%S')} start {algorithm} seed {seed} (free {free_gb():.1f} GB, "
                  f"running {len(running)})", flush=True)
            time.sleep(90)  # let the new process allocate before the next memory check
            continue
        time.sleep(30)
    print(f"ALL DONE; failed: {failed}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
