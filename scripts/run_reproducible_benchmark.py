"""
Unified runner script to execute the complete reproducible fault-adaptation benchmark.
Runs:
  1. Transformer in-context learning agent across seeds (with validation-selected checkpointing).
  2. Independent reactive robust baseline across seeds.
  3. Automated statistical report and publication figure generation (95% bootstrap CIs + IQM).
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, cwd=None):
    print(f"\n[RUNNING] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with returncode {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run reproducible continuous-control fault benchmark.")
    parser.add_argument("--quick", action="store_true", help="Run short smoke test (few iterations) to verify end-to-end pipeline.")
    parser.add_argument("--fast", action="store_true", help="Run fast training (~3 minutes per run, 20 iters).")
    parser.add_argument("--iterations", type=int, default=None, help="Number of training iterations.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Random seeds to evaluate.")
    parser.add_argument("--output_dir", type=str, default="./fault_benchmark_results", help="Results output directory.")
    parser.add_argument("--python_bin", type=str, default=sys.executable, help="Python executable path.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    scripts_dir = repo_root / "scripts"
    output_path = repo_root / args.output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    mode_str = "QUICK SMOKE TEST" if args.quick else ("FAST SCIENTIFIC RUN (~3 min)" if args.fast else "FULL SCIENTIFIC RUN")

    print("=" * 70)
    print("REPRODUCIBLE FAULT-ADAPTATION BENCHMARK SUITE")
    print(f"Mode: {mode_str}")
    print(f"Seeds: {args.seeds}")
    print(f"Output directory: {output_path}")
    print("=" * 70)

    # 1. Run Transformer Agent
    for seed in args.seeds:
        print(f"\n>>> Running Transformer Fault Adaptation Agent (Seed {seed}) <<<")
        cmd = [
            args.python_bin,
            str(scripts_dir / "train_transformer_fault_adaptation.py"),
            "--seed", str(seed),
            "--output_dir", str(output_path),
        ]
        if args.fast:
            cmd.append("--fast")
        elif args.quick:
            cmd.append("--smoke_test")
        elif args.iterations:
            cmd.extend(["--iterations", str(args.iterations)])

        run_cmd(cmd, cwd=str(repo_root))

    # 2. Run Reactive Robust Baseline
    for seed in args.seeds:
        print(f"\n>>> Running Reactive Robust Baseline (Seed {seed}) <<<")
        cmd = [
            args.python_bin,
            str(scripts_dir / "train_reactive_robust_baseline.py"),
            "--seed", str(seed),
            "--output_dir", str(output_path),
        ]
        if args.fast:
            cmd.append("--fast")
        elif args.quick:
            cmd.append("--smoke_test")
        elif args.iterations:
            cmd.extend(["--iterations", str(args.iterations)])

        run_cmd(cmd, cwd=str(repo_root))

    # 3. Generate Statistical Report and Figures
    print("\n>>> Generating Bootstrap Statistical Report and Figures <<<")
    paper_dir = repo_root.parent / "paper_research"
    run_cmd([
        args.python_bin,
        str(scripts_dir / "generate_statistical_report.py"),
        "--results_dir", str(output_path),
        "--output_png", str(paper_dir / "fault_adaptation_result.png"),
        "--output_pdf", str(paper_dir / "fault_adaptation_result.pdf"),
    ], cwd=str(repo_root))

    print("\n" + "=" * 70)
    print("BENCHMARK EXECUTION COMPLETE")
    print(f"All logs and checkpoints saved to: {output_path}")
    print(f"Publication figures and tables updated in: {repo_root.parent / 'paper_research'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
