"""Unattended camera-ready pipeline (31 Oct 2026, v3.2).

v1 gated launches on free physical RAM and crashed jobs anyway: on Windows the binding limit is the
commit charge (each job reserves ~3-4 GB that it never touches). v2 starts a job only when, after its
estimated reservation, at least --margin-gb of commit headroom and 1 GB of free VRAM remain, with at most
--max-jobs heavy jobs and 2 minutes between launches.

v3 (17:50 local) changes only scheduling:
  * Ant study shards not yet started run one source seed per shard, so each seed's five branches start
    as soon as that seed's source model exists instead of waiting behind long serial shards. Sharding
    does not change any branch's computation (branch, adaptation and evaluation seeds are per seed).
  * Jobs already running when the pipeline starts are adopted by matching their command lines and are
    then tracked by process id, so a crash of an adopted job is detected and retried.
  * Analyses run as scheduled steps: each task's analysis starts when that task's runs are complete;
    the latency benchmark waits until every other job has finished so the machine is idle. v3.1 (18:00):
    GPU analysis steps wait while any experiment is ready to launch (the commit limit allows ~8 jobs, and
    the pagefile cannot grow safely because C: is nearly full). v3.2 (18:05): per-job estimates lowered to
    the measured private commit (studies 2.66-2.82 GB, sources 2.72-2.79 GB) plus a small allowance.
Any job that fails is retried once (failed outputs are moved to failed_attempts, never deleted).

After everything: budget checks, then wait for the paper to be marked done. Fallback: shut down with a
10-minute warning (`shutdown /a` cancels) if the paper is not marked done within --shutdown-grace-hours.
"""
import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
EXP = ROOT / "experiments/sensor_adaptation_v1"
HC = ROOT / "output/sensor_adaptation/extension_20261031_v1"
ANT = ROOT / "output/sensor_adaptation/ant_20261031_v1"
SRC = ROOT / "fault_benchmark_results_ant"
PILOT = ROOT / "output/sensor_adaptation/pilot_20260916_v1"
LATENCY = ROOT / "output/sensor_adaptation/latency_20261031.json"
STATUS = ROOT / "output/sensor_adaptation/pipeline_status.json"
LOG = ROOT / "output/sensor_adaptation/pipeline.log"
STEPS = ROOT / "output/sensor_adaptation/pipeline_steps"
PAPER_DONE = ROOT / "paper/PAPER_DONE"
RECIPES = {"transformer": "scripts/train_transformer_source_recipe_v1.py",
           "reactive": "scripts/train_reactive_source_recipe_v1.py"}
EST_GB = {"study": 3.3, "source": 2.9, "step": 3.0, "light-step": 1.0}  # v3.2: measured 2.66-2.82 GB per job
SEEDS = list(range(42, 50))
STEP_COMMANDS = {
    "analyze-hc": ([PY, "analyze_extension.py", HC, "--protocol", "protocol_extension.json", "--pilot", PILOT], EXP),
    "bias-hc": ([PY, "analyze_replay_bias.py", HC, "--protocol", "protocol_extension.json"], EXP),
    "analyze-ant": ([PY, "analyze_extension.py", ANT, "--protocol", "protocol_ant.json"], EXP),
    "bias-ant": ([PY, "analyze_replay_bias.py", ANT, "--protocol", "protocol_ant.json"], EXP),
    "latency": ([PY, "benchmark_latency.py", "--trajectories", PILOT, "--out", LATENCY], EXP),
    "assets": ([PY, "scripts/build_camera_ready_assets.py", "--env", f"hc={HC}", "--env", f"ant={ANT}",
                "--latency", LATENCY], ROOT),
}
LIGHT_STEPS = {"analyze-hc", "analyze-ant", "assets"}


class PerformanceInfo(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t),
                ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
                ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t),
                ("HandleCount", ctypes.c_ulong), ("ProcessCount", ctypes.c_ulong), ("ThreadCount", ctypes.c_ulong)]


def commit_headroom_gb():
    info = PerformanceInfo()
    info.cb = ctypes.sizeof(info)
    ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info), info.cb)
    return (info.CommitLimit - info.CommitTotal) * info.PageSize / 2**30


def free_vram_gb():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                                       "--format=csv,noheader,nounits"], text=True, timeout=30)
        used, total = (float(x) for x in out.strip().splitlines()[0].split(","))
        return (total - used) / 1024
    except Exception:
        return 0.0


def log(message):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def read_status(root):
    try:
        path = root / "status.json"
        return json.loads(path.read_text(encoding="utf-8")).get("status") if path.exists() else None
    except (OSError, ValueError):
        return "UNREADABLE"


def value_after(cmd, flag):
    try:
        return cmd[cmd.index(flag) + 1]
    except (ValueError, IndexError):
        return None


class Job:
    def __init__(self, name, kind, args=None, deps=()):
        self.name, self.kind, self.args, self.deps = name, kind, args or {}, list(deps)
        self.proc, self.handle, self.attempts, self.dead, self.adopted = None, None, 0, False, []

    @property
    def est_gb(self):
        if self.kind == "step":
            return EST_GB["light-step"] if self.name in LIGHT_STEPS else EST_GB["step"]
        return EST_GB[self.kind]

    # ----- identity of outputs
    def root(self):
        return self.args["root"] / self.args["shard"] if self.kind == "study" else None

    def source_log(self):
        return SRC / "logs" / f"{self.args['alg']}_seed_{self.args['seed']}.log"

    def source_dir(self):
        return SRC / f"{self.args['alg']}_seed_{self.args['seed']}"

    def step_log(self):
        return ROOT / "output/sensor_adaptation" / f"pipeline_{self.name}.log"

    def done(self):
        if self.kind == "study":
            return read_status(self.root()) == "COMPLETE"
        if self.kind == "source":
            alg, seed = self.args["alg"], self.args["seed"]
            report = SRC / (f"report_seed_{seed}_lambda_0.05.json" if alg == "transformer"
                            else f"reactive_report_seed_{seed}.json")
            return report.exists() and (self.source_dir() / "best_validation_checkpoint.pt").exists()
        marker = STEPS / f"{self.name}.done"
        if not marker.exists() and self.proc is not None and self.proc.poll() == 0:
            STEPS.mkdir(parents=True, exist_ok=True)
            marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")
            log(f"END {self.name}: exit 0")
        return marker.exists()

    # ----- process tracking
    def matches(self, cmd):
        if self.kind == "study":
            return any(t.endswith("run_study.py") for t in cmd) and value_after(cmd, "--output") == str(self.root())
        if self.kind == "source":
            return (any(t.replace("\\", "/").endswith(RECIPES[self.args["alg"]]) for t in cmd)
                    and value_after(cmd, "--seed") == str(self.args["seed"]) and value_after(cmd, "--env") == "Ant-v5")
        return False

    def alive(self):
        if self.proc is not None:
            return self.proc.poll() is None
        for process in self.adopted:
            try:
                if process.is_running():
                    return True
            except psutil.Error:
                pass
        return False

    def failed(self):
        if self.done() or self.alive():
            return False
        return self.proc is not None or bool(self.adopted)  # its process ended without the completed output

    def running(self):
        return not self.dead and not self.done() and self.alive()

    # ----- launching
    def has_leftovers(self):
        if self.kind == "study":
            return self.root().exists()
        if self.kind == "source":
            return self.source_log().exists() or self.source_dir().exists()
        return False

    def archive_failure(self):
        stamp = time.strftime("%H%M%S")
        if self.kind == "study":
            target = self.args["root"] / "failed_attempts" / f"{self.args['shard']}_{stamp}"
            target.parent.mkdir(parents=True, exist_ok=True)
            if self.root().exists():
                shutil.move(str(self.root()), str(target))
            shard_log = self.args["root"] / "logs" / f"{self.args['shard']}.log"
            if shard_log.exists():
                shutil.move(str(shard_log), str(target) + ".log")
        elif self.kind == "source":
            archive = SRC / "logs" / "failed_attempts"
            archive.mkdir(parents=True, exist_ok=True)
            if self.source_log().exists():
                shutil.move(str(self.source_log()), str(archive / f"{self.source_log().stem}_{stamp}.log"))
            if self.source_dir().exists():
                shutil.move(str(self.source_dir()), str(archive / f"{self.source_dir().name}_{stamp}"))
        elif self.step_log().exists():
            if self.handle:
                self.handle.close()
                self.handle = None
            shutil.move(str(self.step_log()), str(self.step_log().with_name(f"{self.step_log().stem}_failed_{stamp}.log")))

    def launch(self):
        if self.handle:
            self.handle.close()
        if self.has_leftovers():
            log(f"{self.name}: moving leftover outputs to failed_attempts before launch")
            self.archive_failure()
        env = dict(os.environ)
        if self.kind == "study":
            (self.args["root"] / "logs").mkdir(parents=True, exist_ok=True)
            self.handle = open(self.args["root"] / "logs" / f"{self.args['shard']}.log", "w", encoding="utf-8")
            cmd = [PY, "-u", "run_study.py", "--protocol", self.args["protocol"], "--algorithms", self.args["alg"],
                   "--seeds", *map(str, self.args["seeds"]), "--output", str(self.root())]
            cwd = EXP
        elif self.kind == "source":
            (SRC / "logs").mkdir(parents=True, exist_ok=True)
            self.handle = open(self.source_log(), "w", encoding="utf-8")
            cmd = [PY, "-u", RECIPES[self.args["alg"]], "--env", "Ant-v5", "--seed", str(self.args["seed"]),
                   "--num_threads", "2", "--output_dir", "fault_benchmark_results_ant"]
            cwd = ROOT
        else:
            self.handle = open(self.step_log(), "w", encoding="utf-8")
            cmd, cwd = STEP_COMMANDS[self.name]
            cmd = [str(c) for c in cmd]
        if self.kind != "step":
            env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        self.proc = subprocess.Popen(cmd, cwd=cwd, stdout=self.handle, stderr=subprocess.STDOUT, env=env)
        self.adopted = []
        self.attempts += 1


def build_jobs():
    jobs = []
    hc_shards = {"transformer_a": [42, 43, 44], "transformer_b": [45, 46, 47], "transformer_c": [48, 49],
                 "reactive_a": [42, 43, 44, 45], "reactive_b": [46, 47, 48, 49]}
    for shard, seeds in hc_shards.items():
        jobs.append(Job(f"hc-{shard}", "study", {"root": HC, "shard": shard, "protocol": "protocol_extension.json",
                                                 "alg": shard.split("_")[0], "seeds": seeds}))
    jobs += [Job(f"src-transformer-{s}", "source", {"alg": "transformer", "seed": s}) for s in SEEDS]
    jobs += [Job(f"src-reactive-{s}", "source", {"alg": "reactive", "seed": s}) for s in SEEDS]
    jobs.append(Job("ant-transformer_a", "study", {"root": ANT, "shard": "transformer_a", "protocol": "protocol_ant.json",
                                                   "alg": "transformer", "seeds": [42, 43, 44]},
                    deps=[f"src-transformer-{s}" for s in (42, 43, 44)]))
    # Reactive studies first: their sources finish last, so they form the critical path.
    jobs += [Job(f"ant-reactive_s{s}", "study", {"root": ANT, "shard": f"reactive_s{s}", "protocol": "protocol_ant.json",
                                                 "alg": "reactive", "seeds": [s]}, deps=[f"src-reactive-{s}"])
             for s in SEEDS]
    jobs += [Job(f"ant-transformer_s{s}", "study", {"root": ANT, "shard": f"transformer_s{s}",
                                                    "protocol": "protocol_ant.json", "alg": "transformer", "seeds": [s]},
                 deps=[f"src-transformer-{s}"]) for s in range(45, 50)]
    hc_jobs = [j.name for j in jobs if j.name.startswith("hc-")]
    ant_jobs = [j.name for j in jobs if j.name.startswith("ant-")]
    experiments = [j.name for j in jobs]
    jobs += [Job("analyze-hc", "step", deps=hc_jobs), Job("bias-hc", "step", deps=hc_jobs),
             Job("analyze-ant", "step", deps=ant_jobs), Job("bias-ant", "step", deps=ant_jobs)]
    jobs.append(Job("latency", "step", deps=experiments + ["analyze-hc", "bias-hc", "analyze-ant", "bias-ant"]))
    jobs.append(Job("assets", "step", deps=["analyze-hc", "bias-hc", "analyze-ant", "bias-ant", "latency"]))
    return jobs


def adopt_running(jobs):
    processes = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if (process.info["name"] or "").lower().startswith("python") and process.info["cmdline"]:
                processes.append((process, process.info["cmdline"]))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    for job in jobs:
        if job.kind == "step" or job.done():
            continue
        matched = [p for p, cmd in processes if job.matches(cmd)]
        if matched:
            job.adopted, job.attempts = matched, 1
            log(f"adopted running {job.name}: pids {[p.pid for p in matched]}")


def verify_budgets(root, expected):
    bad, results = [], list(root.glob("*/*/result.json"))
    for path in results:
        r = json.loads(path.read_text(encoding="utf-8"))
        if r.get("status") != "COMPLETE" or r.get("environment_steps") != 20000 or r.get("updates") != 5000:
            bad.append({"branch": path.parent.name, "steps": r.get("environment_steps"), "updates": r.get("updates")})
    return {"root": str(root.relative_to(ROOT)), "branches": len(results), "expected": expected,
            "all_budgets_exact": not bad and len(results) == expected, "problems": bad}


def write_status(stage, jobs, headroom=None, vram=None):
    STATUS.write_text(json.dumps({
        "stage": stage, "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "commit_headroom_gb": None if headroom is None else round(headroom, 1),
        "free_vram_gb": None if vram is None else round(vram, 1),
        "done": [j.name for j in jobs if j.done()],
        "running": [j.name for j in jobs if j.running()],
        "waiting": [j.name for j in jobs if not j.dead and not j.done() and not j.running()],
        "dead": [j.name for j in jobs if j.dead]}, indent=1) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin-gb", type=float, default=4.0)
    parser.add_argument("--max-jobs", type=int, default=11)
    parser.add_argument("--spacing-s", type=int, default=120)
    parser.add_argument("--shutdown-grace-hours", type=float, default=4.0)
    parser.add_argument("--no-shutdown", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Show the adopted/done/waiting plan and exit")
    args = parser.parse_args()
    jobs = build_jobs()
    by_name = {j.name: j for j in jobs}
    if args.dry_run:
        global log
        log = print  # noqa: F811 - keep the dry run out of the pipeline log
    adopt_running(jobs)
    if args.dry_run:
        for job in jobs:
            state = "done" if job.done() else "running" if job.running() else "waiting"
            print(f"{job.name:22s} {state:8s} est {job.est_gb:.1f} GB deps {job.deps if state == 'waiting' else ''}")
        print(f"headroom {commit_headroom_gb():.1f} GB, VRAM free {free_vram_gb():.1f} GB")
        return
    last_launch = 0.0
    log(f"pipeline v3 start: {len(jobs)} jobs, headroom {commit_headroom_gb():.1f} GB, VRAM free {free_vram_gb():.1f} GB")
    while True:
        try:
            for job in jobs:
                if not job.dead and job.failed():
                    if job.attempts >= 2:
                        job.dead = True
                        job.proc, job.adopted = None, []
                        log(f"PERMANENT FAILURE {job.name}")
                        continue
                    log(f"job failed: {job.name} (attempt {job.attempts}); archiving for one retry")
                    job.archive_failure()
                    job.proc, job.adopted = None, []
                    job.attempts = max(job.attempts, 1)
            running = [j for j in jobs if j.running()]
            waiting = [j for j in jobs if not j.dead and not j.done() and not j.running()]
            headroom, vram = commit_headroom_gb(), free_vram_gb()
            if waiting and time.time() - last_launch >= args.spacing_s and len(running) < args.max_jobs:
                # Experiments in strict priority order. Light analysis steps may start beside them; GPU analysis
                # steps start only when no experiment is ready and waiting, so they never take an experiment's slot.
                experiment_ready = any(j.kind != "step" and all(by_name[d].done() for d in j.deps) for j in waiting)
                for queue in ([j for j in waiting if j.kind != "step"], [j for j in waiting if j.kind == "step"]):
                    launched = False
                    for job in queue:
                        if not all(by_name[d].done() for d in job.deps):
                            continue
                        if job.kind == "step" and job.name not in LIGHT_STEPS and experiment_ready:
                            continue
                        needs_gpu = job.kind != "step" or job.name not in LIGHT_STEPS
                        if headroom - job.est_gb >= args.margin_gb and (vram >= 1.0 or not needs_gpu):
                            job.launch()
                            last_launch, launched = time.time(), True
                            log(f"launched {job.name} attempt {job.attempts} (headroom {headroom:.1f} GB, "
                                f"VRAM {vram:.1f} GB, running {len(running) + 1})")
                        break  # never let a later job of the same queue overtake the first eligible one
                    if launched:
                        break
            write_status("experiments", jobs, headroom, vram)
            if all(j.done() for j in jobs):
                break
            if any(j.dead for j in jobs) and not any(j.running() for j in jobs) and not any(
                    all(by_name[d].done() for d in j.deps) for j in jobs if not j.dead and not j.done()):
                log("stopping: permanent failures and nothing else can run")
                write_status("analysis_failed", jobs)
                sys.exit(1)
        except SystemExit:
            raise
        except Exception:
            log("scheduler error (continuing): " + traceback.format_exc().replace("\n", " | "))
        time.sleep(60)

    log("all experiments and analyses complete")
    checks = [verify_budgets(HC, 80), verify_budgets(ANT, 80)]
    (ROOT / "output/sensor_adaptation/budget_verification.json").write_text(json.dumps(checks, indent=2) + "\n",
                                                                            encoding="utf-8")
    log(f"budget verification exact: {[c['all_budgets_exact'] for c in checks]}")
    write_status("done_waiting_for_paper", jobs)
    log("analyses done; waiting for paper text before fallback shutdown")
    if args.no_shutdown:
        return
    deadline = time.time() + args.shutdown_grace_hours * 3600
    while time.time() < deadline:
        if PAPER_DONE.exists():
            log("paper marked done; Claude performs the shutdown")
            return
        time.sleep(60)
    log("grace period over without PAPER_DONE; fallback shutdown in 10 minutes")
    subprocess.run(["shutdown", "/s", "/t", "600", "/c",
                    "Camera-ready experiments finished. Shutting down in 10 minutes. Run 'shutdown /a' to cancel."])


if __name__ == "__main__":
    main()
