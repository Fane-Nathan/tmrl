#!/usr/bin/env python3
"""
Orchestration and Real-Time World Record Tracking for Dreamer on Track B (Summer 2020 - 01)

World Record Reference:
- Map: Summer 2020 - 01 (UID: XJ_JEjWGoAexDWe8qfaOjEcq5l8)
- Human World Record: AffiTM 19.454s (19,454 ms)
- Author Medal: ~21.500s
- Gold Medal: ~23.000s

Architecture:
- Recurrent Latent Dreamer (RSSM + continuous actor-critic)
- Multimodal Vision & Kinematics Encoder
- Pure continuous action policy (no residual steering lock)
- Dense Track B reward centerline with off-track safety guard
"""

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PYTHON_EXE = sys.executable
WR_TIME_SECONDS = 19.454
GOLD_MEDAL_SECONDS = 23.000
AUTHOR_MEDAL_SECONDS = 21.500

LOG_DIR = REPO_ROOT / "output" / "dreamer_training"
LOG_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS_FILE = LOG_DIR / "track_b_training_progress.json"

# Attach to interactive desktop on Windows
if sys.platform == "win32":
    import ctypes
    try:
        hwinsta = ctypes.windll.user32.OpenWindowStationW("WinSta0", False, 0x10000000)
        if hwinsta:
            ctypes.windll.user32.SetProcessWindowStation(hwinsta)
        hdesk = ctypes.windll.user32.OpenDesktopW("Default", 0, False, 0x10000000)
        if hdesk:
            ctypes.windll.user32.SetThreadDesktop(hdesk)
    except Exception:
        pass


def get_latest_replay(start_epoch=None):
    search_dirs = [
        Path(os.path.expanduser("~/Documents/Trackmania2020/Replays/My Replays")),
        Path(os.path.expanduser("~/Documents/Trackmania2020/Replays/Autosaves")),
        Path(os.path.expanduser("~/Documents/Trackmania/Replays/My Replays")),
    ]
    candidates = []
    for d in search_dirs:
        if d.is_dir():
            for f in d.glob("*.Replay.Gbx"):
                try:
                    mtime = f.stat().st_mtime
                    if start_epoch is None or mtime >= start_epoch - 2.0:
                        candidates.append((mtime, f))
                except OSError:
                    pass
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    latest_file = candidates[0][1]
    m = re.search(r'\((\d+)_(\d+)_(\d+)\)', latest_file.name)
    if m:
        mins, secs, ms = int(m.group(1)), int(m.group(2)), int(m.group(3))
        time_sec = mins * 60.0 + secs + ms / 1000.0
        return time_sec
    return None


class DreamerTrainingManager:
    def __init__(self, target_wr=WR_TIME_SECONDS, max_training_minutes=120):
        self.target_wr = target_wr
        self.max_training_seconds = max_training_minutes * 60
        self.start_time = time.time()
        self.stop_requested = threading.Event()

        self.server_proc = None
        self.trainer_proc = None
        self.worker_proc = None

        self.episodes = []
        self.best_time = float("inf")
        self.milestones_reached = {}

    def log_status(self, msg):
        elapsed = time.time() - self.start_time
        mins, secs = divmod(int(elapsed), 60)
        timestamp = f"[{mins:02d}m{secs:02d}s]"
        print(f"{timestamp} {msg}", flush=True)

    def start_processes(self):
        self.log_status("[*] Starting TMRL Server...")
        self.server_proc = subprocess.Popen(
            [PYTHON_EXE, "-m", "tmrl", "--server"],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Allow server to bind sockets
        time.sleep(2.0)

        self.log_status("[*] Starting TMRL Trainer (Dreamer Latent RSSM & Policy Optimizer)...")
        self.trainer_proc = subprocess.Popen(
            [PYTHON_EXE, "-m", "tmrl", "--trainer"],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        time.sleep(3.0)

        self.log_status("[*] Starting TMRL Rollout Worker (Active Driving on Track B)...")
        env = dict(os.environ, TMRL_UNIVERSAL_REWARD="0")
        self.worker_proc = subprocess.Popen(
            [PYTHON_EXE, "-m", "tmrl", "--worker"],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Start output logging threads
        threading.Thread(target=self._stream_logs, args=(self.server_proc, "SERVER"), daemon=True).start()
        threading.Thread(target=self._stream_logs, args=(self.trainer_proc, "TRAINER"), daemon=True).start()
        threading.Thread(target=self._stream_worker, args=(self.worker_proc,), daemon=True).start()

    def _stream_logs(self, proc, prefix):
        for line in iter(proc.stdout.readline, ""):
            line = line.strip()
            if not line:
                continue
            # Filter noise
            if any(k in line for k in ["loss_world_model", "epoch", "Round", "saved checkpoint", "Episode", "Loaded"]):
                self.log_status(f"[{prefix}] {line}")

    def _stream_worker(self, proc):
        for line in iter(proc.stdout.readline, ""):
            line = line.strip()
            if not line:
                continue
            self.log_status(f"[WORKER] {line}")

            # Check for episode completion or progress
            if "episode" in line.lower() or "finish" in line.lower():
                self._record_worker_line(line)

    def _record_worker_line(self, line):
        now = time.time()
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": now - self.start_time,
            "raw_log": line,
        }
        self.episodes.append(record)

        # Check for replay file update
        latest_replay_time = get_latest_replay(self.start_time)
        if latest_replay_time is not None:
            self.check_milestone(latest_replay_time)

        self._save_progress()

    def _save_progress(self):
        data = {
            "algorithm": "DREAMER",
            "track": "Summer 2020 - 01",
            "world_record_target": self.target_wr,
            "best_time": self.best_time if self.best_time != float("inf") else None,
            "gap_to_wr": (self.best_time - self.target_wr) if self.best_time != float("inf") else None,
            "milestones": self.milestones_reached,
            "total_episodes": len(self.episodes),
            "episodes": self.episodes[-50:],  # keep recent
        }
        try:
            with open(PROGRESS_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def check_milestone(self, lap_time):
        if lap_time < self.best_time:
            self.best_time = lap_time
            gap = lap_time - self.target_wr
            self.log_status(f"[★ NEW BEST] Lap Time: {lap_time:.3f}s | Gap to WR: {gap:+.3f}s")

        if "clean_lap" not in self.milestones_reached:
            self.milestones_reached["clean_lap"] = {
                "achieved_at_seconds": time.time() - self.start_time,
                "lap_time": lap_time,
            }
            self.log_status(f"[🏆 MILESTONE] First Clean Lap Completed: {lap_time:.3f}s!")

        if lap_time <= GOLD_MEDAL_SECONDS and "gold_medal" not in self.milestones_reached:
            self.milestones_reached["gold_medal"] = {
                "achieved_at_seconds": time.time() - self.start_time,
                "lap_time": lap_time,
            }
            self.log_status(f"[🥇 MILESTONE] GOLD MEDAL BEATEN ({lap_time:.3f}s <= {GOLD_MEDAL_SECONDS:.3f}s)!")

        if lap_time <= AUTHOR_MEDAL_SECONDS and "author_medal" not in self.milestones_reached:
            self.milestones_reached["author_medal"] = {
                "achieved_at_seconds": time.time() - self.start_time,
                "lap_time": lap_time,
            }
            self.log_status(f"[🎖️ MILESTONE] AUTHOR MEDAL BEATEN ({lap_time:.3f}s <= {AUTHOR_MEDAL_SECONDS:.3f}s)!")

        if lap_time <= self.target_wr and "world_record" not in self.milestones_reached:
            self.milestones_reached["world_record"] = {
                "achieved_at_seconds": time.time() - self.start_time,
                "lap_time": lap_time,
            }
            self.log_status(f"[👑 WORLD RECORD BEATEN] {lap_time:.3f}s < {self.target_wr:.3f}s! HUMAN CAPABILITY SURPASSED!")

    def run(self):
        self.start_processes()
        self.log_status("=" * 65)
        self.log_status(f" Dreamer Fast Adaptation Benchmark Active")
        self.log_status(f" Target Track: Summer 2020 - 01 (Track B)")
        self.log_status(f" Target World Record: {self.target_wr:.3f}s")
        self.log_status("=" * 65)

        try:
            while not self.stop_requested.is_set():
                time.sleep(1.0)
                elapsed = time.time() - self.start_time
                if elapsed >= self.max_training_seconds:
                    self.log_status("[!] Max training budget reached.")
                    break

                # Check if processes are alive
                for name, proc in [("Server", self.server_proc), ("Trainer", self.trainer_proc), ("Worker", self.worker_proc)]:
                    if proc and proc.poll() is not None:
                        self.log_status(f"[!] {name} process exited with code {proc.poll()}")
                        return
        except KeyboardInterrupt:
            self.log_status("[!] KeyboardInterrupt received. Stopping gracefully...")
        finally:
            self.shutdown()

    def shutdown(self):
        self.stop_requested.set()
        self.log_status("[*] Terminating TMRL processes...")
        for proc in [self.worker_proc, self.trainer_proc, self.server_proc]:
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=3.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        self._save_progress()
        self.log_status("[✓] All processes stopped. Final progress saved.")


def main():
    parser = argparse.ArgumentParser(description="Train Dreamer on Track B with WR tracking")
    parser.add_argument("--minutes", type=int, default=120, help="Training duration budget in minutes")
    args = parser.parse_args()

    manager = DreamerTrainingManager(max_training_minutes=args.minutes)
    manager.run()


if __name__ == "__main__":
    main()
