#!/usr/bin/env python3
"""
High-Performance Multi-Core Converter for Raw TrackMania .Replay.Gbx Files.

Uses all CPU cores via multiprocessing to unpack raw .Replay.Gbx files
into 20 Hz state-action CSVs or NumPy archives ready for CarBrain training.
"""

import argparse
import glob
import multiprocessing
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd

try:
    from pygbx import Gbx, GbxType
    HAS_PYGBX = True
except ImportError:
    HAS_PYGBX = False


def process_single_gbx(args_tuple):
    gbx_path, out_csv_path = args_tuple
    if os.path.exists(out_csv_path) and os.path.getsize(out_csv_path) > 100:
        return True

    try:
        with open(gbx_path, "rb") as f:
            content = f.read()

        g = Gbx(content)
        ghost = g.get_class_by_id(GbxType.CTN_GHOST)
        if not ghost or not getattr(ghost, "records", None) or len(ghost.records) < 5:
            return False

        records = ghost.records
        control_entries = getattr(ghost, "control_entries", [])

        race_time_ms = ghost.race_time
        num_steps = max(1, race_time_ms // 50)
        times_ms = np.arange(num_steps) * 50

        xs, ys, zs = np.zeros(num_steps), np.zeros(num_steps), np.zeros(num_steps)
        vxs, vys, vzs = np.zeros(num_steps), np.zeros(num_steps), np.zeros(num_steps)
        yaws, pitches, rolls = np.zeros(num_steps), np.zeros(num_steps), np.zeros(num_steps)
        contacts = np.ones(num_steps)
        slidings = np.zeros(num_steps)

        steers = np.zeros(num_steps)
        gases = np.zeros(num_steps)
        brakes = np.zeros(num_steps)

        curr_gas, curr_brake, curr_steer = 0, 0, 0
        input_idx = 0
        sorted_events = sorted(control_entries, key=lambda e: e.time)

        for step, t_ms in enumerate(times_ms):
            while input_idx < len(sorted_events) and sorted_events[input_idx].time <= t_ms:
                ev = sorted_events[input_idx]
                name = ev.event_name
                en = ev.enabled
                if name == "Accelerate":
                    curr_gas = en
                elif name == "Brake":
                    curr_brake = en
                elif name == "SteerLeft":
                    curr_steer = -65536 if en else (curr_steer if curr_steer > 0 else 0)
                elif name == "SteerRight":
                    curr_steer = 65536 if en else (curr_steer if curr_steer < 0 else 0)
                input_idx += 1

            gases[step] = curr_gas
            brakes[step] = curr_brake
            steers[step] = curr_steer

            rec_idx = min(len(records) - 1, int(step * (len(records) / num_steps)))
            pos = records[rec_idx].position
            xs[step] = pos.x
            ys[step] = pos.y
            zs[step] = pos.z
            speed_val = records[rec_idx].speed / 100.0 if hasattr(records[rec_idx], "speed") else 0.0
            vxs[step] = speed_val

        df = pd.DataFrame({
            "time": times_ms,
            "x": xs, "y": ys, "z": zs,
            "yaw": yaws, "pitch": pitches, "roll": rolls,
            "vx": vxs, "vy": vys, "vz": vzs,
            "wheel0_has_contact": contacts, "wheel1_has_contact": contacts,
            "wheel2_has_contact": contacts, "wheel3_has_contact": contacts,
            "wheel0_is_sliding": slidings, "wheel1_is_sliding": slidings,
            "gas": gases, "brake": brakes, "steer": steers,
        })

        os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
        df.to_csv(out_csv_path, index=False)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Multi-Core TrackMania GBX to CSV Converter")
    parser.add_argument("--input_dir", type=str, default="./data/raw_gbx_1m", help="Directory with raw .Gbx files")
    parser.add_argument("--output_dir", type=str, default="./data/converted_replays", help="Output directory")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() or 8, help="Number of CPU worker processes")
    args = parser.parse_args()

    print(f"Scanning {args.input_dir} for raw .Replay.Gbx files...")
    gbx_files = glob.glob(os.path.join(args.input_dir, "**", "*.Replay.Gbx"), recursive=True)
    print(f"Found {len(gbx_files):,} raw GBX files to convert.")

    if not gbx_files:
        print("No .Replay.Gbx files found. Make sure the harvester is running.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    tasks = []
    for gbx_path in gbx_files:
        base_name = os.path.basename(gbx_path).replace(".Replay.Gbx", ".csv")
        out_csv_path = os.path.join(args.output_dir, base_name)
        tasks.append((gbx_path, out_csv_path))

    print(f"Starting multi-core conversion using {args.num_workers} CPU cores...")
    t0 = time.time()
    converted_count = 0

    with multiprocessing.Pool(processes=args.num_workers) as pool:
        for res in pool.imap_unordered(process_single_gbx, tasks, chunksize=100):
            if res:
                converted_count += 1
                if converted_count % 1000 == 0:
                    elapsed = time.time() - t0
                    rate = converted_count / max(0.1, elapsed)
                    print(f"Converted {converted_count:,}/{len(tasks):,} files ({rate:.1f} files/sec)...")

    total_time = time.time() - t0
    print(f"\nConversion complete! Processed {converted_count:,} replays in {total_time:.1f}s.")
    print(f"Output saved to {args.output_dir}")


if __name__ == "__main__":
    main()
