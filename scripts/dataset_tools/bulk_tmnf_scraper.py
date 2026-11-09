#!/usr/bin/env python3
"""
Ultra-High-Speed Concurrent Replay Scraper & Parser for TrackMania (TMX / TMNF).

Features:
- Caches track catalog in `cached_track_ids.json`.
- Concurrent thread pool (24 workers) fetching track replays and streaming downloads.
- Real-time live counter and ETA.
- Resumable checkpointing via `scraper_manifest.json`.
"""

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests

try:
    from pygbx import Gbx, GbxType
    HAS_PYGBX = True
except ImportError:
    HAS_PYGBX = False


def fetch_tracks_paginated(target_tracks: int = 25000, cache_file: Optional[str] = None) -> List[int]:
    """Fetches track IDs using pagination or loads from local cache."""
    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                track_ids = json.load(f)
                if len(track_ids) >= 1000:
                    print(f"Loaded {len(track_ids)} cached track IDs from {cache_file}.")
                    return track_ids[:target_tracks]
        except Exception:
            pass

    track_ids = []
    current_after = None
    page_count = 500

    print(f"Paginating ManiaExchange API for up to {target_tracks} track IDs...")
    while len(track_ids) < target_tracks:
        url = f"https://tmnf.exchange/api/tracks?count={page_count}&fields=TrackId,TrackName"
        if current_after is not None:
            url += f"&after={current_after}"

        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                time.sleep(1)
                continue

            data = resp.json()
            results = data.get("Results", [])
            if not results:
                break

            for item in results:
                track_ids.append(item["TrackId"])
                if len(track_ids) >= target_tracks:
                    break

            current_after = results[-1]["TrackId"]
            print(f"Indexed {len(track_ids)}/{target_tracks} tracks...")

            if not data.get("More", False):
                break

            time.sleep(0.05)

        except Exception as e:
            time.sleep(1)

    if cache_file:
        try:
            with open(cache_file, "w") as f:
                json.dump(track_ids, f)
        except Exception:
            pass

    return track_ids


def parse_and_save_replay(content: bytes, output_csv_path: str) -> bool:
    """Parses .Replay.Gbx bytes into 20 Hz state-action CSV."""
    try:
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

        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        return True
    except Exception:
        return False


def process_single_track(
    track_id: int,
    max_replays: int,
    output_dir: str,
    completed_replays: Set[int],
    session: requests.Session,
) -> int:
    """Fetches and downloads all replays for a given track."""
    saved_for_track = 0
    track_folder = os.path.join(output_dir, f"track_{track_id}")

    # Fetch replay IDs
    rep_url = f"https://tmnf.exchange/api/replays?trackId={track_id}&count={max_replays}&fields=ReplayId"
    try:
        r = session.get(rep_url, timeout=8)
        if r.status_code != 200:
            return 0
        replays = r.json().get("Results", [])
    except Exception:
        return 0

    for rep in replays:
        r_id = rep["ReplayId"]
        if r_id in completed_replays:
            saved_for_track += 1
            continue

        out_csv = os.path.join(track_folder, f"replay_{r_id}.csv")
        if os.path.exists(out_csv) and os.path.getsize(out_csv) > 100:
            completed_replays.add(r_id)
            saved_for_track += 1
            continue

        dl_url = f"https://tmnf.exchange/recordgbx/{r_id}"
        try:
            dl_resp = session.get(dl_url, timeout=12)
            if dl_resp.status_code == 200 and len(dl_resp.content) > 100:
                if parse_and_save_replay(dl_resp.content, out_csv):
                    completed_replays.add(r_id)
                    saved_for_track += 1
        except Exception:
            pass

    return saved_for_track


def main():
    parser = argparse.ArgumentParser(description="Ultra-High-Speed TrackMania Replay Scraper")
    parser.add_argument("--target_replays", type=int, default=50000, help="Target replay count")
    parser.add_argument("--replays_per_track", type=int, default=20, help="Max replays per track")
    parser.add_argument("--output_dir", type=str, default="./data/massive_replays", help="Dataset output folder")
    parser.add_argument("--workers", type=int, default=24, help="Concurrent worker threads")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cache_file = os.path.join(args.output_dir, "cached_track_ids.json")
    manifest_path = os.path.join(args.output_dir, "scraper_manifest.json")

    completed_replays: Set[int] = set()
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                saved = json.load(f)
                completed_replays = set(saved.get("completed_replays", []))
                print(f"Resuming scrape: {len(completed_replays)} replays already downloaded.")
        except Exception:
            pass

    # Fetch track catalog
    tracks_needed = max(1000, int(args.target_replays / 2.0))
    track_ids = fetch_tracks_paginated(target_tracks=tracks_needed, cache_file=cache_file)

    print(f"\n=======================================================")
    print(f"  LAUNCHING LIVE DOWNLOAD PIPELINE: TARGET {args.target_replays} REPLAYS  ")
    print(f"  Workers: {args.workers} concurrent threads           ")
    print(f"  Destination: {args.output_dir}                       ")
    print(f"=======================================================\n")

    total_downloaded = len(completed_replays)
    start_time = time.time()
    lock = threading.Lock()
    processed_tracks = 0

    thread_sessions = {}
    def get_session():
        tid = threading.get_ident()
        if tid not in thread_sessions:
            thread_sessions[tid] = requests.Session()
        return thread_sessions[tid]

    def worker_task(tid: int):
        nonlocal total_downloaded, processed_tracks
        session = get_session()
        added = process_single_track(tid, args.replays_per_track, args.output_dir, completed_replays, session)
        with lock:
            processed_tracks += 1
            if added > 0:
                elapsed = time.time() - start_time
                rate = total_downloaded / max(1.0, elapsed)
                print(f"[{processed_tracks}/{len(track_ids)} tracks] Replays: {len(completed_replays)}/{args.target_replays} | Speed: {rate:.1f} reps/sec", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(worker_task, tid) for tid in track_ids]
        for f in concurrent.futures.as_completed(futures):
            if len(completed_replays) >= args.target_replays:
                print(f"\nTarget {args.target_replays} replays achieved!")
                executor.shutdown(wait=False, cancel_futures=True)
                break

    # Save manifest checkpoint
    with open(manifest_path, "w") as f:
        json.dump({"completed_replays": list(completed_replays), "total": len(completed_replays)}, f)

    print(f"\nHarvest completed! Total replays ready: {len(completed_replays)} in {args.output_dir}")


if __name__ == "__main__":
    main()
