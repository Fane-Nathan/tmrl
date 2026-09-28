#!/usr/bin/env python3
"""
Records human expert driving demonstration directly from TrackMania 2020 via OpenPlanet
with synchronized 96x96 4-frame vision capture.
Saves 20 Hz multi-modal state-action-vision sequences formatted for foundation fine-tuning.

Usage:
  python scripts/record_track_demonstration.py --output data/target_track_demos.pt
"""

import argparse
import hashlib
import os
import socket
import struct
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from tmrl.custom.tm.utils.tm_vision import FastWindowCapture


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Record TrackMania Multi-Modal Demonstration")
    parser.add_argument("--port", type=int, default=9000, help="OpenPlanet telemetry port")
    parser.add_argument("--output", type=str, default="data/target_track_demos.pt", help="Path to save recorded demo")
    parser.add_argument("--min_speed", type=float, default=1.0, help="Min speed to begin recording steps (km/h)")
    parser.add_argument(
        "--track-id",
        default=None,
        help="Stable map name/UID stored with every demonstration",
    )
    parser.add_argument(
        "--map-file",
        default=None,
        help="Optional .Map.Gbx file; its absolute path and SHA-256 are recorded",
    )
    args = parser.parse_args()

    map_file = Path(args.map_file).resolve() if args.map_file else None
    if map_file is not None and not map_file.is_file():
        parser.error(f"--map-file does not exist: {map_file}")
    track_id = args.track_id or (map_file.stem if map_file is not None else None)
    if track_id is None:
        print(
            "[!] No --track-id/--map-file supplied. The recording will be marked "
            "as unknown-track and must not be counted as cross-track data."
        )

    struct_str = '<' + 'f' * 11
    nb_bytes = struct.calcsize(struct_str)

    print(f"\n=======================================================")
    print(f"       TRACKMANIA MULTI-MODAL DEMONSTRATION RECORDER   ")
    print(f"=======================================================")
    print(f"[+] Initializing window capture (96x96 grayscale)...")
    win_cap = FastWindowCapture(target_size=(96, 96))
    img_queue = deque(maxlen=4)
    for _ in range(4):
        f = win_cap.grab_frame(grayscale=True).astype(np.float32) / 255.0
        img_queue.append(f)

    print(f"[+] Connecting to OpenPlanet stream at 127.0.0.1:{args.port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect(('127.0.0.1', args.port))
        print(f"[+] Connected to TrackMania 2020! Ready to record.\n")
    except Exception as e:
        print(f"[!] Could not connect to port {args.port}: {e}")
        return

    print(">>> DRIVE NORMALLY IN TRACKMANIA <<<")
    print("    - Drive 1 to 3 clean laps on your current track.")
    print("    - When you're done, press Ctrl+C in this terminal to save!\n")

    episodes = []
    current_imgs = []
    current_states = []
    current_actions = []

    last_pos = None
    data_raw = b''
    step_count = 0
    t_last_step = time.time()

    try:
        while True:
            t_loop_start = time.time()

            # 1. Grab visual frame
            frame = win_cap.grab_frame(grayscale=True).astype(np.float32) / 255.0
            img_queue.append(frame)
            while len(img_queue) < 4:
                img_queue.append(frame)
            obs_img = np.stack(list(img_queue), axis=0)  # (4, 96, 96)

            # 2. Grab telemetry
            while len(data_raw) < nb_bytes:
                chunk = sock.recv(1024)
                if not chunk:
                    raise ConnectionResetError("OpenPlanet disconnected.")
                data_raw += chunk

            div = len(data_raw) // nb_bytes
            packet = data_raw[(div - 1) * nb_bytes : div * nb_bytes]
            data_raw = data_raw[div * nb_bytes :]

            telemetry = struct.unpack(struct_str, packet)
            speed = telemetry[0]
            pos_x = telemetry[2]
            pos_y = telemetry[3]
            pos_z = telemetry[4]
            steer = telemetry[5]
            gas = telemetry[6]
            is_braking = telemetry[7]
            gear = telemetry[9]
            rpm = telemetry[10]

            # Detect race restart (sudden position jump > 50m while driving)
            curr_pos = np.array([pos_x, pos_y, pos_z])
            if last_pos is not None and len(current_states) > 30:
                dist = np.linalg.norm(curr_pos - last_pos)
                if dist > 50.0:  # Player restarted race
                    print(f"\n[+] Lap/Reset detected! Stored episode with {len(current_states)} steps.")
                    episodes.append({
                        "imgs": np.array(current_imgs, dtype=np.float32),
                        "states": np.array(current_states, dtype=np.float32),
                        "actions": np.array(current_actions, dtype=np.float32),
                    })
                    current_imgs = []
                    current_states = []
                    current_actions = []

            last_pos = curr_pos

            # If driving, record the timestep
            if speed > args.min_speed or len(current_states) > 0:
                # 15D state vector matching foundation pretraining
                state = np.zeros(15, dtype=np.float32)
                state[0] = speed / 100.0
                state[1] = steer
                state[2] = gas
                state[3] = gear
                state[4] = rpm / 10000.0
                state[6] = pos_x
                state[7] = pos_y
                state[8] = pos_z
                state[9:13] = 1.0

                # Action: [gas, brake, steer] in [-1, 1]
                gas_act = gas * 2.0 - 1.0
                brake_act = 1.0 if is_braking > 0.5 else -1.0
                steer_act = np.clip(steer, -1.0, 1.0)
                action = np.array([gas_act, brake_act, steer_act], dtype=np.float32)

                current_imgs.append(obs_img)
                current_states.append(state)
                current_actions.append(action)
                step_count += 1

                if step_count % 10 == 0:
                    laps_recorded = len(episodes)
                    cur_steps = len(current_states)
                    sys.stdout.write(
                        f"\r[Recording] Active Lap Steps: {cur_steps:4d} | Total Laps: {laps_recorded} | "
                        f"Speed: {speed:5.1f} km/h | Steer: {steer:+.2f} | Gas: {gas:.2f} | Brk: {is_braking:.0f} "
                    )
                    sys.stdout.flush()

            # Maintain 20 Hz (0.05s per step)
            sleep_time = 0.05 - (time.time() - t_loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n[PAUSED] Recording ended by user.")
    finally:
        sock.close()
        if len(current_states) > 30:
            print(f"[+] Storing final active lap ({len(current_states)} steps)...")
            episodes.append({
                "imgs": np.array(current_imgs, dtype=np.float32),
                "states": np.array(current_states, dtype=np.float32),
                "actions": np.array(current_actions, dtype=np.float32),
            })

        if episodes:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            total_timesteps = sum(len(ep["states"]) for ep in episodes)
            metadata = {
                "schema_version": 2,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "track_id": track_id or "unknown-track",
                "map_file": str(map_file) if map_file is not None else None,
                "map_sha256": sha256_file(map_file) if map_file is not None else None,
                "sampling_hz": 20,
                "image_format": "4x96x96 grayscale float32 in [0,1]",
                "state_format": "15D; speed/100, raw gear, rpm/10000, xyz at 6:9",
                "action_format": "[gas, brake, steer] in [-1,1]",
                "episode_count": len(episodes),
                "transition_count": total_timesteps,
            }
            torch.save(
                {"episodes": episodes, "has_vision": True, "metadata": metadata},
                args.output,
            )
            print(f"[SUCCESS] Saved {len(episodes)} laps ({total_timesteps:,} total timesteps) to {args.output}!")
        else:
            print("[-] No valid driving episodes recorded.")


if __name__ == "__main__":
    main()
