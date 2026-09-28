#!/usr/bin/env python3
"""
Converts recorded expert demonstration positions from data/target_track_demos.pt
into TMRL's reward.pkl reference racing trajectory (0.1m checkpoints).
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pickle
import numpy as np
import torch
import tmrl.config.config_constants as cfg
from tmrl.tools.record import line

def main():
    demo_path = Path("data/target_track_demos.pt")
    if not demo_path.exists():
        print(f"[-] Demo file not found: {demo_path}")
        return

    print(f"[+] Loading recorded demo from {demo_path}...")
    data = torch.load(demo_path, map_location="cpu", weights_only=False)
    episodes = data["episodes"]
    print(f"[+] Found {len(episodes)} episodes.")

    # Choose the fastest / longest clean lap (Episode 1: 1106 steps or Ep 0: 1121 steps)
    best_ep = min(episodes, key=lambda e: len(e["states"]))
    positions = best_ep["states"][:, 6:9]  # pos_x, pos_y, pos_z
    print(f"[+] Using best demo lap with {len(positions)} captured positions.")

    # Interpolate into 0.1m spaced checkpoints matching TMRL standard
    final_positions = [positions[0]]
    dist_between_points = 0.1
    j = 1
    move_by = dist_between_points
    pt1 = final_positions[-1]

    while j < len(positions):
        pt2 = positions[j]
        pt, dst = line(pt1, pt2, move_by)
        if pt is not None:
            final_positions.append(pt)
            move_by = dist_between_points
            pt1 = pt
        else:
            pt1 = pt2
            j += 1
            move_by = dst

    final_positions = np.array(final_positions, dtype=np.float32)
    print(f"[+] Generated {len(final_positions)} checkpoints for the track reward line.")

    destinations = [
        Path(cfg.REWARD_PATH),
        Path("D:/Project/TmrlData/reward/reward.pkl"),
    ]

    for dest in destinations:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            pickle.dump(final_positions, f)
        print(f"[+] Saved reward trajectory to: {dest}")

    # Also build self-play ghost trajectory (x, y, z, speed, timestamp)
    speeds = best_ep["states"][:, 0] * 100.0  # speed in km/h
    T = len(positions)
    timestamps = np.arange(T, dtype=np.float32) * 0.05
    ghost_traj = np.zeros((T, 5), dtype=np.float32)
    ghost_traj[:, :3] = positions
    ghost_traj[:, 3] = speeds
    ghost_traj[:, 4] = timestamps

    self_play_data = {
        "trajectory": ghost_traj,
        "best_distance": float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))),
        "best_time": float(timestamps[-1]),
        "best_finished": True,
    }

    sp_destinations = [
        cfg.REWARD_FOLDER / "self_play_reward.pkl",
        Path("D:/Project/TmrlData/reward/self_play_reward.pkl"),
    ]
    for sp_dest in sp_destinations:
        sp_dest.parent.mkdir(parents=True, exist_ok=True)
        with open(sp_dest, "wb") as f:
            pickle.dump(self_play_data, f)
        print(f"[+] Saved self-play ghost reference to: {sp_dest}")

    print("\n[+] Verification: Testing RewardFunction loading...")
    from tmrl.custom.tm.utils.compute_reward import RewardFunction
    from tmrl.custom.tm.utils.self_play_reward import SelfPlayRewardFunction
    rf = RewardFunction(str(destinations[0]))
    print(f"[+] RewardFunction loaded successfully with datalen={rf.datalen} checkpoints.")
    spf = SelfPlayRewardFunction(reward_data_path=str(sp_destinations[0]))
    print(f"[+] SelfPlayRewardFunction loaded successfully with {len(spf.manager.best_trajectory)} ghost points.")

if __name__ == "__main__":
    main()
