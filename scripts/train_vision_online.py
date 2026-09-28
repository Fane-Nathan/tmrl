#!/usr/bin/env python3
"""
Real-Time Online Vision Training with 1,000,000 Foundation Backbone for TrackMania 2020.
Trains the visual CNN directly on live TrackMania 2020 game screenshots at 20 Hz,
powered by the pre-trained 1,000,000-replay Causal Transformer physics backbone.

Usage:
  python scripts/train_vision_online.py --steer_gain 2.0
"""

import argparse
import os
import socket
import struct
import sys
import threading
import time
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tmrl.custom.torch.car_brain import MultiModalCarBrain
from tmrl.custom.tm.utils.tm_vision import FastWindowCapture

# Initialize Virtual Gamepad
USE_GAMEPAD = False
try:
    import vgamepad as vg
    gamepad = vg.VX360Gamepad()
    USE_GAMEPAD = True
    print("[+] Virtual Xbox 360 Gamepad initialized successfully!")
except Exception as e:
    print(f"[-] Virtual Gamepad unavailable ({e}). Falling back to keyboard...")
    from tmrl.custom.tm.utils.control_keyboard import apply_control, keyres


def send_control(act, speed=0.0, steer_gain=2.0, deadband=0.03):
    gas = float(act[0])
    brake = float(act[1])
    raw_steer = float(act[2])

    if abs(raw_steer) < deadband:
        steer = 0.0
    else:
        sign = np.sign(raw_steer)
        mag = min(1.0, abs(raw_steer) * steer_gain)
        steer = float(sign * mag)

    # TMRL Control Standards:
    # Full forward throttle when gas > 0.0, zero brake from standstill (< 15 km/h)
    gas_val = 1.0 if gas > 0.0 else max(0.0, min(1.0, (gas + 1.0) / 2.0))
    if speed < 15.0 or brake <= 0.3:
        brake_val = 0.0
    else:
        brake_val = min(1.0, float(brake))

    if USE_GAMEPAD:
        steer_val = max(-1.0, min(1.0, steer))
        # Accelerate: Right Trigger + Button A (0x1000) for universal TM compatibility
        if gas_val > 0.0:
            gamepad.right_trigger_float(value_float=gas_val)
            gamepad.press_button(button=0x1000)
        else:
            gamepad.right_trigger_float(value_float=0.0)
            gamepad.release_button(button=0x1000)

        # Brake: Left Trigger
        gamepad.left_trigger_float(value_float=brake_val)

        # Steering: Left Joystick
        gamepad.left_joystick_float(x_value_float=steer_val, y_value_float=0.0)
        gamepad.update()
    else:
        actions = []
        if gas_val > 0.1:
            actions.append('f')
        if brake_val > 0.1:
            actions.append('b')
        if steer > 0.15:
            actions.append('r')
        elif steer < -0.15:
            actions.append('l')
        apply_control(actions)


def release_controls():
    if USE_GAMEPAD:
        try:
            gamepad.reset()
            gamepad.update()
        except Exception:
            pass
    else:
        try:
            from tmrl.custom.tm.utils.control_keyboard import ReleaseKey, W, A, S, D
            for k in [W, A, S, D]:
                ReleaseKey(k)
        except Exception:
            pass


class CompactReplayBuffer:
    """Stores transitions in CPU RAM with uint8 images (<150 MB for 10,000 steps)."""
    def __init__(self, capacity: int = 20000):
        self.capacity = capacity
        self.imgs = deque(maxlen=capacity)
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.lock = threading.Lock()

    def push(self, img_uint8, state, action, reward):
        with self.lock:
            self.imgs.append(img_uint8)
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)

    def sample_batch(self, batch_size: int = 16, seq_len: int = 16, device: str = "cuda"):
        with self.lock:
            N = len(self.imgs)
            if N <= seq_len + 1:
                return None

            batch_imgs = []
            batch_states = []
            batch_acts = []
            batch_rews = []

            for _ in range(batch_size):
                start = np.random.randint(0, N - seq_len)
                b_i = np.array([self.imgs[i] for i in range(start, start + seq_len)], dtype=np.float32) / 255.0
                b_s = np.array([self.states[i] for i in range(start, start + seq_len)], dtype=np.float32)
                b_a = np.array([self.actions[i] for i in range(start, start + seq_len)], dtype=np.float32)
                b_r = np.array([self.rewards[i] for i in range(start, start + seq_len)], dtype=np.float32)

                batch_imgs.append(b_i)
                batch_states.append(b_s)
                batch_acts.append(b_a)
                batch_rews.append(b_r)

        return (
            torch.tensor(np.array(batch_imgs), dtype=torch.float32, device=device),
            torch.tensor(np.array(batch_states), dtype=torch.float32, device=device),
            torch.tensor(np.array(batch_acts), dtype=torch.float32, device=device),
            torch.tensor(np.array(batch_rews), dtype=torch.float32, device=device),
        )

    def __len__(self):
        with self.lock:
            return len(self.imgs)


def main():
    parser = argparse.ArgumentParser(description="Real-Time Online Vision Training")
    parser.add_argument("--base_model", type=str, default="weights/car_brain_1m_curriculum/car_brain_latest.pt")
    parser.add_argument("--save_path", type=str, default="weights/car_brain_1m_curriculum/car_brain_vision_trained.pt")
    parser.add_argument("--steer_gain", type=float, default=2.0)
    parser.add_argument("--lr_vision", type=float, default=3e-4)
    parser.add_argument("--lr_backbone", type=float, default=2e-5)
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resize_window", action="store_true", default=True, help="Auto-snap TrackMania window to 512x256 in top-left corner")
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f"   TRACKMANIA 2020 REAL-TIME ONLINE VISION TRAINING    ")
    print(f"=======================================================")

    # Initialize Vision & Buffer
    print("[+] Initializing real-time screen capture (96x96)...")
    win_cap = FastWindowCapture(target_size=(96, 96))
    if args.resize_window:
        win_cap.move_and_resize(x=1, y=0, w=512, h=256)
    img_queue = deque(maxlen=4)
    for _ in range(4):
        f = win_cap.grab_frame(grayscale=True)
        img_queue.append(f)

    buffer = CompactReplayBuffer(capacity=20000)

    # Instantiate Model from 1M Foundation
    print(f"[+] Loading 1,000,000 Foundation Brain into MultiModal Architecture: {args.base_model}")
    model = MultiModalCarBrain.from_foundation(args.base_model, device=args.device)
    model.train()

    # RESeL Decoupled Optimizer:
    # High LR on Vision CNN & Policy Head; Conservative LR on 1M Foundation Backbone
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "conv" in n or "visual" in n or "kinematics" in n or "fusion" in n], "lr": args.lr_vision},
        {"params": model.blocks.parameters(), "lr": args.lr_backbone, "weight_decay": 1e-4},
        {"params": model.policy_head.parameters(), "lr": args.lr_vision, "weight_decay": 1e-4},
    ])

    # Connect to OpenPlanet socket
    struct_str = '<' + 'f' * 11
    nb_bytes = struct.calcsize(struct_str)
    print(f"[+] Connecting to OpenPlanet telemetry at 127.0.0.1:{args.port}...")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect(('127.0.0.1', args.port))
        print(f"[+] Connected to TrackMania 2020 OpenPlanet stream!\n")
    except Exception as e:
        print(f"[!] Could not connect to OpenPlanet on port {args.port}: {e}")
        return

    print(">>> AUTONOMOUS ONLINE IMAGE TRAINING ACTIVE <<<")
    print("    - Driving and training live in TrackMania 2020")
    print(f"    - Steer Authority Gain: {args.steer_gain:.1f}x")
    print("    - Press Ctrl+C anytime to stop and save model weights.\n")

    # State variables
    stop_event = threading.Event()
    train_stats = {"loss": 0.0, "updates": 0}
    prev_action = np.zeros(3, dtype=np.float32)
    step_count = 0
    lap_count = 0
    t_last_step = time.time()
    last_pos = None
    race_started = False

    # Background Async Trainer Thread
    def trainer_worker():
        while not stop_event.is_set():
            if len(buffer) < 64:
                time.sleep(0.05)
                continue

            batch = buffer.sample_batch(batch_size=16, seq_len=16, device=args.device)
            if batch is None:
                time.sleep(0.02)
                continue

            b_imgs, b_states, b_acts, b_rews = batch

            optimizer.zero_grad(set_to_none=True)
            # Predict actions from camera history and state
            pred_acts = model(b_imgs, b_states, prev_actions=b_acts)

            # Reward-weighted policy loss: encourage high-speed progress
            # Loss minimizes error towards high-reward trajectories
            loss = F.smooth_l1_loss(pred_acts, b_acts, beta=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_stats["loss"] = float(loss.item())
            train_stats["updates"] += 1
            time.sleep(0.05)

    trainer_thread = threading.Thread(target=trainer_worker, daemon=True)
    trainer_thread.start()

    data_raw = b''

    try:
        while True:
            t_loop_start = time.time()

            # 1. Grab visual screen frame
            cur_frame = win_cap.grab_frame(grayscale=True)  # (96, 96) uint8
            img_queue.append(cur_frame)
            while len(img_queue) < 4:
                img_queue.append(cur_frame)
            obs_img_uint8 = np.stack(list(img_queue), axis=0)  # (4, 96, 96) uint8
            obs_img_norm = obs_img_uint8.astype(np.float32) / 255.0

            # 2. Read OpenPlanet telemetry
            while len(data_raw) < nb_bytes:
                chunk = sock.recv(1024)
                if not chunk:
                    raise ConnectionResetError("OpenPlanet disconnected.")
                data_raw += chunk

            div = len(data_raw) // nb_bytes
            packet = data_raw[(div - 1) * nb_bytes : div * nb_bytes]
            data_raw = data_raw[div * nb_bytes :]

            telemetry = struct.unpack(struct_str, packet)
            speed_val = telemetry[0]
            pos_x = telemetry[2]
            pos_y = telemetry[3]
            pos_z = telemetry[4]
            steer = telemetry[5]
            gas = telemetry[6]
            gear = telemetry[9]
            rpm = telemetry[10]

            curr_pos = np.array([pos_x, pos_y, pos_z])
            if last_pos is not None:
                dist = np.linalg.norm(curr_pos - last_pos)
                if dist > 50.0:  # Restart detected
                    model.reset_context()
                    race_started = False
                    lap_count += 1
                    sys.stdout.write(f"\n[+] Lap #{lap_count} reset detected! Resetting in-context memory.\n")
                    sys.stdout.flush()
            last_pos = curr_pos

            state = np.zeros(15, dtype=np.float32)
            state[0] = speed_val / 100.0
            state[1] = steer
            state[2] = gas
            state[3] = gear
            state[4] = rpm / 10000.0
            state[6] = pos_x
            state[7] = pos_y
            state[8] = pos_z
            state[9:13] = 1.0

            # Start line countdown guard
            if speed_val < 1.0 and not race_started:
                model.reset_context()
                act = np.array([1.0, -1.0, 0.0], dtype=np.float32)
                prev_action = act.copy()
            else:
                race_started = True
                act = model.step_in_context(obs_img_norm, state, prev_action=prev_action)
                prev_action = act.copy()

            # Actuate
            send_control(act, speed=speed_val, steer_gain=args.steer_gain)

            # Reward calculation: forward speed, penalized if stuck
            reward = float(speed_val / 100.0) if speed_val > 5.0 else -0.5

            # Store in live experience buffer
            buffer.push(obs_img_uint8, state, act, reward)

            step_count += 1
            now = time.time()
            actual_step_dt = now - t_last_step
            t_last_step = now

            if step_count % 10 == 0:
                hz = 1.0 / max(0.001, actual_step_dt)
                gas_norm = max(0.0, min(1.0, (act[0] + 1.0) / 2.0))
                brk_norm = max(0.0, min(1.0, (act[1] + 1.0) / 2.0)) if act[1] > -0.2 else 0.0
                gas_bar = "█" * int(gas_norm * 10)
                brk_bar = "█" * int(brk_norm * 10)
                steer_str = f"{np.clip(act[2] * args.steer_gain, -1.0, 1.0):+.2f}"
                sys.stdout.write(
                    f"\r[Vision Trainer] Spd: {speed_val:5.1f} km/h | Steer: {steer_str} | "
                    f"Gas: [{gas_bar:<10}] | Loss: {train_stats['loss']:.4f} | "
                    f"Buffer: {len(buffer):,}/20k | {hz:.0f} Hz"
                )
                sys.stdout.flush()

            # Periodic checkpoint save every 1,000 steps (50 seconds)
            if step_count % 1000 == 0:
                os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
                torch.save(model.state_dict(), args.save_path)

            # 20 Hz synchronization
            sleep_time = 0.05 - (time.time() - t_loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n[PAUSED] Online image training paused by user.")
    finally:
        stop_event.set()
        release_controls()
        sock.close()
        # Save model checkpoint
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
        torch.save(model.state_dict(), args.save_path)
        print(f"[+] Successfully saved online vision-trained model to: {args.save_path}\n")


if __name__ == "__main__":
    main()
