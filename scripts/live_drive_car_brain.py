#!/usr/bin/env python3
"""
Multi-Modal In-Game Inference Driver for TrackMania 2020.
Fuses:
1. Real-Time Camera Vision (96x96 4-frame screen capture via native WindowInterface)
2. 1,000,000-Replay Transformer Physics Backbone (CarBrain / MultiModalCarBrain)
3. OpenPlanet Telemetry Stream (127.0.0.1:9000)
4. Virtual Xbox 360 Gamepad (vgamepad) or DirectInput Keyboard

Usage:
  python scripts/live_drive_car_brain.py --multimodal --steer_gain 2.0
"""

import argparse
import os
import socket
import struct
import sys
import time
from collections import deque
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch

from tmrl.custom.torch.car_brain import CarBrain, MultiModalCarBrain
from tmrl.custom.tm.utils.tm_vision import FastWindowCapture

# Try initializing Virtual Gamepad
USE_GAMEPAD = False
try:
    import vgamepad as vg
    gamepad = vg.VX360Gamepad()
    USE_GAMEPAD = True
    print("[+] Virtual Xbox 360 Gamepad initialized successfully!")
    from tmrl.custom.tm.utils.control_gamepad import gamepad_reset
except Exception as e:
    print(f"[-] Virtual Gamepad unavailable ({e}). Falling back to keyboard...")
    from tmrl.custom.tm.utils.control_keyboard import apply_control

from tmrl.custom.tm.utils.control_keyboard import keyres


def send_control(act, speed=0.0, steer_gain=2.0, deadband=0.03):
    """
    Applies continuous actions to TrackMania using TMRL standard mappings.
    """
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
    """Resets all control inputs upon exit."""
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


def main():
    parser = argparse.ArgumentParser(description="Live Multi-Modal TrackMania Driver")
    parser.add_argument("--model_path", type=str, default="weights/car_brain_1m_curriculum/car_brain_multimodal.pt")
    parser.add_argument("--base_model", type=str, default="weights/car_brain_1m_curriculum/car_brain_latest.pt")
    parser.add_argument("--multimodal", action="store_true", default=True, help="Enable real-time camera vision")
    parser.add_argument("--steer_gain", type=float, default=2.0, help="Steering authority multiplier (default: 2.0)")
    parser.add_argument("--port", type=int, default=9000, help="OpenPlanet telemetry port")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resize_window", action="store_true", default=True, help="Auto-snap TrackMania window to 512x256 in top-left corner")
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f"   TRACKMANIA 2020 MULTI-MODAL CAR BRAIN AUTONOMY     ")
    print(f"=======================================================")

    # Initialize Vision System
    win_cap = None
    img_history = deque(maxlen=4)
    dummy_frame = np.zeros((96, 96), dtype=np.float32)

    if args.multimodal:
        print("[+] Initializing real-time screen capture pipeline (96x96 grayscale, 4-frame history)...")
        win_cap = FastWindowCapture(target_size=(96, 96))
        if args.resize_window:
            win_cap.move_and_resize(x=1, y=0, w=512, h=256)
        # Warm up 4 frames
        for _ in range(4):
            f = win_cap.grab_frame(grayscale=True).astype(np.float32) / 255.0
            img_history.append(f)
        print("[+] Vision system online!")

    # Load Model
    if os.path.exists(args.model_path):
        print(f"Loading Multi-Modal Car Brain: {args.model_path}")
        try:
            model = MultiModalCarBrain(d_model=256, n_layers=6, n_heads=8)
            model.load_state_dict(torch.load(args.model_path, map_location=args.device))
            is_multimodal = True
        except Exception as e:
            print(f"[-] Could not load as MultiModalCarBrain ({e}), loading standard CarBrain...")
            model = CarBrain.from_pretrained(args.model_path, device=args.device)
            is_multimodal = False
    else:
        print(f"[+] Initializing Multi-Modal Brain from 1M foundation weights: {args.base_model}")
        model = MultiModalCarBrain.from_foundation(args.base_model, device=args.device)
        is_multimodal = True

    model.to(args.device)
    model.eval()
    print(f"[+] Model loaded on {args.device.upper()} (Multi-Modal Vision: {is_multimodal})!")

    # Connect to OpenPlanet socket
    struct_str = '<' + 'f' * 11
    nb_bytes = struct.calcsize(struct_str)
    print(f"[+] Connecting to OpenPlanet telemetry at 127.0.0.1:{args.port}...")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect(('127.0.0.1', args.port))
        print(f"[+] Connected to TrackMania 2020 OpenPlanet stream!")
    except Exception as e:
        print(f"[!] Could not connect to OpenPlanet on port {args.port}: {e}")
        print("    Please ensure TrackMania 2020 is running with OpenPlanet TMRL_GrabData plugin active.")
        return

    print("\n[READY] Handing full autonomous driving control over to the Car Brain!")
    print(f"        Steering Authority Gain: {args.steer_gain:.1f}x")
    print("        Press Ctrl+C in this terminal anytime to reclaim manual control.\n")

    prev_action = np.zeros(3, dtype=np.float32)
    step_count = 0
    t_last_step = time.time()
    t_last_reload_check = time.time()
    last_mtime = os.path.getmtime(args.model_path) if os.path.exists(args.model_path) else 0.0
    race_started = False
    last_pos = None
    stuck_steps = 0

    data_raw = b''

    try:
        while True:
            t_loop_start = time.time()

            # Dynamic Hot-Reload: Poll disk every 5s for newly trained checkpoints
            if time.time() - t_last_reload_check > 5.0:
                t_last_reload_check = time.time()
                if os.path.exists(args.model_path):
                    cur_mtime = os.path.getmtime(args.model_path)
                    if cur_mtime > last_mtime:
                        try:
                            try:
                                state_dict = torch.load(args.model_path, map_location=args.device, weights_only=True)
                            except Exception:
                                state_dict = torch.load(args.model_path, map_location=args.device, weights_only=False)
                            model.load_state_dict(state_dict)
                            last_mtime = cur_mtime
                            print(f"\n[+] [HOT-RELOAD] Car Brain weights dynamically updated from disk!\n")
                        except Exception:
                            pass

            # 1. Grab game window frame (in parallel to socket read)
            if is_multimodal and win_cap is not None:
                cur_frame = win_cap.grab_frame(grayscale=True).astype(np.float32) / 255.0
                img_history.append(cur_frame)
                while len(img_history) < 4:
                    img_history.append(cur_frame)
                obs_img = np.stack(list(img_history), axis=0)  # (4, 96, 96)

            # 2. Read latest telemetry packet from OpenPlanet socket
            while len(data_raw) < nb_bytes:
                chunk = sock.recv(1024)
                if not chunk:
                    raise ConnectionResetError("OpenPlanet socket disconnected.")
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
                if dist > 50.0:  # Player restarted race / reset to checkpoint
                    model.reset_context()
                    race_started = False
                    sys.stdout.write("\n[+] Track reset detected! Resetting in-context memory.\n")
                    sys.stdout.flush()
            last_pos = curr_pos

            # Construct 15-dimensional state vector
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

            # Automatic Stuck / Wall Crash Reset (TMRL standard failure countdown)
            if race_started and speed_val < 3.0:
                stuck_steps += 1
                if stuck_steps >= 30:  # 1.5s immobilized
                    sys.stdout.write("\n[!] Car stuck against wall / stopped! Auto-resetting race...\n")
                    sys.stdout.flush()
                    if USE_GAMEPAD:
                        gamepad_reset(gamepad)
                    keyres()
                    model.reset_context()
                    race_started = False
                    stuck_steps = 0
                    time.sleep(1.2)
                    continue
            else:
                stuck_steps = 0

            # Start-line countdown guard:
            # If car has not launched yet (speed < 1.0 km/h), keep context clean and steer straight
            if speed_val < 1.0 and not race_started:
                model.reset_context()
                act = np.array([1.0, -1.0, 0.0], dtype=np.float32)  # Full throttle, no brake, straight
                prev_action = act.copy()
            else:
                race_started = True
                if is_multimodal:
                    act = model.step_in_context(obs_img, state, prev_action=prev_action)
                else:
                    act = model.step_in_context(state, prev_action=prev_action)
                prev_action = act.copy()

            # Actuate car with steering gain
            send_control(act, speed=speed_val, steer_gain=args.steer_gain)

            step_count += 1
            now = time.time()
            actual_step_dt = now - t_last_step
            t_last_step = now

            # Print telemetry dashboard every 10 steps (0.5s)
            if step_count % 10 == 0:
                hz = 1.0 / max(0.001, actual_step_dt)
                gas_norm = max(0.0, min(1.0, (act[0] + 1.0) / 2.0))
                brk_norm = max(0.0, min(1.0, (act[1] + 1.0) / 2.0)) if act[1] > -0.2 else 0.0
                gas_bar = "█" * int(gas_norm * 10)
                brk_bar = "█" * int(brk_norm * 10)
                boosted_steer = np.clip(act[2] * args.steer_gain, -1.0, 1.0)
                sys.stdout.write(
                    f"\r[Live Brain] Speed: {speed_val:5.1f} km/h | Gear: {int(gear)} | "
                    f"Gas: [{gas_bar:<10}] | Brk: [{brk_bar:<10}] | Steer: {boosted_steer:+.2f} | {hz:.0f} Hz"
                )
                sys.stdout.flush()

            # Maintain 20 Hz control loop (0.05s per step)
            sleep_time = 0.05 - (time.time() - t_loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n[PAUSED] Autonomy stopped by user. Releasing controls...")
    finally:
        release_controls()
        sock.close()
        print("[+] Controls safely released. TrackMania control returned to human driver.\n")


if __name__ == "__main__":
    main()
