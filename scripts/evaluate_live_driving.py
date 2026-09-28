#!/usr/bin/env python3
"""Run a bounded local driving evaluation, without a trainer or replay server.

Each run saves observations, actual commands, timing, and a finish flag. Results
from assisted modes measure the controller, not the learned policy's ability.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def extract_latest_replay_info(started_epoch=None):
    """Scan TrackMania replay directories for replays saved during this run.
    Extracts the official race time in milliseconds.
    """
    search_dirs = [
        Path(os.path.expanduser("~/Documents/Trackmania2020/Replays/My Replays")),
        Path(os.path.expanduser("~/Documents/Trackmania/Replays/My Replays")),
    ]
    candidates = []
    for d in search_dirs:
        if d.is_dir():
            for f in d.glob("*.Replay.Gbx"):
                try:
                    mtime = f.stat().st_mtime
                    if started_epoch is None or mtime >= started_epoch - 2.0:
                        candidates.append((mtime, f))
                except OSError:
                    pass
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    latest_file = candidates[0][1]

    time_ms = None
    m = re.search(r'\((\d+)_(\d+)_(\d+)\)', latest_file.name)
    if m:
        mins, secs, ms = int(m.group(1)), int(m.group(2)), int(m.group(3))
        time_ms = (mins * 60 + secs) * 1000 + ms

    digest = hashlib.sha256()
    with open(latest_file, "rb") as fh:
        digest.update(fh.read())

    return {
        "replay_file": str(latest_file.resolve()),
        "replay_sha256": digest.hexdigest(),
        "official_race_time_ms": time_ms,
        "official_race_time_seconds": (time_ms / 1000.0) if time_ms is not None else None,
    }


class MotionStallGuard:
    """Detect low displacement without consulting a demonstrated route.

    This is a conservative stop condition, not a progress or collision metric.
    """
    def __init__(self, window_seconds=2.0, min_displacement_m=0.5, grace_seconds=3.0):
        self.window_seconds = window_seconds
        self.min_displacement_m = min_displacement_m
        self.grace_seconds = grace_seconds
        self.samples = deque()
        self.displacement_m = None

    def update(self, elapsed, position):
        position = np.asarray(position, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all() or not np.isfinite(elapsed):
            raise ValueError("Invalid motion-guard observation")
        self.samples.append((elapsed, position.copy()))
        while len(self.samples) > 1 and elapsed - self.samples[1][0] >= self.window_seconds:
            self.samples.popleft()
        age = elapsed - self.samples[0][0]
        self.displacement_m = float(np.linalg.norm(position - self.samples[0][1]))
        return (elapsed >= self.grace_seconds and age >= self.window_seconds
                and self.displacement_m < self.min_displacement_m)


class FrameFreshnessGuard:
    """Stop on identical camera pixels despite material vehicle movement.

    A static start-line image alone is not an error. Pixel freshness is a
    conservative diagnostic, not proof of frame/telemetry synchronization.
    """
    def __init__(self, max_repeat_seconds=0.5, min_motion_m=0.5):
        self.max_repeat_seconds = max_repeat_seconds
        self.min_motion_m = min_motion_m
        self.last_frame = None
        self.changed_at = None
        self.position_at_change = None
        self.max_observed_repeat_seconds = 0.0

    def update(self, elapsed, frame, position):
        frame, position = np.asarray(frame), np.asarray(position, dtype=np.float64)
        if frame.size == 0 or position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Invalid camera freshness observation")
        if self.last_frame is None or not np.array_equal(frame, self.last_frame):
            self.last_frame = frame.copy()
            self.changed_at = elapsed
            self.position_at_change = position.copy()
            return False
        repeated = elapsed - self.changed_at
        self.max_observed_repeat_seconds = max(self.max_observed_repeat_seconds, repeated)
        motion = float(np.linalg.norm(position - self.position_at_change))
        return repeated >= self.max_repeat_seconds and motion >= self.min_motion_m


class CycleDeadlineGuard:
    """Reject repeated gross deadline overruns, allowing isolated startup jitter."""
    def __init__(self, max_cycle_seconds, consecutive_limit=3):
        self.max_cycle_seconds = max_cycle_seconds
        self.consecutive_limit = consecutive_limit
        self.consecutive_misses = 0

    def update(self, duration):
        if not np.isfinite(duration) or duration < 0:
            raise ValueError("Invalid control cycle duration")
        self.consecutive_misses = self.consecutive_misses + 1 if duration > self.max_cycle_seconds else 0
        return self.consecutive_misses >= self.consecutive_limit


def reference_config_for_run(args, safety_config):
    # An unseen-map test must not even read the old trajectory path.
    if args.unseen_map:
        return None
    config = dict(safety_config["TRAJECTORY_ASSIST"])
    config.update(ENABLED=True, MAX_DEVIATION_METERS=args.max_deviation)
    if args.mode == "tracking":
        config.update(STEERING_MODE="pursuit", SEARCH_FORWARD_STEPS=32,
                      LOOKAHEAD_BASE_METERS=6.0, LOOKAHEAD_SPEED_FACTOR=0.25,
                      LOOKAHEAD_MAX_METERS=14.0,
                      PURE_PURSUIT_GAIN=args.tracking_gain,
                      HEADING_MEASUREMENT_WEIGHT=0.8,
                      TRACKING_MAX_SPEED=args.tracking_max_speed)
    return config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("foundation", "trajectory", "tracking", "lidar"), default="foundation")
    parser.add_argument("--unseen-map", action="store_true",
                        help="Explicit policy-only test on the user-loaded map, with no reference path")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--output", type=Path, default=ROOT / "test_output_brain" / "live_runs")
    parser.add_argument("--max-deviation", type=float, default=12.0)
    parser.add_argument("--stall-seconds", type=float, default=2.0)
    parser.add_argument("--record-frames", action="store_true")
    parser.add_argument("--preserve-window-size", action="store_true",
                        help="Capture the current client without resizing a fullscreen game")
    parser.add_argument("--camera-label", default="unknown",
                        help="User-declared camera for dataset provenance; not automatically verified")
    parser.add_argument("--foundation-weights", type=Path,
                        help="Test an isolated candidate without overwriting the deployed checkpoint")
    parser.add_argument("--continuous-actions", action="store_true",
                        help="Do not ternary-decode steering; required for continuous teacher targets")
    parser.add_argument("--signed-analog-triggers", action="store_true",
                        help="Explicit BC trigger decoding: (-1,0,1) -> (0,0.5,1) pressure, no digital accelerator")
    parser.add_argument("--tracking-gain", type=float, default=0.55)
    parser.add_argument("--tracking-max-speed", type=float, default=22.0,
                        help="Conservative tracking speed cap in native telemetry units (m/s)")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="Maximum duration for the run in seconds (overrides default 30s unseen probe cap)")
    parser.add_argument("--map-uid", default=None,
                        help="Expected map UID for benchmark validation (e.g. XJ_JEjWGoAexDWe8qfaOjEcq5l8)")
    parser.add_argument("--map-file", type=Path, default=None,
                        help="Path to the map file for validation and recording")
    parser.add_argument("--world-record-time", type=float, default=19.454,
                        help="World record reference time in seconds for benchmark comparison")
    args = parser.parse_args(argv)
    if args.unseen_map and args.mode != "foundation":
        parser.error("unseen-map requires foundation mode; old-map controllers are not allowed")
    if args.mode != "foundation" and (args.foundation_weights or args.continuous_actions):
        parser.error("foundation-weights and continuous-actions apply only to foundation mode")
    if args.signed_analog_triggers and (args.mode != "foundation" or not args.continuous_actions):
        parser.error("signed-analog-triggers requires foundation mode and continuous-actions")
    if args.steps < 1 or not all(np.isfinite(v) and v > 0 for v in (
            args.max_deviation, args.stall_seconds, args.tracking_gain, args.tracking_max_speed)):
        parser.error("steps, max-deviation and stall-seconds must be positive")
    return args


def main():
    args = parse_args()

    import tmrl.config.config_constants as cfg
    from tmrl.custom.tm.utils.window import preinitialize_dxcam_capture
    preinitialize_dxcam_capture(resize_window=not args.preserve_window_size)
    from tmrl.custom.tm.tm_gym_interfaces import TM2020Interface
    from tmrl.custom.tm.utils.tools import Lidar
    from tmrl.custom.tm.utils.trajectory_assist import DemonstrationTrajectoryAssist

    period = float(cfg.ENV_CONFIG["RTGYM_CONFIG"]["time_step_duration"])
    max_seconds = args.max_seconds if args.max_seconds is not None else (30.0 if args.unseen_map else 300.0)
    if args.unseen_map and args.max_seconds is None and args.steps * period > 30.0:
        raise ValueError("Unseen-map probes are limited to 30 seconds; reduce --steps or specify --max-seconds")
    assist_config = reference_config_for_run(args, cfg.TMRL_CONFIG.get("DRIVING_SAFETY", {}))
    reference = DemonstrationTrajectoryAssist(assist_config) if assist_config is not None else None
    actor = None
    if args.mode == "foundation":
        from scripts.benchmark_live_pipeline import build_actor
        from tmrl.config.config_objects import OBS_PREPROCESSOR
        actor = build_actor("cpu")
        if args.foundation_weights:
            if hasattr(actor, "encoder") and hasattr(actor.encoder, "reload_foundation_weights"):
                actor.encoder.reload_foundation_weights(str(args.foundation_weights.resolve()))
            else:
                try:
                    actor.load(str(args.foundation_weights.resolve()), device="cpu")
                except Exception as e:
                    print(f"[!] Info: actor.load skipped ({e})")
        if hasattr(actor, "encoder"):
            actor.encoder.reload_on_change = False  # a run evaluates one fixed checkpoint
        if args.continuous_actions and hasattr(actor, "foundation_discrete_actions"):
            actor.foundation_discrete_actions = False
        if hasattr(actor, "reset"):
            actor.reset()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + args.mode
    if args.unseen_map:
        run_id += "_unseen"
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = args.output / run_id
    env = TM2020Interface(img_hist_len=cfg.IMG_HIST_LEN, gamepad=True,
                          grayscale=cfg.GRAYSCALE,
                          resize_to=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT),
                          signed_analog_triggers=args.signed_analog_triggers)
    rows, commands, reference_indices, deviations, durations, frames = [], [], [], [], [], []
    observation_times, command_times = [], []
    history = deque(maxlen=cfg.IMG_HIST_LEN)
    previous_actions = deque((np.zeros(3, dtype=np.float32) for _ in range(cfg.ACT_BUF_LEN)),
                             maxlen=cfg.ACT_BUF_LEN)
    result = {"schema_version": 3 if args.unseen_map else 2, "run_id": run_id, "mode": args.mode,
              "assisted": args.mode != "foundation", "requested_steps": args.steps,
              "period_seconds": period, "finished": False, "reason": "step_limit",
              "trajectory_path": str(reference.path) if reference is not None else None,
              "controls_released": False, "unseen_map_probe": args.unseen_map,
              "reference_used_for_control_or_scoring": reference is not None,
              "map_based_reward_used": False}
    result["controller_config"] = assist_config
    result["camera_label"] = args.camera_label
    result["camera_label_source"] = "user_declared_not_automatically_verified"
    result["action_alignment"] = "observation_t_then_command_t"
    result["telemetry_speed_unit"] = "m/s"
    result["capture_backend"] = cfg.CAPTURE_BACKEND
    result["preserve_window_size"] = args.preserve_window_size
    result["gamepad_trigger_encoding"] = ("signed_to_unit_pressure" if args.signed_analog_triggers
                                         else "legacy_positive_clamp")
    result["gamepad_digital_acceleration"] = not args.signed_analog_triggers
    result["runtime_guard_limits"] = {"identical_frame_seconds_while_moving": 0.5,
                                      "frame_guard_min_motion_m": 0.5,
                                      "max_cycle_seconds": 2 * period,
                                      "consecutive_slow_cycles": 3}
    result["trajectory_sha256"] = (hashlib.sha256(reference.path.read_bytes()).hexdigest()
                                   if reference is not None else None)
    if actor is not None:
        if hasattr(actor, "encoder") and hasattr(actor.encoder, "model_file") and actor.encoder.model_file.exists():
            result["foundation_weights_path"] = str(actor.encoder.model_file.resolve())
            result["foundation_weights_sha256"] = hashlib.sha256(actor.encoder.model_file.read_bytes()).hexdigest()
        else:
            result["foundation_weights_path"] = str(args.foundation_weights.resolve()) if args.foundation_weights else None
            result["foundation_weights_sha256"] = hashlib.sha256(args.foundation_weights.read_bytes()).hexdigest() if (args.foundation_weights and args.foundation_weights.exists()) else None
        result["action_decoder"] = "ternary" if getattr(actor, "foundation_discrete_actions", False) else "continuous"
    error = None
    started = None
    frame_guard = FrameFreshnessGuard()
    deadline_guard = CycleDeadlineGuard(max_cycle_seconds=2 * period)
    try:
        env.initialize()
        env.auto_map_cycle = False
        # This test chooses its controller explicitly and bypasses only the
        # environment's assist hook; reset still enforces map/start identity.
        env.trajectory_assist.enabled = False
        if args.unseen_map:
            # Replace only this instance's check: never mutate the shared config
            # or relabel the new map with the old map-file hash/start position.
            env.driving_safety = {"ENABLED": False}
        env.reset()
        result["observed_start_position"] = list(env.latest_data[2:5])
        if args.unseen_map:
            track_id = args.map_uid or "user-loaded-unseen-map"
            map_sha256 = (env._sha256_file(args.map_file) if args.map_file and args.map_file.is_file() else None)
            result.update(
                track_id=track_id,
                map_uid=args.map_uid,
                map_file=str(args.map_file.resolve()) if args.map_file and args.map_file.is_file() else None,
                map_sha256=map_sha256,
                track_identity_check="declared_map_uid_and_file_hash" if args.map_uid else "user_reported_new_map_live_uid_unavailable"
            )
            print(f"Unseen-map probe: track_id={track_id}, map_uid={args.map_uid}, map_sha256={map_sha256}; saved configuration unchanged.", flush=True)
        else:
            result["track_id"] = env.driving_safety.get("EXPECTED_TRACK_ID", "unknown")
            result["map_file"] = env.driving_safety.get("ACTIVE_MAP_PATH")
            result["map_sha256"] = (env._sha256_file(result["map_file"])
                                    if result["map_file"] else None)
            result["map_uid"] = args.map_uid or env.driving_safety.get("EXPECTED_TRACK_ID", "unknown")
            result["track_identity_check"] = "configured_file_hash_and_live_start_position_not_live_map_uid"
        env._trajectory_assist_active = False
        lidar = Lidar(env.window_interface.screenshot()) if args.mode == "lidar" else None
        stalled = 0
        last_progress = 0
        motion_guard = MotionStallGuard(window_seconds=args.stall_seconds)
        started = time.perf_counter()
        for step in range(args.steps):
            cycle = time.perf_counter()
            data, frame = env.grab_data_and_img()
            if not np.isfinite(np.asarray(data, dtype=np.float64)).all():
                result["reason"] = "invalid_telemetry"
                break
            elapsed = time.perf_counter() - started
            index = None
            info = {"deviation_m": None, "emergency_stop": False}
            if reference is not None:
                assisted, info = reference.action(data, np.zeros(3, dtype=np.float32))
                index = info["reference_index"]
                deviations.append(info["deviation_m"])
                reference_indices.append(index)
            rows.append(data)
            observation_times.append(time.perf_counter() - started)
            if args.record_frames:
                frames.append(frame.copy())
            if frame_guard.update(elapsed, frame, data[2:5]):
                result["reason"] = "frozen_camera_while_moving"
                break
            if bool(data[8]):
                result.update(finished=True, reason="finish_signal")
                break
            if info["emergency_stop"]:
                result["reason"] = "deviation_limit"
                break
            if reference is not None:
                stalled = stalled + 1 if index <= last_progress else 0
                last_progress = max(last_progress, index)
                if step > int(3.0 / period) and stalled >= int(args.stall_seconds / period):
                    result["reason"] = "no_reference_progress"
                    break
            elif motion_guard.update(elapsed, data[2:5]):
                result.update(reason="low_displacement", stall_window_displacement_m=motion_guard.displacement_m)
                break
            if args.unseen_map and elapsed >= max_seconds:
                result["reason"] = "time_limit"
                break

            if args.mode in {"trajectory", "tracking"}:
                action = assisted
            elif args.mode == "lidar":
                rays = lidar.lidar_20(env.window_interface.screenshot())
                normalized = rays / (rays.sum() + 0.001)
                steer = -np.tanh(np.dot(np.arange(19) - 9, normalized) * 4.0)
                action = np.asarray([1.0, -1.0, steer], dtype=np.float32)
            else:
                history.append(frame)
                while len(history) < cfg.IMG_HIST_LEN:
                    history.append(frame)
                observation = (np.asarray([data[0]], dtype=np.float32),
                               np.asarray([data[9]], dtype=np.float32),
                               np.asarray([data[10]], dtype=np.float32),
                               np.asarray(history), *previous_actions)
                action = actor.act_(OBS_PREPROCESSOR(observation), test=True)
            action = np.asarray(action, dtype=np.float32).copy()
            if action.shape != (3,) or not np.isfinite(action).all():
                raise ValueError("Controller returned an invalid action")
            action = np.clip(action, -1.0, 1.0)
            # Check before sending another command; finally releases any prior
            # action if the observation/inference loop repeatedly runs too slow.
            if deadline_guard.update(time.perf_counter() - cycle):
                result["reason"] = "control_deadline_misses"
                break
            env.send_control(action)
            command_times.append(time.perf_counter() - started)
            commands.append(action)
            previous_actions.append(action)
            durations.append(time.perf_counter() - cycle)
            if step % 100 == 0:
                print(json.dumps({"step": step, "reference_index": index,
                                  "speed_telemetry": round(data[0], 2),
                                  "deviation_m": round(info["deviation_m"], 3) if reference is not None else None,
                                  "action": action.tolist()}), flush=True)
            remaining = period - (time.perf_counter() - cycle)
            if remaining > 0:
                time.sleep(remaining)
    except (Exception, KeyboardInterrupt) as exc:
        error = exc
        result.update(reason=type(exc).__name__, error=str(exc))
    finally:
        if env.j is not None:
            env._trajectory_assist_active = False
            env.j.reset()
            env.j.update()
            result["controls_released"] = True
        if env.client is not None:
            env.client.close()
        result.update(steps=len(commands), observations=len(rows),
                      elapsed_seconds=0.0 if started is None else time.perf_counter() - started,
                      final_reference_index=reference.index if reference is not None else None,
                      reference_last_index=len(reference.positions) - 1 if reference is not None else None,
                      max_deviation_m=max(deviations) if deviations else None,
                      compute_p95_ms=float(np.percentile(durations, 95) * 1000) if durations else None,
                      finish_signal_observed=any(bool(row[8]) for row in rows),
                      max_identical_frame_seconds=frame_guard.max_observed_repeat_seconds,
                      runtime_guard_triggered=result["reason"] in {
                          "frozen_camera_while_moving", "control_deadline_misses"})
        started_wallclock = time.time() if started is None else (time.time() - (time.perf_counter() - started))
        if result["finished"] or any(bool(row[8]) for row in rows):
            replay_info = extract_latest_replay_info(started_epoch=started_wallclock)
            if replay_info:
                result.update(replay_info)
                result["official_race_time_source"] = "saved_replay_file"
            else:
                elapsed_time_s = result["elapsed_seconds"]
                result["official_race_time_ms"] = int(elapsed_time_s * 1000)
                result["official_race_time_seconds"] = elapsed_time_s
                result["official_race_time_source"] = "telemetry_wallclock_fallback"

            wr_time = getattr(args, "world_record_time", 19.454)
            if result.get("official_race_time_seconds") and wr_time > 0:
                result["world_record_reference_time"] = wr_time
                result["world_record_reference_holder"] = "AffiTM"
                result["record_reference_gap_pct"] = 100.0 * (result["official_race_time_seconds"] / wr_time - 1.0)
        np.savez_compressed(prefix.with_suffix(".npz"),
                            telemetry=np.asarray(rows, dtype=np.float32),
                            actions=np.asarray(commands, dtype=np.float32),
                            reference_indices=np.asarray(reference_indices),
                            deviations_m=np.asarray(deviations),
                            observation_seconds=np.asarray(observation_times),
                            command_seconds=np.asarray(command_times),
                            compute_seconds=np.asarray(durations),
                            frames=np.asarray(frames, dtype=np.uint8))
        prefix.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        print(f"Saved: {prefix.with_suffix('.json')}", flush=True)
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
