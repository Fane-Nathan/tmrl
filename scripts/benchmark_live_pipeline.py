#!/usr/bin/env python3
"""Profile the live Trackmania vision-policy pipeline without sending controls.

The script reads the game window and OpenPlanet telemetry, builds the same
observation used by the rollout worker, and runs the deployed actor.  It never
creates a virtual gamepad and never calls ``send_control``.
"""

from __future__ import annotations

import argparse
import statistics
import hashlib
import json
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import gymnasium.spaces as spaces
import numpy as np

# Running ``python scripts/...`` otherwise prefers an installed TMRL package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tmrl.config.config_constants as cfg
from tmrl.custom.tm.utils.window import (
    WindowInterface,
    preinitialize_dxcam_capture,
)

from tmrl.custom.tm.utils.tools import TM2020OpenPlanetClient
from tmrl.custom.tm.utils.camera_preprocessing import preprocess_camera_frame


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_ms(samples: dict[str, list[float]]) -> None:
    print("\nLive pipeline timings (milliseconds)")
    print(f"{'stage':<24} {'mean':>10} {'p50':>10} {'p95':>10} {'max':>10}")
    for name, values in samples.items():
        millis = [value * 1000.0 for value in values]
        print(
            f"{name:<24} {statistics.fmean(millis):10.3f} "
            f"{percentile(millis, 50):10.3f} "
            f"{percentile(millis, 95):10.3f} {max(millis):10.3f}"
        )


def build_actor(device: str):
    from tmrl.config.config_objects import POLICY
    image_space = spaces.Box(
        low=0.0,
        high=255.0,
        shape=(cfg.IMG_HIST_LEN, cfg.IMG_HEIGHT, cfg.IMG_WIDTH),
        dtype=np.float32,
    )
    action_space = spaces.Box(
        low=-1.0, high=1.0, shape=(3,), dtype=np.float32
    )
    observation_space = spaces.Tuple(
        (
            spaces.Box(0.0, 1000.0, shape=(1,), dtype=np.float32),
            spaces.Box(0.0, 6.0, shape=(1,), dtype=np.float32),
            spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float32),
            image_space,
            *((action_space,) * cfg.ACT_BUF_LEN),
        )
    )
    try:
        actor = POLICY(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
    except TypeError:
        actor = POLICY(
            observation_space=observation_space,
            action_space=action_space,
        ).to(device)
    model_path = Path(cfg.MODEL_PATH_WORKER)
    if model_path.is_file():
        actor.load(str(model_path), device=device)
        if not getattr(actor, "last_load_succeeded", True):
            raise RuntimeError(
                f"Could not load deployed actor {model_path}: "
                f"{getattr(actor, 'last_load_error', 'unknown error')}"
            )
    actor.eval()
    return actor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--preserve-window-size", action="store_true",
                        help="Capture the current client without forcing a small window")
    parser.add_argument("--foundation-weights", type=Path,
                        help="Read-only profiling of an isolated foundation candidate")
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument("--frame-output", type=Path, help="Save the actual first captured frame for inspection")
    parser.add_argument(
        "--unpaced",
        action="store_true",
        help="run as fast as possible instead of matching the configured control period",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if cfg.CUDA_INFERENCE else "cpu",
    )
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0:
        parser.error("--steps must be positive and --warmup cannot be negative")
    # Desktop Duplication must be initialized before creating a CUDA context on
    # hybrid-GPU Windows systems.  This is also the RolloutWorker construction
    # order (environment first, actor second).
    preinitialize_dxcam_capture(resize_window=not args.preserve_window_size)
    import torch
    from tmrl.config.config_objects import OBS_PREPROCESSOR
    capture = WindowInterface("Trackmania")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but torch.cuda.is_available() is false")
    telemetry = TM2020OpenPlanetClient()
    actor = build_actor(args.device)
    if args.foundation_weights:
        actor.encoder.reload_foundation_weights(str(args.foundation_weights.resolve()))
    actor.encoder.reload_on_change = False
    history: deque[np.ndarray] = deque(maxlen=cfg.IMG_HIST_LEN)
    previous_actions = [np.zeros(3, dtype=np.float32) for _ in range(cfg.ACT_BUF_LEN)]
    samples: dict[str, list[float]] = defaultdict(list)
    period = float(cfg.ENV_CONFIG["RTGYM_CONFIG"]["time_step_duration"])

    total_steps = args.warmup + args.steps
    image_hashes = set()
    image_shape = None
    actor.reset()
    print(
        f"Profiling {args.steps} measured steps after {args.warmup} warmups "
        f"on {args.device}; no controls will be sent."
    )
    for step in range(total_steps):
        measured = step >= args.warmup
        cycle_start = time.perf_counter()

        started = time.perf_counter()
        frame = capture.screenshot()
        captured = time.perf_counter()
        if step == 0:
            image_shape = list(frame.shape)
            if args.frame_output:
                args.frame_output.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(args.frame_output), frame):
                    raise RuntimeError("Could not save captured frame")

        image = preprocess_camera_frame(frame, (cfg.IMG_WIDTH, cfg.IMG_HEIGHT), cfg.GRAYSCALE)
        processed_image = time.perf_counter()

        data = telemetry.retrieve_data(timeout=0.5)
        read_telemetry = time.perf_counter()

        history.append(image)
        while len(history) < cfg.IMG_HIST_LEN:
            history.append(image)
        raw_observation = (
            np.asarray([data[0]], dtype=np.float32),
            np.asarray([data[9]], dtype=np.float32),
            np.asarray([data[10]], dtype=np.float32),
            np.asarray(history),
            *previous_actions,
        )
        observation = OBS_PREPROCESSOR(raw_observation)
        assembled = time.perf_counter()

        action = actor.act_(observation, test=False)
        acted = time.perf_counter()
        previous_actions = [*previous_actions[1:], np.asarray(action, dtype=np.float32)]

        if measured:
            image_hashes.add(hashlib.sha256(image.tobytes()).hexdigest())
            samples["window_capture"].append(captured - started)
            samples["resize_and_color"].append(processed_image - captured)
            samples["telemetry"].append(read_telemetry - processed_image)
            samples["observation_build"].append(assembled - read_telemetry)
            samples["actor"].append(acted - assembled)
            samples["end_to_end"].append(acted - cycle_start)
        if not args.unpaced:
            remaining = cycle_start + period - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)

    telemetry.close()
    summarize_ms(samples)
    period_ms = period * 1000.0
    p95_ms = percentile([value * 1000.0 for value in samples["end_to_end"]], 95)
    print(f"\nConfigured period: {period_ms:.1f} ms")
    print(f"End-to-end p95 headroom: {period_ms - p95_ms:.1f} ms")
    print("Controls sent: 0")
    result = {"controls_sent": 0, "steps": args.steps, "capture_shape": image_shape,
              "distinct_policy_images": len(image_hashes), "preserve_window_size": args.preserve_window_size,
              "foundation_weights": str(actor.encoder.model_file.resolve()),
              "stage_p95_ms": {name: percentile([v * 1000 for v in values], 95) for name, values in samples.items()},
              "period_ms": period_ms, "within_period": p95_ms <= period_ms,
              "caveat": "Stationary policy calls may use the launch safeguard, not the full rolling transformer"}
    print(json.dumps(result, indent=2), flush=True)
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
