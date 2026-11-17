"""Record TrackMania 2020 track-center reward checkpoints from OpenPlanet telemetry."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Sequence

import tmrl.config.config_constants as cfg


PATH_REWARD = cfg.REWARD_PATH


def record_reward_dist(
    path_reward: str | Path = PATH_REWARD,
    use_keyboard: bool = False,
) -> None:
    """Capture 3D positions from OpenPlanet telemetry and write resampled checkpoints."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            f"Reward recording requires NumPy (missing: {exc.name}). "
            'Install it with `pip install "tmrl[trackmania]"`.'
        ) from exc

    try:
        from tmrl.custom.tm.utils.tools import TM2020OpenPlanetClient
    except ImportError as exc:
        raise RuntimeError(
            f"Reward recording requires TrackMania runtime dependencies (missing: {exc.name}). "
            'Install them with `pip install "tmrl[trackmania]"`.'
        ) from exc

    keyboard = None
    if use_keyboard:
        try:
            import keyboard as keyboard_module
        except ImportError as exc:
            raise RuntimeError(
                "The --use-keyboard option requires the optional `keyboard` package. "
                "Install it with `pip install keyboard` or omit --use-keyboard."
            ) from exc
        keyboard = keyboard_module

    positions: list[list[float]] = []
    output_path = Path(path_reward)
    client = TM2020OpenPlanetClient()

    try:
        is_recording = False
        while True:
            if not is_recording:
                if not use_keyboard:
                    logging.info("start recording")
                    is_recording = True
                elif keyboard.is_pressed("e"):
                    logging.info("start recording")
                    is_recording = True

            if is_recording:
                try:
                    data = client.retrieve_data(sleep_if_empty=0.01)  # dense points yield a smooth curve
                except (ConnectionError, TimeoutError, OSError) as exc:
                    raise RuntimeError(
                        f"OpenPlanet telemetry failed at 127.0.0.1:9000 ({exc}). "
                        "Ensure TrackMania 2020 and the OpenPlanet TMRL plugin are running."
                    ) from exc

                terminated = bool(data[8])
                early_stop = keyboard.is_pressed("q") if use_keyboard else False

                if early_stop or terminated:
                    logging.info("Computing reward function checkpoints from captured positions...")
                    logging.info(f"Initial number of captured positions: {len(positions)}")
                    if not positions:
                        raise RuntimeError(
                            "No track positions were recorded before stopping; reward file was not written."
                        )

                    captured = np.array(positions)
                    final_positions = [captured[0]]
                    dist_between_points = 0.1
                    j = 1
                    move_by = dist_between_points
                    pt1 = final_positions[-1]
                    while j < len(captured):
                        pt2 = captured[j]
                        pt, dst = line(pt1, pt2, move_by)
                        if pt is not None:  # a point was created
                            final_positions.append(pt)
                            move_by = dist_between_points
                            pt1 = pt
                        else:  # we passed pt2 without creating a new point
                            pt1 = pt2
                            j += 1
                            move_by = dst

                    resampled = np.array(final_positions)
                    logging.info(f"Final number of checkpoints in the reward function: {len(resampled)}")

                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with output_path.open("wb") as handle:
                        pickle.dump(resampled, handle)
                    logging.info("All done")
                    return
                positions.append([data[2], data[3], data[4]])
            else:
                time.sleep(0.05)  # waiting for user to press E
    finally:
        client.close()


def line(pt1: Any, pt2: Any, dist: float) -> tuple[Any, float]:
    """Create a point between ``pt1`` and ``pt2`` at distance ``dist`` from ``pt1``.

    If ``dist`` exceeds the segment length, returns ``(None, remaining_dist)``.
    Otherwise returns ``(point, 0.0)``.
    """
    import numpy as np

    vec = pt2 - pt1
    norm = np.linalg.norm(vec)
    if norm < dist:
        return None, float(dist - norm)
    vec_unit = vec / norm
    pt = pt1 + vec_unit * dist
    return pt, 0.0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for reward checkpoint recording."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=Path(PATH_REWARD),
        help="Output path for the pickled reward checkpoints.",
    )
    parser.add_argument(
        "--use-keyboard",
        action="store_true",
        help="Wait for 'e' to start recording and 'q' to stop recording.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and record TrackMania reward checkpoints."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    try:
        record_reward_dist(path_reward=args.path, use_keyboard=args.use_keyboard)
    except KeyboardInterrupt:
        return 0
    except (RuntimeError, OSError) as exc:
        print(f"Reward recording failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
