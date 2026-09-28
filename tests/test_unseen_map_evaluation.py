"""Unseen-map evaluation must not replay or score against the old route."""
from contextlib import redirect_stderr
import io
import unittest
import numpy as np

from scripts.evaluate_live_driving import (
    CycleDeadlineGuard, FrameFreshnessGuard, MotionStallGuard,
    parse_args, reference_config_for_run,
)


class UnseenMapTests(unittest.TestCase):
    def test_signed_triggers_require_explicit_continuous_foundation(self):
        self.assertFalse(parse_args([]).signed_analog_triggers)
        self.assertTrue(parse_args(["--continuous-actions", "--signed-analog-triggers"]).signed_analog_triggers)
        for args in (["--signed-analog-triggers"], ["--mode", "tracking", "--signed-analog-triggers"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(args)

    def test_preserve_window_size_is_explicit(self):
        self.assertTrue(parse_args(["--unseen-map", "--preserve-window-size"]).preserve_window_size)
        self.assertFalse(parse_args([]).preserve_window_size)

    def test_unseen_map_ignores_missing_old_reference(self):
        args = parse_args(["--unseen-map", "--steps", "200"])
        self.assertIsNone(reference_config_for_run(args, {}))

    def test_unseen_map_cannot_use_reference_controller(self):
        for mode in ("trajectory", "tracking", "lidar"):
            with self.subTest(mode=mode), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--unseen-map", "--mode", mode])

    def test_known_map_keeps_reference_configuration_unchanged(self):
        config = {"TRAJECTORY_ASSIST": {"ENABLED": False, "TRAJECTORY_PATH": "original.npz"}}
        selected = reference_config_for_run(parse_args(["--mode", "tracking"]), config)
        self.assertTrue(selected["ENABLED"])
        self.assertEqual(selected["TRAJECTORY_PATH"], "original.npz")
        self.assertFalse(config["TRAJECTORY_ASSIST"]["ENABLED"])

    def test_motion_guard_allows_launch_then_stops_a_stationary_car(self):
        guard = MotionStallGuard()
        for elapsed in (0, 0.5, 1, 1.5, 2, 2.5):
            self.assertFalse(guard.update(elapsed, [0, 0, 0]))
        self.assertTrue(guard.update(3, [0, 0, 0]))

    def test_motion_guard_does_not_consult_route_direction(self):
        guard = MotionStallGuard()
        for step in range(80):
            elapsed = step * 0.05
            self.assertFalse(guard.update(elapsed, [-elapsed * 10, 5, elapsed * 3]))

    def test_motion_guard_rejects_nonfinite_positions(self):
        with self.assertRaises(ValueError):
            MotionStallGuard().update(0, [float("nan"), 0, 0])

    def test_repeated_frame_is_allowed_while_stationary(self):
        guard = FrameFreshnessGuard()
        frame = np.zeros((96, 96), dtype=np.uint8)
        for elapsed in (0, 1, 2, 5):
            self.assertFalse(guard.update(elapsed, frame, [0, 0, 0]))

    def test_frozen_camera_is_rejected_during_motion(self):
        guard = FrameFreshnessGuard()
        frame = np.zeros((96, 96), dtype=np.uint8)
        self.assertFalse(guard.update(0, frame, [0, 0, 0]))
        self.assertFalse(guard.update(0.25, frame, [1, 0, 0]))
        self.assertTrue(guard.update(0.51, frame, [2, 0, 0]))

    def test_changing_pixels_reset_camera_age(self):
        guard = FrameFreshnessGuard()
        for index in range(20):
            frame = np.full((96, 96), index, dtype=np.uint8)
            self.assertFalse(guard.update(index * 0.1, frame, [index, 0, 0]))

    def test_deadline_guard_allows_jitter_not_persistent_slow_loop(self):
        guard = CycleDeadlineGuard(max_cycle_seconds=0.1)
        for duration in (0.25, 0.03, 0.04, 0.25, 0.26):
            self.assertFalse(guard.update(duration))
        self.assertTrue(guard.update(0.25))


if __name__ == "__main__":
    unittest.main()
