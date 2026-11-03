"""Map-locked demonstration trajectory safety controller."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np


class DemonstrationTrajectoryAssist:
    """Replay demonstrated controls by spatial progress and correct drift.

    This is intentionally a deployment safety layer, not evidence of one-shot
    generalization.  It is enabled only for the map whose identity is already
    checked by ``TM2020Interface.validate_deployment_track``.
    """

    def __init__(self, config=None):
        config = config or {}
        self.enabled = bool(config.get("ENABLED", False))
        self.path = Path(config.get("TRAJECTORY_PATH", ""))
        self.search_backward = int(config.get("SEARCH_BACKWARD_STEPS", 20))
        self.search_forward = int(config.get("SEARCH_FORWARD_STEPS", 160))
        self.action_lead = int(config.get("ACTION_LEAD_STEPS", 1))
        self.steering_mode = str(config.get("STEERING_MODE", "replay"))
        if self.steering_mode not in {"replay", "pursuit"}:
            raise ValueError("STEERING_MODE must be replay or pursuit")
        self.tracking_speed_scale = float(config.get("TRACKING_SPEED_SCALE", 0.65))
        self.tracking_max_speed = float(config.get("TRACKING_MAX_SPEED", 25.0))
        self.tracking_min_speed = float(config.get("TRACKING_MIN_SPEED", 8.0))
        self.speed_feedback_enabled = bool(
            config.get("SPEED_FEEDBACK_ENABLED", True)
        )
        self.speed_lookahead = int(config.get("SPEED_LOOKAHEAD_STEPS", 0))
        self.speed_reengage_margin = float(
            config.get("SPEED_REENGAGE_MARGIN_MPS", config.get("SPEED_REENGAGE_MARGIN_KMH", 4.0))
        )
        self.speed_brake_margin = float(
            config.get("SPEED_BRAKE_MARGIN_MPS", config.get("SPEED_BRAKE_MARGIN_KMH", 8.0))
        )
        self.correction_enabled = bool(config.get("CORRECTION_ENABLED", False))
        self.correction_start = float(config.get("CORRECTION_START_METERS", 0.75))
        self.correction_full = float(config.get("CORRECTION_FULL_METERS", 3.0))
        self.max_deviation = float(config.get("MAX_DEVIATION_METERS", 12.0))
        self.lookahead_base = float(config.get("LOOKAHEAD_BASE_METERS", 8.0))
        self.lookahead_speed = float(config.get("LOOKAHEAD_SPEED_FACTOR", 0.12))
        self.lookahead_max = float(config.get("LOOKAHEAD_MAX_METERS", 22.0))
        self.pursuit_gain = float(config.get("PURE_PURSUIT_GAIN", 1.5))
        self.heading_measurement_weight = float(config.get("HEADING_MEASUREMENT_WEIGHT", 0.35))
        if not 0.0 < self.heading_measurement_weight <= 1.0:
            raise ValueError("HEADING_MEASUREMENT_WEIGHT must be in (0, 1]")
        self.steer_deadzone = float(config.get("STEER_DEADZONE", 0.20))
        self.log_every = max(1, int(config.get("LOG_EVERY_STEPS", 100)))
        if self.search_backward < 0 or self.search_forward < 1 or self.action_lead < 0 or self.speed_lookahead < 0:
            raise ValueError("Search/lead offsets must be nonnegative, with a positive forward window")
        positive = (self.max_deviation, self.lookahead_base, self.lookahead_max,
                    self.pursuit_gain, self.tracking_speed_scale, self.tracking_min_speed,
                    self.tracking_max_speed)
        if not all(np.isfinite(v) and v > 0 for v in positive):
            raise ValueError("Tracking distances, gain and speed limits must be finite and positive")
        if self.tracking_min_speed > self.tracking_max_speed:
            raise ValueError("TRACKING_MIN_SPEED must not exceed TRACKING_MAX_SPEED")
        self.positions = None
        self.actions = None
        self.speeds = None
        self.cumulative_distance = None
        if self.enabled:
            self._load()
        self.reset(log_previous=False)

    def _load(self):
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Trajectory assist file does not exist: {self.path}"
            )
        with np.load(self.path, allow_pickle=False) as payload:
            self.positions = np.asarray(payload["positions"], dtype=np.float32)
            self.actions = np.asarray(payload["actions"], dtype=np.float32)
            # Legacy exports mislabeled native api.Speed values as km/h. They
            # were never converted, so compatibility means renaming, not /3.6.
            speed_key = "speeds_mps" if "speeds_mps" in payload else "speeds_kmh"
            self.speeds = np.asarray(payload[speed_key], dtype=np.float32)
        if self.positions.ndim != 2 or len(self.positions) < 2:
            raise ValueError("Trajectory requires at least two [x, y, z] positions")
        expected = (len(self.positions), 3)
        if self.positions.shape != expected or self.actions.shape != expected:
            raise ValueError(
                "Trajectory assist requires positions/actions shaped [T, 3], "
                f"got {self.positions.shape} and {self.actions.shape}"
            )
        if self.speeds.shape != (len(self.positions),):
            raise ValueError(
                f"Trajectory speeds must be shaped {(len(self.positions),)}, "
                f"got {self.speeds.shape}"
            )
        if not all(np.isfinite(a).all() for a in (self.positions, self.actions, self.speeds)):
            raise ValueError("Trajectory contains non-finite values")
        if np.any(np.abs(self.actions) > 1.0) or np.any(self.speeds < 0):
            raise ValueError("Trajectory actions must be in [-1, 1] and speeds must be nonnegative")
        segments = np.linalg.norm(np.diff(self.positions, axis=0), axis=1)
        self.cumulative_distance = np.concatenate(
            (np.zeros(1, dtype=np.float32), np.cumsum(segments, dtype=np.float32))
        )
        logging.info(
            "Trajectory assist loaded %d demonstrated controls from %s.",
            len(self.positions),
            self.path,
        )

    def reset(self, log_previous=True):
        if log_previous and getattr(self, "steps", 0):
            logging.info(
                "Trajectory assist episode: steps=%d, progress=%d/%d, "
                "max_deviation=%.2fm, interventions=%d.",
                self.steps,
                self.index,
                len(self.positions) - 1,
                self.max_seen_deviation,
                self.interventions,
            )
        self.index = 0
        self.steps = 0
        self.interventions = 0
        self.max_seen_deviation = 0.0
        self.previous_position_xz = None
        self.heading_xz = None
        self.last_info = {}

    @staticmethod
    def _ternary(value, deadzone):
        return 1.0 if value > deadzone else (-1.0 if value < -deadzone else 0.0)

    def _nearest_index(self, position):
        if self.steps == 0:
            start = 0
            stop = min(len(self.positions), self.search_forward)
        else:
            start = max(0, self.index - self.search_backward)
            stop = min(len(self.positions), self.index + self.search_forward)
        offsets = self.positions[start:stop] - position
        local = int(np.argmin(np.einsum("ij,ij->i", offsets, offsets)))
        candidate = start + local
        # Spatial progress is monotonic on a normal lap.  Refusing backwards
        # jumps prevents ambiguous nearby track segments from rewinding control.
        self.index = max(self.index, candidate)
        return self.index

    def _update_heading(self, position_xz, reference_index):
        if self.previous_position_xz is not None:
            displacement = position_xz - self.previous_position_xz
            norm = float(np.linalg.norm(displacement))
            if norm >= 0.10:
                measured = displacement / norm
                if self.heading_xz is None:
                    self.heading_xz = measured
                else:
                    alpha = self.heading_measurement_weight
                    mixed = (1.0 - alpha) * self.heading_xz + alpha * measured
                    mixed_norm = float(np.linalg.norm(mixed))
                    if mixed_norm > 1e-6:
                        self.heading_xz = mixed / mixed_norm
        self.previous_position_xz = position_xz.copy()
        if self.heading_xz is None:
            before = max(0, reference_index - 2)
            after = min(len(self.positions) - 1, reference_index + 5)
            tangent = self.positions[after, [0, 2]] - self.positions[before, [0, 2]]
            tangent_norm = float(np.linalg.norm(tangent))
            self.heading_xz = (
                tangent / tangent_norm
                if tangent_norm > 1e-6
                else np.asarray([0.0, 1.0], dtype=np.float32)
            )
        return self.heading_xz

    def action(self, telemetry, proposed_action):
        if not self.enabled:
            return np.asarray(proposed_action, dtype=np.float32), {}
        position = np.asarray(
            [telemetry[2], telemetry[3], telemetry[4]], dtype=np.float32
        )
        speed = float(telemetry[0])
        if not np.isfinite(position).all() or not np.isfinite(speed):
            raise ValueError("Trajectory controller received non-finite telemetry")
        reference_index = self._nearest_index(position)
        deviation = float(np.linalg.norm(position - self.positions[reference_index]))
        self.max_seen_deviation = max(self.max_seen_deviation, deviation)

        action_index = min(len(self.actions) - 1, reference_index + self.action_lead)
        assisted = self.actions[action_index].astype(np.float32, copy=True)
        assisted[0] = 1.0 if assisted[0] > 0.0 else -1.0
        assisted[1] = 1.0 if assisted[1] > 0.0 else -1.0
        assisted[2] = self._ternary(float(assisted[2]), 0.20)

        speed_index = min(
            len(self.speeds) - 1, reference_index + self.speed_lookahead
        )
        target_speed = float(self.speeds[speed_index])
        speed_override = "reference"
        if self.speed_feedback_enabled:
            if speed < target_speed - self.speed_reengage_margin:
                # A spatially indexed brake can otherwise deadlock: once the
                # live car slows earlier than the demonstrator it stops moving,
                # so the reference index never reaches the later gas action.
                assisted[0:2] = (1.0, -1.0)
                speed_override = "accelerate"
            elif speed > target_speed + self.speed_brake_margin:
                assisted[0:2] = (-1.0, 1.0)
                speed_override = "brake"
        if self.steering_mode == "pursuit":
            # Speeds are in the native OpenPlanet api.Speed units (m/s), despite
            # the historical speeds_kmh archive key. Use the same units on both
            # sides of the feedback loop and a conservative absolute cap.
            target_speed = max(self.tracking_min_speed,
                               min(self.tracking_max_speed,
                                   target_speed * self.tracking_speed_scale))
            if speed < target_speed - 0.75:
                assisted[0:2] = (1.0, -1.0)
                speed_override = "accelerate"
            elif speed > target_speed + 1.5:
                assisted[0:2] = (-1.0, 1.0)
                speed_override = "brake"
            else:
                assisted[0:2] = (-1.0, -1.0)
                speed_override = "coast"

        lookahead = min(
            self.lookahead_max,
            self.lookahead_base + self.lookahead_speed * max(0.0, speed),
        )
        target_distance = self.cumulative_distance[reference_index] + lookahead
        target_index = min(
            len(self.positions) - 1,
            int(np.searchsorted(self.cumulative_distance, target_distance)),
        )
        position_xz = position[[0, 2]]
        heading = self._update_heading(position_xz, reference_index)
        target_vector = self.positions[target_index, [0, 2]] - position_xz
        target_norm = float(np.linalg.norm(target_vector))
        angle = 0.0
        pursuit = float(assisted[2])
        if target_norm > 1e-6:
            target_direction = target_vector / target_norm
            cross = float(
                heading[0] * target_direction[1]
                - heading[1] * target_direction[0]
            )
            dot = float(np.clip(np.dot(heading, target_direction), -1.0, 1.0))
            angle = float(np.arctan2(cross, dot))
            pursuit = float(
                np.clip(self.pursuit_gain * angle / np.deg2rad(25.0), -1.0, 1.0)
            )

        correction_weight = 0.0
        if self.steering_mode == "pursuit":
            assisted[2] = pursuit
            correction_weight = 1.0
        elif self.correction_enabled and deviation > self.correction_start:
            span = max(1e-3, self.correction_full - self.correction_start)
            correction_weight = float(
                np.clip((deviation - self.correction_start) / span, 0.0, 1.0)
            )
            steer = (1.0 - correction_weight) * assisted[2] + correction_weight * pursuit
            # Keep corrective steering analog. Quantizing any non-trivial
            # correction to full lock caused oscillation and live departures.
            assisted[2] = float(np.clip(steer, -1.0, 1.0))

        # Stop accelerating into scenery if recovery has already failed.
        emergency_stop = deviation > self.max_deviation
        if emergency_stop:
            assisted[:] = (-1.0, 1.0, 0.0)

        proposed = np.asarray(proposed_action, dtype=np.float32)
        changed = not np.array_equal(assisted, proposed)
        self.interventions += int(changed)
        self.steps += 1
        self.last_info = {
            "enabled": True,
            "reference_index": int(reference_index),
            "target_index": int(target_index),
            "deviation_m": deviation,
            "speed_mps": speed,
            "target_speed_mps": target_speed,
            "speed_override": speed_override,
            "pursuit_angle_deg": float(np.rad2deg(angle)),
            "correction_weight": correction_weight,
            "intervened": changed,
            "emergency_stop": emergency_stop,
            "applied_action": assisted.copy(),
        }
        if self.steps % self.log_every == 0:
            logging.info(
                "Trajectory assist step=%d progress=%d/%d speed=%.1f/%.1fm/s "
                "deviation=%.2fm speed_mode=%s "
                "policy=%s applied=%s.",
                self.steps,
                reference_index,
                len(self.positions) - 1,
                speed,
                target_speed,
                deviation,
                speed_override,
                np.array2string(proposed, precision=2),
                np.array2string(assisted, precision=2),
            )
        return assisted, self.last_info
