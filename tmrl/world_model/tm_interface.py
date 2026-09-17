import logging

import numpy as np

import tmrl.config.config_constants as cfg
from tmrl.custom.tm.tm_gym_interfaces import TM2020Interface


class WorldModelTM2020Interface(TM2020Interface):
    """TrackMania image interface with a strict reward-trajectory start check.

    TMRL's reward function assumes ``reward.pkl`` belongs to the currently loaded
    track and starts near the race spawn. If a different map is loaded, rewards
    otherwise stay at zero until the failure countdown resets the episode. This
    guard makes that configuration error explicit before samples are collected.
    """

    def __init__(
        self,
        *args,
        validate_reward_start=True,
        reward_start_tolerance=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.validate_reward_start = bool(validate_reward_start)
        self.reward_start_tolerance = (
            float(reward_start_tolerance)
            if reward_start_tolerance is not None
            else float(cfg.REWARD_CONFIG["MAX_STRAY"])
        )

    def _validate_reward_trajectory_start(self):
        if not self.validate_reward_start:
            return

        data = self.client.retrieve_data()
        live_pos = np.asarray([data[2], data[3], data[4]], dtype=np.float64)
        reward_start = np.asarray(self.reward_function.data[0], dtype=np.float64)
        distance = float(np.linalg.norm(live_pos - reward_start))

        if distance > self.reward_start_tolerance:
            raise RuntimeError(
                "Reward trajectory does not match the currently loaded TrackMania track. "
                f"Live spawn={live_pos.tolist()}, reward.pkl start={reward_start.tolist()}, "
                f"distance={distance:.3f}, allowed={self.reward_start_tolerance:.3f}. "
                "Load the track for which this reward.pkl was recorded, or record a new reward "
                "on the current track with `python -m tmrl --record-reward`, then verify it with "
                "`python -m tmrl --check-environment`."
            )

        logging.info(
            "Reward trajectory start validated: live=%s reward_start=%s distance=%.3f <= %.3f",
            np.round(live_pos, 3).tolist(),
            np.round(reward_start, 3).tolist(),
            distance,
            self.reward_start_tolerance,
        )

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._validate_reward_trajectory_start()
        return obs, info
