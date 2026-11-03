"""Observation faults only. Actuator commands and base rewards pass through."""
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SensorFaultEnv(gym.Wrapper):
    FAULTS = {"clean", "sensor_noise", "sensor_delay"}

    def __init__(self, env, fault="clean", noise_std=None, delay=3, onset=0):
        super().__init__(env)
        if fault not in self.FAULTS:
            raise ValueError(f"Unsupported sensor fault: {fault}")
        if not isinstance(env.observation_space, spaces.Box) or len(env.observation_space.shape) != 1:
            raise ValueError("A flat continuous observation space is required")
        if not isinstance(env.action_space, spaces.Box) or len(env.action_space.shape) != 1:
            raise ValueError("A flat continuous action space is required")
        if int(delay) != delay or delay < 0 or int(onset) != onset or onset < 0:
            raise ValueError("delay and onset must be nonnegative integers")
        self.fault, self.delay, self.onset = fault, int(delay), int(onset)
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.noise_std = np.broadcast_to(
            np.asarray(0.0 if noise_std is None else noise_std, dtype=np.float32),
            (self.obs_dim,),
        ).copy()
        if not np.isfinite(self.noise_std).all() or (self.noise_std < 0).any():
            raise ValueError("noise_std must be finite and nonnegative")
        if fault == "sensor_noise" and not np.any(self.noise_std > 0):
            raise ValueError("sensor_noise must actually alter at least one channel")
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.obs_dim + self.act_dim + 2,), dtype=np.float32
        )
        self.sensor_rng = np.random.default_rng()
        self.history = deque(maxlen=self.delay + 1)
        self.step_index = 0
        self.prev_action = np.zeros(self.act_dim, np.float32)
        self.prev_reward = self.prev_done = 0.0

    def _observe(self, true_obs, info):
        true_obs = np.asarray(true_obs, dtype=np.float32).reshape(self.obs_dim).copy()
        self.history.append(true_obs)
        active = self.step_index >= self.onset
        observed = true_obs.copy()
        if active and self.fault == "sensor_noise":
            observed += self.sensor_rng.normal(0.0, self.noise_std).astype(np.float32)
        elif active and self.fault == "sensor_delay":
            observed = self.history[0].copy() if len(self.history) > self.delay else np.zeros_like(true_obs)
        token = np.concatenate((observed, self.prev_action, [self.prev_reward, self.prev_done])).astype(np.float32)
        # Audit-only fields. The learner receives only token, never this dictionary.
        info = dict(info)
        info["sensor_audit"] = {
            "true_observation": true_obs,
            "observed_sensor": observed.copy(),
            "step_index": self.step_index,
            "fault": self.fault,
            "active": active,
            "observation_age": self.delay if active and self.fault == "sensor_delay" else 0,
        }
        return token, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            # Separate stream: sensor draws cannot change physics/reset randomness.
            self.sensor_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 9173]))
        self.history.clear()
        self.step_index = 0
        self.prev_action = np.zeros(self.act_dim, np.float32)
        self.prev_reward = self.prev_done = 0.0
        return self._observe(obs, info)

    def step(self, action):
        command = np.asarray(action, dtype=np.float32).copy()
        obs, reward, terminated, truncated, info = self.env.step(command)
        self.step_index += 1
        self.prev_action = command
        self.prev_reward = float(reward)
        self.prev_done = float(terminated)
        token, info = self._observe(obs, info)
        return token, float(reward), terminated, truncated, info


def make_env(condition, config, onset=0):
    env = gym.make(config["environment"], max_episode_steps=config["episode_horizon"],
                   **config.get("env_kwargs", {}))
    return SensorFaultEnv(env, fault=condition, noise_std=config["noise_std"],
                          delay=config["sensor_delay_steps"], onset=onset)
