#!/usr/bin/env python3
"""
Gymnasium Fault-Injection Wrapper for In-Context Reinforcement Learning Benchmark.

Implements the complete fault taxonomy from Table I of:
"Memory Buys Adaptation, Demonstrations Buy Competence: In-Context Reinforcement Learning Under Real-Time Constraints"

Features:
- Encapsulates any continuous-action Gymnasium environment (e.g. HalfCheetah-v5).
- Packs (s_t, a_{prev}, r_{prev}, d_{prev}) into flat observation space.
- The stored a_{prev} is the *commanded* action, so the discrepancy with state consequence
  forms the meta-learning signal.
- Clean separation of splits:
  * 'train': action_scale (u in [0.5, 1.5]), action_noise (std=0.5)
  * 'val': unseen parameters of train mechanisms (action_scale in [0.2, 0.5) U (1.5, 1.8], action_noise std=0.75)
  * 'test_latency': held-out cross-mechanism action_latency (delay k in {1..5})
  * 'test_dead': held-out within-mechanism extreme dead_actuator (one dim locked to 0)
  * 'test_sign_flip': held-out within-mechanism extreme sign_flip (one dim negated)
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FaultInjectionWrapper(gym.Wrapper):
    """
    Wraps a Box continuous control environment and injects episode-level faults.
    """

    FAULT_NAMES = [
        "none",
        "action_scale",
        "action_noise",
        "sensor_noise",
        "dead_actuator",
        "sign_flip",
        "action_latency",
    ]

    def __init__(
        self,
        env: gym.Env,
        split: str = "train",
        fixed_fault: Optional[str] = None,
        severity: float = 1.0,
        fixed_latency: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(env)
        self.split = split
        self.fixed_fault = fixed_fault
        self.severity = float(severity)
        self.fixed_latency = fixed_latency

        assert isinstance(env.action_space, spaces.Box), "Only Box action spaces supported"
        assert isinstance(env.observation_space, spaces.Box), "Only Box observation spaces supported"

        self.orig_act_dim = int(np.prod(env.action_space.shape))
        self.orig_obs_dim = int(np.prod(env.observation_space.shape))

        # Packed observation dimension: s_t (orig_obs_dim) + a_{t-1} (orig_act_dim) + r_{t-1} (1) + d_{t-1} (1)
        self.packed_obs_dim = self.orig_obs_dim + self.orig_act_dim + 2

        obs_low = np.concatenate([
            env.observation_space.low,
            env.action_space.low,
            np.array([-np.inf, 0.0], dtype=np.float32),
        ])
        obs_high = np.concatenate([
            env.observation_space.high,
            env.action_space.high,
            np.array([np.inf, 1.0], dtype=np.float32),
        ])
        self.observation_space = spaces.Box(
            low=obs_low.astype(np.float32),
            high=obs_high.astype(np.float32),
            dtype=np.float32,
        )

        self.rng = np.random.default_rng(seed)

        # Episode state variables
        self.active_fault: str = "none"
        self.fault_params: Dict[str, Any] = {}
        self.prev_action: np.ndarray = np.zeros(self.orig_act_dim, dtype=np.float32)
        self.prev_reward: float = 0.0
        self.prev_done: float = 0.0
        self.latency_buffer: List[np.ndarray] = []

    def _sample_fault(self) -> None:
        """Draws one unobservable fault per episode according to the split."""
        if self.fixed_fault is not None:
            fault_type = self.fixed_fault
        elif self.split == "train":
            fault_type = self.rng.choice(["action_scale", "action_noise"])
        elif self.split == "val":
            # Validation derived strictly from training mechanisms with unseen parameters
            fault_type = self.rng.choice(["action_scale", "action_noise"])
        elif self.split in ("test_latency", "action_latency"):
            fault_type = "action_latency"
        elif self.split in ("test_dead", "dead_actuator"):
            fault_type = "dead_actuator"
        elif self.split in ("test_sign_flip", "sign_flip"):
            fault_type = "sign_flip"
        elif self.split in ("test_within", "within_mechanism"):
            fault_type = self.rng.choice(["dead_actuator", "sign_flip"])
        else:
            fault_type = "none"

        self.active_fault = fault_type
        self.fault_params = {}

        s = self.severity
        if fault_type == "action_scale":
            if self.split == "val":
                # Unseen parameter ranges: extreme low [0.2, 0.5) or high (1.5, 1.8]
                if self.rng.random() < 0.5:
                    scales = self.rng.uniform(0.2, 0.5, size=self.orig_act_dim)
                else:
                    scales = self.rng.uniform(1.5, 1.8, size=self.orig_act_dim)
            else:
                # Standard training range: u ~ U[1 - 0.5s, 1 + 0.5s]
                low = max(0.01, 1.0 - 0.5 * s)
                high = 1.0 + 0.5 * s
                scales = self.rng.uniform(low, high, size=self.orig_act_dim)
            self.fault_params["scales"] = scales.astype(np.float32)

        elif fault_type == "action_noise":
            # Gaussian std
            std = 0.75 * s if self.split == "val" else 0.5 * s
            self.fault_params["std"] = float(std)

        elif fault_type == "dead_actuator":
            dead_dim = int(self.rng.integers(0, self.orig_act_dim))
            self.fault_params["dead_dim"] = dead_dim

        elif fault_type == "sign_flip":
            flip_dim = int(self.rng.integers(0, self.orig_act_dim))
            self.fault_params["flip_dim"] = flip_dim

        elif fault_type == "action_latency":
            if self.fixed_latency is not None:
                delay = int(self.fixed_latency)
            else:
                delay = int(self.rng.integers(1, 6))  # U{1, ..., 5}
            self.fault_params["delay"] = delay
            self.latency_buffer = [np.zeros(self.orig_act_dim, dtype=np.float32) for _ in range(delay)]

    def _apply_actuator_fault(self, action: np.ndarray) -> np.ndarray:
        """Applies active fault to commanded action."""
        act = action.copy().astype(np.float32)

        if self.active_fault == "action_scale":
            act = act * self.fault_params["scales"]

        elif self.active_fault == "action_noise":
            noise = self.rng.normal(0.0, self.fault_params["std"], size=self.orig_act_dim)
            act = act + noise

        elif self.active_fault == "dead_actuator":
            act[self.fault_params["dead_dim"]] = 0.0

        elif self.active_fault == "sign_flip":
            act[self.fault_params["flip_dim"]] *= -1.0

        elif self.active_fault == "action_latency":
            self.latency_buffer.append(act.copy())
            act = self.latency_buffer.pop(0)

        # Clip to valid environment bounds
        return np.clip(act, self.env.action_space.low, self.env.action_space.high)

    def _pack_observation(self, base_obs: np.ndarray) -> np.ndarray:
        flat_base = np.asarray(base_obs, dtype=np.float32).reshape(-1)
        packed = np.concatenate([
            flat_base,
            self.prev_action,
            np.array([self.prev_reward, self.prev_done], dtype=np.float32),
        ])
        return packed.astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            base_obs, info = self.env.reset(seed=seed, options=options, **kwargs)
        else:
            base_obs, info = self.env.reset(options=options, **kwargs)
        self._sample_fault()
        self.prev_action = np.zeros(self.orig_act_dim, dtype=np.float32)
        self.prev_reward = 0.0
        self.prev_done = 0.0

        packed_obs = self._pack_observation(base_obs)
        info["fault_type"] = self.active_fault
        info["fault_params"] = self.fault_params
        info["fault_id"] = self.get_fault_id()
        return packed_obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        commanded_action = np.asarray(action, dtype=np.float32).copy()
        corrupted_action = self._apply_actuator_fault(commanded_action)

        base_obs, reward, terminated, truncated, info = self.env.step(corrupted_action)

        # Update previous commanded action, reward, and done
        self.prev_action = commanded_action
        self.prev_reward = float(reward)
        self.prev_done = 1.0 if terminated else 0.0

        packed_obs = self._pack_observation(base_obs)
        info["fault_type"] = self.active_fault
        info["fault_params"] = self.fault_params
        info["fault_id"] = self.get_fault_id()
        info["corrupted_action"] = corrupted_action
        return packed_obs, float(reward), terminated, truncated, info

    def get_fault_id(self) -> int:
        """Returns integer index for auxiliary classification head."""
        try:
            return self.FAULT_NAMES.index(self.active_fault)
        except ValueError:
            return 0


def make_fault_env(
    env_id: str = "HalfCheetah-v5",
    split: str = "train",
    fixed_fault: Optional[str] = None,
    seed: Optional[int] = None,
) -> gym.Env:
    """Factory function for creating wrapped fault-injected environments."""
    base_env = gym.make(env_id)
    return FaultInjectionWrapper(base_env, split=split, fixed_fault=fixed_fault, seed=seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test fault environment wrapper")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    print("Testing FaultInjectionWrapper on HalfCheetah-v5...")
    for sp in ["train", "val", "test_latency", "test_dead", "test_sign_flip"]:
        env = make_fault_env("HalfCheetah-v5", split=sp, seed=42)
        obs, info = env.reset()
        assert obs.shape == (25,), f"Expected 25 (17+6+2), got {obs.shape}"
        act = env.action_space.sample()
        next_obs, rew, term, trunc, info = env.step(act)
        assert next_obs.shape == (25,)
        print(f"  Split '{sp:13s}' OK! Active fault: {info['fault_type']:15s} (id={info['fault_id']})")
        env.close()
    print("All splits verified successfully!")
