"""Behavioral checks: sensing, physics isolation, replay boundaries, and learning."""
import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from frozen_models import TransformerRL2Agent
from learner import Learner, configure, features
from replay import extract_transition, load_episodes, sample_batch, save_episodes
from sensor_env import SensorFaultEnv, make_env

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
torch.set_num_threads(1)


class CounterEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(-np.inf, np.inf, (2,), np.float32)
        self.action_space = spaces.Box(-1, 1, (1,), np.float32)
        self.t = 0
        self.last_action = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.array([10.0, 20.0], np.float32), {}

    def step(self, action):
        self.last_action = action.copy()
        self.t += 1
        return np.array([10 + self.t, 20 + self.t], np.float32), 2.0 + float(action[0]), False, self.t == 8, {}


def episode(length=40):
    return {"observations": np.repeat(np.arange(length + 1, dtype=np.float32)[:, None], 25, axis=1),
            "actions": np.zeros((length, 6), np.float32),
            "rewards": np.arange(length, dtype=np.float32),
            "terminated": np.array([0.] * (length - 1) + [1.], np.float32)}


class SensorTests(unittest.TestCase):
    def test_noise_changes_sensor_but_preserves_command_reward_and_bookkeeping(self):
        env = SensorFaultEnv(CounterEnv(), "sensor_noise", noise_std=[1.0, 0.0])
        obs, info = env.reset(seed=13)
        self.assertNotEqual(obs[0], 10.0)
        self.assertEqual(obs[1], 20.0)
        np.testing.assert_array_equal(obs[2:], np.zeros(3))
        obs, reward, term, trunc, info = env.step(np.array([0.3], np.float32))
        np.testing.assert_array_equal(env.unwrapped.last_action, [np.float32(0.3)])
        self.assertAlmostEqual(reward, 2.3)
        self.assertAlmostEqual(obs[2], 0.3)
        self.assertAlmostEqual(obs[3], 2.3)
        self.assertTrue(env.observation_space.contains(obs))

    def test_noise_repeatable_and_independent_of_environment_rng(self):
        env = SensorFaultEnv(CounterEnv(), "sensor_noise", noise_std=1)
        first = env.reset(seed=9)[0]
        second = env.reset(seed=9)[0]
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first[:2], env.reset(seed=10)[0][:2]))

    def test_delay_is_of_reading_not_of_action_or_reward(self):
        env = SensorFaultEnv(CounterEnv(), "sensor_delay", delay=2)
        np.testing.assert_array_equal(env.reset(seed=1)[0][:2], [0, 0])
        first = env.step(np.array([0.4], np.float32))[0]
        np.testing.assert_array_equal(first[:2], [0, 0])
        second = env.step(np.array([-0.7], np.float32))[0]
        np.testing.assert_array_equal(second[:2], [10, 20])
        self.assertAlmostEqual(second[2], -0.7)
        self.assertAlmostEqual(second[3], 1.3)
        np.testing.assert_array_equal(env.reset(seed=1)[0][:2], [0, 0])

    def test_delayed_fault_onset_uses_actual_prefault_history(self):
        env = SensorFaultEnv(CounterEnv(), "sensor_delay", delay=2, onset=3)
        np.testing.assert_array_equal(env.reset(seed=1)[0][:2], [10, 20])
        env.step(np.array([0], np.float32))
        np.testing.assert_array_equal(env.step(np.array([0], np.float32))[0][:2], [12, 22])
        obs, _, _, _, info = env.step(np.array([0], np.float32))
        np.testing.assert_array_equal(obs[:2], [11, 21])
        self.assertTrue(info["sensor_audit"]["active"])

    def test_invalid_faults_fail_instead_of_silently_doing_nothing(self):
        for kwargs in ({"fault": "action_noise"}, {"fault": "sensor_noise", "noise_std": 0},
                       {"fault": "sensor_delay", "delay": -1}, {"onset": -1}):
            with self.assertRaises(ValueError):
                SensorFaultEnv(CounterEnv(), **kwargs)

    def test_real_mujoco_physics_identical_under_identical_commands(self):
        environments = [make_env(condition, CONFIG) for condition in ["clean", "sensor_noise", "sensor_delay"]]
        try:
            for env in environments:
                env.reset(seed=111)
            for step in range(8):
                command = np.full(6, np.sin(step) * 0.4, np.float32)
                results = [env.step(command) for env in environments]
                for result in results[1:]:
                    np.testing.assert_array_equal(result[4]["sensor_audit"]["true_observation"],
                                                  results[0][4]["sensor_audit"]["true_observation"])
                    self.assertEqual(result[1], results[0][1])
            self.assertFalse(np.array_equal(results[0][0][:17], results[1][0][:17]))
            self.assertFalse(np.array_equal(results[0][0][:17], results[2][0][:17]))
        finally:
            for env in environments:
                env.close()


class ReplayTests(unittest.TestCase):
    def test_short_and_final_transitions_have_correct_next_context(self):
        data = episode()
        cur, nxt = extract_transition(data, 0, 32)
        np.testing.assert_array_equal(cur[:, 0], [0])
        np.testing.assert_array_equal(nxt[:, 0], [0, 1])
        cur, nxt = extract_transition(data, 39, 32)
        np.testing.assert_array_equal(cur[:, 0], np.arange(8, 40))
        np.testing.assert_array_equal(nxt[:, 0], np.arange(9, 41))
        cur, nxt = extract_transition(data, 39, 1)
        self.assertEqual(cur[0, 0], 39)
        self.assertEqual(nxt[0, 0], 40)

    def test_mixed_replay_counts_and_terminal_coverage(self):
        data = episode(3)
        batch = sample_batch([data], [data], 400, 32, 0.5, np.random.default_rng(5), "cpu")
        self.assertEqual(batch["origins"].count("old"), 200)
        self.assertEqual(batch["origins"].count("new"), 200)
        self.assertIn(2, batch["positions"])
        self.assertGreater(float(batch["terminated"].sum()), 0)
        self.assertTrue((batch["next_lengths"] <= 4).all())

    def test_replay_roundtrip_without_pickle(self):
        data = episode(3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodes.npz"
            save_episodes(path, [data])
            loaded = load_episodes(path)
            for key in data:
                np.testing.assert_array_equal(data[key], loaded[0][key])


class LegacyBoundaryTests(unittest.TestCase):
    def test_actuator_wrapper_rejects_sensor_labels(self):
        path = HERE.parents[1] / "scripts" / "benchmark_fault_env.py"
        spec = importlib.util.spec_from_file_location("legacy_actuator_wrapper", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for arguments in ({"fixed_fault": "sensor_noise"}, {"fixed_fault": "sensor_delay"},
                          {"split": "test_sensor_noise"}, {"split": "sensor_delay"}):
            with self.assertRaisesRegex(ValueError, "does not implement observation faults"):
                module.FaultInjectionWrapper(CounterEnv(), **arguments)


class LearningTests(unittest.TestCase):
    def test_right_padding_is_invisible_to_last_valid_context(self):
        torch.manual_seed(10)
        model = configure(TransformerRL2Agent(), "transformer", 32)
        short = torch.randn(2, 4, 25)
        padded = torch.cat([short, torch.randn(2, 28, 25)], 1)
        with torch.no_grad():
            a = features(model, "transformer", short, torch.tensor([4, 4]))
            b = features(model, "transformer", padded, torch.tensor([4, 4]))
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)

    def test_real_update_changes_model_and_preserves_unused_positions(self):
        torch.manual_seed(6)
        model = TransformerRL2Agent()
        initial = model.trunk.pos_emb.detach().clone()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.pt"
            torch.save(model.state_dict(), path)
            learner = Learner("transformer", path, CONFIG, "cpu", 5)
            batch = sample_batch([episode(35)], [episode(35)], 8, 32, 0.5, np.random.default_rng(1), "cpu")
            metrics = learner.update(batch)
            self.assertTrue(np.isfinite(list(metrics.values())).all())
            torch.testing.assert_close(initial[:, 32:], learner.model.trunk.pos_emb[:, 32:], rtol=0, atol=0)
            self.assertGreater(float((initial[:, :32] - learner.model.trunk.pos_emb[:, :32]).abs().max().detach()), 0)
            reloaded_source = torch.load(path, weights_only=True)
            torch.testing.assert_close(initial, reloaded_source["trunk.pos_emb"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
