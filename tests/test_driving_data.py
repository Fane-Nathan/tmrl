"""Causal export and episode-isolated training regressions."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from scripts.export_live_driving_demos import load_verified_run
from scripts.train_multitrack_vision_curriculum import select_episode_split, extract_causal_sequence_dataset


class LiveExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "run.json"
        self.result = {"schema_version": 2, "finished": True, "finish_signal_observed": True,
                       "reason": "finish_signal", "controls_released": True,
                       "action_alignment": "observation_t_then_command_t",
                       "telemetry_speed_unit": "m/s", "map_sha256": "a" * 64,
                       "camera_label": "first-person", "run_id": "test_run",
                       "mode": "tracking", "assisted": True, "controller_config": {},
                       "period_seconds": 0.05, "steps": 32, "observations": 33,
                       "max_deviation_m": 1.0}
        self.telemetry = np.zeros((33, 11), dtype=np.float32)
        self.telemetry[-1, 8] = 1
        self.frames = np.broadcast_to(np.arange(33, dtype=np.uint8)[:, None, None], (33, 96, 96)).copy()
        self.actions = np.tile([1.0, -1.0, 0.25], (32, 1)).astype(np.float32)
        self.obs_time = np.arange(33) * 0.05
        self.act_time = self.obs_time[:-1] + 0.001

    def tearDown(self):
        self.temp.cleanup()

    def load(self):
        self.path.write_text(json.dumps(self.result), encoding="utf-8")
        np.savez(self.path.with_suffix(".npz"), telemetry=self.telemetry, frames=self.frames,
                 actions=self.actions, observation_seconds=self.obs_time, command_seconds=self.act_time)
        return load_verified_run(self.path)

    def test_history_and_previous_action_never_look_ahead(self):
        episode, source = self.load()
        np.testing.assert_array_equal(episode["imgs"][:4, :, 0, 0],
                                      [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 2], [0, 1, 2, 3]])
        np.testing.assert_array_equal(episode["previous_actions"][0], np.zeros(3))
        np.testing.assert_array_equal(episode["previous_actions"][1:], self.actions[:-1])
        self.assertEqual(np.count_nonzero(episode["terminated"]), 1)
        self.assertTrue(episode["terminated"][-1])
        self.assertTrue(source["assisted"])

    def test_failed_run_cannot_be_exported_as_success(self):
        self.result["finished"] = False
        with self.assertRaisesRegex(ValueError, "finish-confirmed"):
            self.load()

    def test_command_before_observation_is_rejected(self):
        self.act_time[1] = self.obs_time[1] - 0.001
        with self.assertRaisesRegex(ValueError, "Non-causal"):
            self.load()

    def test_long_sampling_gap_is_rejected(self):
        self.obs_time[15:] += 1
        with self.assertRaisesRegex(ValueError, "sampling gap"):
            self.load()

    def test_missing_frames_are_rejected(self):
        self.frames = np.empty(0, dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "frame per observation"):
            self.load()


class EpisodeSplitTests(unittest.TestCase):
    def test_explicit_splits_are_preserved(self):
        train, validation = {"split": "train"}, {"split": "validation"}
        self.assertEqual(select_episode_split([train, validation], "train"), [train])
        self.assertEqual(select_episode_split([train, validation], "validation"), [validation])

    def test_legacy_data_holds_out_last_entire_episode(self):
        episodes = [{"id": 1}, {"id": 2}, {"id": 3}]
        self.assertEqual(select_episode_split(episodes, "train"), episodes[:2])
        self.assertEqual(select_episode_split(episodes, "validation"), episodes[2:])

    def test_partial_split_metadata_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Every episode"):
            select_episode_split([{"split": "train"}, {}], "train")

    def test_last_window_retained_without_validation_leakage(self):
        episodes = []
        for split, value in (("train", -1), ("validation", 1)):
            episodes.append({"split": split, "imgs": np.zeros((9, 4, 96, 96), dtype=np.uint8),
                             "states": np.zeros((9, 15), dtype=np.float32),
                             "actions": np.full((9, 3), value, dtype=np.float32)})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "demo.pt"
            torch.save({"episodes": episodes}, path)
            train = extract_causal_sequence_dataset(path, window_len=4, stride=4)
            validation = extract_causal_sequence_dataset(path, window_len=4, stride=4, split="validation")
        self.assertEqual(len(validation[0]), 3)  # starts 0, 4, 5 (terminal window)
        self.assertTrue(torch.all(train[-1] == -1))
        self.assertTrue(torch.all(validation[-1] == 1))


if __name__ == "__main__":
    unittest.main()
