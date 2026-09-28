"""Regression checks for the data actually supplied to the pretrained brain."""
import unittest
from unittest.mock import Mock

import numpy as np
import torch
from torch import nn

from scripts.audit_foundation_deployment import worker_observation
from tmrl.custom.torch.foundation_encoder import FoundationVisualTelemetryEncoder


class SpyKinematics(nn.Module):
    def forward(self, x):
        self.last_input = x.detach().clone()
        return torch.zeros((x.shape[0], 128))


class StubBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.kinematics_proj = SpyKinematics()
        self.fusion_proj = nn.Linear(256, 256)

    def encode_visual(self, images):
        return torch.zeros((images.shape[0], 128))


class FoundationObservationContractTests(unittest.TestCase):
    def test_latest_applied_action_conditions_foundation(self):
        encoder = FoundationVisualTelemetryEncoder.__new__(FoundationVisualTelemetryEncoder)
        nn.Module.__init__(encoder)
        encoder.brain = StubBrain()
        encoder.latent_proj = nn.Linear(256, 128)
        encoder.img_height = encoder.img_width = 96
        encoder._maybe_reload_foundation_weights = Mock()
        state = np.zeros(15, dtype=np.float32)
        state[[0, 3, 4]] = [0.5, 3.0, 0.7]
        older = np.asarray([1.0, -1.0, -1.0], dtype=np.float32)
        latest = np.asarray([-1.0, 1.0, 1.0], dtype=np.float32)
        observation = worker_observation(state, np.zeros((4, 96, 96), dtype=np.uint8),
                                         (older, latest))
        self.assertFalse(torch.equal(observation[4], observation[5]))
        # Training-mode forward avoids invoking the irrelevant policy head.
        encoder(observation)
        actual = encoder.brain.kinematics_proj.last_input
        torch.testing.assert_close(actual[0, 15:18], torch.from_numpy(latest))
        torch.testing.assert_close(actual[0, [0, 3, 4]], torch.tensor([0.5, 3.0, 0.7]))


if __name__ == "__main__":
    unittest.main()
