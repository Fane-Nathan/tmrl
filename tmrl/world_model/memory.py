import random
import numpy as np

from tmrl.custom.custom_memories import MemoryTMFull
from tmrl.util import collate_torch


class WorldModelMemory(MemoryTMFull):
    """TMRL image replay memory that yields contiguous temporal windows.

    It reuses MemoryTMFull's compact storage and frame/action-history rebuilding,
    but changes the sampling unit from one transition to a short sequence.
    """

    def __init__(self, *args, horizon=5, **kwargs):
        super().__init__(*args, **kwargs)
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.horizon = int(horizon)

    def _is_valid_start(self, item):
        if item < 0 or item + self.horizon > len(self):
            return False
        eoes = self.data[4]
        context_end = item + self.min_samples + self.horizon - 2
        return not any(eoes[item : context_end + 1])

    def _sample_start(self):
        max_start = len(self) - self.horizon
        if max_start < 0:
            raise RuntimeError(
                f"Replay has {len(self)} transitions but horizon={self.horizon}. Collect more data before training."
            )
        for _ in range(128):
            item = random.randint(0, max_start)
            if self._is_valid_start(item):
                return item
        for item in range(max_start + 1):
            if self._is_valid_start(item):
                return item
        raise RuntimeError("No contiguous world-model sequence is available in replay yet.")

    @staticmethod
    def _stack_observations(observations):
        fields = zip(*observations)
        return tuple(np.stack(tuple(field), axis=0) for field in fields)

    def _get_sequence(self, start):
        observations, actions, rewards, terminated, truncated = [], [], [], [], []
        for offset in range(self.horizon):
            prev_obs, action, reward, new_obs, term, trunc, _ = super().get_transition(start + offset)
            if offset == 0:
                observations.append(prev_obs)
            observations.append(new_obs)
            actions.append(action)
            rewards.append(np.float32(reward))
            terminated.append(np.float32(term))
            truncated.append(np.float32(trunc))
        return (
            self._stack_observations(observations),
            np.stack(actions).astype(np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminated, dtype=np.float32),
            np.asarray(truncated, dtype=np.float32),
        )

    def sample(self):
        batch = [self._get_sequence(self._sample_start()) for _ in range(self.batch_size)]
        return collate_torch(batch, self.device)
