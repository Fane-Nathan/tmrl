# Sequence-based replay buffer for the recurrent (RL²) agent.
#
# Stores transitions in flat lists (like existing memories) and additionally
# tracks per-step episode boundaries so that sample() can draw intra-episode
# windows of length seq_len + 1 = burn_in + train_window + 1.
#
# Storage convention (same as GenericTorchMemory):
#   self.data[0]: prev_act    — action that produced data[1][t]
#   self.data[1]: obs         — tuple of np.ndarray per timestep
#   self.data[2]: reward      — float per timestep
#   self.data[3]: terminated  — bool per timestep
#   self.data[4]: truncated   — bool per timestep
#   self.data[5]: info        — dict per timestep
#   self.data[6]: done        — terminated OR truncated, bool per timestep
#
# Sample contract: sample() returns a dict of torch tensors:
#   obs:        tuple of [B, T+1, *obs_component] tensors
#   a_prev:     [B, T+1, act_dim]
#   r_prev:     [B, T+1]
#   d_prev:     [B, T+1]
#   train_mask: [B, T] — 1.0 on the last train_window positions, 0.0 on burn-in
# Where T = burn_in + train_window. The +1 position is for the next-state hidden
# used by the Q-target.

import random
from typing import Callable, Optional

import numpy as np
import torch

from tmrl.memory import TorchMemory
from tmrl.util import collate_torch


# Sample compressor for the RL² lidar-progress wrapper.
# obs structure: (speed, progress, lidar, a_prev, r_prev, d_prev) — and any
# trailing rtgym action-buffer entries. We keep the last 19 lidar samples and
# splat the rest through with *obs[3:] so trailing elements survive intact.
def get_local_buffer_sample_lidar_progress_rl2(prev_act, obs, rew, terminated,
                                                truncated, info):
    obs_mod = (obs[0], obs[1], obs[2][-19:], *obs[3:])
    rew_mod = np.float32(rew)
    return prev_act, obs_mod, rew_mod, terminated, truncated, info


# Sample compressor for the RL² image wrapper.
# obs structure: (speed, gear, rpm, imgs, a_prev, r_prev, d_prev) — and any
# trailing rtgym action-buffer entries. Keeps only the most recent image; splats
# everything after the imgs element so trailing components survive.
def get_local_buffer_sample_tm20_imgs_rl2(prev_act, obs, rew, terminated,
                                           truncated, info):
    obs_mod = (obs[0], obs[1], obs[2],
               (obs[3][-1] * 256.0).astype(np.uint8),
               *obs[4:])
    return prev_act, obs_mod, rew, terminated, truncated, info


class SequenceMemory(TorchMemory):
    def __init__(self,
                 burn_in: int = 20,
                 train_window: int = 64,
                 memory_size: int = 1_000_000,
                 batch_size: int = 32,
                 dataset_path: str = "",
                 nb_steps: int = 1,
                 sample_preprocessor: Optional[Callable] = None,
                 crc_debug: bool = False,
                 device: str = "cpu"):
        self.burn_in = burn_in
        self.train_window = train_window
        self.seq_len = burn_in + train_window
        self._episodes_dirty = True
        self._ep_starts: list[int] = []
        self._ep_lens: list[int] = []
        self._cumulative_valid: np.ndarray = np.zeros(0, dtype=np.int64)
        self._total_valid_starts: int = 0
        super().__init__(memory_size=memory_size,
                         batch_size=batch_size,
                         dataset_path=dataset_path,
                         nb_steps=nb_steps,
                         sample_preprocessor=sample_preprocessor,
                         crc_debug=crc_debug,
                         device=device)

    # ---- write path --------------------------------------------------------
    def append_buffer(self, buffer):
        d0 = [b[0] for b in buffer.memory]
        d1 = [b[1] for b in buffer.memory]
        d2 = [b[2] for b in buffer.memory]
        d3 = [b[3] for b in buffer.memory]
        d4 = [b[4] for b in buffer.memory]
        d5 = [b[5] for b in buffer.memory]
        d6 = [bool(b[3]) or bool(b[4]) for b in buffer.memory]

        if len(self.data) == 0:
            self.data.extend([d0, d1, d2, d3, d4, d5, d6])
        else:
            self.data[0] += d0
            self.data[1] += d1
            self.data[2] += d2
            self.data[3] += d3
            self.data[4] += d4
            self.data[5] += d5
            self.data[6] += d6

        to_trim = len(self.data[0]) - self.memory_size
        if to_trim > 0:
            for i in range(7):
                self.data[i] = self.data[i][to_trim:]

        self._episodes_dirty = True

    def __len__(self) -> int:
        if len(self.data) == 0:
            return 0
        if self._episodes_dirty:
            self._rebuild_episode_index()
        return self._total_valid_starts

    # ---- episode bookkeeping -----------------------------------------------
    def _rebuild_episode_index(self):
        self._ep_starts.clear()
        self._ep_lens.clear()
        if len(self.data) == 0 or len(self.data[0]) == 0:
            self._cumulative_valid = np.zeros(0, dtype=np.int64)
            self._total_valid_starts = 0
            self._episodes_dirty = False
            return

        dones = self.data[6]
        n = len(dones)
        ep_start = 0
        for t in range(n):
            if dones[t]:
                ep_len = t - ep_start + 1
                self._ep_starts.append(ep_start)
                self._ep_lens.append(ep_len)
                ep_start = t + 1
        if ep_start < n:
            self._ep_starts.append(ep_start)
            self._ep_lens.append(n - ep_start)

        # Valid starts per episode: we draw a window of seq_len + 1 positions
        # [start, start + seq_len] (inclusive). For MVP we require strict
        # intra-episode — no done flag anywhere inside the window. The done
        # step is the last position of an episode of length L (relative pos
        # L - 1). So the last allowed (start + seq_len) is L - 2, i.e.,
        # start <= L - seq_len - 2 ⇒ valid_starts = L - seq_len - 1.
        valid = np.array([max(0, L - self.seq_len - 1) for L in self._ep_lens],
                         dtype=np.int64)
        self._cumulative_valid = np.cumsum(valid)
        self._total_valid_starts = int(valid.sum())
        self._episodes_dirty = False

    def _pick_global_start(self) -> int:
        """Pick a random valid storage index that admits a full window."""
        if self._total_valid_starts <= 0:
            raise RuntimeError("SequenceMemory has no valid sequence starts; "
                               "need at least one episode of length >= "
                               f"{self.seq_len + 1}")
        k = random.randint(0, self._total_valid_starts - 1)
        ep_idx = int(np.searchsorted(self._cumulative_valid, k, side="right"))
        prev_cumulative = self._cumulative_valid[ep_idx - 1] if ep_idx > 0 else 0
        within_ep = k - prev_cumulative
        return self._ep_starts[ep_idx] + within_ep

    # ---- sample path -------------------------------------------------------
    def get_transition(self, item):
        raise NotImplementedError("SequenceMemory does not support 1-step transition "
                                  "sampling. Use sample() directly.")

    def _build_single_sequence(self, start: int) -> dict:
        """Slice self.data[start : start + seq_len + 1] and return one sample."""
        end = start + self.seq_len + 1

        obs_slice = self.data[1][start:end]
        a_slice = self.data[0][start:end]
        r_slice = self.data[2][start:end]
        d_slice = self.data[6][start:end]

        n_components = len(obs_slice[0]) if isinstance(obs_slice[0], (tuple, list)) else 1
        if n_components == 1 and not isinstance(obs_slice[0], (tuple, list)):
            obs_arr = np.stack([np.asarray(o) for o in obs_slice], axis=0)
            obs_out = obs_arr.astype(np.float32, copy=False)
        else:
            obs_out = tuple(
                np.stack([np.asarray(o[c]) for o in obs_slice], axis=0).astype(np.float32, copy=False)
                for c in range(n_components)
            )

        a_prev = np.stack([np.asarray(a, dtype=np.float32) for a in a_slice], axis=0)
        r_prev = np.asarray(r_slice, dtype=np.float32)
        d_prev = np.asarray([float(x) for x in d_slice], dtype=np.float32)

        train_mask = np.zeros((self.seq_len,), dtype=np.float32)
        train_mask[self.burn_in:] = 1.0

        return {
            "obs": obs_out,
            "a_prev": a_prev,
            "r_prev": r_prev,
            "d_prev": d_prev,
            "train_mask": train_mask,
        }

    def sample(self):
        if self._episodes_dirty:
            self._rebuild_episode_index()
        batch = [self._build_single_sequence(self._pick_global_start())
                 for _ in range(self.batch_size)]
        return collate_torch(batch, self.device)


# Smoke test ============================================================================================================

def _smoke_test():
    """Feed synthetic episodes and verify intra-episode sampling + shapes."""
    class _FakeBuffer:
        def __init__(self, memory):
            self.memory = memory
            self.stat_train_return = 0.0
            self.stat_test_return = 0.0
            self.stat_train_steps = 0
            self.stat_test_steps = 0

        def __len__(self):
            return len(self.memory)

    burn_in, train_window = 4, 8
    seq_len = burn_in + train_window
    mem = SequenceMemory(burn_in=burn_in, train_window=train_window,
                         batch_size=3, memory_size=10_000, device="cpu")

    # 3 episodes of lengths 40, 20, 50.
    rng = np.random.default_rng(0)
    transitions = []
    for ep_len in [40, 20, 50]:
        for t in range(ep_len):
            obs = (rng.standard_normal(2).astype(np.float32),
                   rng.standard_normal((3,)).astype(np.float32))
            prev_act = rng.standard_normal(3).astype(np.float32)
            rew = float(rng.standard_normal())
            terminated = (t == ep_len - 1)
            truncated = False
            info = {}
            transitions.append((prev_act, obs, rew, terminated, truncated, info))
    mem.append(_FakeBuffer(transitions))

    # Episode lengths in storage: 40, 20, 50. Valid starts per ep: max(0, L - seq_len - 1)
    #   ep0: 40 - 12 - 1 = 27
    #   ep1: 20 - 12 - 1 = 7
    #   ep2: 50 - 12 - 1 = 37
    # Total: 71
    assert len(mem) == 71, f"len(mem)={len(mem)}, expected 71"

    batch = mem.sample()
    assert isinstance(batch, dict), "expected dict"
    assert "obs" in batch and "a_prev" in batch and "r_prev" in batch
    assert "d_prev" in batch and "train_mask" in batch

    obs = batch["obs"]
    assert isinstance(obs, tuple) and len(obs) == 2
    assert obs[0].shape == (3, seq_len + 1, 2), f"obs[0]: {obs[0].shape}"
    assert obs[1].shape == (3, seq_len + 1, 3), f"obs[1]: {obs[1].shape}"
    assert batch["a_prev"].shape == (3, seq_len + 1, 3), f"a_prev: {batch['a_prev'].shape}"
    assert batch["r_prev"].shape == (3, seq_len + 1)
    assert batch["d_prev"].shape == (3, seq_len + 1)
    assert batch["train_mask"].shape == (3, seq_len)
    assert batch["train_mask"].sum() == 3 * train_window

    # Critically: no `done` inside the seq_len + 1 window (intra-episode guarantee).
    d_inside = batch["d_prev"].numpy()
    assert d_inside.sum() == 0.0, f"d_prev should be all zero inside window: sum={d_inside.sum()}"

    # Reproducibility-spot-check: many samples should never violate intra-episode.
    random.seed(0)
    for _ in range(200):
        start = mem._pick_global_start()
        window_d = [bool(mem.data[6][start + p]) for p in range(seq_len + 1)]
        assert not any(window_d), f"intra-episode violated at start={start}"

    print(f"smoke OK  len(mem)={len(mem)}  obs shapes={[tuple(o.shape) for o in obs]}  "
          f"train_mask.sum()={int(batch['train_mask'].sum())}")


if __name__ == "__main__":
    _smoke_test()
