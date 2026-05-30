# Sequence-based replay buffer for the recurrent (RL2) agent.
#
# Stores transitions in flat lists (like existing memories) and additionally
# tracks per-step episode boundaries so that sample() can draw intra-episode
# windows of length seq_len + 1 = burn_in + train_window + 1.
#
# Storage convention (same as GenericTorchMemory):
#   self.data[0]: prev_act    - action that produced data[1][t]
#   self.data[1]: obs         - tuple of np.ndarray per timestep
#   self.data[2]: reward      - float per timestep
#   self.data[3]: terminated  - bool per timestep
#   self.data[4]: truncated   - bool per timestep
#   self.data[5]: info        - dict per timestep
#   self.data[6]: done        - terminated OR truncated, bool per timestep
#   self.data[7]: task_id     - discrete RL2 task label per timestep
#
# Sample contract: sample() returns a dict of torch tensors:
#   obs:        tuple of [B, T+1, *obs_component] tensors
#   a_prev:     [B, T+1, act_dim]
#   r_prev:     [B, T+1]
#   d_prev:     [B, T+1]  - terminated ONLY; truncation (step cap) does NOT set it
#   train_mask: [B, T] - 1.0 on trainable current positions, 0.0 elsewhere
#   task_id:    [B]
# Where T = burn_in + train_window. The +1 position is for the next-state hidden
# used by the Q-target.
#
# Window policy (terminal-aware, variable length):
#   A window starts at any in-episode index and runs up to AND INCLUDING the
#   episode's last step (so the finish/crash reward enters training). If that last
#   step is a true termination, d_prev=1 there and the Q-target bootstrap is zeroed;
#   if it is a truncation (step cap), d_prev=0 and the target bootstraps from the
#   real next-state - a truncated episode is not a real ending. Episodes shorter
#   than the full window are right-padded to T+1; padding is causally after every
#   real token so it cannot corrupt real hidden states. train_mask is computed
#   per-sample: a current position c is trainable iff burn_in <= c <= real_len - 2
#   (i.e. its next-state c+1 is a real, non-padded step). This replaces the earlier
#   "no done anywhere in window" rule, which excluded all terminals and starved
#   short episodes.

import random
from typing import Callable, Optional

import numpy as np
import torch

from tmrl.memory import TorchMemory
from tmrl.util import collate_torch


def _stack_obs_component(values):
    arr = np.stack([np.asarray(v) for v in values], axis=0)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 256.0
    return arr.astype(np.float32, copy=False)


def _pad_time(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Right-pad a [t, ...] array to [target_len, ...] with zeros on axis 0."""
    t = arr.shape[0]
    if t >= target_len:
        return arr
    pad_width = [(0, target_len - t)] + [(0, 0)] * (arr.ndim - 1)
    return np.pad(arr, pad_width, mode="constant")


def _task_id_from_info(info) -> int:
    if isinstance(info, dict):
        try:
            return int(info.get("rl2_task_id", 0))
        except (TypeError, ValueError):
            return 0
    return 0


# Sample compressor for the RL2 lidar-progress wrapper.
# obs structure: (speed, progress, lidar, a_prev, r_prev, d_prev) - and any
# trailing rtgym action-buffer entries. We keep the last 19 lidar samples and
# splat the rest through with *obs[3:] so trailing elements survive intact.
def get_local_buffer_sample_lidar_progress_rl2(prev_act, obs, rew, terminated,
                                                truncated, info):
    obs_mod = (obs[0], obs[1], obs[2][-19:], *obs[3:])
    rew_mod = np.float32(rew)
    return prev_act, obs_mod, rew_mod, terminated, truncated, info


# Sample compressor for the RL2 image wrapper.
# obs structure: (speed, gear, rpm, imgs, a_prev, r_prev, d_prev) - and any
# trailing rtgym action-buffer entries. Keeps only the most recent image; splats
# everything after the imgs element so trailing components survive.
def get_local_buffer_sample_tm20_imgs_rl2(prev_act, obs, rew, terminated,
                                           truncated, info):
    obs_mod = (obs[0], obs[1], obs[2],
               np.clip(obs[3][-1:] * 256.0, 0, 255).astype(np.uint8),
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
        d7 = [_task_id_from_info(b[5]) for b in buffer.memory]

        if len(self.data) == 0:
            self.data.extend([d0, d1, d2, d3, d4, d5, d6, d7])
        else:
            if len(self.data) < 8:
                self.data.append([0 for _ in range(len(self.data[0]))])
            self.data[0] += d0
            self.data[1] += d1
            self.data[2] += d2
            self.data[3] += d3
            self.data[4] += d4
            self.data[5] += d5
            self.data[6] += d6
            self.data[7] += d7

        to_trim = len(self.data[0]) - self.memory_size
        if to_trim > 0:
            for i in range(len(self.data)):
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

        # Valid starts per episode. A window starting at in-episode offset `off`
        # spans real positions [off, min(off + seq_len, L - 1)], i.e. real_len =
        # min(seq_len + 1, L - off), and may include the terminal at the end.
        # It yields >= 1 trainable current position when real_len >= burn_in + 2
        # (current burn_in needs a real next-state at burn_in + 1). That holds for
        # off <= L - burn_in - 2, so valid_starts = max(0, L - burn_in - 1).
        valid = np.array([max(0, L - self.burn_in - 1) for L in self._ep_lens],
                         dtype=np.int64)
        self._cumulative_valid = np.cumsum(valid)
        self._total_valid_starts = int(valid.sum())
        self._episodes_dirty = False

    def _pick_global_start(self) -> tuple[int, int]:
        """Pick a random valid (start, real_len) admitting >= 1 trainable position.

        real_len is the number of real (non-padded) steps in the window:
        min(seq_len + 1, episode_len - within_ep_offset). The window runs up to
        and including the episode terminal when the episode ends within reach.
        """
        if self._total_valid_starts <= 0:
            raise RuntimeError("SequenceMemory has no valid sequence starts; "
                               "need at least one episode of length >= "
                               f"{self.burn_in + 2}")
        k = random.randint(0, self._total_valid_starts - 1)
        ep_idx = int(np.searchsorted(self._cumulative_valid, k, side="right"))
        prev_cumulative = self._cumulative_valid[ep_idx - 1] if ep_idx > 0 else 0
        within_ep = k - prev_cumulative
        start = self._ep_starts[ep_idx] + within_ep
        real_len = min(self.seq_len + 1, self._ep_lens[ep_idx] - within_ep)
        return start, real_len

    # ---- sample path -------------------------------------------------------
    def get_transition(self, item):
        raise NotImplementedError("SequenceMemory does not support 1-step transition "
                                  "sampling. Use sample() directly.")

    def _build_single_sequence(self, start: int, real_len: int) -> dict:
        """Slice self.data[start : start + real_len], right-pad to seq_len + 1.

        real_len real steps are read (possibly ending on the episode terminal);
        the remainder up to seq_len + 1 is zero-padded. train_mask marks the
        current positions whose next-state is a real (non-padded) step.
        """
        total = self.seq_len + 1
        end = start + real_len

        obs_slice = self.data[1][start:end]
        a_slice = self.data[0][start:end]
        r_slice = self.data[2][start:end]
        # d_prev carries terminated-ONLY (data[3]), not term|trunc (data[6]).
        # Truncation (e.g. episode-step cap) must bootstrap, not zero the target;
        # and inference packs d_prev = terminated only, so this also keeps the
        # embedder's RL2 done-feature consistent between train and inference.
        # Episode segmentation still uses data[6] in _rebuild_episode_index.
        term_slice = self.data[3][start:end]
        task_id = int(self.data[7][start]) if len(self.data) > 7 else 0

        n_components = len(obs_slice[0]) if isinstance(obs_slice[0], (tuple, list)) else 1
        if n_components == 1 and not isinstance(obs_slice[0], (tuple, list)):
            obs_out = _pad_time(_stack_obs_component(obs_slice), total)
        else:
            obs_out = tuple(
                _pad_time(_stack_obs_component([o[c] for o in obs_slice]), total)
                for c in range(n_components)
            )

        a_prev = _pad_time(
            np.stack([np.asarray(a, dtype=np.float32) for a in a_slice], axis=0), total)
        r_prev = _pad_time(np.asarray(r_slice, dtype=np.float32), total)
        d_prev = _pad_time(np.asarray([float(x) for x in term_slice], dtype=np.float32), total)

        # Current position c is trainable iff its next-state c+1 is real:
        # burn_in <= c <= real_len - 2.
        train_mask = np.zeros((self.seq_len,), dtype=np.float32)
        last_train = real_len - 2
        if last_train >= self.burn_in:
            train_mask[self.burn_in:last_train + 1] = 1.0

        return {
            "obs": obs_out,
            "a_prev": a_prev,
            "r_prev": r_prev,
            "d_prev": d_prev,
            "train_mask": train_mask,
            "task_id": np.asarray(task_id, dtype=np.int64),
        }

    def sample(self):
        if self._episodes_dirty:
            self._rebuild_episode_index()
        batch = [self._build_single_sequence(*self._pick_global_start())
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

    # ep0-2 terminate (crash/finish); ep3 truncates at the step cap (terminated=False).
    episodes = [(40, "term"), (20, "term"), (50, "term"), (30, "trunc")]
    rng = np.random.default_rng(0)
    transitions = []
    for ep_idx, (ep_len, ending) in enumerate(episodes):
        for t in range(ep_len):
            obs = (rng.standard_normal(2).astype(np.float32),
                   rng.standard_normal((3,)).astype(np.float32))
            prev_act = rng.standard_normal(3).astype(np.float32)
            rew = float(rng.standard_normal())
            last = (t == ep_len - 1)
            terminated = last and ending == "term"
            truncated = last and ending == "trunc"
            info = {"rl2_task_id": ep_idx}
            transitions.append((prev_act, obs, rew, terminated, truncated, info))
    mem.append(_FakeBuffer(transitions))

    # Storage: ep0 0..39 (term), ep1 40..59 (term), ep2 60..109 (term), ep3 110..139 (trunc).
    # Valid starts per ep: max(0, L - burn_in - 1) = max(0, L - 5)
    #   ep0: 35   ep1: 15   ep2: 45   ep3: 25   -> total 120
    assert len(mem) == 120, f"len(mem)={len(mem)}, expected 120"

    batch = mem.sample()
    assert isinstance(batch, dict), "expected dict"
    assert "obs" in batch and "a_prev" in batch and "r_prev" in batch
    assert "d_prev" in batch and "train_mask" in batch and "task_id" in batch

    obs = batch["obs"]
    assert isinstance(obs, tuple) and len(obs) == 2
    assert obs[0].shape == (3, seq_len + 1, 2), f"obs[0]: {obs[0].shape}"
    assert obs[1].shape == (3, seq_len + 1, 3), f"obs[1]: {obs[1].shape}"
    assert batch["a_prev"].shape == (3, seq_len + 1, 3), f"a_prev: {batch['a_prev'].shape}"
    assert batch["r_prev"].shape == (3, seq_len + 1)
    assert batch["d_prev"].shape == (3, seq_len + 1)
    assert batch["train_mask"].shape == (3, seq_len)
    assert batch["task_id"].shape == (3,)
    assert batch["task_id"].min() >= 0 and batch["task_id"].max() <= 3

    # Full non-terminal window: start=0, real_len=13. Terminal (idx 39) out of reach.
    full = mem._build_single_sequence(0, seq_len + 1)
    assert full["d_prev"].sum() == 0.0, "early full window must contain no done"
    assert full["train_mask"].sum() == train_window, \
        f"full-window train positions: {full['train_mask'].sum()}"
    assert full["train_mask"][:burn_in].sum() == 0.0, "burn-in must be masked"

    # Terminal window: start=33 in ep0 -> real_len=7, terminal (idx 39) at real pos 6.
    real_len = min(seq_len + 1, 40 - 33)
    term = mem._build_single_sequence(33, real_len)
    assert term["d_prev"][real_len - 1] == 1.0, "terminal step must carry done=1"
    assert term["d_prev"][:real_len - 1].sum() == 0.0, "no done before terminal"
    assert term["d_prev"][real_len:].sum() == 0.0, "padded done must be 0"
    # Trainable current positions: [burn_in, real_len-2] -> real_len-burn_in-1.
    assert term["train_mask"].sum() == real_len - burn_in - 1, \
        f"terminal-window train positions: {term['train_mask'].sum()}"
    for comp in term["obs"]:  # right-padding is zero on every obs component
        assert comp[real_len:].sum() == 0.0, "obs padding must be zero"

    # Truncation window: ep3 (storage 110..139) ends in TRUNCATION, not termination.
    # Even though it is an episode boundary (data[6]=1), d_prev (terminated-only)
    # must be 0 there so the Q-target bootstraps instead of being zeroed.
    trunc_real_len = 12
    trunc_start = 139 - (trunc_real_len - 1)  # 128
    trunc_win = mem._build_single_sequence(trunc_start, trunc_real_len)
    assert bool(mem.data[6][139]) and not bool(mem.data[3][139]), \
        "ep3 last step must be truncated (data[6]=1) but not terminated (data[3]=0)"
    assert trunc_win["d_prev"][trunc_real_len - 1] == 0.0, \
        "truncation must NOT set d_prev (else the bootstrap is wrongly zeroed)"
    assert trunc_win["d_prev"].sum() == 0.0, "no terminated step in the truncation window"
    assert trunc_win["train_mask"].sum() == trunc_real_len - burn_in - 1

    # Sampling integrity: done only ever at the last real position; the window
    # never crosses an episode boundary; every sample has >= 1 train position.
    random.seed(0)
    for _ in range(500):
        start, rl = mem._pick_global_start()
        assert rl >= burn_in + 2, f"real_len {rl} too short to train at start={start}"
        dones_before_last = [bool(mem.data[6][start + p]) for p in range(rl - 1)]
        assert not any(dones_before_last), \
            f"done before last real pos at start={start}, real_len={rl}"

    print(f"smoke OK  len(mem)={len(mem)}  obs shapes={[tuple(o.shape) for o in obs]}  "
          f"full_mask={int(full['train_mask'].sum())}  term_mask={int(term['train_mask'].sum())}  "
          f"term_done@{real_len - 1}  trunc_d_prev@last={trunc_win['d_prev'][trunc_real_len - 1]:.0f}")


if __name__ == "__main__":
    _smoke_test()
