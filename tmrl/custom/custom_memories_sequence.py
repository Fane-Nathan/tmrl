# Sequence-based replay buffer for the recurrent (RL²) agent.
#
# Skeleton file. Implementation lands in task #3.
#
# Stores transitions in flat lists (like existing memories) but additionally
# tracks per-step episode_id and step_idx so that sample() can draw
# intra-episode windows of length burn_in + train_window.

from random import randrange
from typing import Callable, Optional

import numpy as np
import torch

from tmrl.memory import TorchMemory
from tmrl.util import collate_torch


class SequenceMemory(TorchMemory):
    """Replay buffer that samples intra-episode sequences of length
    burn_in + train_window for the recurrent RL² agent.

    Stored items per timestep (same convention as existing memories):
      - prev_act:   action that produced the current obs
      - obs:        tuple of arrays
      - rew:        scalar
      - terminated: bool
      - truncated:  bool
      - info:       dict (unused at training time)

    Additional bookkeeping:
      - ep_id[t]: int — increments at every reset
      - ep_start_step[ep_id]: int — index into the flat buffer where episode begins
      - ep_len[ep_id]: int — length of episode (in stored steps)

    sample() draws random episodes weighted by length, then a random valid start
    such that [start, start + burn_in + train_window) lies fully inside one episode.
    """

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
        super().__init__(memory_size=memory_size,
                         batch_size=batch_size,
                         dataset_path=dataset_path,
                         nb_steps=nb_steps,
                         sample_preprocessor=sample_preprocessor,
                         crc_debug=crc_debug,
                         device=device)
        self.burn_in = burn_in
        self.train_window = train_window
        self.seq_len = burn_in + train_window
        # Episode bookkeeping is rebuilt lazily from self.data when sampling.
        self._episodes_dirty = True
        self._ep_starts: list[int] = []
        self._ep_lens: list[int] = []

    # ----- write path -------------------------------------------------------
    def append_buffer(self, buffer):
        """Append a Buffer of transitions to self.data and mark episode index dirty."""
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    # ----- sample path ------------------------------------------------------
    def _rebuild_episode_index(self):
        """Walk self.data once and recompute self._ep_starts, self._ep_lens.

        An episode boundary is any t where data[t].done == True (terminated or
        truncated). The next index begins a new episode.
        """
        raise NotImplementedError

    def get_transition(self, item: int):
        """Required by TorchMemory base class.

        For SequenceMemory we override sample_indices instead, so this can
        return a single transition like the existing memories do (used only
        if the base class falls back to per-transition sampling).
        """
        raise NotImplementedError

    def sample_indices(self):
        """Override the base class so we sample sequences instead of transitions.

        Returns a list of self.batch_size 'sequence starts' encoded as ints into
        the flat self.data, with the guarantee that each start admits a full
        seq_len window inside one episode.
        """
        raise NotImplementedError

    def sample(self):
        """Returns a dict with tensors shaped [B, T, ...]:
            obs:        tuple of [B, T, ...] tensors (one per obs component)
            act:        [B, T, act_dim]    (the action taken at step t)
            a_prev:     [B, T, act_dim]    (the action taken at step t-1)
            rew:        [B, T]
            r_prev:     [B, T]
            done:       [B, T]
            d_prev:     [B, T]
            obs2:       same as obs but shifted by +1
            train_mask: [B, T] — 1 on the last train_window positions, 0 on burn-in
        """
        raise NotImplementedError
