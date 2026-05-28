# Transformer-based RL² actor + REDQ critics.
#
# Skeleton file. Implementation lands in task #2.
#
# Architecture:
#   TimestepEmbedder:     [obs_flat || a_prev || r_prev || d_prev] -> d_model
#   CausalTransformerTrunk: 2-layer causal Transformer (d_model=64, 2 heads, FFN=128)
#   TransformerActorHead: last-token -> mu, log_std (squashed Gaussian)
#   TransformerCriticHead:last-token + last-action -> scalar Q
#   TransformerREDQActorCritic: trunk + actor head + n critic heads (REDQ)

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from tmrl.util import prod
from tmrl.actor import TorchActorModule


LOG_STD_MAX = 2
LOG_STD_MIN = -20


def _flatten_obs(obs):
    """Flatten a tuple-obs into a single [B, T, D] tensor.

    obs is a tuple of tensors each shaped [B, T, ...]. Returns [B, T, sum(prod(rest))].
    """
    raise NotImplementedError


class TimestepEmbedder(nn.Module):
    """Embeds (obs, a_prev, r_prev, d_prev) at each timestep into d_model."""

    def __init__(self, obs_dim: int, act_dim: int, d_model: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.d_model = d_model
        self.proj = nn.Linear(obs_dim + act_dim + 2, d_model)  # +2 for r_prev, d_prev

    def forward(self, obs_flat: torch.Tensor, a_prev: torch.Tensor,
                r_prev: torch.Tensor, d_prev: torch.Tensor) -> torch.Tensor:
        """obs_flat: [B,T,obs_dim]; a_prev: [B,T,act_dim]; r_prev,d_prev: [B,T,1]."""
        raise NotImplementedError


class CausalTransformerTrunk(nn.Module):
    """Decoder-only causal Transformer. Returns per-token hidden states [B,T,d_model]."""

    def __init__(self, d_model: int = 64, n_layers: int = 2, n_heads: int = 2,
                 ffn_dim: int = 128, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: [B,T,d_model]. Returns [B,T,d_model]. Uses a causal mask if attn_mask is None."""
        raise NotImplementedError


class TransformerActorHead(TorchActorModule):
    """Reads the trunk's last token -> squashed-Gaussian action."""

    def __init__(self, observation_space, action_space, trunk: CausalTransformerTrunk,
                 embedder: TimestepEmbedder):
        super().__init__(observation_space, action_space)
        self.trunk = trunk
        self.embedder = embedder
        d_model = trunk.d_model
        dim_act = action_space.shape[0]
        self.mu_layer = nn.Linear(d_model, dim_act)
        self.log_std_layer = nn.Linear(d_model, dim_act)
        self.act_limit = float(action_space.high[0])

    def forward(self, obs_seq, test: bool = False, with_logprob: bool = True):
        """obs_seq is (obs_tuple_at_each_step, a_prev_seq, r_prev_seq, d_prev_seq).

        Each element shaped [B,T,...]. Returns (pi_action[B,act_dim], logp_pi[B] or None)
        — the action at the *last* timestep of each sequence.
        """
        raise NotImplementedError

    def act(self, obs, test: bool = False):
        """Inference: obs has T=1 (single step)."""
        raise NotImplementedError


class TransformerCriticHead(nn.Module):
    """Reads the trunk's last token + the action -> scalar Q."""

    def __init__(self, d_model: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(d_model + act_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, last_hidden: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """last_hidden: [B,d_model]; act: [B,act_dim]. Returns [B]."""
        raise NotImplementedError


class TransformerREDQActorCritic(nn.Module):
    """Shared trunk + actor head + n REDQ critic heads.

    For the recurrent agent's train step we expose:
      - embed(obs_seq): runs the embedder -> [B,T,d_model]
      - trunk(x):       runs the trunk    -> [B,T,d_model]
      - actor_head(last_hidden, test, with_logprob): -> (pi[B,act_dim], logp[B] or None)
      - q_head_i(last_hidden, act): -> [B]
    The actor's .forward / .act paths use the same components internally.
    """

    def __init__(self, observation_space, action_space, n: int = 10,
                 d_model: int = 64, n_layers: int = 2, n_heads: int = 2,
                 ffn_dim: int = 128, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        try:
            obs_dim = sum(prod(s for s in space.shape) for space in observation_space)
            self.tuple_obs = True
        except TypeError:
            obs_dim = prod(observation_space.shape)
            self.tuple_obs = False
        act_dim = action_space.shape[0]

        self.embedder = TimestepEmbedder(obs_dim=obs_dim, act_dim=act_dim, d_model=d_model)
        self.trunk = CausalTransformerTrunk(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            ffn_dim=ffn_dim, max_len=max_len, dropout=dropout,
        )
        self.actor = TransformerActorHead(observation_space, action_space,
                                          trunk=self.trunk, embedder=self.embedder)
        self.n = n
        self.qs = nn.ModuleList([
            TransformerCriticHead(d_model=d_model, act_dim=act_dim) for _ in range(n)
        ])

    def act(self, obs, test: bool = False):
        raise NotImplementedError
