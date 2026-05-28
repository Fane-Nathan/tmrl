# Transformer-based RL² actor + REDQ critics.
#
# Architecture:
#   TimestepEmbedder:       [obs_flat || a_prev || r_prev || d_prev] -> d_model
#   CausalTransformerTrunk: 2-layer causal Transformer (d_model=64, 2 heads, FFN=128)
#   TransformerActorHead:   last-token (optionally detached) -> mu, log_std
#   TransformerCriticHead:  last-token + action -> scalar Q
#   TransformerREDQActorCritic: trunk + actor head + n REDQ critic heads (shared trunk)
#
# Design notes:
# - The actor receives the trunk output with stop-grad by default. With UTD=20
#   the trunk would otherwise be 20x dominated by the critic loss. Detach keeps
#   the trunk a pure critic-feature extractor for the actor.
# - act() uses a deque history of recent (obs, a_prev, r_prev, d_prev) and runs
#   a full forward pass each inference. No KV cache for MVP — at d_model=64 the
#   recompute cost is negligible.
# - The model is shape-agnostic at the boundary: TimestepEmbedder receives the
#   flattened obs tensor, so swapping lidar -> image only requires a different
#   obs-flatten step in the env wrapper / training agent.

from collections import deque
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


def _compute_obs_dim(observation_space) -> Tuple[int, bool]:
    try:
        obs_dim = int(sum(prod(s for s in space.shape) for space in observation_space))
        tuple_obs = True
    except TypeError:
        obs_dim = int(prod(observation_space.shape))
        tuple_obs = False
    return obs_dim, tuple_obs


def flatten_obs_seq(obs_seq, tuple_obs: bool) -> torch.Tensor:
    """Flatten a sequence-shaped obs to [B, T, obs_dim].

    obs_seq is either:
      - a tuple of tensors each shaped [B, T, *obs_component_shape], or
      - a single tensor shaped [B, T, *obs_shape]
    """
    if tuple_obs:
        parts = []
        for o in obs_seq:
            b, t = o.shape[0], o.shape[1]
            parts.append(o.reshape(b, t, -1))
        return torch.cat(parts, dim=-1)
    b, t = obs_seq.shape[0], obs_seq.shape[1]
    return obs_seq.reshape(b, t, -1)


class TimestepEmbedder(nn.Module):
    """Embeds (obs, a_prev, r_prev, d_prev) at each timestep into d_model."""

    def __init__(self, obs_dim: int, act_dim: int, d_model: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.d_model = d_model
        self.proj = nn.Linear(obs_dim + act_dim + 2, d_model)

    def forward(self, obs_flat: torch.Tensor, a_prev: torch.Tensor,
                r_prev: torch.Tensor, d_prev: torch.Tensor) -> torch.Tensor:
        # obs_flat: [B, T, obs_dim]
        # a_prev:   [B, T, act_dim]
        # r_prev:   [B, T] or [B, T, 1]
        # d_prev:   [B, T] or [B, T, 1]
        if r_prev.dim() == 2:
            r_prev = r_prev.unsqueeze(-1)
        if d_prev.dim() == 2:
            d_prev = d_prev.unsqueeze(-1)
        x = torch.cat([obs_flat, a_prev, r_prev, d_prev], dim=-1)
        return self.proj(x)


class CausalTransformerTrunk(nn.Module):
    """Decoder-only causal Transformer. Returns per-token hidden states [B, T, d_model]."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        b, t, _ = x.shape
        if t > self.max_len:
            raise ValueError(f"Sequence length {t} exceeds max_len {self.max_len}")
        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        x = x + self.pos_emb(pos)
        causal_mask = torch.triu(
            torch.full((t, t), float("-inf"), device=x.device), diagonal=1,
        )
        return self.encoder(x, mask=causal_mask, is_causal=True)


def _squashed_gaussian_sample(mu: torch.Tensor, log_std: torch.Tensor,
                              act_limit: float, test: bool, with_logprob: bool
                              ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Sample from a tanh-squashed Gaussian.

    mu, log_std: [..., act_dim] — leading shape can be [B] or [B, T].
    Returns (action, logp) where logp has the leading shape (no act_dim).
    """
    log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
    std = torch.exp(log_std)
    dist = Normal(mu, std)
    if test:
        u = mu
    else:
        u = dist.rsample()
    if with_logprob:
        logp = dist.log_prob(u).sum(dim=-1)
        # Spinup's numerically stable tanh correction; axis=-1 to support [B,T,act_dim].
        logp = logp - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(dim=-1)
    else:
        logp = None
    a = torch.tanh(u) * act_limit
    return a, logp


class TransformerActorHead(TorchActorModule):
    """Reads the trunk's last token -> squashed-Gaussian action.

    Stores a deque of recent (obs_flat, a_prev, r_prev, d_prev) for inference,
    so act() can run a single forward pass over the history window.
    """

    def __init__(self, observation_space, action_space, trunk: CausalTransformerTrunk,
                 embedder: TimestepEmbedder, tuple_obs: bool, obs_dim: int):
        super().__init__(observation_space, action_space)
        self.trunk = trunk
        self.embedder = embedder
        self.tuple_obs = tuple_obs
        self.obs_dim = obs_dim
        d_model = trunk.d_model
        dim_act = action_space.shape[0]
        self.act_dim = dim_act
        self.mu_layer = nn.Linear(d_model, dim_act)
        self.log_std_layer = nn.Linear(d_model, dim_act)
        self.act_limit = float(action_space.high[0])
        self._history: deque = deque(maxlen=trunk.max_len)

    def reset_history(self):
        self._history.clear()

    def forward(self, obs_seq, a_prev_seq, r_prev_seq, d_prev_seq,
                test: bool = False, with_logprob: bool = True,
                detach_trunk: bool = False, return_full: bool = False):
        """Sequence forward used at training time.

        obs_seq:    tuple of [B, T, ...] tensors (or single [B, T, ...] tensor)
        a_prev_seq: [B, T, act_dim]
        r_prev_seq: [B, T] or [B, T, 1]
        d_prev_seq: [B, T] or [B, T, 1]

        If return_full=False (default): returns (pi[B, act_dim], logp[B] or None)
            using the last token of each sequence.
        If return_full=True: returns (pi[B, T, act_dim], logp[B, T] or None, hidden[B, T, d_model]).
        """
        obs_flat = flatten_obs_seq(obs_seq, self.tuple_obs)
        x = self.embedder(obs_flat, a_prev_seq, r_prev_seq, d_prev_seq)
        hidden = self.trunk(x)
        head_in = hidden.detach() if detach_trunk else hidden
        if return_full:
            mu = self.mu_layer(head_in)
            log_std = self.log_std_layer(head_in)
            a, logp = _squashed_gaussian_sample(mu, log_std, self.act_limit, test, with_logprob)
            return a, logp, hidden
        mu = self.mu_layer(head_in[:, -1])
        log_std = self.log_std_layer(head_in[:, -1])
        a, logp = _squashed_gaussian_sample(mu, log_std, self.act_limit, test, with_logprob)
        return a, logp

    def _history_to_seq(self, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stack the history deque into [1, T, ...] tensors. Pads on the left with zeros if empty."""
        if len(self._history) == 0:
            zeros_obs = torch.zeros((1, 1, self.obs_dim), device=device)
            zeros_a = torch.zeros((1, 1, self.act_dim), device=device)
            zeros_r = torch.zeros((1, 1, 1), device=device)
            ones_d = torch.ones((1, 1, 1), device=device)  # treat empty history as "just reset"
            return zeros_obs, zeros_a, zeros_r, ones_d
        obs_list, a_list, r_list, d_list = zip(*self._history)
        obs_t = torch.stack(obs_list, dim=0).unsqueeze(0)
        a_t = torch.stack(a_list, dim=0).unsqueeze(0)
        r_t = torch.stack(r_list, dim=0).unsqueeze(0)
        d_t = torch.stack(d_list, dim=0).unsqueeze(0)
        return obs_t, a_t, r_t, d_t

    @torch.no_grad()
    def act(self, obs, test: bool = False):
        """Inference. obs is the unaugmented env obs (tuple or array).

        The env wrapper is responsible for appending (a_prev, r_prev, d_prev) into the
        actor's history via push_transition() *before* calling act(). On the very first
        call after reset (no prior step), the history is empty and we treat d_prev=1.
        """
        device = next(self.parameters()).device
        if self.tuple_obs:
            parts = []
            for o in obs:
                t = torch.as_tensor(o, device=device, dtype=torch.float32).reshape(-1)
                parts.append(t)
            obs_flat = torch.cat(parts, dim=-1)
        else:
            obs_flat = torch.as_tensor(obs, device=device, dtype=torch.float32).reshape(-1)
        # Append the *current* obs with the most recent (a, r, d) already in history;
        # but our convention is the caller pushes (obs, a_prev, r_prev, d_prev) BEFORE
        # act(). So we expect the most recent entry to be this obs. To keep act() simple
        # and stateful, we'll push here if the caller forgot — see push_transition for the
        # canonical path.
        # We assume the caller has already pushed. Run forward over the deque:
        obs_t, a_t, r_t, d_t = self._history_to_seq(device)
        a, _ = self.forward(obs_t, a_t, r_t, d_t,
                            test=test, with_logprob=False, detach_trunk=False)
        return a.squeeze(0).cpu().numpy()

    def push_transition(self, obs, a_prev, r_prev: float, d_prev: float):
        """Append a new step to the actor's history deque. Called by the env wrapper
        immediately before calling act() so the deque ends with the current obs and
        the action/reward/done from the previous step."""
        device = next(self.parameters()).device
        if self.tuple_obs:
            parts = []
            for o in obs:
                t = torch.as_tensor(o, device=device, dtype=torch.float32).reshape(-1)
                parts.append(t)
            obs_flat = torch.cat(parts, dim=-1)
        else:
            obs_flat = torch.as_tensor(obs, device=device, dtype=torch.float32).reshape(-1)
        a_t = torch.as_tensor(a_prev, device=device, dtype=torch.float32).reshape(-1)
        r_t = torch.as_tensor([float(r_prev)], device=device, dtype=torch.float32)
        d_t = torch.as_tensor([float(d_prev)], device=device, dtype=torch.float32)
        self._history.append((obs_flat, a_t, r_t, d_t))


class TransformerCriticHead(nn.Module):
    """Reads a trunk hidden state + an action -> scalar Q."""

    def __init__(self, d_model: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(d_model + act_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, hidden: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        # hidden: [..., d_model]; act: [..., act_dim]. Returns [...].
        x = torch.cat([hidden, act], dim=-1)
        return self.q(x).squeeze(-1)


class TransformerREDQActorCritic(nn.Module):
    """Shared trunk + actor head + n REDQ critic heads."""

    def __init__(self, observation_space, action_space, n: int = 10,
                 d_model: int = 64, n_layers: int = 2, n_heads: int = 2,
                 ffn_dim: int = 128, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        obs_dim, tuple_obs = _compute_obs_dim(observation_space)
        act_dim = int(action_space.shape[0])

        self.embedder = TimestepEmbedder(obs_dim=obs_dim, act_dim=act_dim, d_model=d_model)
        self.trunk = CausalTransformerTrunk(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            ffn_dim=ffn_dim, max_len=max_len, dropout=dropout,
        )
        self.actor = TransformerActorHead(
            observation_space, action_space,
            trunk=self.trunk, embedder=self.embedder,
            tuple_obs=tuple_obs, obs_dim=obs_dim,
        )
        self.n = n
        self.qs = nn.ModuleList([
            TransformerCriticHead(d_model=d_model, act_dim=act_dim) for _ in range(n)
        ])

    def encode(self, obs_seq, a_prev_seq, r_prev_seq, d_prev_seq) -> torch.Tensor:
        """Run embedder + trunk over a sequence. Returns hidden [B, T, d_model]."""
        if self.actor.tuple_obs:
            obs_flat = flatten_obs_seq(obs_seq, True)
        else:
            obs_flat = flatten_obs_seq(obs_seq, False)
        x = self.embedder(obs_flat, a_prev_seq, r_prev_seq, d_prev_seq)
        return self.trunk(x)

    def act(self, obs, test: bool = False):
        return self.actor.act(obs, test=test)


# Smoke test ============================================================================================================

def _smoke_test():
    """Construct the model on a fake lidar-like obs space and run a forward+backward.

    Catches shape bugs and confirms gradients flow.
    """
    import gymnasium.spaces as spaces

    obs_space = spaces.Tuple((
        spaces.Box(low=0.0, high=1.0, shape=(1,)),       # speed
        spaces.Box(low=0.0, high=1.0, shape=(1,)),       # progress
        spaces.Box(low=0.0, high=1.0, shape=(1, 19)),    # lidar
    ))
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))

    model = TransformerREDQActorCritic(obs_space, act_space, n=4, d_model=32, n_layers=2,
                                       n_heads=2, ffn_dim=64, max_len=128)
    B, T = 5, 84
    obs_seq = (
        torch.randn(B, T, 1),
        torch.randn(B, T, 1),
        torch.randn(B, T, 1, 19),
    )
    a_prev = torch.randn(B, T, 3).tanh()
    r_prev = torch.randn(B, T)
    d_prev = torch.zeros(B, T)

    pi_seq, logp_seq, hidden = model.actor(obs_seq, a_prev, r_prev, d_prev,
                                           return_full=True, detach_trunk=True)
    assert pi_seq.shape == (B, T, 3), f"pi_seq: {pi_seq.shape}"
    assert logp_seq.shape == (B, T), f"logp_seq: {logp_seq.shape}"
    assert hidden.shape == (B, T, 32), f"hidden: {hidden.shape}"

    q0 = model.qs[0](hidden, a_prev)
    assert q0.shape == (B, T), f"q0: {q0.shape}"

    loss = (q0.mean() + logp_seq.mean()) * 0.5
    loss.backward()
    print(f"smoke OK  loss={loss.item():.4f}  pi_seq.shape={tuple(pi_seq.shape)}  q.shape={tuple(q0.shape)}")

    # Inference path:
    model_eval = model.eval()
    actor = model_eval.actor
    actor.reset_history()
    fake_obs = (np.zeros((1,), dtype=np.float32),
                np.zeros((1,), dtype=np.float32),
                np.zeros((1, 19), dtype=np.float32))
    actor.push_transition(fake_obs, np.zeros(3, dtype=np.float32), 0.0, 1.0)
    a = actor.act(fake_obs, test=True)
    assert a.shape == (3,), f"a: {a.shape}"
    print(f"inference OK  a={a}")


if __name__ == "__main__":
    _smoke_test()
