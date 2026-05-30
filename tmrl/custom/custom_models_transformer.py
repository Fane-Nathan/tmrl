# Transformer-based RL2 actor + REDQ critics.
#
# Architecture:
#   VisualEncoder (opt):    [B*T, C, H, W] -> [B*T, cnn_out_dim]  (per-timestep CNN)
#   TimestepEmbedder:       [scalar_flat || visual_emb || a_prev || r_prev || d_prev] -> d_model
#   CausalTransformerTrunk: 2-layer causal Transformer (d_model=64, 2 heads, FFN=128)
#   TransformerActorHead:   last-token (optionally detached) -> mu, log_std
#   TransformerCriticHead:  last-token + action -> scalar Q
#   TransformerREDQActorCritic: trunk + actor head + n REDQ critic heads (shared trunk)
#
# Design notes:
# - The actor receives the trunk output with stop-grad by default. With UTD=20
#   the trunk would otherwise be 20x dominated by the critic loss. Detach keeps
#   the trunk a pure critic-feature extractor for the actor.
# - act() uses a deque history of recent (obs_flat, img, a_prev, r_prev, d_prev)
#   and runs a full forward pass each inference. No KV cache for MVP - at
#   d_model=64 the recompute cost is negligible.
# - When an image component is detected in the observation space (any component
#   with ndim >= 3), a VisualEncoder CNN is created and applied per-timestep
#   before the TimestepEmbedder projection. For scalar-only obs (e.g. lidar),
#   no CNN is created and behavior is identical to the original implementation.

from collections import deque
from math import floor
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


# ---------------------------------------------------------------------------
# Observation space analysis
# ---------------------------------------------------------------------------

def _rl2_meta_start(spaces_list, action_space):
    if action_space is None or len(spaces_list) < 4:
        return None
    if tuple(spaces_list[-3].shape) != tuple(action_space.shape):
        return None
    if tuple(spaces_list[-2].shape) != (1,) or tuple(spaces_list[-1].shape) != (1,):
        return None
    return len(spaces_list) - 3


def _analyze_obs_space(observation_space, action_space=None):
    """Analyze observation space to identify image vs scalar components.

    Returns:
        scalar_dim (int): total flattened dim of non-RL2 scalar components
        tuple_obs (bool): whether obs is a Tuple space
        img_idx (int or None): index of the image component (ndim >= 3), or None
        img_shape (tuple or None): shape of the image component, or None
        meta_start (int or None): index where (a_prev, r_prev, d_prev) starts
    """
    try:
        spaces_list = list(observation_space)
        tuple_obs = True
    except TypeError:
        # Single space, not a Tuple
        shape = observation_space.shape
        if len(shape) >= 3:
            # Single image obs - unusual but handle it
            return 0, False, 0, shape, None
        return int(prod(shape)), False, None, None, None

    img_idx = None
    img_shape = None
    scalar_dim = 0
    meta_start = _rl2_meta_start(spaces_list, action_space)

    for i, space in enumerate(spaces_list):
        if meta_start is not None and i >= meta_start:
            continue
        if len(space.shape) >= 2 and prod(space.shape) > 64:
            # Heuristic: 2D+ component with > 64 elements is an image.
            # This catches (1, 64, 64) grayscale and (H, W, 3) color.
            # Lidar (1, 19) has prod=19 so it won't match.
            if img_idx is not None:
                raise ValueError(
                    f"Multiple image components detected at indices {img_idx} and {i}. "
                    "Only one image component is supported."
                )
            img_idx = i
            img_shape = space.shape
        else:
            scalar_dim += int(prod(s for s in space.shape))

    return scalar_dim, tuple_obs, img_idx, img_shape, meta_start


def _compute_obs_dim(observation_space):
    """Legacy helper: returns (obs_dim, tuple_obs) by flattening everything.

    Kept for backward compatibility with code that doesn't need image separation.
    """
    try:
        obs_dim = int(sum(prod(s for s in space.shape) for space in observation_space))
        tuple_obs = True
    except TypeError:
        obs_dim = int(prod(observation_space.shape))
        tuple_obs = False
    return obs_dim, tuple_obs


def flatten_obs_seq(obs_seq, tuple_obs: bool, img_idx: int = None, meta_start: int = None):
    """Flatten a sequence-shaped obs to [B, T, scalar_dim] (+ optional image tensor).

    obs_seq is either:
      - a tuple of tensors each shaped [B, T, *obs_component_shape], or
      - a single tensor shaped [B, T, *obs_shape]

    If img_idx is not None, the component at that index is returned separately
    as a raw tensor (not flattened), and only the scalar components are concatenated.

    Returns:
        If img_idx is None: (scalar_flat, None)
            scalar_flat: [B, T, scalar_dim]
        If img_idx is set: (scalar_flat, img_tensor)
            scalar_flat: [B, T, scalar_dim]  (without the image component)
            img_tensor:  [B, T, *img_shape]  (raw image data)
    """
    if not tuple_obs:
        if img_idx is not None and img_idx == 0:
            # Single-component image obs (unusual)
            return torch.zeros(obs_seq.shape[0], obs_seq.shape[1], 0,
                               device=obs_seq.device), obs_seq
        b, t = obs_seq.shape[0], obs_seq.shape[1]
        return obs_seq.reshape(b, t, -1), None

    parts = []
    img_tensor = None
    for i, o in enumerate(obs_seq):
        if meta_start is not None and i >= meta_start:
            continue
        if i == img_idx:
            img_tensor = o
            continue
        b, t = o.shape[0], o.shape[1]
        parts.append(o.reshape(b, t, -1))

    if parts:
        scalar_flat = torch.cat(parts, dim=-1)
    else:
        # All components are images (shouldn't happen, but be safe)
        b, t = obs_seq[0].shape[0], obs_seq[0].shape[1]
        scalar_flat = torch.zeros(b, t, 0, device=obs_seq[0].device)

    return scalar_flat, img_tensor


# ---------------------------------------------------------------------------
# Visual encoder (per-timestep CNN)
# ---------------------------------------------------------------------------

def _conv2d_out_dims(conv_layer, h_in, w_in):
    h_out = floor((h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] *
                   (conv_layer.kernel_size[0] - 1) - 1) / conv_layer.stride[0] + 1)
    w_out = floor((w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] *
                   (conv_layer.kernel_size[1] - 1) - 1) / conv_layer.stride[1] + 1)
    return h_out, w_out


class VisualEncoder(nn.Module):
    """Lightweight per-timestep CNN for image observations.

    Takes [N, C, H, W] and produces [N, cnn_out_dim].
    Designed for grayscale 64x64 input but adapts to other sizes via
    AdaptiveAvgPool2d.
    """

    def __init__(self, in_channels: int = 1, cnn_out_dim: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.cnn_out_dim = cnn_out_dim

        self.conv1 = nn.Conv2d(in_channels, 32, 8, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, cnn_out_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, C, H, W] -> [N, cnn_out_dim]"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)           # [N, 64, 1, 1]
        x = x.view(x.size(0), -1)  # [N, 64]
        x = F.relu(self.fc(x))     # [N, cnn_out_dim]
        return x

    def encode_seq(self, img: torch.Tensor) -> torch.Tensor:
        """Process a sequence of images: [B, T, ...] -> [B, T, cnn_out_dim].

        Reshapes to [B*T, C, H, W], runs the CNN, then reshapes back.
        """
        b, t = img.shape[0], img.shape[1]
        h, w = img.shape[-2], img.shape[-1]
        if img.dim() == 4:
            c = 1
        elif img.dim() == 5:
            c = img.shape[2]
        else:
            mid_dims = img.shape[2:-2]
            c = 1
            for d in mid_dims:
                c *= d

        img = img.reshape(b, t, c, h, w)
        flat = img.reshape(b * t, c, h, w)
        enc = self.forward(flat)
        return enc.reshape(b, t, -1)


# ---------------------------------------------------------------------------
# Timestep embedder
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """Embeds (obs, a_prev, r_prev, d_prev) at each timestep into d_model.

    If a visual_encoder is provided, the image component is encoded separately
    via CNN and concatenated with scalar features before projection.
    """

    def __init__(self, scalar_dim: int, act_dim: int, d_model: int = 64,
                 visual_encoder: Optional[VisualEncoder] = None):
        super().__init__()
        self.scalar_dim = scalar_dim
        self.act_dim = act_dim
        self.d_model = d_model
        self.visual_encoder = visual_encoder

        cnn_dim = visual_encoder.cnn_out_dim if visual_encoder is not None else 0
        input_dim = scalar_dim + cnn_dim + act_dim + 2  # +2 for r_prev, d_prev
        self.proj = nn.Linear(input_dim, d_model)

    def forward(self, scalar_flat: torch.Tensor, a_prev: torch.Tensor,
                r_prev: torch.Tensor, d_prev: torch.Tensor,
                img: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        scalar_flat: [B, T, scalar_dim]
        a_prev:      [B, T, act_dim]
        r_prev:      [B, T] or [B, T, 1]
        d_prev:      [B, T] or [B, T, 1]
        img:         [B, T, C, H, W] or None
        """
        if r_prev.dim() == 2:
            r_prev = r_prev.unsqueeze(-1)
        if d_prev.dim() == 2:
            d_prev = d_prev.unsqueeze(-1)

        parts = [scalar_flat, a_prev, r_prev, d_prev]

        if self.visual_encoder is not None:
            assert img is not None, "TimestepEmbedder has a visual_encoder but no img was provided"
            visual_emb = self.visual_encoder.encode_seq(img)  # [B, T, cnn_out_dim]
            parts.insert(1, visual_emb)  # scalar, visual, a_prev, r_prev, d_prev

        x = torch.cat(parts, dim=-1)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Causal Transformer trunk
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Task encoder
# ---------------------------------------------------------------------------

class TaskEncoder(nn.Module):
    """Infers a task embedding z from causal driving-memory hidden states."""

    def __init__(self, d_model: int, z_dim: int = 8, num_tasks: int = 5,
                 hidden: int = 128):
        super().__init__()
        self.z_dim = int(z_dim)
        self.num_tasks = int(num_tasks)
        self.encoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.z_dim),
        )
        self.classifier = nn.Linear(self.z_dim, self.num_tasks)

    def forward(self, hidden_seq: torch.Tensor):
        z = self.encoder(hidden_seq)
        task_logits = self.classifier(z)
        return z, task_logits


def _condition_hidden(hidden: torch.Tensor, z: Optional[torch.Tensor],
                      z_dim: int) -> torch.Tensor:
    if z_dim <= 0:
        return hidden
    if z is None:
        z = hidden.new_zeros(*hidden.shape[:-1], z_dim)
    elif hidden.dim() == z.dim() + 1:
        z = z.unsqueeze(1).expand(*hidden.shape[:-1], z.shape[-1])
    return torch.cat([hidden, z], dim=-1)


# ---------------------------------------------------------------------------
# Squashed Gaussian sampling
# ---------------------------------------------------------------------------

def _squashed_gaussian_sample(mu: torch.Tensor, log_std: torch.Tensor,
                              act_limit: float, test: bool, with_logprob: bool
                              ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Sample from a tanh-squashed Gaussian.

    mu, log_std: [..., act_dim] - leading shape can be [B] or [B, T].
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


# ---------------------------------------------------------------------------
# Actor head
# ---------------------------------------------------------------------------

class TransformerActorHead(TorchActorModule):
    """Reads the trunk's last token -> squashed-Gaussian action.

    Stores a deque of recent (scalar_flat, img, a_prev, r_prev, d_prev) for
    inference, so act() can run a single forward pass over the history window.
    """

    def __init__(self, observation_space, action_space, trunk: CausalTransformerTrunk,
                 embedder: TimestepEmbedder, tuple_obs: bool, scalar_dim: int,
                 img_idx: int = None, img_shape: tuple = None, meta_start: int = None,
                 task_encoder: Optional[TaskEncoder] = None, z_dim: int = 0):
        super().__init__(observation_space, action_space)
        self.trunk = trunk
        self.embedder = embedder
        self.task_encoder = task_encoder
        self.z_dim = int(z_dim)
        self.tuple_obs = tuple_obs
        self.scalar_dim = scalar_dim
        self.img_idx = img_idx
        self.img_shape = img_shape
        self.meta_start = meta_start
        self.has_cnn = img_idx is not None
        d_model = trunk.d_model
        dim_act = action_space.shape[0]
        self.act_dim = dim_act
        head_dim = d_model + self.z_dim
        self.mu_layer = nn.Linear(head_dim, dim_act)
        self.log_std_layer = nn.Linear(head_dim, dim_act)
        self.act_limit = float(action_space.high[0])
        self._history: deque = deque(maxlen=trunk.max_len)

    # For backward compat with code that reads self.obs_dim
    @property
    def obs_dim(self):
        return self.scalar_dim

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
        scalar_flat, img = flatten_obs_seq(obs_seq, self.tuple_obs, self.img_idx, self.meta_start)
        x = self.embedder(scalar_flat, a_prev_seq, r_prev_seq, d_prev_seq, img=img)
        hidden = self.trunk(x)
        head_hidden = hidden.detach() if detach_trunk else hidden
        z_seq = None
        if self.task_encoder is not None:
            z_seq, _ = self.task_encoder(head_hidden)
        head_in = _condition_hidden(head_hidden, z_seq, self.z_dim)
        if return_full:
            mu = self.mu_layer(head_in)
            log_std = self.log_std_layer(head_in)
            a, logp = _squashed_gaussian_sample(mu, log_std, self.act_limit, test, with_logprob)
            return a, logp, hidden
        mu = self.mu_layer(head_in[:, -1])
        log_std = self.log_std_layer(head_in[:, -1])
        a, logp = _squashed_gaussian_sample(mu, log_std, self.act_limit, test, with_logprob)
        return a, logp

    def _history_to_seq(self, device):
        """Stack the history deque into [1, T, ...] tensors."""
        if len(self._history) == 0:
            zeros_scalar = torch.zeros((1, 1, self.scalar_dim), device=device)
            zeros_a = torch.zeros((1, 1, self.act_dim), device=device)
            zeros_r = torch.zeros((1, 1, 1), device=device)
            ones_d = torch.ones((1, 1, 1), device=device)
            if self.has_cnn:
                zeros_img = torch.zeros((1, 1, *self.img_shape), device=device)
                return zeros_scalar, zeros_a, zeros_r, ones_d, zeros_img
            return zeros_scalar, zeros_a, zeros_r, ones_d, None

        if self.has_cnn:
            scalar_list, img_list, a_list, r_list, d_list = zip(*self._history)
            img_t = torch.stack(img_list, dim=0).unsqueeze(0)
        else:
            scalar_list, a_list, r_list, d_list = zip(*self._history)
            img_t = None

        scalar_t = torch.stack(scalar_list, dim=0).unsqueeze(0)
        a_t = torch.stack(a_list, dim=0).unsqueeze(0)
        r_t = torch.stack(r_list, dim=0).unsqueeze(0)
        d_t = torch.stack(d_list, dim=0).unsqueeze(0)
        return scalar_t, a_t, r_t, d_t, img_t

    @torch.no_grad()
    def act(self, obs, test: bool = False):
        """Inference. obs is a tuple where the last 3 elements are
        (a_prev_arr, r_prev_arr, d_prev_arr) - packed in by the RL2 env wrapper.
        """
        assert isinstance(obs, (tuple, list)) and len(obs) >= 4, \
            "RL2 actor expects a tuple obs with at least 4 elements (last 3 = a_prev, r_prev, d_prev)"
        device = next(self.parameters()).device
        a_prev_arr = obs[-3]
        r_prev_arr = obs[-2]
        d_prev_arr = obs[-1]
        r_prev_scalar = float(np.asarray(r_prev_arr).reshape(-1)[0])
        d_prev_scalar = float(np.asarray(d_prev_arr).reshape(-1)[0])

        if d_prev_scalar >= 0.5:
            self._history.clear()

        self.push_transition(obs, a_prev_arr, r_prev_scalar, d_prev_scalar)

        scalar_t, a_t, r_t, d_t, img_t = self._history_to_seq(device)
        x = self.embedder(scalar_t, a_t, r_t, d_t, img=img_t)
        hidden = self.trunk(x)
        z_last = None
        if self.task_encoder is not None:
            z_seq, _ = self.task_encoder(hidden)
            z_last = z_seq[:, -1]
        head_in = _condition_hidden(hidden[:, -1], z_last, self.z_dim)
        mu = self.mu_layer(head_in)
        log_std = self.log_std_layer(head_in)
        a, _ = _squashed_gaussian_sample(mu, log_std, self.act_limit,
                                         test=test, with_logprob=False)
        return a.squeeze(0).cpu().numpy()

    def head_from_hidden(self, hidden: torch.Tensor, z: Optional[torch.Tensor] = None,
                         test: bool = False,
                         with_logprob: bool = True):
        """Apply mu/log_std/squashed-Gaussian sampling to a pre-computed hidden state.

        hidden: [..., d_model]. Returns (action[..., act_dim], logp[...]) or (action, None).
        Used by the agent to sample next-state actions from target-trunk hiddens
        without re-running the actor's full forward pipe.
        """
        head_in = _condition_hidden(hidden, z, self.z_dim)
        mu = self.mu_layer(head_in)
        log_std = self.log_std_layer(head_in)
        return _squashed_gaussian_sample(mu, log_std, self.act_limit, test, with_logprob)

    def push_transition(self, obs, a_prev, r_prev: float, d_prev: float):
        """Append a new step to the actor's history deque."""
        device = next(self.parameters()).device

        # Separate image from scalar components
        if self.has_cnn:
            scalar_parts = []
            img_tensor = None
            for i, o in enumerate(obs):
                if self.meta_start is not None and i >= self.meta_start:
                    continue
                t = torch.as_tensor(np.asarray(o), device=device, dtype=torch.float32)
                if i == self.img_idx:
                    # Keep image in its spatial shape (C, H, W)
                    img_tensor = t
                else:
                    scalar_parts.append(t.reshape(-1))
            scalar_flat = torch.cat(scalar_parts, dim=-1) if scalar_parts else \
                torch.zeros(0, device=device)
        else:
            if self.tuple_obs:
                parts = []
                for i, o in enumerate(obs):
                    if self.meta_start is not None and i >= self.meta_start:
                        continue
                    t = torch.as_tensor(o, device=device, dtype=torch.float32).reshape(-1)
                    parts.append(t)
                scalar_flat = torch.cat(parts, dim=-1)
            else:
                scalar_flat = torch.as_tensor(obs, device=device, dtype=torch.float32).reshape(-1)
            img_tensor = None

        a_t = torch.as_tensor(a_prev, device=device, dtype=torch.float32).reshape(-1)
        r_t = torch.as_tensor([float(r_prev)], device=device, dtype=torch.float32)
        d_t = torch.as_tensor([float(d_prev)], device=device, dtype=torch.float32)

        if self.has_cnn:
            self._history.append((scalar_flat, img_tensor, a_t, r_t, d_t))
        else:
            self._history.append((scalar_flat, a_t, r_t, d_t))


# ---------------------------------------------------------------------------
# Critic head
# ---------------------------------------------------------------------------

class TransformerCriticHead(nn.Module):
    """Reads a trunk hidden state + an action -> scalar Q."""

    def __init__(self, d_model: int, act_dim: int, hidden: int = 128,
                 z_dim: int = 0):
        super().__init__()
        self.z_dim = int(z_dim)
        self.q = nn.Sequential(
            nn.Linear(d_model + act_dim + self.z_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, hidden: torch.Tensor, act: torch.Tensor,
                z: Optional[torch.Tensor] = None) -> torch.Tensor:
        # hidden: [..., d_model]; act: [..., act_dim]. Returns [...].
        hidden_z = _condition_hidden(hidden, z, self.z_dim)
        x = torch.cat([hidden_z, act], dim=-1)
        return self.q(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Full actor-critic
# ---------------------------------------------------------------------------

class TransformerREDQActorCritic(nn.Module):
    """Shared trunk + actor head + n REDQ critic heads."""

    def __init__(self, observation_space, action_space, n: int = 10,
                 d_model: int = 64, n_layers: int = 2, n_heads: int = 2,
                 ffn_dim: int = 128, max_len: int = 128, dropout: float = 0.1,
                 task_conditioning: bool = False, task_z_dim: int = 8,
                 num_tasks: int = 5):
        super().__init__()
        scalar_dim, tuple_obs, img_idx, img_shape, meta_start = _analyze_obs_space(observation_space, action_space)
        act_dim = int(action_space.shape[0])
        self.task_conditioning = bool(task_conditioning and task_z_dim > 0 and num_tasks > 1)
        self.z_dim = int(task_z_dim) if self.task_conditioning else 0
        self.num_tasks = int(num_tasks) if self.task_conditioning else 0

        # Create visual encoder if image component detected
        if img_idx is not None:
            # Determine number of input channels
            # img_shape could be (C, H, W) for grayscale or (H, W, C) for color
            # Our preprocessor ensures (C, H, W) format
            in_channels = img_shape[0] if len(img_shape) == 3 else 1
            self.visual_encoder = VisualEncoder(in_channels=in_channels, cnn_out_dim=128)
        else:
            self.visual_encoder = None

        self.embedder = TimestepEmbedder(
            scalar_dim=scalar_dim, act_dim=act_dim, d_model=d_model,
            visual_encoder=self.visual_encoder,
        )
        self.trunk = CausalTransformerTrunk(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            ffn_dim=ffn_dim, max_len=max_len, dropout=dropout,
        )
        self.task_encoder = (
            TaskEncoder(d_model=d_model, z_dim=self.z_dim, num_tasks=self.num_tasks)
            if self.task_conditioning else None
        )
        self.actor = TransformerActorHead(
            observation_space, action_space,
            trunk=self.trunk, embedder=self.embedder,
            tuple_obs=tuple_obs, scalar_dim=scalar_dim,
            img_idx=img_idx, img_shape=img_shape,
            meta_start=meta_start,
            task_encoder=self.task_encoder,
            z_dim=self.z_dim,
        )
        self.n = n
        self.qs = nn.ModuleList([
            TransformerCriticHead(d_model=d_model, act_dim=act_dim, z_dim=self.z_dim)
            for _ in range(n)
        ])

        # Store obs space info for encode()
        self._tuple_obs = tuple_obs
        self._img_idx = img_idx
        self._meta_start = meta_start

    def encode(self, obs_seq, a_prev_seq, r_prev_seq, d_prev_seq) -> torch.Tensor:
        """Run embedder + trunk over a sequence. Returns hidden [B, T, d_model]."""
        scalar_flat, img = flatten_obs_seq(obs_seq, self._tuple_obs, self._img_idx, self._meta_start)
        x = self.embedder(scalar_flat, a_prev_seq, r_prev_seq, d_prev_seq, img=img)
        return self.trunk(x)

    def infer_task(self, hidden_seq: torch.Tensor):
        if self.task_encoder is None:
            return None, None
        return self.task_encoder(hidden_seq)

    def act(self, obs, test: bool = False):
        return self.actor.act(obs, test=test)


# ---------------------------------------------------------------------------
# Standalone actor for rollout worker
# ---------------------------------------------------------------------------

class TransformerActorOnly(TorchActorModule):
    """Standalone actor policy for the rollout worker.

    The rollout worker constructs `POLICY(observation_space, action_space)` and
    calls `.act(obs)` on it. It owns its own embedder + causal trunk + actor
    head; the submodule attribute names (`embedder`, `trunk`, `mu_layer`,
    `log_std_layer`) match the attribute names of `TransformerActorHead`, so its
    state_dict keys are identical to what the trainer saves when it exports
    `model.actor` (which is the actor head with `trunk`/`embedder` as referenced
    submodules). No key remapping needed at load time.

    Hyperparameters are read from `tmrl.config.config_constants`.
    """

    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        import tmrl.config.config_constants as cfg
        scalar_dim, tuple_obs, img_idx, img_shape, meta_start = _analyze_obs_space(observation_space, action_space)
        act_dim = int(action_space.shape[0])
        task_conditioning = bool(
            cfg.RL2_TASK_CONDITIONING and
            cfg.RL2_TASK_Z_DIM > 0 and
            cfg.RL2_TASK_NUM_TASKS > 1
        )
        self.z_dim = int(cfg.RL2_TASK_Z_DIM) if task_conditioning else 0
        self.num_tasks = int(cfg.RL2_TASK_NUM_TASKS) if task_conditioning else 0

        # Create visual encoder if image component detected
        if img_idx is not None:
            in_channels = img_shape[0] if len(img_shape) == 3 else 1
            visual_encoder = VisualEncoder(in_channels=in_channels, cnn_out_dim=128)
        else:
            visual_encoder = None

        self.embedder = TimestepEmbedder(
            scalar_dim=scalar_dim, act_dim=act_dim,
            d_model=cfg.RL2_TRANSFORMER_D_MODEL,
            visual_encoder=visual_encoder,
        )
        self.trunk = CausalTransformerTrunk(
            d_model=cfg.RL2_TRANSFORMER_D_MODEL,
            n_layers=cfg.RL2_TRANSFORMER_LAYERS,
            n_heads=cfg.RL2_TRANSFORMER_HEADS,
            ffn_dim=cfg.RL2_TRANSFORMER_FFN,
            max_len=cfg.RL2_TRANSFORMER_MAX_LEN,
            dropout=0.1,
        )
        self.task_encoder = (
            TaskEncoder(
                d_model=cfg.RL2_TRANSFORMER_D_MODEL,
                z_dim=self.z_dim,
                num_tasks=self.num_tasks,
            )
            if task_conditioning else None
        )
        head_dim = cfg.RL2_TRANSFORMER_D_MODEL + self.z_dim
        self.mu_layer = nn.Linear(head_dim, act_dim)
        self.log_std_layer = nn.Linear(head_dim, act_dim)
        self.act_limit = float(action_space.high[0])
        self.tuple_obs = tuple_obs
        self.scalar_dim = scalar_dim
        self.img_idx = img_idx
        self.img_shape = img_shape
        self.meta_start = meta_start
        self.has_cnn = img_idx is not None
        self.act_dim = act_dim
        self._history: deque = deque(maxlen=self.trunk.max_len)

    # For backward compat
    @property
    def obs_dim(self):
        return self.scalar_dim

    def reset_history(self):
        self._history.clear()

    @torch.no_grad()
    def act(self, obs, test: bool = False):
        assert isinstance(obs, (tuple, list)) and len(obs) >= 4, \
            "RL2 actor expects a tuple obs with last 3 elements = (a_prev, r_prev, d_prev)"
        device = next(self.parameters()).device
        a_prev_arr = obs[-3]
        r_prev_arr = obs[-2]
        d_prev_arr = obs[-1]
        r_prev_scalar = float(np.asarray(r_prev_arr).reshape(-1)[0])
        d_prev_scalar = float(np.asarray(d_prev_arr).reshape(-1)[0])

        if d_prev_scalar >= 0.5:
            self._history.clear()

        # Separate image from scalar components
        if self.has_cnn:
            scalar_parts = []
            img_tensor = None
            for i, o in enumerate(obs):
                if self.meta_start is not None and i >= self.meta_start:
                    continue
                t = torch.as_tensor(np.asarray(o), device=device, dtype=torch.float32)
                if i == self.img_idx:
                    img_tensor = t
                else:
                    scalar_parts.append(t.reshape(-1))
            scalar_flat = torch.cat(scalar_parts, dim=-1) if scalar_parts else \
                torch.zeros(0, device=device)
        else:
            if self.tuple_obs:
                parts = []
                for i, o in enumerate(obs):
                    if self.meta_start is not None and i >= self.meta_start:
                        continue
                    t = torch.as_tensor(o, device=device, dtype=torch.float32).reshape(-1)
                    parts.append(t)
                scalar_flat = torch.cat(parts, dim=-1)
            else:
                scalar_flat = torch.as_tensor(obs, device=device, dtype=torch.float32).reshape(-1)
            img_tensor = None

        a_t_now = torch.as_tensor(a_prev_arr, device=device, dtype=torch.float32).reshape(-1)
        r_t_now = torch.as_tensor([r_prev_scalar], device=device, dtype=torch.float32)
        d_t_now = torch.as_tensor([d_prev_scalar], device=device, dtype=torch.float32)

        if self.has_cnn:
            self._history.append((scalar_flat, img_tensor, a_t_now, r_t_now, d_t_now))
        else:
            self._history.append((scalar_flat, a_t_now, r_t_now, d_t_now))

        if len(self._history) == 0:
            zeros_scalar = torch.zeros((1, 1, self.scalar_dim), device=device)
            zeros_a = torch.zeros((1, 1, self.act_dim), device=device)
            zeros_r = torch.zeros((1, 1, 1), device=device)
            ones_d = torch.ones((1, 1, 1), device=device)
            img_t = None
            if self.has_cnn:
                img_t = torch.zeros((1, 1, *self.img_shape), device=device)
            scalar_t, a_t, r_t, d_t = zeros_scalar, zeros_a, zeros_r, ones_d
        else:
            if self.has_cnn:
                scalar_list, img_list, a_list, r_list, d_list = zip(*self._history)
                img_t = torch.stack(img_list, dim=0).unsqueeze(0)
            else:
                scalar_list, a_list, r_list, d_list = zip(*self._history)
                img_t = None
            scalar_t = torch.stack(scalar_list, dim=0).unsqueeze(0)
            a_t = torch.stack(a_list, dim=0).unsqueeze(0)
            r_t = torch.stack(r_list, dim=0).unsqueeze(0)
            d_t = torch.stack(d_list, dim=0).unsqueeze(0)

        x = self.embedder(scalar_t, a_t, r_t, d_t, img=img_t)
        hidden = self.trunk(x)
        z_last = None
        if self.task_encoder is not None:
            z_seq, _ = self.task_encoder(hidden)
            z_last = z_seq[:, -1]
        head_in = _condition_hidden(hidden[:, -1], z_last, self.z_dim)
        mu = self.mu_layer(head_in)
        log_std = self.log_std_layer(head_in)
        a, _ = _squashed_gaussian_sample(mu, log_std, self.act_limit,
                                         test=test, with_logprob=False)
        return a.squeeze(0).cpu().numpy()


# ===========================================================================
# Smoke test
# ===========================================================================

def _smoke_test():
    """Construct the model on a fake lidar-like obs space and run a forward+backward.

    Catches shape bugs and confirms gradients flow.
    """
    import gymnasium.spaces as spaces

    # --- Test 1: Scalar (lidar) obs space ---
    print("=== Test 1: Scalar (lidar) obs ===")
    obs_space = spaces.Tuple((
        spaces.Box(low=0.0, high=1.0, shape=(1,)),       # speed
        spaces.Box(low=0.0, high=1.0, shape=(1,)),       # progress
        spaces.Box(low=0.0, high=1.0, shape=(1, 19)),    # lidar
    ))
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))

    model = TransformerREDQActorCritic(
        obs_space, act_space, n=4, d_model=32, n_layers=2,
        n_heads=2, ffn_dim=64, max_len=128,
        task_conditioning=True, task_z_dim=8, num_tasks=5,
    )
    assert model.visual_encoder is None, "No CNN for scalar obs"
    assert model.task_encoder is not None, "Task encoder should be created"
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
    z_seq, task_logits = model.infer_task(hidden)
    assert z_seq.shape == (B, T, 8), f"z_seq: {z_seq.shape}"
    assert task_logits.shape == (B, T, 5), f"task_logits: {task_logits.shape}"

    q0 = model.qs[0](hidden, a_prev, z_seq)
    assert q0.shape == (B, T), f"q0: {q0.shape}"

    task_targets = torch.randint(0, 5, (B, T), device=task_logits.device)
    task_loss = F.cross_entropy(
        task_logits.reshape(-1, task_logits.shape[-1]),
        task_targets.reshape(-1),
    )
    loss = (q0.mean() + logp_seq.mean()) * 0.5 + 0.05 * task_loss
    loss.backward()
    task_grad_ok = all(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.task_encoder.parameters()
                       if p.requires_grad)
    assert task_grad_ok, "Gradients must flow through the task encoder"
    print(f"  smoke OK  loss={loss.item():.4f}  pi_seq.shape={tuple(pi_seq.shape)}  q.shape={tuple(q0.shape)}")

    # Inference path: scalar obs
    obs_space_aug = spaces.Tuple((
        spaces.Box(low=0.0, high=1.0, shape=(1,)),
        spaces.Box(low=0.0, high=1.0, shape=(1,)),
        spaces.Box(low=0.0, high=1.0, shape=(1, 19)),
        spaces.Box(low=-1.0, high=1.0, shape=(3,)),  # a_prev
        spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),  # r_prev
        spaces.Box(low=0.0, high=1.0, shape=(1,)),  # d_prev
    ))
    model_inf = TransformerREDQActorCritic(
        obs_space_aug, act_space, n=4, d_model=32,
        n_layers=2, n_heads=2, ffn_dim=64, max_len=128,
        task_conditioning=True, task_z_dim=8, num_tasks=5,
    )
    actor = model_inf.actor
    actor.reset_history()
    fake_obs = (np.zeros((1,), dtype=np.float32),
                np.zeros((1,), dtype=np.float32),
                np.zeros((1, 19), dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                np.zeros(1, dtype=np.float32),
                np.ones(1, dtype=np.float32))
    a = actor.act(fake_obs, test=True)
    assert a.shape == (3,), f"a: {a.shape}"
    print(f"  inference OK  a={a}")

    # --- Test 2: Image obs space ---
    print("\n=== Test 2: Image obs (grayscale 64x64) ===")
    obs_space_img = spaces.Tuple((
        spaces.Box(low=0.0, high=1.0, shape=(1,)),              # speed
        spaces.Box(low=0.0, high=6.0, shape=(1,)),              # gear
        spaces.Box(low=0.0, high=np.inf, shape=(1,)),           # rpm
        spaces.Box(low=0.0, high=1.0, shape=(1, 64, 64)),       # grayscale image
    ))
    act_space_img = spaces.Box(low=-1.0, high=1.0, shape=(3,))

    model_img = TransformerREDQActorCritic(
        obs_space_img, act_space_img, n=4,
        d_model=32, n_layers=2, n_heads=2,
        ffn_dim=64, max_len=128,
        task_conditioning=True, task_z_dim=8, num_tasks=5,
    )
    assert model_img.visual_encoder is not None, "CNN should be created for image obs"
    print(f"  visual_encoder: {model_img.visual_encoder}")
    n_cnn_params = sum(p.numel() for p in model_img.visual_encoder.parameters())
    print(f"  CNN params: {n_cnn_params:,}")

    B, T = 4, 20
    obs_seq_img = (
        torch.randn(B, T, 1),
        torch.randn(B, T, 1),
        torch.randn(B, T, 1),
        torch.randn(B, T, 1, 64, 64),   # image
    )
    a_prev_img = torch.randn(B, T, 3).tanh()
    r_prev_img = torch.randn(B, T)
    d_prev_img = torch.zeros(B, T)

    pi_img, logp_img, hidden_img = model_img.actor(
        obs_seq_img, a_prev_img, r_prev_img, d_prev_img,
        return_full=True, detach_trunk=True,
    )
    assert pi_img.shape == (B, T, 3), f"pi_img: {pi_img.shape}"
    assert logp_img.shape == (B, T), f"logp_img: {logp_img.shape}"
    assert hidden_img.shape == (B, T, 32), f"hidden_img: {hidden_img.shape}"
    z_img, task_logits_img = model_img.infer_task(hidden_img)
    assert z_img.shape == (B, T, 8), f"z_img: {z_img.shape}"
    assert task_logits_img.shape == (B, T, 5), f"task_logits_img: {task_logits_img.shape}"

    q0_img = model_img.qs[0](hidden_img, a_prev_img, z_img)
    assert q0_img.shape == (B, T), f"q0_img: {q0_img.shape}"

    loss_img = (q0_img.mean() + logp_img.mean()) * 0.5
    loss_img.backward()

    # Check gradients flow through CNN
    cnn_grad_ok = all(p.grad is not None and p.grad.abs().sum() > 0
                      for p in model_img.visual_encoder.parameters()
                      if p.requires_grad)
    print(f"  CNN gradients flow: {cnn_grad_ok}")
    assert cnn_grad_ok, "Gradients must flow through the CNN"
    print(f"  smoke OK  loss={loss_img.item():.4f}  pi.shape={tuple(pi_img.shape)}  q.shape={tuple(q0_img.shape)}")

    # Inference path: image obs (with a_prev, r_prev, d_prev appended by RL2 wrapper)
    obs_space_img_aug = spaces.Tuple((
        spaces.Box(low=0.0, high=1.0, shape=(1,)),              # speed
        spaces.Box(low=0.0, high=6.0, shape=(1,)),              # gear
        spaces.Box(low=0.0, high=np.inf, shape=(1,)),           # rpm
        spaces.Box(low=0.0, high=1.0, shape=(1, 64, 64)),       # image
        spaces.Box(low=-1.0, high=1.0, shape=(3,)),             # a_prev
        spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),       # r_prev
        spaces.Box(low=0.0, high=1.0, shape=(1,)),              # d_prev
    ))
    model_img_inf = TransformerREDQActorCritic(
        obs_space_img_aug, act_space_img, n=4,
        d_model=32, n_layers=2, n_heads=2,
        ffn_dim=64, max_len=128,
        task_conditioning=True, task_z_dim=8, num_tasks=5,
    )
    actor_img = model_img_inf.actor
    actor_img.reset_history()
    fake_img_obs = (
        np.zeros((1,), dtype=np.float32),         # speed
        np.zeros((1,), dtype=np.float32),         # gear
        np.zeros((1,), dtype=np.float32),         # rpm
        np.zeros((1, 64, 64), dtype=np.float32),  # image
        np.zeros(3, dtype=np.float32),             # a_prev
        np.zeros(1, dtype=np.float32),             # r_prev
        np.ones(1, dtype=np.float32),              # d_prev (episode start)
    )
    a_img = actor_img.act(fake_img_obs, test=True)
    assert a_img.shape == (3,), f"a_img: {a_img.shape}"
    # Do a few steps to test history accumulation
    for step in range(5):
        fake_step = (
            np.random.randn(1).astype(np.float32),
            np.random.randn(1).astype(np.float32),
            np.random.randn(1).astype(np.float32),
            np.random.randn(1, 64, 64).astype(np.float32),
            np.random.randn(3).astype(np.float32),
            np.random.randn(1).astype(np.float32),
            np.zeros(1, dtype=np.float32),  # d_prev=0 (not done)
        )
        a_img = actor_img.act(fake_step, test=False)
        assert a_img.shape == (3,), f"step {step}: a_img: {a_img.shape}"
    print(f"  inference OK  history_len={len(actor_img._history)}  a={a_img}")

    # Total param count comparison
    total_params_scalar = sum(p.numel() for p in model.parameters())
    total_params_img = sum(p.numel() for p in model_img.parameters())
    print(f"\n=== Param comparison ===")
    print(f"  Scalar (lidar) model: {total_params_scalar:,} params")
    print(f"  Image (CNN) model:    {total_params_img:,} params")
    print(f"  CNN overhead:         {total_params_img - total_params_scalar:,} params")

    print("\nAll smoke tests passed!")


if __name__ == "__main__":
    _smoke_test()
