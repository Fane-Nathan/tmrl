"""Checkpoint-compatible model definitions extracted without changes from the prior implementation.
See model_provenance.json. Experiment behavior is configured in learner.py.
"""
from __future__ import annotations
import math
from collections import deque
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class TimestepEmbedder(nn.Module):
    def __init__(self, in_dim: int = 25, d_model: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # PyTorch causal flash attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 2, d_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerTrunk(nn.Module):
    def __init__(
        self,
        in_dim: int = 25,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 2,
        d_ff: int = 128,
        max_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedder = TimestepEmbedder(in_dim=in_dim, d_model=d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.embedder(x) + self.pos_emb[:, :T, :]
        for block in self.blocks:
            h = block(h)
        return self.ln_f(h)


class TaskEncoder(nn.Module):
    def __init__(self, d_model: int = 64, z_dim: int = 8, num_classes: int = 7):
        super().__init__()
        self.z_head = nn.Linear(d_model, z_dim)
        self.cls_head = nn.Linear(z_dim, num_classes)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = F.relu(self.z_head(h))
        logits = self.cls_head(z)
        return z, logits


class GaussianActorHead(nn.Module):
    def __init__(self, in_dim: int = 72, act_dim: int = 6, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)

    def forward(self, x: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        feat = self.net(x)
        mu = self.mu(feat)
        log_std = torch.clamp(self.log_std(feat), -20, 2)
        std = torch.exp(log_std)
        dist = Normal(mu, std)

        if deterministic:
            raw_act = mu
            log_prob = None
        else:
            raw_act = dist.rsample()
            log_prob = dist.log_prob(raw_act).sum(dim=-1, keepdim=True)
            # Tanh squashing correction
            log_prob -= (2.0 * (math.log(2.0) - raw_act - F.softplus(-2.0 * raw_act))).sum(dim=-1, keepdim=True)

        action = torch.tanh(raw_act)
        return action, log_prob


class CriticHead(nn.Module):
    def __init__(self, in_dim: int = 78, hidden_dim: int = 256, use_layernorm: bool = True):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerRL2Agent(nn.Module):
    def __init__(
        self,
        obs_dim: int = 25,
        act_dim: int = 6,
        d_model: int = 64,
        z_dim: int = 8,
        num_critics: int = 10,
        subset_m: int = 2,
        use_layernorm: bool = True,
        lambda_task: float = 0.05,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.d_model = d_model
        self.z_dim = z_dim
        self.num_critics = num_critics
        self.subset_m = subset_m
        self.lambda_task = lambda_task

        # Trunk & Task Encoder
        self.trunk = TransformerTrunk(in_dim=obs_dim, d_model=d_model)
        self.task_encoder = TaskEncoder(d_model=d_model, z_dim=z_dim, num_classes=7)

        # Actor head reads detached trunk + task embedding
        self.actor = GaussianActorHead(in_dim=d_model + z_dim, act_dim=act_dim)

        # Critic heads read trunk + task embedding + action
        critic_in_dim = d_model + z_dim + act_dim
        self.critics = nn.ModuleList([
            CriticHead(in_dim=critic_in_dim, hidden_dim=256, use_layernorm=use_layernorm)
            for _ in range(num_critics)
        ])

        # Target critics (copy of critics + trunk)
        self.target_trunk = TransformerTrunk(in_dim=obs_dim, d_model=d_model)
        self.target_task_encoder = TaskEncoder(d_model=d_model, z_dim=z_dim, num_classes=7)
        self.target_critics = nn.ModuleList([
            CriticHead(in_dim=critic_in_dim, hidden_dim=256, use_layernorm=use_layernorm)
            for _ in range(num_critics)
        ])
        self.target_trunk.load_state_dict(self.trunk.state_dict())
        self.target_task_encoder.load_state_dict(self.task_encoder.state_dict())
        self.target_critics.load_state_dict(self.critics.state_dict())

        for p in self.target_trunk.parameters():
            p.requires_grad = False
        for p in self.target_task_encoder.parameters():
            p.requires_grad = False
        for p in self.target_critics.parameters():
            p.requires_grad = False

        # In-context inference deque
        self.context_deque: deque = deque(maxlen=64)

    def reset_context(self) -> None:
        self.context_deque.clear()

    def step_in_context(
        self,
        obs: np.ndarray,
        adaptive: bool = True,
        deterministic: bool = True,
        device: str = "cpu",
    ) -> np.ndarray:
        if not adaptive:
            self.reset_context()

        self.context_deque.append(obs)
        seq_np = np.asarray(self.context_deque, dtype=np.float32)
        seq = torch.from_numpy(seq_np).unsqueeze(0).to(device)
        with torch.no_grad():
            h = self.trunk(seq)
            z, _ = self.task_encoder(h)
            h_last = h[:, -1:]
            z_last = z[:, -1:]
            feat = torch.cat([h_last, z_last], dim=-1)
            act, _ = self.actor(feat, deterministic=deterministic)
            return act[0, 0].cpu().numpy()


class MLPGaussianActor(nn.Module):
    def __init__(self, obs_dim: int = 25, act_dim: int = 6, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)

    def forward(self, x: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        feat = self.net(x)
        mu = self.mu(feat)
        log_std = torch.clamp(self.log_std(feat), -20, 2)
        std = torch.exp(log_std)
        dist = Normal(mu, std)

        if deterministic:
            raw_act = mu
            log_prob = None
        else:
            raw_act = dist.rsample()
            log_prob = dist.log_prob(raw_act).sum(dim=-1, keepdim=True)
            log_prob -= (2.0 * (math.log(2.0) - raw_act - F.softplus(-2.0 * raw_act))).sum(dim=-1, keepdim=True)

        action = torch.tanh(raw_act)
        return action, log_prob


class MLPCritic(nn.Module):
    def __init__(self, obs_dim: int = 25, act_dim: int = 6, hidden_dim: int = 256, use_layernorm: bool = True):
        super().__init__()
        layers = []
        layers.append(nn.Linear(obs_dim + act_dim, hidden_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1))


class ReactiveREDQAgent(nn.Module):
    def __init__(
        self,
        obs_dim: int = 25,
        act_dim: int = 6,
        num_critics: int = 10,
        subset_m: int = 2,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_critics = num_critics
        self.subset_m = subset_m

        self.actor = MLPGaussianActor(obs_dim, act_dim)
        self.critics = nn.ModuleList([
            MLPCritic(obs_dim, act_dim, use_layernorm=use_layernorm)
            for _ in range(num_critics)
        ])
        self.target_critics = nn.ModuleList([
            MLPCritic(obs_dim, act_dim, use_layernorm=use_layernorm)
            for _ in range(num_critics)
        ])
        self.target_critics.load_state_dict(self.critics.state_dict())
        for p in self.target_critics.parameters():
            p.requires_grad = False

    def act(self, obs: np.ndarray, deterministic: bool = True, device: str = "cpu") -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            act, _ = self.actor(obs_t, deterministic=deterministic)
            return act[0].cpu().numpy()
