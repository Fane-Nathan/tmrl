#!/usr/bin/env python3
"""
Sequence-Level Transformer RL^2 Agent for In-Context Fault Adaptation.

Implements the architecture and training protocol from Section III & IV of:
"Memory Buys Adaptation, Demonstrations Buy Competence: In-Context Reinforcement Learning Under Real-Time Constraints"

Key features & Methodological fixes:
1. Causal Transformer Trunk: 2 layers, 2 heads, d_model=64, d_ff=128, pre-norm, GELU, pos-emb.
2. Stop-Gradient to Actor Head: trunk is shaped by value gradients (and auxiliary task loss).
3. REDQ Critics with LayerNorm: N=10 critic heads sharing trunk, M=2 random subset for min-pessimism.
   LayerNorm on critic heads prevents value divergence.
4. Auxiliary Task Classification Head: with configurable weight lambda_task (default 0.05, 0.0 for ablation).
5. CLEAN VALIDATION-ONLY CHECKPOINT SELECTION (Zero Test Leakage):
   - Periodic evaluation runs solely on the Validation split (unseen parameters of train faults).
   - Held-out test conditions (action_latency, dead_actuator, sign_flip) are evaluated ONCE on the frozen
     validation-selected checkpoint.
6. Adaptive vs. No-History Evaluation:
   - Computes G = R_adaptive - R_no_history over deterministic evaluation episodes.
   - Saves all individual episode returns to JSON for complete auditability and bootstrap CI calculation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

# Local imports
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark_fault_env import FaultInjectionWrapper, make_fault_env
from scripts.confirmatory_logging import (
    compute_file_sha256,
    setup_run_directory,
    update_manifest,
)


# ==============================================================================
# Model Architecture
# ==============================================================================

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

    def forward(self, x: torch.Tensor, block_history_for_last_token: bool = False) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        dropout_p = self.dropout.p if self.training else 0.0
        if block_history_for_last_token:
            # Context-matched ablation: preserve sequence length and positional
            # index, but prevent the current query from reading prior tokens.
            allowed = torch.tril(torch.ones((T, T), dtype=torch.bool, device=x.device))
            allowed[-1, :] = False
            allowed[-1, -1] = True
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=allowed,
                dropout_p=dropout_p,
                is_causal=False,
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=dropout_p,
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

    def forward(self, x: torch.Tensor, block_history_for_last_token: bool = False) -> torch.Tensor:
        x = x + self.attn(
            self.ln1(x),
            block_history_for_last_token=block_history_for_last_token,
        )
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

    def forward(self, x: torch.Tensor, block_history_for_last_token: bool = False) -> torch.Tensor:
        B, T, _ = x.shape
        if T > self.pos_emb.shape[1]:
            raise ValueError(
                f"Sequence length {T} exceeds positional-embedding capacity {self.pos_emb.shape[1]}"
            )
        h = self.embedder(x) + self.pos_emb[:, :T, :]
        for block in self.blocks:
            h = block(
                h,
                block_history_for_last_token=block_history_for_last_token,
            )
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
        context_window: int = 32,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.d_model = d_model
        self.z_dim = z_dim
        self.num_critics = num_critics
        self.subset_m = subset_m
        self.lambda_task = lambda_task
        self.context_window = int(context_window)
        if self.context_window < 2:
            raise ValueError("context_window must be at least 2")

        # Trunk & Task Encoder
        self.trunk = TransformerTrunk(
            in_dim=obs_dim,
            d_model=d_model,
            max_len=self.context_window,
        )
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
        self.target_trunk = TransformerTrunk(
            in_dim=obs_dim,
            d_model=d_model,
            max_len=self.context_window,
        )
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

        # Never use absolute positional embeddings beyond the horizon exercised
        # during training.
        self.context_deque: deque = deque(maxlen=self.context_window)

    def reset_context(self) -> None:
        self.context_deque.clear()

    def step_in_context(
        self,
        obs: np.ndarray,
        adaptive: bool = True,
        history_blocked: bool = False,
        deterministic: bool = True,
        device: str = "cpu",
    ) -> np.ndarray:
        if not adaptive:
            self.reset_context()

        self.context_deque.append(obs)
        seq_np = np.asarray(self.context_deque, dtype=np.float32)
        seq = torch.from_numpy(seq_np).unsqueeze(0).to(device)
        with torch.no_grad():
            h = self.trunk(
                seq,
                block_history_for_last_token=history_blocked,
            )
            z, _ = self.task_encoder(h)
            h_last = h[:, -1:]
            z_last = z[:, -1:]
            feat = torch.cat([h_last, z_last], dim=-1)
            act, _ = self.actor(feat, deterministic=deterministic)
            return act[0, 0].cpu().numpy()


# ==============================================================================
# Trajectory Replay Buffer
# ==============================================================================

class SequenceReplayBuffer:
    def __init__(self, max_transitions: int = 1_000_000, max_seq_len: int = 32, burn_in: int = 8):
        self.max_transitions = max_transitions
        self.max_seq_len = max_seq_len
        self.burn_in = burn_in

        self.episodes: List[Dict[str, np.ndarray]] = []
        self.total_steps = 0

    def add_episode(self, obs: np.ndarray, acts: np.ndarray, rews: np.ndarray, dones: np.ndarray, fault_id: int):
        ep_len = len(rews)
        if ep_len == 0:
            return
        ep = {
            "obs": np.asarray(obs, dtype=np.float32),
            "acts": np.asarray(acts, dtype=np.float32),
            "rews": np.asarray(rews, dtype=np.float32).reshape(-1, 1),
            "dones": np.asarray(dones, dtype=np.float32).reshape(-1, 1),
            "fault_id": int(fault_id),
            "length": ep_len,
        }
        self.episodes.append(ep)
        self.total_steps += ep_len

        while self.total_steps > self.max_transitions and len(self.episodes) > 1:
            removed = self.episodes.pop(0)
            self.total_steps -= removed["length"]

    def sample_batch(self, batch_size: int, device: str = "cpu") -> Dict[str, torch.Tensor]:
        obs_seqs, act_seqs, rew_seqs, done_seqs = [], [], [], []
        valid_seqs, valid_lengths, fault_ids = [], [], []
        L = self.max_seq_len

        for _ in range(batch_size):
            ep = random.choice(self.episodes)
            T = ep["length"]

            if T < L:
                transition_pad = L - T
                o = np.pad(ep["obs"], ((0, transition_pad), (0, 0)), mode="edge")
                a = np.pad(ep["acts"], ((0, transition_pad), (0, 0)), mode="constant")
                r = np.pad(ep["rews"], ((0, transition_pad), (0, 0)), mode="constant")
                d = np.pad(
                    ep["dones"],
                    ((0, transition_pad), (0, 0)),
                    mode="constant",
                    constant_values=1.0,
                )
                valid = np.concatenate(
                    [
                        np.ones(T, dtype=np.float32),
                        np.zeros(transition_pad, dtype=np.float32),
                    ]
                )
                start = 0
            else:
                # T-L is the final valid start and includes the terminal
                # transition. random.randint is inclusive at both ends.
                start = random.randint(0, T - L)
                o = ep["obs"]
                a = ep["acts"]
                r = ep["rews"]
                d = ep["dones"]
                valid = np.ones(L, dtype=np.float32)

            obs_seqs.append(o[start:start + L + 1])
            act_seqs.append(a[start:start + L])
            rew_seqs.append(r[start:start + L])
            done_seqs.append(d[start:start + L])
            valid_seqs.append(valid)
            valid_lengths.append(min(T, L))
            fault_ids.append(ep["fault_id"])

        return {
            "obs": torch.as_tensor(np.stack(obs_seqs), dtype=torch.float32, device=device),
            "acts": torch.as_tensor(np.stack(act_seqs), dtype=torch.float32, device=device),
            "rews": torch.as_tensor(np.stack(rew_seqs), dtype=torch.float32, device=device),
            "dones": torch.as_tensor(np.stack(done_seqs), dtype=torch.float32, device=device),
            "valid": torch.as_tensor(np.stack(valid_seqs), dtype=torch.float32, device=device),
            "valid_lengths": torch.as_tensor(valid_lengths, dtype=torch.long, device=device),
            "fault_ids": torch.as_tensor(fault_ids, dtype=torch.long, device=device),
        }


# ==============================================================================
# Evaluation Protocol
# ==============================================================================

# ==============================================================================
# Evaluation Protocol
# ==============================================================================

def evaluate_policy_on_split(
    agent: TransformerRL2Agent,
    split: str,
    episodes: int = 25,
    adaptive: Optional[bool] = None,
    mode: str = "adaptive",
    device: str = "cpu",
    base_seed: int = 50000,
) -> Tuple[float, List[float], List[int]]:
    """
    Evaluates the frozen agent on a specific fault split without parameter updates.
    Uses per-episode deterministic seeds for strictly reproducible, paired evaluation (CRN).
    """
    if adaptive is not None:
        mode = "adaptive" if adaptive else "context_reset"
    if mode not in {"adaptive", "history_blocked", "context_reset"}:
        raise ValueError(f"Unknown evaluation mode: {mode}")

    agent.eval()
    env = make_fault_env("HalfCheetah-v5", split=split, seed=base_seed)
    returns: List[float] = []
    lengths: List[int] = []

    for ep in range(episodes):
        ep_seed = base_seed + ep * 100
        obs, _ = env.reset(seed=ep_seed)
        agent.reset_context()
        ep_ret = 0.0
        ep_len = 0
        term, trunc = False, False

        while not (term or trunc):
            action = agent.step_in_context(
                obs,
                adaptive=(mode != "context_reset"),
                history_blocked=(mode == "history_blocked"),
                deterministic=True,
                device=device,
            )
            obs, rew, term, trunc, _ = env.step(action)
            ep_ret += rew
            ep_len += 1

        returns.append(float(ep_ret))
        lengths.append(int(ep_len))

    env.close()
    return float(np.mean(returns)), returns, lengths


# ==============================================================================
# Training Pipeline
# ==============================================================================

def train_transformer_agent(
    seed: int = 42,
    iterations: int = 100,
    collection_episodes_per_iter: int = 4,
    gradient_steps_per_iter: int = 100,
    utd: int = 10,
    batch_size: int = 64,
    burn_in: int = 8,
    train_seq_len: int = 24,
    lambda_task: float = 0.05,
    use_layernorm: bool = True,
    val_interval: int = 10,
    val_episodes: int = 10,
    test_episodes: int = 25,
    base_eval_seed: int = 50000,
    num_threads: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "./fault_benchmark_results",
    config_path: Optional[str] = None,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    manifest_path = Path(output_dir) / "confirmatory_manifest.csv"
    os.makedirs(output_dir, exist_ok=True)

    config_dict: Dict[str, Any] = {}
    protocol_version = "unversioned"
    protocol_sha256 = ""
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        protocol_version = str(config_dict.get("protocol_version", "unversioned"))
        protocol_sha256 = compute_file_sha256(config_path)
        budget = config_dict.get("training_budget", {})
        iterations = budget.get("iterations", iterations)
        collection_episodes_per_iter = budget.get("collection_episodes_per_iter", collection_episodes_per_iter)
        gradient_steps_per_iter = budget.get("gradient_steps_per_iter", gradient_steps_per_iter)
        utd = budget.get("utd", utd)
        val_interval = budget.get("val_interval", val_interval)
        val_episodes = budget.get("val_episodes", val_episodes)
        eval_proto = config_dict.get("evaluation_protocol", {})
        test_episodes = eval_proto.get("episodes_per_condition", test_episodes)
        base_eval_seed = eval_proto.get("base_eval_seed", base_eval_seed)
        opt_cfg = config_dict.get("optimization", {})
        batch_size = int(opt_cfg.get("batch_size", batch_size))
        burn_in = int(opt_cfg.get("burn_in", burn_in))
        train_seq_len = int(opt_cfg.get("train_seq_len", train_seq_len))
        arch_cfg = config_dict.get("transformer_architecture", {})
        lambda_task = float(arch_cfg.get("lambda_task", lambda_task))
        use_layernorm = bool(arch_cfg.get("use_layernorm", use_layernorm))

    if smoke_test:
        iterations = 2
        collection_episodes_per_iter = 1
        gradient_steps_per_iter = 5
        utd = 2
        val_episodes = 2
        test_episodes = 2
        val_interval = 2

    # Reproducibility run directory
    run_name = f"transformer_seed_{seed}"
    run_config = {
        "algorithm": "transformer_rl2",
        "seed": seed,
        "iterations": iterations,
        "collection_episodes_per_iter": collection_episodes_per_iter,
        "gradient_steps_per_iter": gradient_steps_per_iter,
        "utd": utd,
        "batch_size": batch_size,
        "burn_in": burn_in,
        "train_seq_len": train_seq_len,
        "lambda_task": lambda_task,
        "use_layernorm": use_layernorm,
        "val_interval": val_interval,
        "val_episodes": val_episodes,
        "test_episodes": test_episodes,
        "base_eval_seed": base_eval_seed,
        "device": device,
        "protocol_version": protocol_version,
        "protocol_sha256": protocol_sha256,
        "context_window": burn_in + train_seq_len,
    }
    run_dir = setup_run_directory(output_dir, run_name, run_config, seed, REPO_ROOT)

    train_start_str = time.strftime("%Y-%m-%d %H:%M:%S")
    update_manifest(manifest_path, {
        "algorithm": "transformer_rl2",
        "seed": seed,
        "status": "TRAINING",
        "train_start": train_start_str,
        "train_end": "",
        "training_iterations": iterations,
        "environment_steps": 0,
        "validation_checkpoint": "",
        "checkpoint_sha256": "",
        "validation_score": "",
        "test_completed": "False",
        "raw_results_file": "",
        "notes": "Controlled benchmark (NODE/RESeL disabled)",
    })

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    seq_len = burn_in + train_seq_len
    configured_context = int(
        config_dict.get("transformer_architecture", {}).get("context_window", seq_len)
    )
    if configured_context != seq_len:
        raise ValueError(
            "context_window must equal burn_in + train_seq_len "
            f"({configured_context} != {seq_len})"
        )
    train_env = make_fault_env("HalfCheetah-v5", split="train", seed=seed)

    agent = TransformerRL2Agent(
        obs_dim=train_env.observation_space.shape[0],
        act_dim=train_env.action_space.shape[0],
        d_model=64,
        z_dim=8,
        num_critics=10,
        subset_m=2,
        use_layernorm=use_layernorm,
        lambda_task=lambda_task,
        context_window=seq_len,
    ).to(device)

    # Optimizers
    actor_opt = torch.optim.Adam(agent.actor.parameters(), lr=3e-4)
    critic_params = list(agent.critics.parameters()) + list(agent.trunk.parameters()) + list(agent.task_encoder.parameters())
    critic_opt = torch.optim.Adam(critic_params, lr=1.5e-4)

    # Automatic Entropy Tuning
    target_entropy = -float(train_env.action_space.shape[0])
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_opt = torch.optim.Adam([log_alpha], lr=3e-4)

    replay = SequenceReplayBuffer(max_transitions=1_000_000, max_seq_len=seq_len, burn_in=burn_in)

    best_val_score = -float("inf")
    best_val_iter = 0
    best_checkpoint_path = run_dir / "best_validation_checkpoint.pt"
    top_checkpoint_path = Path(output_dir) / f"best_val_seed_{seed}.pt"
    val_history = []
    total_env_steps = 0

    val_metrics_file = run_dir / "validation_metrics.csv"
    with open(val_metrics_file, "w", newline="", encoding="utf-8") as f:
        f.write("iteration,val_return,best_so_far\n")

    print(f"[{time.strftime('%X')}] Starting Seed {seed} | Device: {device} | Lambda_task: {lambda_task} | LayerNorm: {use_layernorm}", flush=True)

    first_training_reset = True
    for it in range(1, iterations + 1):
        t_it_start = time.time()
        # 1. Collection Phase on TRAIN split
        agent.train()
        for _ in range(collection_episodes_per_iter):
            obs_list, act_list, rew_list, done_list = [], [], [], []
            if first_training_reset:
                obs, info = train_env.reset(seed=seed)
                first_training_reset = False
            else:
                obs, info = train_env.reset()
            agent.reset_context()
            term, trunc = False, False

            while not (term or trunc):
                obs_list.append(obs)
                act = agent.step_in_context(obs, adaptive=True, deterministic=False, device=device)
                next_obs, rew, term, trunc, _ = train_env.step(act)
                act_list.append(act)
                rew_list.append(rew)
                done_list.append(term)
                obs = next_obs
                total_env_steps += 1

            obs_list.append(obs)
            replay.add_episode(
                obs=np.stack(obs_list),
                acts=np.stack(act_list),
                rews=np.array(rew_list),
                dones=np.array(done_list),
                fault_id=info["fault_id"],
            )

        # 2. Gradient Updates
        if replay.total_steps >= batch_size * seq_len:
            for _ in range(gradient_steps_per_iter):
                for _ in range(utd):
                    batch = replay.sample_batch(batch_size, device=device)
                    obs_seq = batch["obs"][:, :seq_len]
                    next_obs_seq = batch["obs"][:, 1:seq_len + 1]
                    act_seq = batch["acts"]
                    rew_seq = batch["rews"]
                    done_seq = batch["dones"]
                    valid_seq = batch["valid"]
                    valid_lengths = batch["valid_lengths"]
                    fault_ids = batch["fault_ids"]

                    h_seq = agent.trunk(obs_seq)
                    z_seq, cls_logits = agent.task_encoder(h_seq)

                    with torch.no_grad():
                        # Match deployment context. For transitions 0..L-2,
                        # keep the same prefix and advance one token. For the
                        # final transition, slide the full context window by one.
                        h_target_prefix = agent.target_trunk(obs_seq)
                        z_target_prefix, _ = agent.target_task_encoder(h_target_prefix)
                        h_target_sliding = agent.target_trunk(next_obs_seq)
                        z_target_sliding, _ = agent.target_task_encoder(h_target_sliding)
                        h_next = torch.cat(
                            [h_target_prefix[:, 1:], h_target_sliding[:, -1:]],
                            dim=1,
                        )
                        z_next = torch.cat(
                            [z_target_prefix[:, 1:], z_target_sliding[:, -1:]],
                            dim=1,
                        )
                        actor_next_feat = torch.cat([h_next, z_next], dim=-1)
                        next_act_seq, next_log_prob = agent.actor(actor_next_feat, deterministic=False)

                        target_in = torch.cat([h_next, z_next, next_act_seq], dim=-1)
                        sampled_indices = random.sample(range(agent.num_critics), agent.subset_m)
                        q_targets = [agent.target_critics[idx](target_in) for idx in sampled_indices]
                        q_min = torch.min(torch.stack(q_targets, dim=0), dim=0).values

                        alpha = log_alpha.exp().detach()
                        target_y = rew_seq + (1.0 - done_seq) * 0.99 * (q_min - alpha * next_log_prob)

                    critic_in = torch.cat([h_seq, z_seq, act_seq], dim=-1)
                    loss_mask = valid_seq[:, burn_in:].unsqueeze(-1)
                    mask_denom = loss_mask.sum().clamp_min(1.0)
                    critic_losses = []
                    for critic in agent.critics:
                        q_pred = critic(critic_in)
                        sq_error = (q_pred[:, burn_in:] - target_y[:, burn_in:]).pow(2)
                        c_loss = (sq_error * loss_mask).sum() / mask_denom
                        critic_losses.append(c_loss)
                    total_critic_loss = sum(critic_losses)

                    cls_loss = torch.zeros(1, device=device)
                    if agent.lambda_task > 0.0:
                        last_valid = (valid_lengths - 1).clamp(min=0, max=seq_len - 1)
                        batch_idx = torch.arange(cls_logits.shape[0], device=device)
                        cls_loss = F.cross_entropy(cls_logits[batch_idx, last_valid], fault_ids)

                    total_loss = total_critic_loss + agent.lambda_task * cls_loss

                    critic_opt.zero_grad(set_to_none=True)
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(critic_params, 10.0)
                    critic_opt.step()

                    with torch.no_grad():
                        for param, target_param in zip(agent.trunk.parameters(), agent.target_trunk.parameters()):
                            target_param.data.mul_(0.995).add_(param.data, alpha=0.005)
                        for param, target_param in zip(agent.task_encoder.parameters(), agent.target_task_encoder.parameters()):
                            target_param.data.mul_(0.995).add_(param.data, alpha=0.005)
                        for param, target_param in zip(agent.critics.parameters(), agent.target_critics.parameters()):
                            target_param.data.mul_(0.995).add_(param.data, alpha=0.005)

                # Policy & Alpha Update
                h_det = h_seq.detach()
                z_det = z_seq.detach()
                actor_feat = torch.cat([h_det, z_det], dim=-1)
                pred_act, log_pi = agent.actor(actor_feat, deterministic=False)

                actor_in = torch.cat([h_det, z_det, pred_act], dim=-1)
                q1_pred = agent.critics[0](actor_in)
                q2_pred = agent.critics[1](actor_in)
                q_actor_min = torch.min(q1_pred, q2_pred)

                alpha = log_alpha.exp()
                actor_objective = (
                    (alpha.detach() * log_pi[:, burn_in:]) - q_actor_min[:, burn_in:]
                )
                actor_mask = valid_seq[:, burn_in:].unsqueeze(-1)
                actor_denom = actor_mask.sum().clamp_min(1.0)
                actor_loss = (actor_objective * actor_mask).sum() / actor_denom

                actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(agent.actor.parameters(), 10.0)
                actor_opt.step()

                alpha_objective = -(
                    log_alpha * (log_pi[:, burn_in:].detach() + target_entropy)
                )
                alpha_loss = (alpha_objective * actor_mask).sum() / actor_denom
                alpha_opt.zero_grad(set_to_none=True)
                alpha_loss.backward()
                alpha_opt.step()

        # 3. Clean Periodic Evaluation on VALIDATION SPLIT ONLY
        if it % val_interval == 0 or it == iterations:
            base_validation_seed = int(
                config_dict.get("evaluation_protocol", {}).get("base_validation_seed", 40000)
            )
            val_score, _, _ = evaluate_policy_on_split(
                agent,
                split="val",
                episodes=val_episodes,
                mode="adaptive",
                device=device,
                base_seed=base_validation_seed,
            )
            val_history.append({"iter": it, "val_return": val_score})
            improved = val_score > best_val_score
            if improved:
                best_val_score = val_score
                best_val_iter = it
                torch.save(agent.state_dict(), str(best_checkpoint_path))
                torch.save(agent.state_dict(), str(top_checkpoint_path))

            with open(val_metrics_file, "a", newline="", encoding="utf-8") as f:
                f.write(f"{it},{val_score:.4f},{best_val_score:.4f}\n")

            print(f"  Iter {it:3d}/{iterations} ({time.time()-t_it_start:.1f}s) | Val Return: {val_score:+.2f} {'[BEST]' if improved else ''}", flush=True)
        else:
            print(f"  Iter {it:3d}/{iterations} ({time.time()-t_it_start:.1f}s) | Training step complete", flush=True)

    train_end_str = time.strftime("%Y-%m-%d %H:%M:%S")

    # 4. FINAL TEST EVALUATION: Load best validation checkpoint and evaluate on held-out test splits
    print(f"\n[{time.strftime('%X')}] Finalizing Seed {seed}: Loading best validation checkpoint (val_return={best_val_score:.2f} at iter {best_val_iter})...", flush=True)
    if os.path.exists(best_checkpoint_path):
        agent.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
    elif os.path.exists(top_checkpoint_path):
        agent.load_state_dict(torch.load(top_checkpoint_path, map_location=device))

    checkpoint_hash = compute_file_sha256(best_checkpoint_path) if os.path.exists(best_checkpoint_path) else "NO_CHECKPOINT_HASH"

    checkpoint_metadata = {
        "seed": seed,
        "best_iteration": best_val_iter,
        "best_val_score": best_val_score,
        "checkpoint_sha256": checkpoint_hash,
        "frozen_timestamp": train_end_str,
        "selection_rule": "argmax_validation_score",
        "validation_split": "unseen parameters of action_scale & action_noise",
        "protocol_version": protocol_version,
        "protocol_sha256": protocol_sha256,
    }
    with open(run_dir / "checkpoint_metadata.json", "w") as f:
        json.dump(checkpoint_metadata, f, indent=2)

    raw_episodes_csv = run_dir / "final_test_raw_episodes.csv"
    with open(raw_episodes_csv, "w", newline="", encoding="utf-8") as f:
        f.write("algorithm,seed,fault_condition,mode,episode_id,environment_seed,return,episode_length,checkpoint_hash\n")

    test_results: Dict[str, Any] = {}
    test_conditions = ["test_latency", "test_dead", "test_sign_flip"]

    for condition in test_conditions:
        ad_mean, ad_raw, ad_lens = evaluate_policy_on_split(
            agent, split=condition, episodes=test_episodes, mode="adaptive",
            device=device, base_seed=base_eval_seed
        )
        hb_mean, hb_raw, hb_lens = evaluate_policy_on_split(
            agent, split=condition, episodes=test_episodes, mode="history_blocked",
            device=device, base_seed=base_eval_seed
        )
        cr_mean, cr_raw, cr_lens = evaluate_policy_on_split(
            agent, split=condition, episodes=test_episodes, mode="context_reset",
            device=device, base_seed=base_eval_seed
        )
        gap = ad_mean - hb_mean
        reset_gap = ad_mean - cr_mean
        test_results[condition] = {
            "adaptive_mean": ad_mean,
            "history_blocked_mean": hb_mean,
            "no_history_mean": hb_mean,
            "context_reset_mean": cr_mean,
            "adaptation_gap": gap,
            "context_reset_gap": reset_gap,
            "raw_adaptive": ad_raw,
            "raw_history_blocked": hb_raw,
            "raw_no_history": hb_raw,
            "raw_context_reset": cr_raw,
            "raw_gaps": [a - b for a, b in zip(ad_raw, hb_raw)],
            "raw_context_reset_gaps": [a - r for a, r in zip(ad_raw, cr_raw)],
            "raw_lengths_adaptive": ad_lens,
            "raw_lengths_history_blocked": hb_lens,
            "raw_lengths_no_history": hb_lens,
            "raw_lengths_context_reset": cr_lens,
        }

        with open(raw_episodes_csv, "a", newline="", encoding="utf-8") as f:
            for ep_idx in range(len(ad_raw)):
                ep_s = base_eval_seed + ep_idx * 100
                f.write(f"transformer_rl2,{seed},{condition},adaptive,{ep_idx},{ep_s},{ad_raw[ep_idx]:.4f},{ad_lens[ep_idx]},{checkpoint_hash}\n")
                f.write(f"transformer_rl2,{seed},{condition},history_blocked,{ep_idx},{ep_s},{hb_raw[ep_idx]:.4f},{hb_lens[ep_idx]},{checkpoint_hash}\n")
                f.write(f"transformer_rl2,{seed},{condition},context_reset,{ep_idx},{ep_s},{cr_raw[ep_idx]:.4f},{cr_lens[ep_idx]},{checkpoint_hash}\n")

        endpoint_tag = "PRIMARY ENDPOINT" if condition == "test_latency" else "SECONDARY ENDPOINT"
        print(
            f"  [{endpoint_tag}] '{condition:15s}' | Adaptive: {ad_mean:+.2f} | "
            f"History-Blocked: {hb_mean:+.2f} | Gap (G): {gap:+.2f} | "
            f"Context-Reset: {cr_mean:+.2f}",
            flush=True,
        )

    summary = {
        "algorithm": "transformer_rl2",
        "seed": seed,
        "protocol_version": protocol_version,
        "protocol_sha256": protocol_sha256,
        "primary_ablation": "history_blocked_context_matched",
        "lambda_task": lambda_task,
        "use_layernorm": use_layernorm,
        "best_val_score": best_val_score,
        "best_val_iter": best_val_iter,
        "checkpoint_sha256": checkpoint_hash,
        "val_history": val_history,
        "test_results": test_results,
    }

    with open(run_dir / "final_test_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    report_path = Path(output_dir) / f"report_seed_{seed}_lambda_{lambda_task}.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    update_manifest(manifest_path, {
        "algorithm": "transformer_rl2",
        "seed": seed,
        "status": "COMPLETE",
        "train_start": train_start_str,
        "train_end": train_end_str,
        "training_iterations": iterations,
        "environment_steps": total_env_steps,
        "validation_checkpoint": str(best_checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "validation_score": round(best_val_score, 4),
        "test_completed": "True",
        "raw_results_file": str(raw_episodes_csv),
        "notes": f"Primary latency gap G = {test_results['test_latency']['adaptation_gap']:+.4f}",
    })

    print(f"Saved complete run report to {report_path}\n", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sequence Transformer RL^2 Agent")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default=None, help="Path to confirmatory_protocol.json")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--collection_episodes_per_iter", type=int, default=4)
    parser.add_argument("--gradient_steps_per_iter", type=int, default=100)
    parser.add_argument("--utd", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_interval", type=int, default=10)
    parser.add_argument("--val_episodes", type=int, default=10)
    parser.add_argument("--test_episodes", type=int, default=25)
    parser.add_argument("--lambda_task", type=float, default=0.05)
    parser.add_argument("--use_layernorm", action="store_true", default=True)
    parser.add_argument("--num_threads", type=int, default=4, help="Number of CPU threads per process")
    parser.add_argument("--fast", action="store_true", help="Fast training for development/benchmark verification")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./fault_benchmark_results")
    args = parser.parse_args()

    if args.fast:
        iters = 20
        coll = 2
        grads = 25
        utd = 4
        val_int = 5
        val_ep = 5
        test_ep = 25
    else:
        iters = args.iterations
        coll = args.collection_episodes_per_iter
        grads = args.gradient_steps_per_iter
        utd = args.utd
        val_int = args.val_interval
        val_ep = args.val_episodes
        test_ep = args.test_episodes

    train_transformer_agent(
        seed=args.seed,
        iterations=iters,
        collection_episodes_per_iter=coll,
        gradient_steps_per_iter=grads,
        utd=utd,
        batch_size=args.batch_size,
        val_interval=val_int,
        val_episodes=val_ep,
        test_episodes=test_ep,
        num_threads=args.num_threads,
        lambda_task=args.lambda_task,
        use_layernorm=args.use_layernorm,
        config_path=args.config,
        smoke_test=args.smoke_test,
        output_dir=args.output_dir,
    )

