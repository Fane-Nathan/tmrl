#!/usr/bin/env python3
"""
Pretraining TrackMania Causal Transformer "Car Brain" on Kaggle.

Self-contained training script designed to run on Kaggle GPUs (T4 / P100 / A100)
using the public dataset: `catalystgma/trackmania-replays` (1,000+ replays across 100+ tracks).

Features:
- Discovers all track replay CSVs across directories.
- Track-level Train/Val Split (e.g. 85% train tracks, 15% unseen held-out tracks).
- 20 Hz Subsampling (downsamples 100 Hz dataset by 5 to match TMRL 20 Hz control loop).
- Feature normalization with saved statistics.
- Autoregressive next-action prediction training with causal masking.
- Evaluates In-Context Learning (ICL) on held-out tracks: L=1 (no-history) vs L=64 (full-history).
- Exports: `car_brain.safetensors`, `car_brain.pt`, `norm_stats.json`.
"""

import argparse
import glob
import json
import math
import os
import random
from collections import deque
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from safetensors.torch import save_file as save_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


# Ensure local repository root takes precedence
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from tmrl.custom.torch.car_brain import CarBrain
except ImportError:
    # Standalone fallback if run directly in an environment without tmrl installed
    class TimestepEmbedder(nn.Module):
        def __init__(self, state_dim=15, action_dim=3, d_model=128):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(state_dim + action_dim + 2, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        def forward(self, states, prev_actions=None, prev_rewards=None, prev_dones=None):
            B, T = states.shape[0], states.shape[1]
            if prev_actions is None:
                prev_actions = torch.zeros(B, T, 3, device=states.device, dtype=states.dtype)
            if prev_rewards is None:
                prev_rewards = torch.zeros(B, T, 1, device=states.device, dtype=states.dtype)
            if prev_dones is None:
                prev_dones = torch.zeros(B, T, 1, device=states.device, dtype=states.dtype)
            concat_input = torch.cat([states, prev_actions, prev_rewards, prev_dones], dim=-1)
            return self.proj(concat_input)

    class CausalSelfAttention(nn.Module):
        def __init__(self, d_model=128, n_heads=4, dropout=0.1):
            super().__init__()
            self.d_model, self.n_heads = d_model, n_heads
            self.head_dim = d_model // n_heads
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)
        def forward(self, x):
            B, T, C = x.shape
            q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.tril(torch.ones((T, T), device=x.device)).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask == 0, float("-inf"))
            attn = self.dropout(F.softmax(scores, dim=-1))
            out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)
            return self.out_proj(out)

    class TransformerBlock(nn.Module):
        def __init__(self, d_model=128, n_heads=4, d_ff=256, dropout=0.1):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = CausalSelfAttention(d_model, n_heads, dropout)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_ff, d_model), nn.Dropout(dropout),
            )
        def forward(self, x):
            return x + self.mlp(self.ln2(x + self.attn(self.ln1(x))))

    class CarBrainPolicyHead(nn.Module):
        def __init__(self, d_model=128, action_dim=3, hidden_dim=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )
        def forward(self, h):
            return torch.tanh(self.net(h))

    class CarBrain(nn.Module):
        def __init__(self, state_dim=15, action_dim=3, d_model=128, n_layers=4, n_heads=4, d_ff=256, max_context_len=64, dropout=0.1):
            super().__init__()
            self.state_dim, self.action_dim, self.d_model = state_dim, action_dim, d_model
            self.n_layers, self.n_heads, self.max_context_len = n_layers, n_heads, max_context_len
            self.embedder = TimestepEmbedder(state_dim=state_dim, action_dim=action_dim, d_model=d_model)
            self.pos_emb = nn.Parameter(torch.zeros(1, max_context_len, d_model))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)
            self.blocks = nn.ModuleList([TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout) for _ in range(n_layers)])
            self.ln_f = nn.LayerNorm(d_model)
            self.policy_head = CarBrainPolicyHead(d_model=d_model, action_dim=action_dim)
            self._history_states, self._history_prev_actions = deque(maxlen=max_context_len), deque(maxlen=max_context_len)
            self._history_prev_rewards, self._history_prev_dones = deque(maxlen=max_context_len), deque(maxlen=max_context_len)
            self.adaptive = True
        def forward(self, states, prev_actions=None, prev_rewards=None, prev_dones=None):
            B, T = states.shape[0], states.shape[1]
            tokens = self.embedder(states, prev_actions, prev_rewards, prev_dones) + self.pos_emb[:, :T, :]
            x = tokens
            for block in self.blocks: x = block(x)
            return self.policy_head(self.ln_f(x))
        def reset_context(self):
            self._history_states.clear()
            self._history_prev_actions.clear()
            self._history_prev_rewards.clear()
            self._history_prev_dones.clear()
        def step_in_context(self, state, prev_action=None, prev_reward=0.0, prev_done=False, device=None):
            if not self.adaptive: self.reset_context()
            st = torch.from_numpy(state).float() if isinstance(state, np.ndarray) else state.float()
            pa = torch.from_numpy(prev_action).float() if isinstance(prev_action, np.ndarray) else (prev_action.float() if prev_action is not None else torch.zeros(self.action_dim))
            self._history_states.append(st)
            self._history_prev_actions.append(pa)
            self._history_prev_rewards.append(torch.tensor([prev_reward], dtype=torch.float32))
            self._history_prev_dones.append(torch.tensor([1.0 if prev_done else 0.0], dtype=torch.float32))
            s_seq = torch.stack(list(self._history_states), dim=0).unsqueeze(0)
            pa_seq = torch.stack(list(self._history_prev_actions), dim=0).unsqueeze(0)
            pr_seq = torch.stack(list(self._history_prev_rewards), dim=0).unsqueeze(0)
            pd_seq = torch.stack(list(self._history_prev_dones), dim=0).unsqueeze(0)
            dev = device if device is not None else next(self.parameters()).device
            with torch.no_grad():
                out = self.forward(s_seq.to(dev), pa_seq.to(dev), pr_seq.to(dev), pd_seq.to(dev))
            return out[0, -1].cpu().numpy()
        def save_pretrained(self, save_dir, filename_prefix="car_brain"):
            os.makedirs(save_dir, exist_ok=True)
            paths = {}
            cfg = {"state_dim": self.state_dim, "action_dim": self.action_dim, "d_model": self.d_model, "n_layers": self.n_layers, "n_heads": self.n_heads, "max_context_len": self.max_context_len}
            with open(os.path.join(save_dir, "config.json"), "w") as f: json.dump(cfg, f, indent=2)
            pt_path = os.path.join(save_dir, f"{filename_prefix}.pt")
            torch.save(self.state_dict(), pt_path)
            paths["pt"] = pt_path
            if HAS_SAFETENSORS:
                st_path = os.path.join(save_dir, f"{filename_prefix}.safetensors")
                save_safetensors({k: v.contiguous() for k, v in self.state_dict().items()}, st_path)
                paths["safetensors"] = st_path
            return paths


# ==============================================================================
# Dataset & Preprocessing
# ==============================================================================

STATE_COLS = [
    "vx", "vy", "vz",
    "yaw", "pitch", "roll",
    "x", "y", "z",
    "wheel0_has_contact", "wheel1_has_contact", "wheel2_has_contact", "wheel3_has_contact",
    "wheel0_is_sliding", "wheel1_is_sliding",
]


class TrackManiaReplayDataset(Dataset):
    """
    Loads TrackMania replay CSVs, sub-samples to 20 Hz, extracts sliding windows of length L.
    """
    def __init__(
        self,
        csv_files: List[str],
        window_len: int = 64,
        downsample_stride: int = 5,  # 100 Hz -> 20 Hz
        norm_stats: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.window_len = window_len
        self.downsample_stride = downsample_stride
        self.episodes: List[Tuple[np.ndarray, np.ndarray]] = []  # List of (states, actions)

        raw_states_list = []

        print(f"Loading {len(csv_files)} replay CSV files (downsample stride={downsample_stride})...")
        for fpath in csv_files:
            try:
                df = pd.read_csv(fpath, skipinitialspace=True)
                df.columns = [c.strip() for c in df.columns]
                # Check required columns
                missing = [c for c in ["steer", "gas", "brake"] if c not in df.columns]
                if missing:
                    continue

                # Filter and sort by time
                if "time" in df.columns:
                    df = df.sort_values("time").reset_index(drop=True)

                # Subsample 100 Hz -> 20 Hz
                df_sub = df.iloc[::downsample_stride].reset_index(drop=True)
                if len(df_sub) < window_len // 2:
                    continue

                # Extract state features
                state_features = []
                for col in STATE_COLS:
                    if col in df_sub.columns:
                        state_features.append(df_sub[col].values.astype(np.float32))
                    else:
                        state_features.append(np.zeros(len(df_sub), dtype=np.float32))
                states = np.stack(state_features, axis=-1)  # (N, 15)

                # Extract & normalize actions:
                # steer is in [-65536, 65536] -> map to [-1, 1]
                # gas in [0, 1], brake in [0, 1] -> map to [-1, 1]
                steer = (df_sub["steer"].values.astype(np.float32) / 65536.0).clip(-1.0, 1.0)
                gas = df_sub["gas"].values.astype(np.float32).clip(0.0, 1.0)
                brake = df_sub["brake"].values.astype(np.float32).clip(0.0, 1.0)
                # Action representation: [gas, brake, steer] in [-1, 1]^3
                actions = np.stack([gas * 2.0 - 1.0, brake * 2.0 - 1.0, steer], axis=-1)

                self.episodes.append((states, actions))
                raw_states_list.append(states)
            except Exception as e:
                continue

        if not self.episodes:
            raise ValueError(f"No valid replay files loaded from given list of {len(csv_files)} paths.")

        # Compute or use normalization statistics
        all_states = np.concatenate(raw_states_list, axis=0)
        if norm_stats is None:
            self.mean = np.mean(all_states, axis=0, keepdims=True).astype(np.float32)
            self.std = (np.std(all_states, axis=0, keepdims=True) + 1e-5).astype(np.float32)
        else:
            self.mean = norm_stats["mean"]
            self.std = norm_stats["std"]

        # Normalize episodes
        self.normalized_episodes = []
        for states, actions in self.episodes:
            norm_s = (states - self.mean) / self.std
            self.normalized_episodes.append((norm_s.astype(np.float32), actions.astype(np.float32)))

        # Build index of valid window starts
        self.windows = []
        for ep_idx, (states, _) in enumerate(self.normalized_episodes):
            L = len(states)
            if L >= window_len:
                for start_idx in range(0, L - window_len + 1, max(1, window_len // 4)):
                    self.windows.append((ep_idx, start_idx))
            else:
                self.windows.append((ep_idx, 0))

        print(f"Dataset ready: {len(self.normalized_episodes)} episodes, {len(self.windows)} sample windows.")

    def get_norm_stats(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean, "std": self.std}

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        ep_idx, start_idx = self.windows[idx]
        states, actions = self.normalized_episodes[ep_idx]
        L = len(states)

        if L >= self.window_len:
            s_seq = states[start_idx : start_idx + self.window_len]
            a_seq = actions[start_idx : start_idx + self.window_len]
        else:
            # Right-pad shorter sequences
            pad_len = self.window_len - L
            s_seq = np.pad(states, ((0, pad_len), (0, 0)), mode="edge")
            a_seq = np.pad(actions, ((0, pad_len), (0, 0)), mode="edge")

        # Prev actions: a_{t-1} with zero init at t=0
        prev_a_seq = np.zeros_like(a_seq)
        prev_a_seq[1:] = a_seq[:-1]

        return (
            torch.from_numpy(s_seq),
            torch.from_numpy(prev_a_seq),
            torch.from_numpy(a_seq),
        )


# ==============================================================================
# Training & In-Context Evaluation
# ==============================================================================

def evaluate_in_context_adaptation(
    model: CarBrain,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluates next-action prediction error across context lengths:
    - L=1 (No-History / reactive)
    - L=16 (Short Context)
    - L=64 (Full In-Context History)
    
    If L=64 error < L=1 error, in-context adaptation is active!
    """
    model.eval()
    losses_l1, losses_l16, losses_l64 = [], [], []

    with torch.no_grad():
        for states, prev_actions, target_actions in val_loader:
            states = states.to(device)
            prev_actions = prev_actions.to(device)
            target_actions = target_actions.to(device)
            B, T, _ = states.shape

            # 1. Full context forward pass (L=64)
            pred_l64 = model(states, prev_actions)
            # Evaluate on the final timestep
            loss_64 = F.mse_loss(pred_l64[:, -1], target_actions[:, -1]).item()
            losses_l64.append(loss_64)

            # 2. Short context forward pass (L=16)
            if T >= 16:
                pred_l16 = model(states[:, -16:], prev_actions[:, -16:])
                loss_16 = F.mse_loss(pred_l16[:, -1], target_actions[:, -1]).item()
                losses_l16.append(loss_16)

            # 3. No-history forward pass (L=1, memoryless)
            pred_l1 = model(states[:, -1:], prev_actions[:, -1:])
            loss_1 = F.mse_loss(pred_l1[:, -1], target_actions[:, -1]).item()
            losses_l1.append(loss_1)

    mean_l1 = float(np.mean(losses_l1))
    mean_l16 = float(np.mean(losses_l16)) if losses_l16 else mean_l1
    mean_l64 = float(np.mean(losses_l64))

    adaptation_ratio = mean_l64 / (mean_l1 + 1e-8)
    return {
        "mse_l1_no_history": mean_l1,
        "mse_l16_short_context": mean_l16,
        "mse_l64_full_context": mean_l64,
        "adaptation_ratio": adaptation_ratio,  # < 1.0 confirms in-context learning
    }


def main():
    parser = argparse.ArgumentParser(description="Pretrain TrackMania Car Brain on Kaggle")
    parser.add_argument("--data_dir", type=str, default="/kaggle/input/trackmania-replays", help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="./output_car_brain", help="Output directory for weights")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--context_len", type=int, default=64, help="Transformer context window")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Discover all CSVs
    all_csvs = glob.glob(os.path.join(args.data_dir, "**", "*.csv"), recursive=True)
    if not all_csvs:
        # Check local fallback
        all_csvs = glob.glob(os.path.join("./data", "**", "*.csv"), recursive=True)

    print(f"Discovered {len(all_csvs)} total CSV files.")
    if not all_csvs:
        print("Warning: No CSV files found. Creating synthetic demo files to verify pipeline.")
        all_csvs = []
        for i in range(6):
            track_dir = f"./demo_data/track_{i}"
            os.makedirs(track_dir, exist_ok=True)
            df_dummy = pd.DataFrame({
                "time": np.arange(400) * 10,
                "vx": np.sin(np.arange(400) * 0.1) * 20,
                "vy": np.zeros(400),
                "vz": np.cos(np.arange(400) * 0.1) * 20,
                "yaw": np.sin(np.arange(400) * 0.05),
                "pitch": np.zeros(400),
                "roll": np.zeros(400),
                "x": np.arange(400) * 2.0,
                "y": np.zeros(400),
                "z": np.arange(400) * 1.5,
                "wheel0_has_contact": np.ones(400),
                "wheel1_has_contact": np.ones(400),
                "wheel2_has_contact": np.ones(400),
                "wheel3_has_contact": np.ones(400),
                "wheel0_is_sliding": np.zeros(400),
                "wheel1_is_sliding": np.zeros(400),
                "steer": (np.sin(np.arange(400) * 0.2) * 65536).astype(int),
                "gas": np.ones(400, dtype=int),
                "brake": np.zeros(400, dtype=int),
            })
            p = f"{track_dir}/replay_0.csv"
            df_dummy.to_csv(p, index=False)
            all_csvs.append(p)

    # Group files by track / directory to do track-level split
    tracks_dict = {}
    for f in all_csvs:
        track_key = os.path.dirname(os.path.abspath(f))
        tracks_dict.setdefault(track_key, []).append(f)

    track_keys = list(tracks_dict.keys())
    random.shuffle(track_keys)
    if len(track_keys) > 1:
        val_split_count = max(1, int(len(track_keys) * 0.15))
        # Ensure at least one train track
        val_split_count = min(val_split_count, len(track_keys) - 1)
        val_tracks = set(track_keys[:val_split_count])
        train_tracks = set(track_keys[val_split_count:])
    else:
        # Fallback if only 1 track exists
        train_tracks = set(track_keys)
        val_tracks = set(track_keys)

    train_files = [f for t in train_tracks for f in tracks_dict[t]]
    val_files = [f for t in val_tracks for f in tracks_dict[t]]
    print(f"Track split: {len(train_tracks)} train tracks ({len(train_files)} files), {len(val_tracks)} held-out test tracks ({len(val_files)} files).")

    # Build Datasets
    train_dataset = TrackManiaReplayDataset(train_files, window_len=args.context_len)
    norm_stats = train_dataset.get_norm_stats()
    val_dataset = TrackManiaReplayDataset(val_files, window_len=args.context_len, norm_stats=norm_stats)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Instantiate CarBrain
    model = CarBrain(
        state_dim=len(STATE_COLS),
        action_dim=3,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_context_len=args.context_len,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    history = []

    print(f"\nStarting training for {args.epochs} epochs on {device}...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for states, prev_actions, target_actions in train_loader:
            states = states.to(device)
            prev_actions = prev_actions.to(device)
            target_actions = target_actions.to(device)

            optimizer.zero_grad()
            pred_actions = model(states, prev_actions)
            # Autoregressive action prediction loss
            loss = F.mse_loss(pred_actions, target_actions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        if epoch_losses:
            scheduler.step()
            train_loss = float(np.mean(epoch_losses))
        else:
            train_loss = 0.0

        # Evaluate on held-out tracks
        icl_metrics = evaluate_in_context_adaptation(model, val_loader, device)
        val_l64 = icl_metrics["mse_l64_full_context"]
        val_l1 = icl_metrics["mse_l1_no_history"]
        ratio = icl_metrics["adaptation_ratio"]

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train MSE: {train_loss:.5f} | "
            f"Val L=64: {val_l64:.5f} | "
            f"Val L=1: {val_l1:.5f} | "
            f"ICL Ratio: {ratio:.4f} {'(Adaptation Active!)' if ratio < 0.95 else ''}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            **icl_metrics,
        })

        if val_l64 < best_val_loss:
            best_val_loss = val_l64
            # Save best checkpoints
            pt_path = os.path.join(args.output_dir, "car_brain.pt")
            torch.save(model.state_dict(), pt_path)
            if HAS_SAFETENSORS:
                st_path = os.path.join(args.output_dir, "car_brain.safetensors")
                state_dict_contig = {k: v.contiguous() for k, v in model.state_dict().items()}
                save_safetensors(state_dict_contig, st_path)

    # Save normalization stats and report
    norm_save_path = os.path.join(args.output_dir, "norm_stats.json")
    with open(norm_save_path, "w") as f:
        json.dump({
            "mean": norm_stats["mean"].tolist(),
            "std": norm_stats["std"].tolist(),
            "state_cols": STATE_COLS,
        }, f, indent=2)

    report_path = os.path.join(args.output_dir, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nPretraining Complete! Best Val L=64 MSE: {best_val_loss:.5f}")
    print(f"Saved model to: {args.output_dir}/car_brain.safetensors and car_brain.pt")


if __name__ == "__main__":
    main()
