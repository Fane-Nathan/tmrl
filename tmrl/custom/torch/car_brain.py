"""
TrackMania Causal Transformer "Car Brain" for In-Context Learning.

Implements an autoregressive sequence model designed for pretraining on multi-track
driving demonstrations (e.g., TMNF / TM2020) and evaluating in-context adaptation.

Features:
- TimestepEmbedder: Projects continuous telemetry into token embeddings.
- CausalTransformerTrunk: Multi-layer pre-norm causal self-attention.
- CarBrainPolicyHead: Continuous action prediction [gas, brake, steer] in [-1, 1]^3.
- Stateful Rolling Context: In-memory deque for real-time closed-loop rollouts.
- Adaptive vs. No-History Toggle: Supports the paper's no-history ablation.
- SafeTensors & PyTorch serialization: Clean weight export/import.
"""

from __future__ import annotations

import math
import os
from collections import deque
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from safetensors.torch import load_file as load_safetensors
    from safetensors.torch import save_file as save_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


class TimestepEmbedder(nn.Module):
    """
    Embeds multi-modal timestep features [state_t, action_{t-1}, reward_{t-1}, done_{t-1}]
    into a d_model dimensional token.
    """
    def __init__(
        self,
        state_dim: int = 15,
        action_dim: int = 3,
        d_model: int = 128,
        include_prev_action: bool = True,
        include_prev_reward: bool = True,
        include_prev_done: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.include_prev_action = include_prev_action
        self.include_prev_reward = include_prev_reward
        self.include_prev_done = include_prev_done

        total_input_dim = state_dim
        if include_prev_action:
            total_input_dim += action_dim
        if include_prev_reward:
            total_input_dim += 1
        if include_prev_done:
            total_input_dim += 1

        self.proj = nn.Sequential(
            nn.Linear(total_input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        states: torch.Tensor,
        prev_actions: Optional[torch.Tensor] = None,
        prev_rewards: Optional[torch.Tensor] = None,
        prev_dones: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            states: (B, T, state_dim) or (B, state_dim)
            prev_actions: (B, T, action_dim) or (B, action_dim)
            prev_rewards: (B, T, 1) or (B, 1)
            prev_dones: (B, T, 1) or (B, 1)
        Returns:
            tokens: (B, T, d_model) or (B, d_model)
        """
        parts = [states]
        B = states.shape[0]
        T = states.shape[1] if states.dim() == 3 else 1

        if self.include_prev_action:
            if prev_actions is None:
                prev_actions = torch.zeros(
                    (B, T, self.action_dim) if states.dim() == 3 else (B, self.action_dim),
                    device=states.device,
                    dtype=states.dtype,
                )
            parts.append(prev_actions)

        if self.include_prev_reward:
            if prev_rewards is None:
                prev_rewards = torch.zeros(
                    (B, T, 1) if states.dim() == 3 else (B, 1),
                    device=states.device,
                    dtype=states.dtype,
                )
            parts.append(prev_rewards)

        if self.include_prev_done:
            if prev_dones is None:
                prev_dones = torch.zeros(
                    (B, T, 1) if states.dim() == 3 else (B, 1),
                    device=states.device,
                    dtype=states.dtype,
                )
            parts.append(prev_dones)

        concat_input = torch.cat(parts, dim=-1)
        return self.proj(concat_input)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with pre-norm and dropout.
    Enforces causal masking so token t cannot attend to token t+k.
    """
    def __init__(self, d_model: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)

        if mask is None:
            # Fast causal FlashAttention on Tensor Cores
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(attn_out)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with causal attention and MLP."""
    def __init__(self, d_model: int = 128, n_heads: int = 4, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.mlp(self.ln2(x))
        return x


class CarBrainPolicyHead(nn.Module):
    """
    Maps transformer output tokens to continuous action predictions:
    [gas, brake, steer] in [-1, 1]^3.
    """
    def __init__(self, d_model: int = 128, action_dim: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raw = self.net(hidden_states)
        return torch.tanh(raw)


class CarBrain(nn.Module):
    """
    Full TrackMania Car Brain:
    - Ingests sequences of states, past actions, and rewards.
    - Causal transformer trunk processes context.
    - Policy head outputs action predictions.
    - Stateful deque for closed-loop in-context inference.
    - Adaptive toggle for No-History ablation.
    """
    def __init__(
        self,
        state_dim: int = 15,
        action_dim: int = 3,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 256,
        max_context_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_context_len = max_context_len

        self.embedder = TimestepEmbedder(
            state_dim=state_dim,
            action_dim=action_dim,
            d_model=d_model,
        )
        self.pos_emb = nn.Parameter(torch.zeros(1, max_context_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.policy_head = CarBrainPolicyHead(d_model=d_model, action_dim=action_dim)

        # In-context inference rolling buffers
        self._history_states: deque = deque(maxlen=max_context_len)
        self._history_prev_actions: deque = deque(maxlen=max_context_len)
        self._history_prev_rewards: deque = deque(maxlen=max_context_len)
        self._history_prev_dones: deque = deque(maxlen=max_context_len)

        # Flag for the paper's No-History ablation
        self.adaptive: bool = True

    def forward(
        self,
        states: torch.Tensor,
        prev_actions: Optional[torch.Tensor] = None,
        prev_rewards: Optional[torch.Tensor] = None,
        prev_dones: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Batch sequence forward pass (used during offline pretraining).
        states: (B, T, state_dim)
        Returns:
            actions: (B, T, action_dim) in [-1, 1]^3
        """
        B, T, _ = states.shape
        assert T <= self.max_context_len, f"Sequence length {T} exceeds max_context_len {self.max_context_len}"

        tokens = self.embedder(states, prev_actions, prev_rewards, prev_dones)  # (B, T, d_model)
        tokens = tokens + self.pos_emb[:, :T, :]

        x = tokens
        for block in self.blocks:
            x = block(x)
        h = self.ln_f(x)
        actions = self.policy_head(h)
        return actions

    def reset_context(self) -> None:
        """Clears the in-memory context history (call at episode resets)."""
        self._history_states.clear()
        self._history_prev_actions.clear()
        self._history_prev_rewards.clear()
        self._history_prev_dones.clear()

    def step_in_context(
        self,
        state: Union[np.ndarray, torch.Tensor],
        prev_action: Optional[Union[np.ndarray, torch.Tensor]] = None,
        prev_reward: float = 0.0,
        prev_done: bool = False,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """
        Real-time closed-loop action selection at 20 Hz.
        Uses rolling context buffer if self.adaptive=True.
        If self.adaptive=False (No-History mode), clears context before stepping.
        """
        if not self.adaptive:
            self.reset_context()

        # Convert state to tensor
        if isinstance(state, np.ndarray):
            state_t = torch.from_numpy(state).float()
        else:
            state_t = state.float()

        if prev_action is None:
            prev_act_t = torch.zeros(self.action_dim, dtype=torch.float32)
        elif isinstance(prev_action, np.ndarray):
            prev_act_t = torch.from_numpy(prev_action).float()
        else:
            prev_act_t = prev_action.float()

        prev_rew_t = torch.tensor([prev_reward], dtype=torch.float32)
        prev_done_t = torch.tensor([1.0 if prev_done else 0.0], dtype=torch.float32)

        # Append to rolling history
        self._history_states.append(state_t)
        self._history_prev_actions.append(prev_act_t)
        self._history_prev_rewards.append(prev_rew_t)
        self._history_prev_dones.append(prev_done_t)

        # Assemble sequence tensor
        states_seq = torch.stack(list(self._history_states), dim=0).unsqueeze(0)  # (1, T, D)
        prev_actions_seq = torch.stack(list(self._history_prev_actions), dim=0).unsqueeze(0)
        prev_rewards_seq = torch.stack(list(self._history_prev_rewards), dim=0).unsqueeze(0)
        prev_dones_seq = torch.stack(list(self._history_prev_dones), dim=0).unsqueeze(0)

        dev = device if device is not None else next(self.parameters()).device
        states_seq = states_seq.to(dev)
        prev_actions_seq = prev_actions_seq.to(dev)
        prev_rewards_seq = prev_rewards_seq.to(dev)
        prev_dones_seq = prev_dones_seq.to(dev)

        with torch.no_grad():
            actions_seq = self.forward(
                states_seq,
                prev_actions=prev_actions_seq,
                prev_rewards=prev_rewards_seq,
                prev_dones=prev_dones_seq,
            )
            # Latest timestep action
            act = actions_seq[0, -1].cpu().numpy()

        return act

    act = step_in_context

    def save_pretrained(self, save_dir: str, filename_prefix: str = "car_brain") -> Dict[str, str]:
        """Saves model weights in SafeTensors and PyTorch formats along with config.json."""
        import json
        os.makedirs(save_dir, exist_ok=True)
        paths = {}

        # Save configuration
        config = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "max_context_len": self.max_context_len,
        }
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        paths["config"] = config_path

        # PyTorch checkpoint
        pt_path = os.path.join(save_dir, f"{filename_prefix}.pt")
        torch.save(self.state_dict(), pt_path)
        paths["pt"] = pt_path

        # SafeTensors format
        if HAS_SAFETENSORS:
            st_path = os.path.join(save_dir, f"{filename_prefix}.safetensors")
            state_dict = {k: v.contiguous() for k, v in self.state_dict().items()}
            save_safetensors(state_dict, st_path)
            paths["safetensors"] = st_path

        return paths

    @classmethod
    def from_pretrained(
        cls,
        filepath: str,
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        d_model: Optional[int] = None,
        n_layers: Optional[int] = None,
        n_heads: Optional[int] = None,
        d_ff: Optional[int] = None,
        max_context_len: Optional[int] = None,
        device: str = "cpu",
    ) -> "CarBrain":
        """Loads a CarBrain instance from .safetensors or .pt, auto-reading config.json if available."""
        import json
        model_dir = os.path.dirname(os.path.abspath(filepath))
        cfg_path = os.path.join(model_dir, "config.json")

        cfg = {}
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r") as f:
                    cfg = json.load(f)
            except Exception:
                pass

        final_state_dim = state_dim or cfg.get("state_dim", 15)
        final_action_dim = action_dim or cfg.get("action_dim", 3)
        final_d_model = d_model or cfg.get("d_model", 128)
        final_n_layers = n_layers or cfg.get("n_layers", 4)
        final_n_heads = n_heads or cfg.get("n_heads", 4)
        final_d_ff = d_ff or cfg.get("d_ff", 256)
        final_max_context_len = max_context_len or cfg.get("max_context_len", 64)

        model = cls(
            state_dim=final_state_dim,
            action_dim=final_action_dim,
            d_model=final_d_model,
            n_layers=final_n_layers,
            n_heads=final_n_heads,
            d_ff=final_d_ff,
            max_context_len=final_max_context_len,
        )

        if filepath.endswith(".safetensors"):
            if not HAS_SAFETENSORS:
                raise ImportError("safetensors is required to load .safetensors file.")
            state_dict = load_safetensors(filepath, device=device)
        else:
            state_dict = torch.load(filepath, map_location=device)

        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        return model


class MultiModalCarBrain(nn.Module):
    """
    Multi-Modal Foundation Brain for TrackMania 2020.
    Fuses real-time 96x96 4-frame visual camera input (via TMRL native CNN)
    with the 1,000,000-replay Transformer physics backbone.
    """
    def __init__(
        self,
        img_channels: int = 4,
        img_size: Tuple[int, int] = (96, 96),
        state_dim: int = 15,
        action_dim: int = 3,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 512,
        max_context_len: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.max_context_len = max_context_len

        # 1. Native TMRL CNN Visual Encoder: 4x96x96 -> 128-dim
        # conv1: 4 -> 64 (kernel 8, stride 2) -> 45x45
        # conv2: 64 -> 64 (kernel 4, stride 2) -> 21x21
        # conv3: 64 -> 128 (kernel 4, stride 2) -> 9x9
        # conv4: 128 -> 128 (kernel 4, stride 2) -> 3x3
        # 128 * 3 * 3 = 1152 -> 128
        self.conv1 = nn.Conv2d(img_channels, 64, kernel_size=8, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=4, stride=2)
        self.visual_fc = nn.Linear(128 * 3 * 3, 128)
        self.visual_ln = nn.LayerNorm(128)

        # 2. Kinematics & Inputs Projection: (15 state + 3 action + 1 rew + 1 done = 20) -> 128-dim
        self.kinematics_proj = nn.Sequential(
            nn.Linear(state_dim + action_dim + 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # 3. Multi-Modal Token Fusion: (128 visual + 128 kinematics) -> d_model (256)
        self.fusion_proj = nn.Linear(128 + 128, d_model)

        # 4. Foundation Transformer Backbone (from 1,000,000 replays)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_context_len, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        # 5. Policy Head
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

        # In-context rolling inference queues
        self._history_imgs: deque = deque(maxlen=max_context_len)
        self._history_states: deque = deque(maxlen=max_context_len)
        self._history_prev_actions: deque = deque(maxlen=max_context_len)

    def encode_visual(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: (B, C, H, W) or (B, T, C, H, W) in [0, 1]
        Returns: (B, 128) or (B, T, 128)
        """
        orig_shape = imgs.shape
        if imgs.dim() == 5:
            B, T, C, H, W = orig_shape
            x = imgs.view(B * T, C, H, W)
        else:
            B, T = orig_shape[0], 1
            x = imgs

        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = h.reshape(h.size(0), -1)
        v = self.visual_ln(F.relu(self.visual_fc(h)))

        if len(orig_shape) == 5:
            v = v.view(B, T, 128)
        return v

    def forward(
        self,
        imgs: torch.Tensor,
        states: torch.Tensor,
        prev_actions: Optional[torch.Tensor] = None,
        prev_rewards: Optional[torch.Tensor] = None,
        prev_dones: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        imgs: (B, T, 4, 96, 96)
        states: (B, T, state_dim)
        Returns: actions (B, T, action_dim) in [-1, 1]^3; optionally
        (actions, pre-tanh logits) for saturation-resistant trigger supervision.
        """
        B, T, _ = states.shape
        v_tokens = self.encode_visual(imgs)  # (B, T, 128)

        if prev_actions is None:
            prev_actions = torch.zeros(B, T, self.action_dim, device=states.device, dtype=states.dtype)
        if prev_rewards is None:
            prev_rewards = torch.zeros(B, T, 1, device=states.device, dtype=states.dtype)
        if prev_dones is None:
            prev_dones = torch.zeros(B, T, 1, device=states.device, dtype=states.dtype)

        kin_input = torch.cat([states, prev_actions, prev_rewards, prev_dones], dim=-1)
        k_tokens = self.kinematics_proj(kin_input)  # (B, T, 128)

        # Fused multi-modal tokens
        fused = self.fusion_proj(torch.cat([v_tokens, k_tokens], dim=-1))  # (B, T, 256)
        tokens = fused + self.pos_emb[:, :T, :]

        x = tokens
        for block in self.blocks:
            x = block(x)
        h = self.ln_f(x)
        if return_logits:
            # Keep checkpoint names and the deployed inference path unchanged.
            # FP32 tanh avoids half-precision endpoint rounding; trigger BCE on
            # logits also supplies useful gradients for confidently wrong gas.
            logits = self.policy_head[:-1](h).float()
            return self.policy_head[-1](logits), logits
        return self.policy_head(h)

    def reset_context(self) -> None:
        """Clears in-context history at lap start / reset."""
        self._history_imgs.clear()
        self._history_states.clear()
        self._history_prev_actions.clear()

    def step_in_context(
        self,
        img: Union[np.ndarray, torch.Tensor],
        state: Union[np.ndarray, torch.Tensor],
        prev_action: Optional[Union[np.ndarray, torch.Tensor]] = None,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """
        Real-time closed loop step at 20 Hz.
        img: (4, 96, 96) in [0, 1]
        state: (15,) kinematics vector
        """
        if isinstance(img, np.ndarray):
            img_t = torch.from_numpy(img).float()
        else:
            img_t = img.float()

        if isinstance(state, np.ndarray):
            state_t = torch.from_numpy(state).float()
        else:
            state_t = state.float()

        if prev_action is None:
            prev_act_t = torch.zeros(self.action_dim, dtype=torch.float32)
        elif isinstance(prev_action, np.ndarray):
            prev_act_t = torch.from_numpy(prev_action).float()
        else:
            prev_act_t = prev_action.float()

        self._history_imgs.append(img_t)
        self._history_states.append(state_t)
        self._history_prev_actions.append(prev_act_t)

        imgs_seq = torch.stack(list(self._history_imgs), dim=0).unsqueeze(0)    # (1, T, 4, 96, 96)
        states_seq = torch.stack(list(self._history_states), dim=0).unsqueeze(0)  # (1, T, 15)
        actions_seq = torch.stack(list(self._history_prev_actions), dim=0).unsqueeze(0)  # (1, T, 3)

        dev = device if device is not None else next(self.parameters()).device
        imgs_seq = imgs_seq.to(dev)
        states_seq = states_seq.to(dev)
        actions_seq = actions_seq.to(dev)

        with torch.no_grad():
            act_seq = self.forward(imgs_seq, states_seq, prev_actions=actions_seq)
            return act_seq[0, -1].cpu().numpy()

    @classmethod
    def from_foundation(
        cls,
        foundation_weights_path: str = "weights/car_brain_1m_curriculum/car_brain_latest.pt",
        device: Optional[str] = None,
    ) -> "MultiModalCarBrain":
        """
        Constructs MultiModalCarBrain and initializes its Transformer backbone and policy head
        directly from the 1,000,000-replay pre-trained foundation weights.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = cls(
            img_channels=4,
            img_size=(96, 96),
            state_dim=15,
            action_dim=3,
            d_model=256,
            n_layers=6,
            n_heads=8,
            d_ff=512,
            max_context_len=64,
        )

        if os.path.exists(foundation_weights_path):
            print(f"[+] Loading 1,000,000 foundation weights into MultiModalCarBrain backbone from: {foundation_weights_path}")
            base_dict = torch.load(foundation_weights_path, map_location="cpu")
            model_dict = model.state_dict()

            # Transfer matching transformer blocks, LayerNorm, and policy head
            transferred = 0
            for k, v in base_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    model_dict[k] = v
                    transferred += 1

            model.load_state_dict(model_dict)
            print(f"[+] Successfully transferred {transferred} foundation weight tensors into Transformer backbone!")
        else:
            print(f"[-] Foundation weights '{foundation_weights_path}' not found, initializing fresh.")

        model.to(device)
        return model
