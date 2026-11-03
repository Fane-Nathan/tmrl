"""
TMRL ActorModule wrapper for the Causal Transformer Car Brain.

Integrates `CarBrain` into TMRL's distributed rollout and training loop.
Maintains rolling context during 20 Hz execution and supports the
No-History ablation (adaptive vs. no-history) directly in TrackMania.
"""

from __future__ import annotations

from typing import Optional, Union, Tuple
import numpy as np
import torch
import torch.nn as nn

from tmrl.core.torch.actor import TorchActorModule
from tmrl.custom.torch.car_brain import CarBrain


class CarBrainActor(TorchActorModule):
    """
    TMRL ActorModule backed by a Causal Transformer Car Brain.
    
    Compatible with TMRL rollout workers and evaluation loops.
    """
    def __init__(
        self,
        observation_space,
        action_space,
        state_dim: int = 15,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 256,
        max_context_len: int = 64,
        pretrained_path: Optional[str] = None,
        device: str = "cpu",
    ):
        super().__init__(observation_space, action_space, device=device)
        self.action_dim = action_space.shape[0]
        self.state_dim = state_dim
        self.device = device

        self.car_brain = CarBrain(
            state_dim=state_dim,
            action_dim=self.action_dim,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_context_len=max_context_len,
        ).to(device)

        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)

        self._last_action = np.zeros(self.action_dim, dtype=np.float32)

    def load_pretrained(self, filepath: str) -> None:
        """Loads weights from .safetensors or .pt."""
        loaded = CarBrain.from_pretrained(
            filepath=filepath,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            d_model=self.car_brain.d_model,
            n_layers=self.car_brain.n_layers,
            n_heads=self.car_brain.n_heads,
            max_context_len=self.car_brain.max_context_len,
            device=self.device,
        )
        self.car_brain.load_state_dict(loaded.state_dict())
        self.car_brain.to(self.device)

    def set_adaptive(self, adaptive: bool) -> None:
        """Enables or disables in-context history (No-History ablation)."""
        self.car_brain.adaptive = adaptive
        if not adaptive:
            self.car_brain.reset_context()

    def reset_context(self) -> None:
        """Resets in-context rolling history (call on episode reset)."""
        self.car_brain.reset_context()
        self._last_action = np.zeros(self.action_dim, dtype=np.float32)

    def _extract_state_vector(self, obs) -> torch.Tensor:
        """
        Extracts or formats a 1D state tensor of size (state_dim) from TMRL obs.
        Handles both tuple observations: (speed, gear, rpm, images/telem, act1, act2)
        and flat tensors.
        """
        if isinstance(obs, (tuple, list)):
            # Typical TMRL observation tuple:
            # (speed, gear, rpm, ...) + act1, act2
            scalars = []
            for item in obs:
                if isinstance(item, torch.Tensor):
                    flat = torch.flatten(item, start_dim=1 if item.dim() > 1 else 0)
                    scalars.append(flat)
                elif isinstance(item, np.ndarray):
                    t = torch.from_numpy(item).float().to(self.device)
                    flat = torch.flatten(t, start_dim=1 if t.dim() > 1 else 0)
                    scalars.append(flat)
            cat_feat = torch.cat(scalars, dim=-1)
            # Match state_dim by slicing or padding
            feat_dim = cat_feat.shape[-1]
            if feat_dim > self.state_dim:
                return cat_feat[:, :self.state_dim]
            elif feat_dim < self.state_dim:
                pad = torch.zeros((cat_feat.shape[0], self.state_dim - feat_dim), device=cat_feat.device)
                return torch.cat([cat_feat, pad], dim=-1)
            return cat_feat
        elif isinstance(obs, torch.Tensor):
            flat = torch.flatten(obs, start_dim=1 if obs.dim() > 1 else 0)
            feat_dim = flat.shape[-1]
            if feat_dim > self.state_dim:
                return flat[:, :self.state_dim]
            elif feat_dim < self.state_dim:
                pad = torch.zeros((flat.shape[0], self.state_dim - feat_dim), device=flat.device)
                return torch.cat([flat, pad], dim=-1)
            return flat
        else:
            arr = np.asarray(obs, dtype=np.float32).reshape(-1)
            feat_dim = len(arr)
            if feat_dim > self.state_dim:
                arr = arr[:self.state_dim]
            elif feat_dim < self.state_dim:
                arr = np.pad(arr, (0, self.state_dim - feat_dim))
            return torch.from_numpy(arr).unsqueeze(0).to(self.device)

    def forward(self, obs, test=False):
        """
        Batched forward pass conforming to TMRL TorchActorModule.
        """
        state_vec = self._extract_state_vector(obs)  # (B, state_dim)
        # If running batched sequence (B, T, D) vs single step (B, D)
        if state_vec.dim() == 2:
            state_seq = state_vec.unsqueeze(1)  # (B, 1, D)
        else:
            state_seq = state_vec

        actions = self.car_brain(state_seq)
        return actions[:, -1], None  # (action, logp_pi)

    def act(self, obs, test=False) -> np.ndarray:
        """
        Real-time closed-loop action step at 20 Hz.
        Called by TMRL worker during episode rollouts.
        """
        state_vec = self._extract_state_vector(obs)  # (1, state_dim)
        state_np = state_vec[0].detach().cpu().numpy()

        action = self.car_brain.step_in_context(
            state=state_np,
            prev_action=self._last_action,
            device=torch.device(self.device),
        )
        self._last_action = action.copy()
        return action
