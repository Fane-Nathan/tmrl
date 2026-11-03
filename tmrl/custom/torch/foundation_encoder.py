"""
Foundation Visual-Telemetry Encoder for PyTorch Dreamer.
Bridges the 1,000,000-replay MultiModalCarBrain perception and kinematics
representations into Dreamer's Latent RSSM World Model.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tmrl.custom.torch.car_brain import MultiModalCarBrain


class FoundationVisualTelemetryEncoder(nn.Module):
    """
    PyTorch Visual-Telemetry Encoder powered by the 1M Foundation Model.
    Reuses MultiModalCarBrain's pre-trained 4-layer visual CNN, kinematics
    tokenizer, and multi-modal fusion layer to provide structured 128D
    latent observation embeddings e_t to Dreamer's RSSM world model.
    """
    def __init__(
        self,
        img_channels: int = 4,
        img_height: int = 96,
        img_width: int = 96,
        latent_dim: int = 128,
        model_path: str = "weights/car_brain_1m_curriculum/car_brain_multimodal.pt",
        base_foundation_path: str = "weights/car_brain_1m_curriculum/car_brain_latest.pt",
        freeze_foundation: bool = True,
        reload_on_change: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_height = img_height
        self.img_width = img_width
        self.freeze_foundation = bool(freeze_foundation)
        self.reload_on_change = bool(reload_on_change)

        # Load MultiModalCarBrain with 1M foundation weights
        model_file = Path(model_path)
        base_file = Path(base_foundation_path)

        if model_file.exists():
            print(f"[+] FoundationEncoder: Loading pre-trained MultiModalCarBrain from {model_file}")
            self.brain = MultiModalCarBrain(d_model=256, n_layers=6, n_heads=8)
            self.brain.load_state_dict(
                self._read_state_dict(model_file, "cpu"), strict=True
            )
        elif base_file.exists():
            print(f"[+] FoundationEncoder: Initializing from 1M foundation physics weights: {base_file}")
            self.brain = MultiModalCarBrain.from_foundation(str(base_file), device="cpu")
        else:
            print("[-] FoundationEncoder: No pre-trained weights found! Initializing fresh MultiModalCarBrain.")
            self.brain = MultiModalCarBrain(d_model=256, n_layers=6, n_heads=8)

        # Projection from 256-dim fused multi-modal token to Dreamer's 128-dim latent space
        self.latent_proj = nn.Sequential(
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )

        self.model_file = Path(model_path)
        self._last_mtime = self.model_file.stat().st_mtime if self.model_file.exists() else 0.0
        self._last_check_time = 0.0
        self._last_foundation_action = None
        self._token_history = []

        self.set_foundation_trainable(not self.freeze_foundation)
        if self.freeze_foundation:
            print("[+] FoundationEncoder: Foundation backbone frozen.")

    @staticmethod
    def _read_state_dict(model_file: Path, device):
        try:
            state = torch.load(str(model_file), map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(str(model_file), map_location=device)
        if not isinstance(state, dict):
            raise TypeError(
                f"Foundation checkpoint must contain a state dict, got {type(state)!r}"
            )
        return state

    def set_foundation_trainable(self, trainable: bool) -> None:
        """Freeze/unfreeze only the demonstrated driving policy backbone."""
        self.freeze_foundation = not bool(trainable)
        for parameter in self.brain.parameters():
            parameter.requires_grad_(bool(trainable))
        if self.freeze_foundation:
            # The rollout worker does not call ``eval()`` on actors.  Keep dropout
            # and normalization in the demonstrated policy deterministic anyway.
            self.brain.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "freeze_foundation", False):
            self.brain.eval()
        return self

    def reset_context(self) -> None:
        """Clear the causal action context at an environment episode boundary."""
        self._token_history = []
        self._last_foundation_action = None

    def reload_foundation_weights(self, model_path: Optional[str] = None) -> None:
        """Restore the canonical demonstrated policy without touching Dreamer."""
        model_file = Path(model_path) if model_path is not None else self.model_file
        if not model_file.exists():
            raise FileNotFoundError(
                f"Canonical foundation checkpoint does not exist: {model_file}"
            )
        device = next(self.brain.parameters()).device
        state = self._read_state_dict(model_file, device)
        self.brain.load_state_dict(state, strict=True)
        self.model_file = model_file
        self._last_mtime = model_file.stat().st_mtime
        self.reset_context()
        self.set_foundation_trainable(not self.freeze_foundation)

    def _maybe_reload_foundation_weights(self) -> None:
        """Hot-reload a newer canonical checkpoint during no-grad inference."""
        if not getattr(self, "reload_on_change", True) or torch.is_grad_enabled():
            return
        import time

        now = time.time()
        if now - getattr(self, "_last_check_time", 0.0) <= 5.0:
            return
        self._last_check_time = now
        model_file = getattr(self, "model_file", None)
        if model_file is None or not model_file.exists():
            return
        try:
            mtime = model_file.stat().st_mtime
            if mtime > getattr(self, "_last_mtime", 0.0):
                self.reload_foundation_weights()
                logging.info(
                    "FoundationEncoder hot-reloaded canonical checkpoint %s",
                    model_file,
                )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            logging.warning(
                "FoundationEncoder could not hot-reload %s: %s",
                model_file,
                exc,
            )

    def forward(self, obs_tuple) -> torch.Tensor:
        """
        obs_tuple: (speed, gear, rpm, images, [act1, act2])
        images shape: (B, 4, 96, 96) in [0, 1] or [0, 255]
        Returns:
            embedding: (B, latent_dim)
        """
        # The RolloutWorker invokes actors under torch.no_grad(), even though it
        # leaves the module's ``training`` flag set.  Key reloads to grad mode so
        # a trainer can never swap weights in the middle of backpropagation.
        self._maybe_reload_foundation_weights()

        device = next(self.parameters()).device
        speed = torch.as_tensor(obs_tuple[0], device=device)
        gear = torch.as_tensor(obs_tuple[1], device=device)
        rpm = torch.as_tensor(obs_tuple[2], device=device)
        imgs = torch.as_tensor(obs_tuple[3], device=device)

        # RTGym appends its deque oldest -> newest. Training conditions on
        # a[t-1], so using obs[4] when there are two actions adds one full
        # control period of delay and disagrees with the demonstration loader.
        act1 = torch.as_tensor(obs_tuple[-1], device=device) if len(obs_tuple) > 4 and obs_tuple[-1] is not None else None

        if speed.dim() == 1:
            speed = speed.unsqueeze(-1)
        if gear.dim() == 1:
            gear = gear.unsqueeze(-1)
        if rpm.dim() == 1:
            rpm = rpm.unsqueeze(-1)

        speed = speed.float()
        gear = gear.float()
        rpm = rpm.float()

        # Format image tensors: (B, 4, 96, 96) in [0, 1]
        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(0)
        if imgs.dim() == 4 and imgs.shape[-1] == 4:
            # (B, H, W, C) -> (B, C, H, W)
            imgs = imgs.permute(0, 3, 1, 2)

        if imgs.dtype == torch.uint8 or imgs.max() > 1.5:
            imgs = imgs.float() / 255.0
        else:
            imgs = imgs.float()

        B = imgs.shape[0]
        if imgs.shape[-2:] != (self.img_height, self.img_width):
            imgs = F.interpolate(
                imgs,
                size=(self.img_height, self.img_width),
                mode="bilinear",
                align_corners=False,
            )

        # 1. MultiModal Visual Tokenizer (4-layer CNN + LayerNorm)
        v_tokens = self.brain.encode_visual(imgs)  # (B, 128)

        # 2. Kinematics 15D State Vector matching Foundation pre-training
        # Target normalization matching target_track_demos.pt: speed / 100.0, gear 1-4, rpm / 10000.0
        if speed.max() > 1.5:  # Raw km/h (e.g. 50.0 - 400.0)
            norm_speed = speed / 100.0
        elif speed.max() < 0.8:  # TMRL obs_preprocessor already divided by 1000.0 (e.g. 0.05 = 50 km/h)
            norm_speed = speed * 10.0
        else:
            norm_speed = speed

        norm_rpm = rpm / 10000.0 if rpm.max() > 1.5 else rpm
        norm_gear = gear if gear.max() > 1.5 else gear * 10.0

        states = torch.zeros((B, 15), dtype=torch.float32, device=device)
        states[:, 0:1] = norm_speed
        states[:, 3:4] = norm_gear
        states[:, 4:5] = norm_rpm
        states[:, 9:13] = 1.0  # surface/tires

        if act1 is not None and isinstance(act1, torch.Tensor):
            if act1.dim() == 1:
                act1 = act1.unsqueeze(0)
            act1 = act1.view(B, -1)
            prev_act = act1[:, :3].float() if act1.shape[-1] >= 3 else torch.zeros((B, 3), dtype=torch.float32, device=device)
        else:
            prev_act = torch.zeros((B, 3), dtype=torch.float32, device=device)

        # Kinematics input: [state (15), action (3), reward (1), done (1)] -> 20D
        kin_input = torch.cat([
            states,
            prev_act,
            torch.zeros(B, 1, device=device, dtype=torch.float32),
            torch.zeros(B, 1, device=device, dtype=torch.float32),
        ], dim=-1)

        k_tokens = self.brain.kinematics_proj(kin_input)  # (B, 128)

        # 3. Multi-modal Fusion
        fused = self.brain.fusion_proj(torch.cat([v_tokens, k_tokens], dim=-1))  # (B, 256)

        # 4. Foundation Action Prediction:
        # Pretrained causal transformer backbone predicts expert action prior [gas, brake, steer] in [-1, 1]
        # Start-line launch safeguard: at standstill (<5 km/h / norm_speed < 0.05), guarantee straight full-gas acceleration
        # World-model training calls this encoder on flattened B*T tensors.  A
        # streaming transformer history is meaningful only for live no-grad
        # inference and previously mixed incompatible batch shapes in training.
        if torch.is_grad_enabled():
            self._last_foundation_action = None
        elif (norm_speed < 0.05).all():
            self.reset_context()
            self._last_foundation_action = torch.tensor(
                [[1.0, -1.0, 0.0]], dtype=torch.float32, device=device
            ).expand(B, 3)
        else:
            if self._token_history and self._token_history[0].shape[0] != B:
                self.reset_context()
            # Append current fused token to rolling temporal sequence (maxlen=16)
            self._token_history.append(fused.detach())
            if len(self._token_history) > 16:
                self._token_history.pop(0)

            with torch.no_grad():
                T_ctx = len(self._token_history)
                ctx_tokens = torch.stack(self._token_history, dim=1)  # (B, T_ctx, 256)
                tokens = ctx_tokens + self.brain.pos_emb[:, :T_ctx, :]
                x = tokens
                for block in self.brain.blocks:
                    x = block(x)
                h = self.brain.ln_f(x)
                # Output latest step action
                self._last_foundation_action = self.brain.policy_head(h)[:, -1].detach()

        # 5. Latent Embedding for RSSM
        embedding = self.latent_proj(fused)  # (B, 128)
        return embedding

    def get_foundation_action(self) -> Optional[torch.Tensor]:
        """Returns the most recently inferred foundation action tensor [B, 3] in [-1, 1]."""
        return getattr(self, "_last_foundation_action", None)
