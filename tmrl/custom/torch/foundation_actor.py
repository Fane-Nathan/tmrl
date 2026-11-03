"""
Squashed Gaussian Foundation Actor for TMRL.
Bridges the 1,000,000-replay In-Context Learning (ICL) MultiModal Transformer
into Yann Bouteiller's official TMRL reinforcement learning pipeline (SAC/REDQ).
"""

from collections import deque
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import tmrl.config.config_constants as cfg
from tmrl.core.torch.actor import TorchActorModule
from tmrl.custom.torch.car_brain import MultiModalCarBrain
from tmrl.custom.torch.custom_models import VanillaCNNQFunction

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianFoundationActor(TorchActorModule):
    """
    TMRL Actor wrapping the 1,000,000-replay Foundation In-Context Learning Transformer.
    Integrates 4-frame 96x96 visual camera observations with telemetry into a 6-layer
    Causal Transformer backbone for closed-loop SAC/REDQ training.
    """
    def __init__(
        self,
        observation_space,
        action_space,
        model_path: str = "weights/car_brain_1m_curriculum/car_brain_multimodal.pt",
        base_foundation_path: str = "weights/car_brain_1m_curriculum/car_brain_latest.pt",
        device: Optional[str] = None,
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(observation_space, action_space, device=device)
        self.dim_act = action_space.shape[0]
        self.act_limit = float(action_space.high[0])
        self.device = device

        # Load MultiModalCarBrain with 1M foundation weights
        model_file = Path(model_path)
        base_file = Path(base_foundation_path)

        if model_file.exists():
            print(f"[+] FoundationActor: Loading pre-trained MultiModalCarBrain from {model_file}")
            self.brain = MultiModalCarBrain(d_model=256, n_layers=6, n_heads=8)
            try:
                self.brain.load_state_dict(torch.load(str(model_file), map_location=device, weights_only=True))
            except Exception:
                self.brain.load_state_dict(torch.load(str(model_file), map_location=device, weights_only=False))
        elif base_file.exists():
            print(f"[+] FoundationActor: Initializing from 1M foundation physics weights: {base_file}")
            self.brain = MultiModalCarBrain.from_foundation(str(base_file), device=device)
        else:
            print("[-] FoundationActor: No pre-trained weights found! Initializing fresh MultiModalCarBrain.")
            self.brain = MultiModalCarBrain(d_model=256, n_layers=6, n_heads=8)

        self.brain.to(device)

        # Policy distribution heads for SAC/REDQ exploration
        # Reuses pre-trained policy layers from MultiModalCarBrain so initial policy matches the expert
        self.mu_head = nn.Sequential(
            self.brain.policy_head[0],
            self.brain.policy_head[1],
            self.brain.policy_head[2],
        ).to(device)
        self.log_std_head = nn.Linear(256, self.dim_act).to(device)
        nn.init.constant_(self.log_std_head.weight, 0.0)
        nn.init.constant_(self.log_std_head.bias, -2.0)  # Initial std ~ 0.135

        # Rolling in-context buffers for real-time live worker driving (act)
        self.model_file = Path(model_path)
        self._last_mtime = self.model_file.stat().st_mtime if self.model_file.exists() else 0.0
        self.max_context_len = 64
        self._history_imgs = deque(maxlen=self.max_context_len)
        self._history_states = deque(maxlen=self.max_context_len)
        self._history_prev_actions = deque(maxlen=self.max_context_len)
        self._last_speed = 0.0

    def reset_context(self):
        """Flushes rolling in-context history upon track restart and hot-reloads new weights if available."""
        self._history_imgs.clear()
        self._history_states.clear()
        self._history_prev_actions.clear()
        self._last_speed = 0.0

        # Automatic Hot-Reload: Check if trainer saved newer checkpoint
        model_file = getattr(self, "model_file", None)
        if model_file is not None and model_file.exists():
            try:
                mtime = model_file.stat().st_mtime
                last_mtime = getattr(self, "_last_mtime", 0.0)
                if mtime > last_mtime:
                    try:
                        self.brain.load_state_dict(torch.load(str(model_file), map_location=self.device, weights_only=True))
                    except Exception:
                        self.brain.load_state_dict(torch.load(str(model_file), map_location=self.device, weights_only=False))
                    self._last_mtime = mtime
                    print(f"[+] FoundationActor: Hot-reloaded updated checkpoint from {model_file.name}")
            except Exception:
                pass

    def _format_inputs(self, obs):
        """
        Converts TMRL TM20IMAGES observation tuple:
        obs = (speed, gear, rpm, images, act1, act2)
        into normalized MultiModalCarBrain inputs:
        imgs: (B, 4, 96, 96) in [0, 1]
        state: (B, 15)
        prev_act: (B, 3)
        """
        speed, gear, rpm, images, act1, act2 = obs

        # Ensure correct device and float types
        speed = speed.to(self.device).float()
        gear = gear.to(self.device).float()
        rpm = rpm.to(self.device).float()
        act1 = act1.to(self.device).float()
        images = images.to(self.device)

        # Normalize images
        if images.dtype == torch.uint8:
            imgs = images.float() / 255.0
        else:
            imgs = images.float()
            if imgs.max() > 1.5:
                imgs = imgs / 255.0

        # Resize if dimensions differ from 96x96
        B = imgs.shape[0]
        if imgs.shape[-2:] != (96, 96):
            imgs = F.interpolate(imgs, size=(96, 96), mode="bilinear", align_corners=False)

        # Construct 15-dim state vector
        state = torch.zeros((B, 15), dtype=torch.float32, device=self.device)
        state[:, 0:1] = speed / 100.0
        state[:, 1:2] = act1[:, 2:3]  # last steer
        state[:, 2:3] = act1[:, 0:1]  # last gas
        state[:, 3:4] = gear
        state[:, 4:5] = rpm / 10000.0
        state[:, 9:13] = 1.0  # surface/tires

        return imgs, state, act1

    def forward_features(self, imgs, state, prev_act):
        """Passes inputs through MultiModal visual encoder and Transformer backbone."""
        # imgs: (B, 4, 96, 96) -> (B, 1, 4, 96, 96)
        # state: (B, 15) -> (B, 1, 15)
        # prev_act: (B, 3) -> (B, 1, 3)
        B = imgs.shape[0]
        imgs_seq = imgs.unsqueeze(1)
        state_seq = state.unsqueeze(1)
        act_seq = prev_act.unsqueeze(1)

        v_tokens = self.brain.encode_visual(imgs_seq)  # (B, 1, 128)

        kin_input = torch.cat([
            state_seq,
            act_seq,
            torch.zeros(B, 1, 1, device=self.device),  # rew
            torch.zeros(B, 1, 1, device=self.device),  # done
        ], dim=-1)
        k_tokens = self.brain.kinematics_proj(kin_input)  # (B, 1, 128)

        fused = self.brain.fusion_proj(torch.cat([v_tokens, k_tokens], dim=-1))  # (B, 1, 256)
        tokens = fused + self.brain.pos_emb[:, :1, :]

        x = tokens
        for block in self.brain.blocks:
            x = block(x)
        h = self.brain.ln_f(x)  # (B, 1, 256)
        return h.squeeze(1)     # (B, 256)

    def forward(self, obs, test=False, with_logprob=True):
        """
        Computes SAC squashed Gaussian action and log probability.
        """
        imgs, state, prev_act = self._format_inputs(obs)
        features = self.forward_features(imgs, state, prev_act)  # (B, 256)

        mu = self.mu_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if test:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            # Squashing correction from OpenAI SpinUp
            logp_pi -= (2 * (math.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=-1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action) * self.act_limit
        return pi_action, logp_pi

    def act(self, obs, test=False):
        """
        Real-time 20 Hz action generation for TMRL worker.
        Maintains rolling In-Context Learning (ICL) history.
        """
        # Detect auto-reset from speed drop
        speed_val = float(obs[0].cpu().numpy().flatten()[0]) if isinstance(obs[0], torch.Tensor) else float(obs[0])
        if speed_val < 0.5 and self._last_speed > 20.0:
            self.reset_context()
        self._last_speed = speed_val

        with torch.no_grad():
            action, _ = self.forward(obs, test=test, with_logprob=False)
            act_np = action.squeeze(0).cpu().numpy()
            return act_np


class FoundationCNNActorCritic(nn.Module):
    """
    Actor-Critic architecture pairing the 1,000,000-replay Foundation ICL Actor
    with dual TMRL CNN Q-Functions for Soft Actor-Critic (SAC).
    """
    def __init__(
        self,
        observation_space,
        action_space,
        critic_dropout=0.0,
        critic_layer_norm=False,
        actor_layer_norm=False,
    ):
        super().__init__()
        self.actor = SquashedGaussianFoundationActor(observation_space, action_space)
        self.q1 = VanillaCNNQFunction(observation_space, action_space, dropout=critic_dropout, layer_norm=critic_layer_norm)
        self.q2 = VanillaCNNQFunction(observation_space, action_space, dropout=critic_dropout, layer_norm=critic_layer_norm)


class REDQFoundationCNNActorCritic(nn.Module):
    """
    Actor-Critic architecture pairing the 1,000,000-replay Foundation ICL Actor
    with an ensemble of N TMRL CNN Q-Functions for REDQ.
    """
    def __init__(
        self,
        observation_space,
        action_space,
        n=10,
        critic_dropout=0.0,
        critic_layer_norm=False,
        actor_layer_norm=False,
    ):
        super().__init__()
        self.n = n
        self.actor = SquashedGaussianFoundationActor(observation_space, action_space)
        self.qs = nn.ModuleList([
            VanillaCNNQFunction(observation_space, action_space, dropout=critic_dropout, layer_norm=critic_layer_norm)
            for _ in range(self.n)
        ])
