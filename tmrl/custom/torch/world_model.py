import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from typing import Tuple, Dict


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithmic transformation for scale-invariant regression."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse symmetric exponential transformation with numerical clamping."""
    return torch.sign(x) * torch.expm1(torch.clamp(torch.abs(x), 0.0, 15.0))


class TorchVisualTelemetryEncoder(nn.Module):
    """
    PyTorch Visual-Telemetry Encoder:
    Encodes stacked camera images plus [speed, gear, rpm] telemetry into a latent vector.
    """
    def __init__(self,
                 img_channels: int = 4,
                 img_height: int = 96,
                 img_width: int = 96,
                 latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        # Conv layers (parity with TMRL / PyTorch)
        self.conv1 = nn.Conv2d(img_channels, 32, kernel_size=8, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=0)

        # Telemetry MLP
        self.telem_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        def conv_out(size: int, kernel: int, stride: int) -> int:
            return (size - kernel) // stride + 1

        conv_h, conv_w = img_height, img_width
        for kernel, stride in ((8, 2), (4, 2), (4, 2), (4, 2)):
            conv_h = conv_out(conv_h, kernel, stride)
            conv_w = conv_out(conv_w, kernel, stride)
        if conv_h < 1 or conv_w < 1:
            raise ValueError(
                f"Image size {img_height}x{img_width} is too small for the world-model encoder."
            )

        self.fc_fused = nn.Sequential(
            nn.Linear(128 * conv_h * conv_w + 64, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU()
        )

    def forward(self, obs_tuple) -> torch.Tensor:
        """
        obs_tuple: (speed, gear, rpm, images, [act1, act2])
        images shape: (B, 4, 96, 96) in [0, 1] or [0, 255]
        """
        speed = obs_tuple[0]
        gear = obs_tuple[1]
        rpm = obs_tuple[2]
        imgs = obs_tuple[3]

        if speed.dim() == 1:
            speed = speed.unsqueeze(-1)
        if gear.dim() == 1:
            gear = gear.unsqueeze(-1)
        if rpm.dim() == 1:
            rpm = rpm.unsqueeze(-1)

        if imgs.dim() == 4 and imgs.shape[-1] == 4:
            # (B, H, W, C) -> (B, C, H, W)
            imgs = imgs.permute(0, 3, 1, 2)

        if imgs.dtype == torch.uint8 or imgs.max() > 1.0:
            imgs = imgs.float() / 255.0

        # Conv backbone
        x = F.relu(self.conv1(imgs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        img_feat = torch.flatten(x, start_dim=1)

        # Telemetry backbone
        speed_norm = speed / 300.0
        gear_norm = gear / 5.0
        rpm_norm = rpm / 10000.0
        telem = torch.cat([speed_norm, gear_norm, rpm_norm], dim=-1)
        telem_feat = self.telem_mlp(telem)

        # Fused Latent Embedding
        fused = torch.cat([img_feat, telem_feat], dim=-1)
        return self.fc_fused(fused)


class TorchLatentRSSM(nn.Module):
    """
    Recurrent State Space Model:
    Maintains deterministic belief h_t (GRU) + stochastic state z_t (Gaussian prior/posterior).
    """
    def __init__(self, latent_dim: int = 128, action_dim: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Deterministic Recurrent Unit
        self.gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)

        # Stochastic Transition Prior: p(z_t | h_t)
        self.prior_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )

        # Stochastic Posterior (Filtering): q(z_t | h_t, e_t)
        self.post_fc = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )

    def compute_prior(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stats = self.prior_fc(h)
        mean, log_std = torch.chunk(stats, 2, dim=-1)
        log_std = torch.clamp(log_std, -3.0, 1.0)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(mean)
        return z, mean, std

    def compute_posterior(self, h: torch.Tensor, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = torch.cat([h, e], dim=-1)
        stats = self.post_fc(inputs)
        mean, log_std = torch.chunk(stats, 2, dim=-1)
        log_std = torch.clamp(log_std, -3.0, 1.0)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(mean)
        return z, mean, std

    def step_deterministic(self, h: torch.Tensor, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([z, a], dim=-1)
        return self.gru(inputs, h)


class TorchLatentWorldModel(nn.Module):
    """
    Full Latent World Model: Encoder + RSSM + Prediction Heads
    """
    def __init__(self,
                 img_channels: int = 4,
                 img_height: int = 96,
                 img_width: int = 96,
                 latent_dim: int = 128,
                 action_dim: int = 3,
                 hidden_dim: int = 256,
                 policy_feature_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.policy_feature_dim = policy_feature_dim

        self.encoder = TorchVisualTelemetryEncoder(
            img_channels=img_channels,
            img_height=img_height,
            img_width=img_width,
            latent_dim=latent_dim,
        )
        self.rssm = TorchLatentRSSM(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim)

        feat_dim = hidden_dim + latent_dim

        # Predictor heads
        self.reward_head = nn.Linear(feat_dim, 1)
        self.continue_head = nn.Linear(feat_dim, 1)
        self.embedding_head = nn.Linear(feat_dim, latent_dim)
        # Align imagined states with the feature vector consumed by the exact
        # action head that is broadcast to rollout workers.
        # Predict signed symlog features.  The deployed CNN's internal features
        # can be large because legacy telemetry inputs are not normalized; the
        # transform keeps this auxiliary target numerically balanced.
        self.policy_feature_head = nn.Linear(feat_dim, policy_feature_dim)

        # Zero init predictor heads to prevent exploding predictions at init
        nn.init.zeros_(self.reward_head.weight)
        nn.init.zeros_(self.reward_head.bias)
        nn.init.zeros_(self.continue_head.weight)
        nn.init.zeros_(self.continue_head.bias)

    def get_feature(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([h, z], dim=-1)

    def predict_reward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.reward_head(feat)

    def predict_continuation(self, feat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.continue_head(feat))

    def predict_embedding(self, feat: torch.Tensor) -> torch.Tensor:
        return self.embedding_head(feat)

    def predict_policy_feature(self, feat: torch.Tensor) -> torch.Tensor:
        return symexp(self.predict_policy_feature_symlog(feat))

    def predict_policy_feature_symlog(self, feat: torch.Tensor) -> torch.Tensor:
        return self.policy_feature_head(feat)

    def imagine_step(self,
                     h: torch.Tensor,
                     z: torch.Tensor,
                     a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Takes one imagined step in latent dynamics:
        Returns: (next_h, next_z, pred_reward, pred_continue, feat)
        """
        next_h = self.rssm.step_deterministic(h, z, a)
        next_z, _, _ = self.rssm.compute_prior(next_h)
        feat = self.get_feature(next_h, next_z)
        pred_reward = symexp(self.predict_reward(feat))
        pred_cont = self.predict_continuation(feat)
        return next_h, next_z, pred_reward, pred_cont, feat


class TorchLatentAdversaryProposer(nn.Module):
    """
    Experimental bounded latent-perturbation module.

    A perturbation is not an AZR/PAIRED task unless a simulator can validate
    and repeatedly execute the corresponding scenario.
    """
    def __init__(self, feat_dim: int = 384, perturbation_dim: int = 128, max_magnitude: float = 0.25):
        super().__init__()
        self.max_magnitude = max_magnitude
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(128, perturbation_dim)
        self.log_std_head = nn.Linear(128, perturbation_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(feat)
        return torch.tanh(self.mean_head(hidden)) * self.max_magnitude

    def sample(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a bounded proposal and return its matched log probability."""
        hidden = self.trunk(feat)
        mean = self.mean_head(hidden)
        log_std = torch.clamp(self.log_std_head(hidden), -5.0, 1.0)
        distribution = Normal(mean, torch.exp(log_std))
        # A score-function proposer update consumes this matched log-probability,
        # so the sampled task itself must be treated as constant in that loss.
        raw = distribution.sample()
        squashed = torch.tanh(raw)
        perturbation = squashed * self.max_magnitude
        # Change-of-variables correction for tanh.  The constant action scale
        # can be omitted because it has no gradient with respect to parameters.
        log_prob = distribution.log_prob(raw) - torch.log(1.0 - squashed.square() + 1e-6)
        return perturbation, log_prob.sum(dim=-1)
