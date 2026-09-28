"""Torch Dreamer-style recurrent control for the production TMRL pipeline.

This module is intentionally separate from the SAC/AZR hybrid.  The deployed
actor owns the observation encoder, RSSM filter, and latent policy; therefore
the same recurrent controller trained on imagined trajectories is the one that
acts in TrackMania.
"""

import itertools
import logging
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.optim import Adam

import tmrl.config.config_constants as cfg
from tmrl.core.training import TrainingAgent
from tmrl.core.torch.actor import TorchActorModule
from tmrl.core.util import cached_property
from tmrl.custom.torch.utils.nn import copy_shared, no_grad
from tmrl.custom.torch.azr import ReplayLatentTaskBuffer, compute_azr_learnability
from tmrl.custom.torch.world_model import (
    TorchLatentAdversaryProposer,
    TorchLatentRSSM,
    TorchVisualTelemetryEncoder,
    symlog,
    symexp,
)


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class TorchDreamerActor(TorchActorModule):
    """Recurrent latent actor deployed directly in the RolloutWorker."""

    def __init__(
        self,
        observation_space,
        action_space,
        latent_dim=128,
        hidden_dim=256,
        policy_hidden_dim=256,
        img_channels=None,
        img_height=None,
        img_width=None,
        device="cpu",
        use_foundation_encoder=True,
        foundation_weights_path="weights/car_brain_1m_curriculum/car_brain_multimodal.pt",
        freeze_foundation=True,
        foundation_only=True,
        foundation_discrete_actions=False,
        foundation_steer_threshold=0.05,
        residual_scale=0.25,
        reload_foundation_on_actor_load=True,
    ):
        super().__init__(observation_space, action_space, device=device)
        img_channels = int(img_channels or cfg.IMG_HIST_LEN)
        img_height = int(img_height or cfg.IMG_HEIGHT)
        img_width = int(img_width or cfg.IMG_WIDTH)
        action_dim = int(action_space.shape[0])

        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = action_dim
        self.use_foundation_encoder = bool(use_foundation_encoder)
        self.foundation_weights_path = str(foundation_weights_path)
        self.freeze_foundation = bool(freeze_foundation)
        self.foundation_only = bool(foundation_only)
        self.foundation_discrete_actions = bool(foundation_discrete_actions)
        self.foundation_steer_threshold = float(foundation_steer_threshold)
        self.reload_foundation_on_actor_load = bool(
            reload_foundation_on_actor_load
        )
        self.residual_scale = float(residual_scale)
        if self.foundation_only and not self.use_foundation_encoder:
            raise ValueError("foundation_only requires use_foundation_encoder=True")
        if not 0.0 <= self.foundation_steer_threshold < 1.0:
            raise ValueError("foundation_steer_threshold must be in [0, 1)")

        if self.use_foundation_encoder:
            from tmrl.custom.torch.foundation_encoder import FoundationVisualTelemetryEncoder
            self.encoder = FoundationVisualTelemetryEncoder(
                img_channels=img_channels,
                img_height=img_height,
                img_width=img_width,
                latent_dim=self.latent_dim,
                model_path=foundation_weights_path,
                freeze_foundation=self.freeze_foundation,
            )
        else:
            self.encoder = TorchVisualTelemetryEncoder(
                img_channels=img_channels,
                img_height=img_height,
                img_width=img_width,
                latent_dim=self.latent_dim,
            )

        self.rssm = TorchLatentRSSM(
            latent_dim=self.latent_dim,
            action_dim=action_dim,
            hidden_dim=self.hidden_dim,
        )
        feature_dim = self.hidden_dim + self.latent_dim
        self.policy = nn.Sequential(
            nn.Linear(feature_dim, policy_hidden_dim),
            nn.LayerNorm(policy_hidden_dim),
            nn.SiLU(),
            nn.Linear(policy_hidden_dim, policy_hidden_dim),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(policy_hidden_dim, action_dim)
        self.log_std_head = nn.Linear(policy_hidden_dim, action_dim)

        # Under Residual Foundation RL, the foundation model provides the base trajectory.
        # Policy head starts with zero bias so initial action is 100% expert foundation steering.
        nn.init.zeros_(self.mu_head.weight)
        if hasattr(self.mu_head, "bias") and self.mu_head.bias is not None:
            self.mu_head.bias.data[0] = 1.0   # gas
            self.mu_head.bias.data[1] = -1.0  # brake
            self.mu_head.bias.data[2] = 0.0   # steer
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, -1.0)

        action_scale = torch.as_tensor(action_space.high, dtype=torch.float32)
        self.register_buffer("action_scale", action_scale)
        self.to(device)
        self.reset()

    @property
    def feature_dim(self):
        return self.hidden_dim + self.latent_dim

    def reset(self):
        """Forget recurrent state at a real environment episode boundary."""
        self._h = None
        self._z = None
        self._previous_action = None
        reset_encoder = getattr(getattr(self, "encoder", None), "reset_context", None)
        if reset_encoder is not None:
            reset_encoder()

    def initial(self, batch_size, device=None, dtype=None):
        parameter = next(self.parameters())
        device = device or parameter.device
        dtype = dtype or parameter.dtype
        h = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        z = torch.zeros(batch_size, self.latent_dim, device=device, dtype=dtype)
        action = torch.zeros(batch_size, self.action_dim, device=device, dtype=dtype)
        return h, z, action

    def observe(
        self,
        obs,
        h,
        z,
        previous_action,
        is_first=None,
        sample=True,
    ):
        """Filter a real observation into the recurrent posterior state."""
        dev = next(self.parameters()).device
        h = h.to(dev)
        z = z.to(dev)
        previous_action = previous_action.to(dev)
        obs = tuple(x.to(dev) for x in obs)
        next_h = self.rssm.step_deterministic(h, z, previous_action)
        if is_first is not None:
            mask = is_first.to(device=next_h.device, dtype=torch.bool).reshape(-1, 1)
            next_h = torch.where(mask, torch.zeros_like(next_h), next_h)
        embedding = self.encoder(obs)
        posterior_z, mean, std = self.rssm.compute_posterior(next_h, embedding)
        if not sample:
            posterior_z = mean
        return next_h, posterior_z, mean, std, embedding

    def forward_features(self, feature, test=False, with_logprob=True):
        hidden = self.policy(feature)
        mean = self.mu_head(hidden)
        log_std = torch.clamp(self.log_std_head(hidden), LOG_STD_MIN, LOG_STD_MAX)
        distribution = Normal(mean, torch.exp(log_std))
        raw_action = mean if test else distribution.rsample()
        if with_logprob:
            log_prob = distribution.log_prob(raw_action).sum(dim=-1)
            log_prob -= (
                2.0
                * (np.log(2.0) - raw_action - F.softplus(-2.0 * raw_action))
            ).sum(dim=-1)
        else:
            log_prob = None
        action = torch.tanh(raw_action) * self.action_scale
        return action, log_prob

    def policy_parameters(self):
        return itertools.chain(
            self.policy.parameters(),
            self.mu_head.parameters(),
            self.log_std_head.parameters(),
        )

    def act(self, obs, test=False):
        dev = next(self.parameters()).device
        obs = tuple(torch.as_tensor(x, device=dev) if not isinstance(x, torch.Tensor) else x.to(dev) for x in obs)
        batch_size = obs[0].shape[0]

        # Safe deployment gate: until the residual controller passes a real
        # closed-loop evaluation, drive with exactly the demonstrated policy.
        # In particular, Dreamer's random/stochastic initial residual must not
        # be allowed to turn or brake the live car.
        if getattr(self, "foundation_only", False):
            self.encoder(obs)
            foundation_action = self.encoder.get_foundation_action()
            if foundation_action is None:
                raise RuntimeError(
                    "Foundation-only deployment produced no foundation action"
                )
            if not torch.isfinite(foundation_action).all():
                raise RuntimeError(
                    "Foundation-only deployment produced a non-finite action"
                )
            action = torch.clamp(foundation_action, -1.0, 1.0)
            if getattr(self, "foundation_discrete_actions", False):
                # The target demonstrations were recorded from keyboard inputs:
                # gas/brake are binary and steering is ternary.  Applying the
                # raw regression magnitude (for example +0.06 for a target +1)
                # causes systematic understeer in closed loop.
                steer_threshold = float(
                    getattr(self, "foundation_steer_threshold", 0.05)
                )
                gas = torch.where(
                    action[:, 0:1] > 0.0,
                    torch.ones_like(action[:, 0:1]),
                    -torch.ones_like(action[:, 0:1]),
                )
                brake = torch.where(
                    action[:, 1:2] > 0.0,
                    torch.ones_like(action[:, 1:2]),
                    -torch.ones_like(action[:, 1:2]),
                )
                steer = torch.where(
                    action[:, 2:3] > steer_threshold,
                    torch.ones_like(action[:, 2:3]),
                    torch.where(
                        action[:, 2:3] < -steer_threshold,
                        -torch.ones_like(action[:, 2:3]),
                        torch.zeros_like(action[:, 2:3]),
                    ),
                )
                action = torch.cat((gas, brake, steer), dim=-1)
            self._previous_action = action.detach()
            return action.squeeze(0).detach().cpu().numpy()

        if (
            self._h is None
            or self._h.shape[0] != batch_size
            or self._h.device != dev
        ):
            h, z, previous_action = self.initial(
                batch_size, device=dev, dtype=torch.float32
            )
            is_first = torch.ones(batch_size, device=dev, dtype=torch.bool)
        else:
            h, z, previous_action = self._h, self._z, self._previous_action
            is_first = torch.zeros(batch_size, device=dev, dtype=torch.bool)

        h, z, _, _, _ = self.observe(
            obs,
            h,
            z,
            previous_action,
            is_first=is_first,
            sample=not test,
        )
        feature = torch.cat([h, z], dim=-1)
        action, _ = self.forward_features(feature, test=test, with_logprob=False)

        # Residual Foundation RL Integration (only applied if explicitly enabled):
        if getattr(self, "use_residual_action", False) and getattr(self, "use_foundation_encoder", False) and hasattr(self.encoder, "get_foundation_action"):
            foundation_action = self.encoder.get_foundation_action()
            if foundation_action is not None:
                scale = getattr(self, "residual_scale", 0.25)
                c_gas = torch.clamp(foundation_action[:, 0:1] + action[:, 0:1] * scale, -1.0, 1.0)
                c_steer = torch.clamp(foundation_action[:, 2:3] + action[:, 2:3] * scale, -1.0, 1.0)
                c_brake = torch.maximum(foundation_action[:, 1:2], action[:, 1:2])
                action = torch.cat([c_gas, c_brake, c_steer], dim=-1)

        self._h = h.detach()
        self._z = z.detach()
        self._previous_action = action.detach()
        return action.squeeze(0).detach().cpu().numpy()

    def load(self, path, device):
        """Load Dreamer state while protecting the canonical driving policy.

        The foundation checkpoint is the reviewed, demonstrated controller.
        Broadcasts contain a copy for serialization compatibility, but that copy
        may be stale or may have been altered by an older trainer.  Ignore those
        keys and restore them from the canonical checkpoint on disk.
        """
        self.device = device
        self.to(device)
        self.last_load_succeeded = False
        self.last_load_error = None
        try:
            state = torch.load(path, map_location=device, weights_only=True)
            if not isinstance(state, dict):
                raise TypeError(
                    f"Dreamer actor checkpoint must be a state dict, got {type(state)!r}"
                )

            protected_prefix = "encoder.brain."
            protect_foundation = bool(
                getattr(self, "use_foundation_encoder", False)
                and getattr(self, "reload_foundation_on_actor_load", True)
            )
            load_state = {
                key: value
                for key, value in state.items()
                if not (protect_foundation and key.startswith(protected_prefix))
            }
            expected = self.state_dict()
            expected_keys = {
                key
                for key in expected
                if not (protect_foundation and key.startswith(protected_prefix))
            }
            missing = sorted(expected_keys - load_state.keys())
            unexpected = sorted(load_state.keys() - expected_keys)
            mismatched = sorted(
                key
                for key in expected_keys & load_state.keys()
                if expected[key].shape != load_state[key].shape
            )
            if missing or unexpected or mismatched:
                raise RuntimeError(
                    "incompatible Dreamer actor state "
                    f"(missing={missing[:5]}, unexpected={unexpected[:5]}, "
                    f"shape_mismatch={mismatched[:5]})"
                )
            self.load_state_dict(load_state, strict=not protect_foundation)
            if protect_foundation:
                self.encoder.reload_foundation_weights(
                    getattr(self, "foundation_weights_path", None)
                )
            self.last_load_succeeded = True
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.last_load_error = str(exc)
            logging.warning(
                "Ignoring incompatible actor weights while waiting for the first "
                "Dreamer broadcast: %s",
                exc,
            )
        self.reset()
        return self


class DreamerPredictionHeads(nn.Module):
    """World-model likelihood heads used only by the Trainer."""

    def __init__(self, feature_dim, latent_dim, img_channels, reconstruction_size=24):
        super().__init__()
        self.img_channels = int(img_channels)
        self.reconstruction_size = int(reconstruction_size)
        self.reward = nn.Linear(feature_dim, 1)
        self.continuation = nn.Linear(feature_dim, 1)
        self.embedding = nn.Linear(feature_dim, latent_dim)
        self.telemetry = nn.Linear(feature_dim, 3)
        pixels = self.img_channels * self.reconstruction_size * self.reconstruction_size
        self.observation = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.SiLU(),
            nn.Linear(512, pixels),
        )
        nn.init.zeros_(self.reward.weight)
        nn.init.zeros_(self.reward.bias)
        nn.init.zeros_(self.continuation.weight)
        nn.init.zeros_(self.continuation.bias)
        self.foundation_action = nn.Linear(feature_dim, 3)
        nn.init.zeros_(self.foundation_action.weight)
        nn.init.zeros_(self.foundation_action.bias)

    def image_logits(self, feature):
        logits = self.observation(feature)
        return logits.reshape(
            feature.shape[0],
            self.img_channels,
            self.reconstruction_size,
            self.reconstruction_size,
        )


class DreamerValue(nn.Module):
    """Symlog state-value model for imagined lambda returns."""

    def __init__(self, feature_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feature):
        return self.net(feature).squeeze(-1)


class PercentileReturnNormalizer:
    """EMA percentile range used to keep imagined actor targets well scaled."""

    def __init__(self, rate=0.01, low=0.05, high=0.95, minimum_scale=1.0):
        self.rate = float(rate)
        self.low = float(low)
        self.high = float(high)
        self.minimum_scale = float(minimum_scale)
        self.low_ema = None
        self.high_ema = None

    @torch.no_grad()
    def update(self, values):
        flat = values.detach().float().reshape(-1)
        quantiles = torch.quantile(
            flat,
            torch.tensor([self.low, self.high], device=flat.device),
        )
        low, high = quantiles[0], quantiles[1]
        if self.low_ema is None:
            self.low_ema = low
            self.high_ema = high
        else:
            self.low_ema = (1.0 - self.rate) * self.low_ema + self.rate * low
            self.high_ema = (1.0 - self.rate) * self.high_ema + self.rate * high
        return torch.clamp(
            self.high_ema - self.low_ema,
            min=self.minimum_scale,
        )


def lambda_returns(rewards, discounts, next_values, lambda_=0.95):
    """Compute reverse-time generalized lambda returns for ``[time, batch]``."""
    if rewards.shape != discounts.shape or rewards.shape != next_values.shape:
        raise ValueError("reward, discount, and next-value tensors must have equal shape")
    result = []
    carry = next_values[-1]
    for t in reversed(range(rewards.shape[0])):
        carry = rewards[t] + discounts[t] * (
            (1.0 - lambda_) * next_values[t] + lambda_ * carry
        )
        result.append(carry)
    return torch.stack(list(reversed(result)), dim=0)


@dataclass(eq=0)
class TorchDreamerAgent(TrainingAgent):
    """Sequence-trained Dreamer controller with latent imagination updates."""

    observation_space: type
    action_space: type
    device: str = None
    latent_dim: int = 128
    hidden_dim: int = 256
    policy_hidden_dim: int = 256
    reconstruction_size: int = 24
    lr_world_model: float = 3e-4
    lr_actor: float = 8e-5
    lr_critic: float = 8e-5
    gamma: float = 0.99
    lambda_: float = 0.95
    free_nats: float = 1.0
    dyn_kl_scale: float = 1.0
    rep_kl_scale: float = 0.1
    observation_scale: float = 1.0
    telemetry_scale: float = 1.0
    embedding_scale: float = 1.0
    imagination_horizon: int = 8
    imagination_batch_size: int = 64
    burn_in: int = 5
    world_model_warmup_steps: int = 2000
    entropy_scale: float = 3e-4
    target_polyak: float = 0.99
    grad_clip: float = 100.0
    return_normalizer_rate: float = 0.01
    enable_azr_imagination: bool = False
    lr_adversary: float = 1e-4
    solver_attempts: int = 4
    tasks_per_proposal: int = 4
    task_batch_size: int = 8
    task_buffer_capacity: int = 1024
    task_max_age: int = 2000
    task_proposal_interval: int = 8
    min_task_continuation: float = 0.5
    max_latent_perturbation: float = 0.25
    target_quantile: float = 0.5
    azr_seed: int = 0
    use_foundation_encoder: bool = True
    foundation_weights_path: str = "weights/car_brain_1m_curriculum/car_brain_multimodal.pt"
    lr_foundation: float = 0.0
    freeze_foundation: bool = True
    foundation_only: bool = True
    foundation_discrete_actions: bool = False
    foundation_steer_threshold: float = 0.05
    residual_scale: float = 0.25
    reload_foundation_on_actor_load: bool = True

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.actor)))

    def __post_init__(self):
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_mode = "DREAMER"
        self.actor = TorchDreamerActor(
            observation_space=self.observation_space,
            action_space=self.action_space,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            policy_hidden_dim=self.policy_hidden_dim,
            img_channels=cfg.IMG_HIST_LEN,
            img_height=cfg.IMG_HEIGHT,
            img_width=cfg.IMG_WIDTH,
            use_foundation_encoder=self.use_foundation_encoder,
            foundation_weights_path=self.foundation_weights_path,
            freeze_foundation=self.freeze_foundation,
            foundation_only=self.foundation_only,
            foundation_discrete_actions=self.foundation_discrete_actions,
            foundation_steer_threshold=self.foundation_steer_threshold,
            residual_scale=self.residual_scale,
            reload_foundation_on_actor_load=self.reload_foundation_on_actor_load,
        ).to(self.device)
        feature_dim = self.hidden_dim + self.latent_dim
        self.heads = DreamerPredictionHeads(
            feature_dim=feature_dim,
            latent_dim=self.latent_dim,
            img_channels=cfg.IMG_HIST_LEN,
            reconstruction_size=self.reconstruction_size,
        ).to(self.device)
        self.critic = DreamerValue(feature_dim).to(self.device)
        self.target_critic = no_grad(deepcopy(self.critic))

        self.world_model_optimizer = self._build_world_model_optimizer()

        self.actor_optimizer = Adam(list(self.actor.policy_parameters()), lr=self.lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr_critic)
        self.return_normalizer = PercentileReturnNormalizer(
            rate=self.return_normalizer_rate
        )
        self.adversary = TorchLatentAdversaryProposer(
            feat_dim=feature_dim,
            perturbation_dim=self.latent_dim,
            max_magnitude=self.max_latent_perturbation,
        ).to(self.device)
        self.adversary_optimizer = Adam(
            self.adversary.parameters(), lr=self.lr_adversary
        )
        self.task_buffer = ReplayLatentTaskBuffer(
            capacity=self.task_buffer_capacity,
            max_age=self.task_max_age,
            seed=self.azr_seed,
        )
        self.world_model_updates = 0
        self.train_steps = 0
        self._activation_logged = False

        for name in (
            "imagination_horizon",
            "imagination_batch_size",
            "solver_attempts",
            "tasks_per_proposal",
            "task_batch_size",
            "task_buffer_capacity",
            "task_max_age",
            "task_proposal_interval",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.target_quantile <= 1.0:
            raise ValueError("target_quantile must be in [0, 1]")

    def _build_world_model_optimizer(self):
        if self.use_foundation_encoder and hasattr(self.actor.encoder, "brain"):
            self.actor.encoder.set_foundation_trainable(
                not self.freeze_foundation
            )
            model_param_groups = []
            if not self.freeze_foundation:
                model_param_groups.append(
                    {
                        "params": self.actor.encoder.brain.parameters(),
                        "lr": self.lr_foundation,
                    }
                )
            model_param_groups.extend(
                [
                    {
                        "params": self.actor.encoder.latent_proj.parameters(),
                        "lr": self.lr_world_model,
                    },
                    {
                        "params": self.actor.rssm.parameters(),
                        "lr": self.lr_world_model,
                    },
                    {"params": self.heads.parameters(), "lr": self.lr_world_model},
                ]
            )
            return Adam(model_param_groups)

        model_parameters = itertools.chain(
            self.actor.encoder.parameters(),
            self.actor.rssm.parameters(),
            self.heads.parameters(),
        )
        return Adam(model_parameters, lr=self.lr_world_model)

    def configure_foundation_safety(
        self,
        *,
        freeze_foundation=True,
        foundation_only=True,
        foundation_discrete_actions=False,
        foundation_steer_threshold=0.05,
        residual_scale=0.25,
        reload_weights=True,
    ):
        """Apply deployment safety settings to new or legacy checkpoints."""
        if not self.use_foundation_encoder or not hasattr(self.actor.encoder, "brain"):
            return
        self.freeze_foundation = bool(freeze_foundation)
        self.foundation_only = bool(foundation_only)
        self.foundation_discrete_actions = bool(foundation_discrete_actions)
        self.foundation_steer_threshold = float(foundation_steer_threshold)
        self.residual_scale = float(residual_scale)
        self.actor.freeze_foundation = self.freeze_foundation
        self.actor.foundation_only = self.foundation_only
        self.actor.foundation_discrete_actions = self.foundation_discrete_actions
        self.actor.foundation_steer_threshold = self.foundation_steer_threshold
        self.actor.residual_scale = self.residual_scale
        self.actor.foundation_weights_path = self.foundation_weights_path
        self.actor.reload_foundation_on_actor_load = bool(
            getattr(self, "reload_foundation_on_actor_load", True)
        )
        self.actor.encoder.freeze_foundation = self.freeze_foundation
        self.actor.encoder.reload_on_change = True
        if reload_weights:
            self.actor.encoder.reload_foundation_weights(
                self.foundation_weights_path
            )
        self.actor.encoder.set_foundation_trainable(
            not self.freeze_foundation
        )
        self.world_model_optimizer = self._build_world_model_optimizer()
        # ``model_nograd`` is a descriptor-backed shared copy.  Refresh it so
        # non-tensor deployment flags migrate along with a loaded checkpoint.
        self.model_nograd = no_grad(copy_shared(self.actor))

    def get_actor(self):
        return self.model_nograd

    @staticmethod
    def _gaussian_kl(p_mean, p_std, q_mean, q_std):
        p_var = p_std.square()
        q_var = q_std.square()
        elem = (
            torch.log(q_std / (p_std + 1e-6))
            + (p_var + (p_mean - q_mean).square()) / (2.0 * q_var + 1e-6)
            - 0.5
        )
        return elem.sum(dim=-1)

    @staticmethod
    def _flatten_observation_sequence(obs):
        batch, time = obs[0].shape[:2]
        return tuple(
            tensor.reshape(batch * time, *tensor.shape[2:]) for tensor in obs
        )

    @staticmethod
    def _observation_at(obs, time_index):
        return tuple(tensor[:, time_index] for tensor in obs)

    def _train_world_model(self, batch):
        if len(batch) != 7:
            raise ValueError(
                "TorchDreamerAgent requires sequence batches with an is_first mask"
            )
        obs, actions, rewards, next_obs, terminated, truncated, is_first = batch
        batch_size, sequence_length = actions.shape[:2]
        if self.burn_in >= sequence_length:
            raise ValueError("Dreamer burn_in must be shorter than replay sequences")

        flat_next_obs = self._flatten_observation_sequence(next_obs)
        next_embeddings = self.actor.encoder(flat_next_obs).reshape(
            batch_size, sequence_length, self.latent_dim
        )
        initial_obs = self._observation_at(obs, 0)
        h, z, previous_action = self.actor.initial(
            batch_size, device=actions.device, dtype=actions.dtype
        )
        h, z, _, _, _ = self.actor.observe(
            initial_obs,
            h,
            z,
            previous_action,
            is_first=is_first[:, 0],
            sample=True,
        )

        features = []
        posterior_h = []
        posterior_z = []
        prior_means = []
        prior_stds = []
        posterior_means = []
        posterior_stds = []
        for t in range(sequence_length):
            h = self.actor.rssm.step_deterministic(h, z, actions[:, t])
            _, prior_mean, prior_std = self.actor.rssm.compute_prior(h)
            z, posterior_mean, posterior_std = self.actor.rssm.compute_posterior(
                h, next_embeddings[:, t]
            )
            feature = torch.cat([h, z], dim=-1)
            features.append(feature)
            posterior_h.append(h)
            posterior_z.append(z)
            prior_means.append(prior_mean)
            prior_stds.append(prior_std)
            posterior_means.append(posterior_mean)
            posterior_stds.append(posterior_std)

        feature_sequence = torch.stack(features, dim=1)
        flat_feature = feature_sequence.reshape(-1, feature_sequence.shape[-1])
        reward_prediction = self.heads.reward(flat_feature).reshape(
            batch_size, sequence_length
        )
        continuation_logits = self.heads.continuation(flat_feature).reshape(
            batch_size, sequence_length
        )
        embedding_prediction = self.heads.embedding(flat_feature).reshape(
            batch_size, sequence_length, self.latent_dim
        )

        reward_targets = symlog(rewards.reshape(batch_size, sequence_length))
        reward_loss = F.smooth_l1_loss(reward_prediction, reward_targets)
        done = torch.maximum(terminated.float(), truncated.float()).reshape(batch_size, sequence_length)
        continuation_target = 1.0 - done
        continuation_loss = F.binary_cross_entropy_with_logits(
            continuation_logits, continuation_target
        )
        embedding_loss = F.mse_loss(
            embedding_prediction, next_embeddings.detach()
        )

        telemetry_prediction = self.heads.telemetry(flat_feature).reshape(
            batch_size, sequence_length, 3
        )
        telemetry_target = torch.cat(
            [
                next_obs[0] / 300.0,
                next_obs[1] / 5.0,
                next_obs[2] / 10000.0,
            ],
            dim=-1,
        )
        telemetry_loss = F.smooth_l1_loss(
            telemetry_prediction, telemetry_target
        )

        image_target = next_obs[3].reshape(
            batch_size * sequence_length,
            *next_obs[3].shape[2:],
        ).float()
        if image_target.max() > 1.0:
            image_target = image_target / 255.0
        image_target = F.interpolate(
            image_target,
            size=(self.reconstruction_size, self.reconstruction_size),
            mode="bilinear",
            align_corners=False,
        )
        image_logits = self.heads.image_logits(flat_feature)
        observation_loss = F.binary_cross_entropy_with_logits(
            image_logits, image_target
        )

        prior_mean = torch.stack(prior_means, dim=1)
        prior_std = torch.stack(prior_stds, dim=1)
        posterior_mean = torch.stack(posterior_means, dim=1)
        posterior_std = torch.stack(posterior_stds, dim=1)
        dynamics_kl = self._gaussian_kl(
            posterior_mean.detach(), posterior_std.detach(), prior_mean, prior_std
        )
        representation_kl = self._gaussian_kl(
            posterior_mean, posterior_std, prior_mean.detach(), prior_std.detach()
        )
        free_nats = torch.as_tensor(
            self.free_nats, device=actions.device, dtype=actions.dtype
        )
        dynamics_kl_loss = torch.maximum(dynamics_kl, free_nats).mean()
        representation_kl_loss = torch.maximum(
            representation_kl, free_nats
        ).mean()

        if hasattr(self.heads, "foundation_action"):
            pred_f_action = torch.tanh(self.heads.foundation_action(flat_feature))
            action_dim = self.actor.action_dim
            foundation_action_loss = F.smooth_l1_loss(
                pred_f_action, actions.reshape(-1, action_dim)
            )
        else:
            foundation_action_loss = torch.zeros((), device=actions.device)

        world_model_loss = (
            reward_loss
            + continuation_loss
            + self.embedding_scale * embedding_loss
            + self.telemetry_scale * telemetry_loss
            + self.observation_scale * observation_loss
            + self.dyn_kl_scale * dynamics_kl_loss
            + self.rep_kl_scale * representation_kl_loss
            + foundation_action_loss
        )
        start_h = torch.stack(posterior_h, dim=1).detach()
        start_z = torch.stack(posterior_z, dim=1).detach()

        self.world_model_optimizer.zero_grad(set_to_none=True)
        world_model_loss.backward()
        model_parameters = [
            parameter
            for parameter in itertools.chain(
                self.actor.encoder.parameters(),
                self.actor.rssm.parameters(),
                self.heads.parameters(),
            )
            if parameter.requires_grad
        ]
        torch.nn.utils.clip_grad_norm_(model_parameters, self.grad_clip)
        self.world_model_optimizer.step()
        self.world_model_updates += 1

        metrics = {
            "loss_world_model": world_model_loss.detach().item(),
            "loss_wm_reward": reward_loss.detach().item(),
            "loss_wm_continue": continuation_loss.detach().item(),
            "loss_wm_embedding": embedding_loss.detach().item(),
            "loss_wm_telemetry": telemetry_loss.detach().item(),
            "loss_wm_observation": observation_loss.detach().item(),
            "loss_wm_kl_dyn": dynamics_kl_loss.detach().item(),
            "loss_wm_kl_rep": representation_kl_loss.detach().item(),
            "loss_wm_foundation_action": foundation_action_loss.detach().item(),
        }
        return start_h, start_z, metrics

    def _set_world_model_grad(self, enabled):
        for module in (self.actor.encoder, self.actor.rssm, self.heads):
            for parameter in module.parameters():
                parameter.requires_grad_(enabled)
        if (
            getattr(self, "freeze_foundation", True)
            and hasattr(self.actor.encoder, "brain")
        ):
            for parameter in self.actor.encoder.brain.parameters():
                parameter.requires_grad_(False)

    def _sample_imagination_starts(self, h_sequence, z_sequence):
        h = h_sequence[:, self.burn_in:].reshape(-1, self.hidden_dim)
        z = z_sequence[:, self.burn_in:].reshape(-1, self.latent_dim)
        count = min(self.imagination_batch_size, h.shape[0])
        indices = torch.randperm(h.shape[0], device=h.device)[:count]
        return h[indices], z[indices]

    def _empty_azr_metrics(self):
        return {
            "azr_ready": 0.0,
            "azr_tasks_proposed": 0.0,
            "azr_tasks_accepted": 0.0,
            "azr_task_buffer_size": float(len(self.task_buffer)),
            "azr_mean_pass_rate": 0.0,
            "azr_mean_learnability": 0.0,
            "azr_mean_survival": 0.0,
            "azr_imagination_starts": 0.0,
            "loss_adversary": 0.0,
        }

    def _rollout_task_attempts(self, h, z, attempts):
        """Evaluate repeated solver attempts at identical stored latent tasks."""
        task_count = h.shape[0]
        h = h.unsqueeze(0).expand(attempts, -1, -1).reshape(
            attempts * task_count, self.hidden_dim
        )
        z = z.unsqueeze(0).expand(attempts, -1, -1).reshape(
            attempts * task_count, self.latent_dim
        )
        cumulative_return = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
        survival = torch.ones_like(cumulative_return)
        discount = 1.0
        for _ in range(self.imagination_horizon):
            feature = torch.cat([h, z], dim=-1)
            action, _ = self.actor.forward_features(feature)
            if getattr(self.actor, "use_residual_action", False) and getattr(self, "use_foundation_encoder", False) and hasattr(self.heads, "foundation_action"):
                base_action = torch.tanh(self.heads.foundation_action(feature)).detach()
                scale = getattr(self.actor, "residual_scale", 0.25)
                c_gas = torch.clamp(base_action[:, 0:1] + action[:, 0:1] * scale, -1.0, 1.0)
                c_steer = torch.clamp(base_action[:, 2:3] + action[:, 2:3] * scale, -1.0, 1.0)
                c_brake = torch.maximum(base_action[:, 1:2], action[:, 1:2])
                full_action = torch.cat([c_gas, c_brake, c_steer], dim=-1)
            else:
                full_action = action
            h = self.actor.rssm.step_deterministic(h, z, full_action)
            z, _, _ = self.actor.rssm.compute_prior(h)
            next_feature = torch.cat([h, z], dim=-1)
            reward = symexp(self.heads.reward(next_feature)).squeeze(-1)
            reward = reward.clamp(-10.0, 10.0)
            continuation = torch.sigmoid(
                self.heads.continuation(next_feature)
            ).squeeze(-1)
            cumulative_return = cumulative_return + discount * survival * reward
            survival = survival * continuation
            discount *= self.gamma
        shape = (attempts, task_count)
        return cumulative_return.reshape(shape), survival.reshape(shape)

    def _propose_azr_tasks(self, h_sequence, z_sequence):
        metrics = self._empty_azr_metrics()
        h = h_sequence[:, self.burn_in:].reshape(-1, self.hidden_dim)
        z = z_sequence[:, self.burn_in:].reshape(-1, self.latent_dim)
        count = min(self.tasks_per_proposal, h.shape[0])
        indices = torch.randperm(h.shape[0], device=h.device)[:count]
        h = h[indices].detach()
        z = z[indices].detach()
        feature = torch.cat([h, z], dim=-1)
        perturbation, proposal_log_prob = self.adversary.sample(feature)
        task_z = z + perturbation

        with torch.no_grad():
            baseline_returns, _ = self._rollout_task_attempts(
                h, z, self.solver_attempts
            )
            target_return = torch.quantile(
                baseline_returns, self.target_quantile, dim=0
            )
            returns, survival = self._rollout_task_attempts(
                h, task_z.detach(), self.solver_attempts
            )
            successes = (
                (returns > target_return.unsqueeze(0))
                & (survival >= self.min_task_continuation)
            ).float()
            learnability, pass_rate = compute_azr_learnability(successes)
            mean_survival = survival.mean(dim=0)
            finite = (
                torch.isfinite(returns).all(dim=0)
                & torch.isfinite(survival).all(dim=0)
                & torch.isfinite(target_return)
            )
            in_support = perturbation.detach().abs().amax(dim=-1) <= (
                self.max_latent_perturbation + 1e-6
            )
            valid = finite & in_support & (
                mean_survival >= self.min_task_continuation
            )

        signal = learnability.detach() * valid.float()
        adversary_loss = torch.zeros((), device=h.device)
        if signal.sum() > 0.0:
            adversary_loss = -(
                signal * proposal_log_prob
            ).sum() / valid.float().sum().clamp_min(1.0)
            self.adversary_optimizer.zero_grad(set_to_none=True)
            adversary_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.adversary.parameters(), 10.0)
            self.adversary_optimizer.step()

        accepted = self.task_buffer.add_batch(
            h=h,
            z=task_z,
            target_returns=target_return,
            priorities=learnability,
            pass_rates=pass_rate,
            mean_survival=mean_survival,
            valid=valid,
            current_step=self.train_steps,
        )
        metrics.update(
            azr_tasks_proposed=float(count),
            azr_tasks_accepted=float(accepted),
            azr_task_buffer_size=float(len(self.task_buffer)),
            azr_mean_pass_rate=pass_rate.mean().item(),
            azr_mean_learnability=learnability.mean().item(),
            azr_mean_survival=mean_survival.mean().item(),
            loss_adversary=adversary_loss.detach().item(),
        )
        return metrics

    def _sample_azr_starts(self):
        metrics = self._empty_azr_metrics()
        if len(self.task_buffer) == 0:
            return None, None, metrics
        task_ids, h, z, target, _ = self.task_buffer.sample(
            self.task_batch_size,
            self.device,
            current_step=self.train_steps,
        )
        with torch.no_grad():
            returns, survival = self._rollout_task_attempts(
                h, z, self.solver_attempts
            )
            successes = (
                (returns > target.unsqueeze(0))
                & (survival >= self.min_task_continuation)
            ).float()
            learnability, pass_rate = compute_azr_learnability(successes)
            mean_survival = survival.mean(dim=0)
            valid = (
                torch.isfinite(returns).all(dim=0)
                & torch.isfinite(survival).all(dim=0)
                & (mean_survival >= self.min_task_continuation)
            )
            priorities = torch.where(
                valid, learnability, torch.zeros_like(learnability)
            )
        self.task_buffer.update(
            task_ids,
            priorities,
            pass_rate,
            mean_survival,
            current_step=self.train_steps,
        )
        active = priorities > 0.0
        h = h[active]
        z = z[active]
        metrics.update(
            azr_task_buffer_size=float(len(self.task_buffer)),
            azr_mean_pass_rate=pass_rate.mean().item(),
            azr_mean_learnability=learnability.mean().item(),
            azr_mean_survival=mean_survival.mean().item(),
            azr_imagination_starts=float(active.sum().item()),
        )
        if not active.any():
            return None, None, metrics
        return h.detach(), z.detach(), metrics

    def _merge_imagination_starts(self, replay_h, replay_z, task_h, task_z):
        if task_h is None:
            return replay_h, replay_z
        reserve = min(
            task_h.shape[0],
            max(1, self.imagination_batch_size // 2),
        )
        replay_count = max(self.imagination_batch_size - reserve, 0)
        return (
            torch.cat([replay_h[:replay_count], task_h[:reserve]], dim=0),
            torch.cat([replay_z[:replay_count], task_z[:reserve]], dim=0),
        )

    def _train_imagination(self, h, z):
        self._set_world_model_grad(False)
        try:
            rewards = []
            discounts = []
            next_values = []
            log_probs = []
            state_features = []
            weights = []
            weight = torch.ones(h.shape[0], device=h.device, dtype=h.dtype)
            for _ in range(self.imagination_horizon):
                feature = torch.cat([h, z], dim=-1)
                state_features.append(feature)
                weights.append(weight)
                action, log_prob = self.actor.forward_features(feature)
                if getattr(self.actor, "use_residual_action", False) and getattr(self, "use_foundation_encoder", False) and hasattr(self.heads, "foundation_action"):
                    base_action = torch.tanh(self.heads.foundation_action(feature)).detach()
                    scale = getattr(self.actor, "residual_scale", 0.25)
                    c_gas = torch.clamp(base_action[:, 0:1] + action[:, 0:1] * scale, -1.0, 1.0)
                    c_steer = torch.clamp(base_action[:, 2:3] + action[:, 2:3] * scale, -1.0, 1.0)
                    c_brake = torch.maximum(base_action[:, 1:2], action[:, 1:2])
                    full_action = torch.cat([c_gas, c_brake, c_steer], dim=-1)
                else:
                    full_action = action
                h = self.actor.rssm.step_deterministic(h, z, full_action)
                z, _, _ = self.actor.rssm.compute_prior(h)
                next_feature = torch.cat([h, z], dim=-1)
                reward = symexp(self.heads.reward(next_feature)).squeeze(-1)
                continuation = torch.sigmoid(
                    self.heads.continuation(next_feature)
                ).squeeze(-1)
                reward = reward.clamp(-10.0, 10.0)
                discount = self.gamma * continuation
                value = symexp(self.target_critic(next_feature))
                rewards.append(reward)
                discounts.append(discount)
                next_values.append(value)
                log_probs.append(log_prob)
                weight = weight * discount

            rewards = torch.stack(rewards, dim=0)
            discounts = torch.stack(discounts, dim=0)
            next_values = torch.stack(next_values, dim=0)
            log_probs = torch.stack(log_probs, dim=0)
            weights = torch.stack(weights, dim=0)
            returns = lambda_returns(
                rewards, discounts, next_values, lambda_=self.lambda_
            )
            return_scale = self.return_normalizer.update(returns)
            normalized_returns = returns / return_scale
            actor_objective = normalized_returns - self.entropy_scale * log_probs
            actor_loss = -(weights.detach() * actor_objective).mean()

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.policy_parameters()), self.grad_clip
            )
            self.actor_optimizer.step()

            state_features = torch.stack(state_features, dim=0).detach()
            value_prediction = self.critic(state_features.reshape(-1, state_features.shape[-1]))
            value_target = symlog(returns.detach()).reshape(-1)
            value_error = F.smooth_l1_loss(
                value_prediction,
                value_target,
                reduction="none",
            ).reshape_as(weights)
            critic_loss = (weights.detach() * value_error).mean()
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
            self.critic_optimizer.step()

            with torch.no_grad():
                for parameter, target in zip(
                    self.critic.parameters(), self.target_critic.parameters()
                ):
                    target.data.mul_(self.target_polyak)
                    target.data.add_((1.0 - self.target_polyak) * parameter.data)
        finally:
            self._set_world_model_grad(True)

        return {
            "loss_actor": actor_loss.detach().item(),
            "loss_critic": critic_loss.detach().item(),
            "mean_imagined_return": returns.detach().mean().item(),
            "mean_imagined_survival": weights.detach().mean().item(),
            "imagined_return_scale": return_scale.detach().item(),
            "imagined_policy_updates": 1.0,
        }

    def train(self, batch):
        h_sequence, z_sequence, metrics = self._train_world_model(batch)
        self.train_steps += 1
        ready = self.world_model_updates >= self.world_model_warmup_steps
        azr_metrics = self._empty_azr_metrics()
        azr_ready = ready and self.enable_azr_imagination
        azr_metrics["azr_ready"] = float(azr_ready)
        metrics.update(
            dreamer_ready=float(ready),
            dreamer_warmup_updates=float(self.world_model_updates),
            dreamer_warmup_remaining=float(
                max(self.world_model_warmup_steps - self.world_model_updates, 0)
            ),
            loss_actor=0.0,
            loss_critic=0.0,
            mean_imagined_return=0.0,
            mean_imagined_survival=0.0,
            imagined_return_scale=0.0,
            imagined_policy_updates=0.0,
        )
        if ready and not self._activation_logged:
            logging.info(
                "Dreamer imagination activated after %s world-model updates; "
                "the recurrent actor and slow critic now train in latent rollouts.",
                self.world_model_updates,
            )
            self._activation_logged = True
        if ready:
            start_h, start_z = self._sample_imagination_starts(
                h_sequence, z_sequence
            )
            if azr_ready and self.train_steps % self.task_proposal_interval == 0:
                proposal_metrics = self._propose_azr_tasks(h_sequence, z_sequence)
                azr_metrics.update(proposal_metrics)
                azr_metrics["azr_ready"] = 1.0
            if azr_ready:
                task_h, task_z, refresh_metrics = self._sample_azr_starts()
                proposal_counts = {
                    key: azr_metrics[key]
                    for key in (
                        "azr_tasks_proposed",
                        "azr_tasks_accepted",
                        "loss_adversary",
                    )
                }
                azr_metrics.update(refresh_metrics)
                azr_metrics.update(proposal_counts)
                azr_metrics["azr_ready"] = 1.0
                start_h, start_z = self._merge_imagination_starts(
                    start_h, start_z, task_h, task_z
                )
            metrics.update(self._train_imagination(start_h, start_z))
        azr_metrics["azr_task_buffer_size"] = float(len(self.task_buffer))
        metrics.update(azr_metrics)
        return metrics
