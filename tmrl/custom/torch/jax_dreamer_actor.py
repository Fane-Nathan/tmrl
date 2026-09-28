"""Torch/CUDA rollout mirror for a JAX-trained Dreamer actor.

The trainer exports portable NumPy parameters from Flax NNX. The Windows
worker loads them into this shape-identical Torch module, avoiding unsupported
native-Windows JAX GPU execution while preserving the trained computation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tmrl.core.torch.actor import TorchActorModule


ACTOR_BUNDLE_MAGIC = "tmrl-jax-dreamer-actor"
ACTOR_BUNDLE_VERSION = 4
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
POLICY_MEAN_BOUND = 5.0
RSSM_MEAN_BOUND = 5.0
RSSM_STD_MIN = 0.1
RSSM_STD_MAX = 2.0


def _parameter_checksum(parameters):
    digest = hashlib.sha256()
    for name in sorted(parameters):
        value = np.ascontiguousarray(parameters[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _validate_portable_parameters(parameters):
    bad = []
    for name, value in parameters.items():
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.inexact) and not np.isfinite(array).all():
            bad.append(str(name))
    if bad:
        preview = ", ".join(bad[:8])
        suffix = "" if len(bad) <= 8 else f" (+{len(bad) - 8} more)"
        raise FloatingPointError(
            f"portable actor contains non-finite parameters: {preview}{suffix}"
        )


class TorchPortableEncoder(nn.Module):
    def __init__(self, img_channels, img_height, img_width, latent_dim, channels):
        super().__init__()
        self.img_channels = int(img_channels)
        channels = tuple(int(value) for value in channels)
        self.conv1 = nn.Conv2d(img_channels, channels[0], 8, stride=2, padding=0)
        self.conv2 = nn.Conv2d(channels[0], channels[1], 4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(channels[1], channels[2], 4, stride=2, padding=0)
        self.conv4 = nn.Conv2d(channels[2], channels[3], 4, stride=2, padding=0)
        self.telemetry_fc = nn.Linear(3, 32)

        def conv_out(size, kernel, stride):
            return (size - kernel) // stride + 1

        conv_h, conv_w = int(img_height), int(img_width)
        for kernel, stride in ((8, 2), (4, 2), (4, 2), (4, 2)):
            conv_h = conv_out(conv_h, kernel, stride)
            conv_w = conv_out(conv_w, kernel, stride)
        self.fc_fuse = nn.Linear(channels[3] * conv_h * conv_w + 32, latent_dim)

    def forward_packed(self, telemetry, images):
        if images.ndim == 4 and images.shape[-1] == self.img_channels:
            images = images.permute(0, 3, 1, 2)
        return self.forward_channel_first(telemetry, images)

    def forward_channel_first(self, telemetry, images):
        integer_images = not torch.is_floating_point(images)
        images = images.float()
        if integer_images:
            images = images / 255.0
        visual = F.relu(self.conv1(images))
        visual = F.relu(self.conv2(visual))
        visual = F.relu(self.conv3(visual))
        visual = F.relu(self.conv4(visual)).flatten(start_dim=1)
        telemetry = F.relu(self.telemetry_fc(telemetry))
        return F.relu(self.fc_fuse(torch.cat((visual, telemetry), dim=-1)))

    def forward(self, obs):
        speed, gear, rpm, images, *_ = obs
        return self.forward_packed(torch.cat((speed, gear, rpm), dim=-1), images)


class TorchPortableRSSM(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc_h = nn.Linear(latent_dim + action_dim + hidden_dim, hidden_dim)
        self.fc_prior_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_prior_std = nn.Linear(hidden_dim, latent_dim)
        self.fc_post_mean = nn.Linear(hidden_dim + latent_dim, latent_dim)
        self.fc_post_std = nn.Linear(hidden_dim + latent_dim, latent_dim)

    def step_deterministic(self, previous_h, previous_z, action):
        return torch.tanh(
            self.fc_h(torch.cat((previous_h, previous_z, action), dim=-1))
        )

    def posterior(self, h, embedding, sample=True):
        inputs = torch.cat((h, embedding), dim=-1)
        raw_mean = self.fc_post_mean(inputs)
        mean = RSSM_MEAN_BOUND * torch.tanh(raw_mean / RSSM_MEAN_BOUND)
        std = RSSM_STD_MIN + (RSSM_STD_MAX - RSSM_STD_MIN) * torch.sigmoid(
            self.fc_post_std(inputs)
        )
        if sample:
            z = mean + std * torch.randn_like(mean)
        else:
            z = mean
        return z, mean


class TorchPortablePolicy(nn.Module):
    def __init__(self, feature_dim, hidden_dim, action_dim, action_scale):
        super().__init__()
        self.hidden_1 = nn.Linear(feature_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.hidden_2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.register_buffer(
            "action_scale", torch.as_tensor(action_scale, dtype=torch.float32)
        )

    def forward(self, feature, test=False):
        hidden = F.silu(self.norm(self.hidden_1(feature)))
        hidden = F.silu(self.hidden_2(hidden))
        raw_mean = self.mu_head(hidden)
        mean = POLICY_MEAN_BOUND * torch.tanh(raw_mean / POLICY_MEAN_BOUND)
        log_std = torch.clamp(self.log_std_head(hidden), LOG_STD_MIN, LOG_STD_MAX)
        if test:
            raw_action = mean
        else:
            raw_action = mean + torch.exp(log_std) * torch.randn_like(mean)
        return torch.tanh(raw_action) * self.action_scale


class TorchPortableRolloutCore(nn.Module):
    """One graph for the recurrent filter, encoder, and deployed policy."""

    def __init__(self, encoder, rssm, policy, deterministic):
        super().__init__()
        self.encoder = encoder
        self.rssm = rssm
        self.policy = policy
        self.deterministic = bool(deterministic)

    def forward(self, telemetry, images, h, z, previous_action):
        h = self.rssm.step_deterministic(h, z, previous_action)
        embedding = self.encoder.forward_channel_first(telemetry, images)
        z, _ = self.rssm.posterior(
            h,
            embedding,
            sample=not self.deterministic,
        )
        action = self.policy(
            torch.cat((h, z), dim=-1),
            test=self.deterministic,
        )
        return h, z, action


class TorchJAXDreamerActor(TorchActorModule):
    """Real-time Torch mirror loaded only from versioned JAX actor bundles."""

    def __init__(
        self,
        observation_space,
        action_space,
        latent_dim=128,
        hidden_dim=256,
        policy_hidden_dim=256,
        img_channels=4,
        img_height=96,
        img_width=96,
        encoder_channels=(16, 32, 64, 64),
        realtime_cpu_tuning=False,
        realtime_cpu_threads=8,
        realtime_cpu_affinity_count=8,
        realtime_high_priority=True,
        device="cpu",
    ):
        super().__init__(observation_space, action_space, device=device)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_space.shape[0])
        self.img_channels = int(img_channels)
        self.img_height = int(img_height)
        self.img_width = int(img_width)
        self.policy_hidden_dim = int(policy_hidden_dim)
        self.encoder_channels = tuple(int(value) for value in encoder_channels)
        self.realtime_cpu_tuning = bool(realtime_cpu_tuning)
        self.realtime_cpu_threads = max(1, int(realtime_cpu_threads))
        self.realtime_cpu_affinity_count = max(
            1, int(realtime_cpu_affinity_count)
        )
        self.realtime_high_priority = bool(realtime_high_priority)
        if self.realtime_cpu_tuning:
            self._configure_realtime_cpu_runtime()
        self.encoder = TorchPortableEncoder(
            self.img_channels,
            self.img_height,
            self.img_width,
            self.latent_dim,
            self.encoder_channels,
        )
        self.rssm = TorchPortableRSSM(
            self.latent_dim, self.action_dim, self.hidden_dim
        )
        self.policy = TorchPortablePolicy(
            self.hidden_dim + self.latent_dim,
            self.policy_hidden_dim,
            self.action_dim,
            np.asarray(action_space.high, dtype=np.float32),
        )
        object.__setattr__(self, "_deterministic_core", None)
        object.__setattr__(self, "_stochastic_core", None)
        self.last_load_succeeded = False
        self.last_load_error = None
        self.nonfinite_action_count = 0
        self.reset()

    def _configure_realtime_cpu_runtime(self):
        """Best-effort worker tuning; only enabled by the production config."""

        torch.set_num_threads(self.realtime_cpu_threads)
        selected_affinity = None
        priority_changed = False
        try:
            import psutil

            process = psutil.Process()
            eligible = process.cpu_affinity()
            if eligible:
                selected_affinity = eligible[-self.realtime_cpu_affinity_count :]
                process.cpu_affinity(selected_affinity)
            if self.realtime_high_priority and os.name == "nt":
                process.nice(psutil.HIGH_PRIORITY_CLASS)
                priority_changed = True
        except Exception as exc:
            logging.warning("Could not fully tune rollout CPU runtime: %s", exc)
        logging.info(
            "Rollout CPU runtime: torch threads=%d, affinity=%s, high priority=%s.",
            self.realtime_cpu_threads,
            selected_affinity,
            priority_changed,
        )

    def architecture_manifest(self):
        return {
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "policy_hidden_dim": self.policy_hidden_dim,
            "action_dim": self.action_dim,
            "img_channels": self.img_channels,
            "img_height": self.img_height,
            "img_width": self.img_width,
            "encoder_channels": list(self.encoder_channels),
            "normalized_observations": True,
            "rssm_state_activation": "tanh",
            "rssm_mean_bound": RSSM_MEAN_BOUND,
            "rssm_std_range": [RSSM_STD_MIN, RSSM_STD_MAX],
            "policy_mean_bound": POLICY_MEAN_BOUND,
            "action_scale": self.policy.action_scale.detach().cpu().tolist(),
        }

    def reset(self):
        self._h = None
        self._z = None
        self._previous_action = None

    def _safe_action_result(self, action):
        result = action[0].detach().cpu().numpy().astype(np.float32, copy=False)
        if np.isfinite(result).all():
            return result
        self.nonfinite_action_count += 1
        logging.error(
            "Portable JAX Dreamer actor produced a non-finite action; "
            "resetting recurrent state and applying a zero action."
        )
        self.reset()
        return np.zeros((self.action_dim,), dtype=np.float32)

    def _compile_rollout_cores(self, telemetry, images):
        h = torch.zeros(
            (1, self.hidden_dim), device=telemetry.device, dtype=torch.float32
        )
        z = torch.zeros(
            (1, self.latent_dim), device=telemetry.device, dtype=torch.float32
        )
        previous_action = torch.zeros(
            (1, self.action_dim), device=telemetry.device, dtype=torch.float32
        )
        examples = (telemetry, images, h, z, previous_action)
        deterministic = torch.jit.trace(
            TorchPortableRolloutCore(
                self.encoder, self.rssm, self.policy, deterministic=True
            ).eval(),
            examples,
            check_trace=False,
            strict=False,
        )
        stochastic = torch.jit.trace(
            TorchPortableRolloutCore(
                self.encoder, self.rssm, self.policy, deterministic=False
            ).eval(),
            examples,
            check_trace=False,
            strict=False,
        )
        object.__setattr__(self, "_deterministic_core", deterministic)
        object.__setattr__(self, "_stochastic_core", stochastic)

    def act(self, obs, test=False):
        batch_size = obs[0].shape[0]
        if batch_size != 1:
            raise ValueError("TorchJAXDreamerActor rollout inference requires batch size 1")
        if self._h is None:
            h = torch.zeros(
                (1, self.hidden_dim), device=obs[0].device, dtype=torch.float32
            )
            z = torch.zeros(
                (1, self.latent_dim), device=obs[0].device, dtype=torch.float32
            )
            previous_action = torch.zeros(
                (1, self.action_dim), device=obs[0].device, dtype=torch.float32
            )
            is_first = True
        else:
            h, z, previous_action = self._h, self._z, self._previous_action
            is_first = False
        h = self.rssm.step_deterministic(h, z, previous_action)
        if is_first:
            h = torch.zeros_like(h)
        embedding = self.encoder(obs)
        z, _ = self.rssm.posterior(h, embedding, sample=not test)
        action = self.policy(torch.cat((h, z), dim=-1), test=test)
        self._h = h
        self._z = z
        self._previous_action = action
        return self._safe_action_result(action)

    def act_(self, obs, test=False):
        """Fast batch-one collation for the real-time TrackMania worker."""

        telemetry_np = np.concatenate(
            tuple(np.asarray(value, dtype=np.float32).reshape(-1) for value in obs[:3])
        )
        image_np = np.asarray(obs[3])
        if not image_np.flags.c_contiguous:
            image_np = np.ascontiguousarray(image_np)
        telemetry = torch.from_numpy(telemetry_np).unsqueeze(0)
        images = torch.from_numpy(image_np).unsqueeze(0)
        if str(self.device) != "cpu":
            telemetry = telemetry.to(self.device)
            images = images.to(self.device)
        with torch.inference_mode():
            if self._deterministic_core is None or self._stochastic_core is None:
                self._compile_rollout_cores(telemetry, images)
            if self._h is None:
                h = torch.zeros(
                    (1, self.hidden_dim),
                    device=telemetry.device,
                    dtype=torch.float32,
                )
                z = torch.zeros(
                    (1, self.latent_dim),
                    device=telemetry.device,
                    dtype=torch.float32,
                )
                previous_action = torch.zeros(
                    (1, self.action_dim),
                    device=telemetry.device,
                    dtype=torch.float32,
                )
                is_first = True
            else:
                h, z, previous_action = self._h, self._z, self._previous_action
                is_first = False
            if is_first:
                # At an episode boundary h is defined as zero; avoid evaluating
                # a recurrent transition whose result would be discarded.
                h = torch.zeros_like(h)
            core = self._deterministic_core if test else self._stochastic_core
            if is_first:
                embedding = self.encoder.forward_packed(telemetry, images)
                z, _ = self.rssm.posterior(h, embedding, sample=not test)
                action = self.policy(torch.cat((h, z), dim=-1), test=test)
            else:
                h, z, action = core(telemetry, images, h, z, previous_action)
            self._h = h
            self._z = z
            self._previous_action = action
            return self._safe_action_result(action)

    def warmup(self):
        sample = tuple(
            np.zeros(space.shape, dtype=space.dtype)
            for space in self.observation_space
        )
        self.reset()
        # Compile from the current parameter snapshot. A subsequent bundle load
        # invalidates and rebuilds these graphs.
        for _ in range(5):
            self.act_(sample, test=True)
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(device=self.device)
        self.reset()

    @staticmethod
    def _copy_linear(module, parameters, prefix):
        kernel = torch.from_numpy(np.asarray(parameters[f"{prefix}/kernel"]).T.copy())
        bias = torch.from_numpy(np.asarray(parameters[f"{prefix}/bias"]).copy())
        if tuple(module.weight.shape) != tuple(kernel.shape):
            raise ValueError(f"shape mismatch for {prefix}/kernel")
        module.weight.copy_(kernel)
        module.bias.copy_(bias)

    @staticmethod
    def _copy_conv(module, parameters, prefix):
        source = np.asarray(parameters[f"{prefix}/kernel"])
        kernel = torch.from_numpy(np.transpose(source, (3, 2, 0, 1)).copy())
        bias = torch.from_numpy(np.asarray(parameters[f"{prefix}/bias"]).copy())
        if tuple(module.weight.shape) != tuple(kernel.shape):
            raise ValueError(f"shape mismatch for {prefix}/kernel")
        module.weight.copy_(kernel)
        module.bias.copy_(bias)

    def _load_portable_parameters(self, parameters):
        with torch.no_grad():
            for name in ("conv1", "conv2", "conv3", "conv4"):
                self._copy_conv(
                    getattr(self.encoder, name),
                    parameters,
                    f"world_model/encoder/{name}",
                )
            self._copy_linear(
                self.encoder.telemetry_fc,
                parameters,
                "world_model/encoder/telemetry_fc",
            )
            self._copy_linear(
                self.encoder.fc_fuse,
                parameters,
                "world_model/encoder/fc_fuse",
            )
            for name in (
                "fc_h",
                "fc_prior_mean",
                "fc_prior_std",
                "fc_post_mean",
                "fc_post_std",
            ):
                self._copy_linear(
                    getattr(self.rssm, name), parameters, f"world_model/rssm/{name}"
                )
            for name in ("hidden_1", "hidden_2", "mu_head", "log_std_head"):
                self._copy_linear(
                    getattr(self.policy, name), parameters, f"policy/{name}"
                )
            self.policy.norm.weight.copy_(
                torch.from_numpy(np.asarray(parameters["policy/norm/scale"]).copy())
            )
            self.policy.norm.bias.copy_(
                torch.from_numpy(np.asarray(parameters["policy/norm/bias"]).copy())
            )

    def load(self, path, device):
        self.device = device
        self.last_load_succeeded = False
        self.last_load_error = None
        try:
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
            if not isinstance(payload, dict) or payload.get("magic") != ACTOR_BUNDLE_MAGIC:
                raise ValueError("not a portable JAX Dreamer actor bundle")
            if payload.get("version") != ACTOR_BUNDLE_VERSION:
                raise ValueError(f"unsupported actor bundle version: {payload.get('version')}")
            if payload.get("architecture") != self.architecture_manifest():
                raise ValueError("portable actor architecture mismatch")
            parameters = payload.get("parameters")
            if not isinstance(parameters, dict):
                raise ValueError("portable actor parameters are missing")
            if _parameter_checksum(parameters) != payload.get("parameter_checksum"):
                raise ValueError("portable actor parameter checksum failed")
            _validate_portable_parameters(parameters)
            self._load_portable_parameters(parameters)
            self.to(device)
            object.__setattr__(self, "_deterministic_core", None)
            object.__setattr__(self, "_stochastic_core", None)
            self.reset()
            self.warmup()
            self.last_load_succeeded = True
        except Exception as exc:
            self.last_load_error = str(exc)
            logging.warning(
                "Ignoring incompatible actor weights while waiting for the first "
                "portable JAX Dreamer broadcast: %s",
                exc,
            )
        return self
