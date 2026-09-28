"""Deployed JAX/Flax NNX Dreamer foundation for the continual pipeline.

This module implements M1 only: one recurrent foundation actor trained from
real episode-safe replay and latent imagination. Expert growth, routing,
protected replay, consolidation, and AZR remain disabled until later gates.
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import pickle
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from tmrl.core.jax.actor import NNXActorModule
from tmrl.core.jax.util import get_rngs
from tmrl.core.training import TrainingAgent
from tmrl.custom.jax.world_model import LatentWorldModel, symlog, symexp


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
POLICY_MEAN_BOUND = 5.0
ACTOR_BUNDLE_MAGIC = "tmrl-jax-dreamer-actor"
ACTOR_BUNDLE_VERSION = 4
MAX_CONSECUTIVE_NONFINITE_GRADIENTS = 1_000_000_000


def _scalar_sequence(value: jnp.ndarray) -> jnp.ndarray:
    if value.ndim >= 3 and value.shape[-1] == 1:
        return value[..., 0]
    return value


def gaussian_kl(p_mean, p_std, q_mean, q_std):
    p_var = jnp.square(p_std)
    q_var = jnp.square(q_std)
    element = (
        jnp.log(q_std / (p_std + 1e-6))
        + (p_var + jnp.square(p_mean - q_mean)) / (2.0 * q_var + 1e-6)
        - 0.5
    )
    return jnp.sum(element, axis=-1)


def lambda_returns(rewards, discounts, next_values, lambda_=0.95):
    if rewards.shape != discounts.shape or rewards.shape != next_values.shape:
        raise ValueError("reward, discount, and next-value arrays must match")
    outputs = []
    carry = next_values[-1]
    for time_index in reversed(range(rewards.shape[0])):
        carry = rewards[time_index] + discounts[time_index] * (
            (1.0 - lambda_) * next_values[time_index] + lambda_ * carry
        )
        outputs.append(carry)
    return jnp.stack(outputs[::-1], axis=0)


def portable_parameter_checksum(parameters):
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


def validate_portable_parameters(parameters, label="portable actor"):
    """Reject checksummed-but-non-finite weights before they reach a worker."""

    bad = []
    for name, value in parameters.items():
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.inexact) and not np.isfinite(array).all():
            bad.append(str(name))
    if bad:
        preview = ", ".join(bad[:8])
        suffix = "" if len(bad) <= 8 else f" (+{len(bad) - 8} more)"
        raise FloatingPointError(
            f"{label} contains non-finite parameters: {preview}{suffix}"
        )


def nonfinite_nnx_state_paths(module):
    """Return floating-point NNX state paths containing NaN or infinity."""

    bad = []
    for variable_path, variable in nnx.to_flat_state(nnx.state(module)):
        raw_value = jax.device_get(variable[...])
        try:
            is_inexact = jnp.issubdtype(raw_value.dtype, jnp.inexact)
        except TypeError:
            is_inexact = False
        if not is_inexact:
            continue
        value = np.asarray(raw_value)
        if not np.isfinite(value).all():
            bad.append("/".join(str(part) for part in variable_path))
    return bad


def assert_finite_nnx_state(module, label="NNX state"):
    bad = nonfinite_nnx_state_paths(module)
    if bad:
        preview = ", ".join(bad[:8])
        suffix = "" if len(bad) <= 8 else f" (+{len(bad) - 8} more)"
        raise FloatingPointError(
            f"{label} contains non-finite values: {preview}{suffix}"
        )


def _tree_all_finite(tree):
    checks = [
        jnp.all(jnp.isfinite(value))
        for value in jax.tree.leaves(tree)
        if jnp.issubdtype(value.dtype, jnp.inexact)
    ]
    if not checks:
        return jnp.asarray(True)
    return jnp.all(jnp.stack(checks))


def _assert_finite_training_metrics(metrics, stage, finite_flags=()):
    host_metrics = {
        name: np.asarray(jax.device_get(value)) for name, value in metrics.items()
    }
    bad = [
        name
        for name, value in host_metrics.items()
        if np.issubdtype(value.dtype, np.inexact) and not np.isfinite(value).all()
    ]
    rejected = [
        name
        for name in finite_flags
        if name not in host_metrics or not bool(np.asarray(host_metrics[name]).all())
    ]
    if bad or rejected:
        details = []
        if bad:
            details.append("non-finite metrics=" + ",".join(bad))
        if rejected:
            details.append("non-finite gradients=" + ",".join(rejected))
        raise FloatingPointError(
            f"{stage} update rejected; " + "; ".join(details)
        )


def _safe_optimizer(inner):
    return optax.apply_if_finite(
        inner,
        max_consecutive_errors=MAX_CONSECUTIVE_NONFINITE_GRADIENTS,
    )


class JAXDreamerPolicy(nnx.Module):
    def __init__(
        self,
        feature_dim: int,
        action_scale,
        hidden_dim: int = 256,
        rngs: nnx.Rngs | None = None,
    ):
        rngs = rngs or get_rngs()
        self.hidden_1 = nnx.Linear(feature_dim, hidden_dim, rngs=rngs)
        self.norm = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.hidden_2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.mu_head = nnx.Linear(hidden_dim, len(action_scale), rngs=rngs)
        self.log_std_head = nnx.Linear(hidden_dim, len(action_scale), rngs=rngs)
        self.action_scale = tuple(float(value) for value in action_scale)

    def __call__(self, feature, rngs: nnx.Rngs, test=False, with_logprob=True):
        hidden = jax.nn.silu(self.norm(self.hidden_1(feature)))
        hidden = jax.nn.silu(self.hidden_2(hidden))
        raw_mean = self.mu_head(hidden)
        mean = POLICY_MEAN_BOUND * jnp.tanh(raw_mean / POLICY_MEAN_BOUND)
        log_std = jnp.clip(self.log_std_head(hidden), LOG_STD_MIN, LOG_STD_MAX)
        std = jnp.exp(log_std)
        noise = jax.random.normal(rngs.noise(), mean.shape)
        raw_action = jnp.where(jnp.asarray(test), mean, mean + std * noise)
        if with_logprob:
            log_prob = (
                -0.5 * jnp.square((raw_action - mean) / (std + 1e-8))
                - log_std
                - 0.5 * jnp.log(2.0 * jnp.pi)
            ).sum(axis=-1)
            log_prob -= (
                2.0
                * (jnp.log(2.0) - raw_action - jax.nn.softplus(-2.0 * raw_action))
            ).sum(axis=-1)
        else:
            log_prob = jnp.zeros(mean.shape[:-1], dtype=mean.dtype)
        scale = jnp.asarray(self.action_scale, dtype=mean.dtype)
        return jnp.tanh(raw_action) * scale, log_prob


class JAXDreamerValue(nnx.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 256, rngs=None):
        rngs = rngs or get_rngs()
        self.hidden_1 = nnx.Linear(feature_dim, hidden_dim, rngs=rngs)
        self.norm = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.hidden_2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, feature):
        hidden = jax.nn.silu(self.norm(self.hidden_1(feature)))
        hidden = jax.nn.silu(self.hidden_2(hidden))
        return self.output(hidden).squeeze(-1)


@nnx.jit
def _compiled_actor_step(actor, obs, test):
    h = actor.rollout_h[...]
    z = actor.rollout_z[...]
    previous_action = actor.rollout_action[...]
    is_first = actor.rollout_is_first[...]
    next_h = actor.world_model.rssm.step_deterministic(h, z, previous_action)
    next_h = jnp.where(is_first[:, None], jnp.zeros_like(next_h), next_h)
    embedding = actor.world_model.encoder(obs)
    sampled_z, posterior_mean, _ = actor.world_model.rssm.compute_posterior(
        next_h, embedding, actor.rngs
    )
    next_z = jnp.where(jnp.asarray(test), posterior_mean, sampled_z)
    feature = actor.world_model.get_feature(next_h, next_z)
    action, _ = actor.policy(
        feature,
        rngs=actor.rngs,
        test=test,
        with_logprob=False,
    )
    actor.rollout_h[...] = next_h
    actor.rollout_z[...] = next_z
    actor.rollout_action[...] = action
    actor.rollout_is_first[...] = jnp.zeros_like(is_first)
    return action


class JAXDreamerActor(NNXActorModule):
    """Recurrent NNX actor whose latent policy is optimized in imagination."""

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
        rngs_seed=0,
        internal_rngs_seed=1000,
        device=None,
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
        init_rngs = get_rngs(rngs_seed, rngs_seed + 1, rngs_seed + 2)
        self.rngs = get_rngs(
            internal_rngs_seed,
            internal_rngs_seed + 1,
            internal_rngs_seed + 2,
        )
        self.world_model = LatentWorldModel(
            img_hist_len=self.img_channels,
            img_height=self.img_height,
            img_width=self.img_width,
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            encoder_channels=self.encoder_channels,
            normalized_observations=True,
            stable_rssm=True,
            rngs=init_rngs,
        )
        self.policy = JAXDreamerPolicy(
            feature_dim=self.hidden_dim + self.latent_dim,
            action_scale=np.asarray(action_space.high, dtype=np.float32),
            hidden_dim=self.policy_hidden_dim,
            rngs=init_rngs,
        )
        self.rollout_h = nnx.Variable(
            jnp.zeros((1, self.hidden_dim), dtype=jnp.float32)
        )
        self.rollout_z = nnx.Variable(
            jnp.zeros((1, self.latent_dim), dtype=jnp.float32)
        )
        self.rollout_action = nnx.Variable(
            jnp.zeros((1, self.action_dim), dtype=jnp.float32)
        )
        self.rollout_is_first = nnx.Variable(jnp.ones((1,), dtype=jnp.bool_))

    @property
    def feature_dim(self):
        return self.hidden_dim + self.latent_dim

    def initial(self, batch_size):
        return (
            jnp.zeros((batch_size, self.hidden_dim), dtype=jnp.float32),
            jnp.zeros((batch_size, self.latent_dim), dtype=jnp.float32),
            jnp.zeros((batch_size, self.action_dim), dtype=jnp.float32),
        )

    def reset(self):
        self.rollout_h[...] = jnp.zeros_like(self.rollout_h[...])
        self.rollout_z[...] = jnp.zeros_like(self.rollout_z[...])
        self.rollout_action[...] = jnp.zeros_like(self.rollout_action[...])
        self.rollout_is_first[...] = jnp.ones_like(self.rollout_is_first[...])

    def act(self, obs, test=False):
        if obs[0].shape[0] != 1:
            raise ValueError("JAXDreamerActor rollout inference requires batch size 1")
        action = _compiled_actor_step(self, obs, jnp.asarray(test))
        result = np.asarray(action[0])
        if not np.isfinite(result).all():
            self.reset()
            raise FloatingPointError("JAX Dreamer actor produced a non-finite action")
        return result.astype(np.float32, copy=False)

    def warmup(self):
        sample = tuple(
            np.zeros(space.shape, dtype=space.dtype)
            for space in self.observation_space
        )
        self.reset()
        self.act_(sample, test=True)
        self.act_(sample, test=True)
        self.reset()

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
            "rssm_mean_bound": 5.0,
            "rssm_std_range": [0.1, 2.0],
            "policy_mean_bound": POLICY_MEAN_BOUND,
            "action_scale": list(self.policy.action_scale),
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.reset()
        _, state = nnx.split(self)
        flat_parameters = nnx.to_flat_state(nnx.state(self, nnx.Param))
        parameters = {
            "/".join(str(part) for part in variable_path): np.asarray(
                jax.device_get(variable[...])
            )
            for variable_path, variable in flat_parameters
        }
        validate_portable_parameters(parameters)
        payload = {
            "magic": ACTOR_BUNDLE_MAGIC,
            "version": ACTOR_BUNDLE_VERSION,
            "architecture": self.architecture_manifest(),
            "state": jax.device_get(state),
            "parameters": parameters,
            "parameter_checksum": portable_parameter_checksum(parameters),
        }
        handle = tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        )
        tmp_path = Path(handle.name)
        try:
            with handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def load(self, path, device):
        try:
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
            if not isinstance(payload, dict) or payload.get("magic") != ACTOR_BUNDLE_MAGIC:
                raise ValueError("not a JAX Dreamer actor bundle")
            if payload.get("version") != ACTOR_BUNDLE_VERSION:
                raise ValueError(f"unsupported actor bundle version: {payload.get('version')}")
            if payload.get("architecture") != self.architecture_manifest():
                expected = json.dumps(self.architecture_manifest(), sort_keys=True)
                received = json.dumps(payload.get("architecture"), sort_keys=True)
                raise ValueError(
                    f"actor architecture mismatch; expected {expected}, received {received}"
                )
            parameters = payload.get("parameters")
            if not isinstance(parameters, dict):
                raise ValueError("actor bundle has no portable parameters")
            if portable_parameter_checksum(parameters) != payload.get(
                "parameter_checksum"
            ):
                raise ValueError("actor bundle parameter checksum failed")
            validate_portable_parameters(parameters)
            nnx.update(self, payload["state"])
            assert_finite_nnx_state(self, "loaded JAX actor")
            self.to_device(device)
            self.reset()
        except Exception as exc:
            logging.warning(
                "Ignoring incompatible actor weights while waiting for the first "
                "JAX Dreamer broadcast: %s",
                exc,
            )
        return self

    def to_device(self, device: str):
        self.reset()
        return super().to_device(device)


class JAXDreamerAgent(nnx.Module, TrainingAgent):
    """M1 JAX Dreamer agent with real-replay and latent-imagination updates."""

    def __init__(
        self,
        observation_space,
        action_space,
        device=None,
        latent_dim=128,
        hidden_dim=256,
        policy_hidden_dim=256,
        lr_world_model=1e-4,
        lr_actor=3e-5,
        lr_critic=3e-5,
        gamma=0.99,
        lambda_=0.95,
        free_nats=1.0,
        dyn_kl_scale=1.0,
        rep_kl_scale=0.1,
        telemetry_scale=1.0,
        embedding_scale=1.0,
        imagination_horizon=8,
        imagination_batch_size=64,
        burn_in=5,
        world_model_warmup_steps=2000,
        entropy_scale=3e-4,
        target_polyak=0.99,
        grad_clip=10.0,
        encoder_channels=(16, 32, 64, 64),
        seed=0,
    ):
        TrainingAgent.__init__(self, observation_space, action_space, device)
        self.checkpoint_mode = "CONTINUAL_DREAMER_JAX"
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.gamma = float(gamma)
        self.lambda_ = float(lambda_)
        self.free_nats = float(free_nats)
        self.dyn_kl_scale = float(dyn_kl_scale)
        self.rep_kl_scale = float(rep_kl_scale)
        self.telemetry_scale = float(telemetry_scale)
        self.embedding_scale = float(embedding_scale)
        self.imagination_horizon = int(imagination_horizon)
        self.imagination_batch_size = int(imagination_batch_size)
        self.burn_in = int(burn_in)
        self.world_model_warmup_steps = int(world_model_warmup_steps)
        self.entropy_scale = float(entropy_scale)
        self.target_polyak = float(target_polyak)
        self.grad_clip = float(grad_clip)
        if self.imagination_horizon < 1 or self.imagination_batch_size < 1:
            raise ValueError("imagination horizon and batch size must be positive")

        image_shape = observation_space[3].shape
        if len(image_shape) != 3:
            raise ValueError("JAX Dreamer requires stacked grayscale image observations")
        if image_shape[0] <= 8:
            img_channels, img_height, img_width = image_shape
        else:
            img_height, img_width, img_channels = image_shape
        self.rngs = get_rngs(seed + 10, seed + 11, seed + 12)
        self.actor = JAXDreamerActor(
            observation_space=observation_space,
            action_space=action_space,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            policy_hidden_dim=policy_hidden_dim,
            encoder_channels=encoder_channels,
            img_channels=img_channels,
            img_height=img_height,
            img_width=img_width,
            rngs_seed=seed,
            internal_rngs_seed=seed + 1000,
        )
        self.critic = JAXDreamerValue(
            self.actor.feature_dim,
            hidden_dim=policy_hidden_dim,
            rngs=get_rngs(seed + 20, seed + 21, seed + 22),
        )
        self.target_critic = nnx.clone(self.critic)
        self.world_optimizer = nnx.Optimizer(
            self.actor.world_model,
            _safe_optimizer(
                optax.chain(
                    optax.clip_by_global_norm(self.grad_clip),
                    optax.adam(lr_world_model),
                )
            ),
            wrt=nnx.Param,
        )
        self.actor_optimizer = nnx.Optimizer(
            self.actor.policy,
            _safe_optimizer(
                optax.chain(
                    optax.clip_by_global_norm(self.grad_clip),
                    optax.adam(lr_actor),
                )
            ),
            wrt=nnx.Param,
        )
        self.critic_optimizer = nnx.Optimizer(
            self.critic,
            _safe_optimizer(
                optax.chain(
                    optax.clip_by_global_norm(self.grad_clip),
                    optax.adam(lr_critic),
                )
            ),
            wrt=nnx.Param,
        )
        self.world_model_updates = nnx.Variable(jnp.asarray(0, dtype=jnp.int32))
        self.train_steps = nnx.Variable(jnp.asarray(0, dtype=jnp.int32))

    def get_actor(self):
        actor = nnx.clone(self.actor)
        actor.reset()
        return actor

    @staticmethod
    def _flatten_observation_sequence(obs):
        batch, sequence = obs[0].shape[:2]
        return tuple(
            value.reshape(batch * sequence, *value.shape[2:]) for value in obs
        )

    @staticmethod
    def _observation_at(obs, time_index):
        return tuple(value[:, time_index] for value in obs)

    @nnx.jit
    def _train_world_model(self, batch):
        if len(batch) != 7:
            raise ValueError("JAXDreamerAgent requires an is_first sequence mask")
        obs, actions, rewards, next_obs, terminated, truncated, is_first = batch
        batch_size, sequence_length = actions.shape[:2]
        if self.burn_in >= sequence_length:
            raise ValueError("Dreamer burn_in must be shorter than replay sequences")

        flat_next_obs = self._flatten_observation_sequence(next_obs)
        initial_obs = self._observation_at(obs, 0)

        def loss_fn(world_model, rngs):
            next_embeddings = world_model.encoder(flat_next_obs).reshape(
                batch_size, sequence_length, self.latent_dim
            )
            h, z, previous_action = self.actor.initial(batch_size)
            h = world_model.rssm.step_deterministic(h, z, previous_action)
            h = jnp.where(is_first[:, 0, None], jnp.zeros_like(h), h)
            initial_embedding = world_model.encoder(initial_obs)
            z, _, _ = world_model.rssm.compute_posterior(h, initial_embedding, rngs)

            features = []
            posterior_h = []
            posterior_z = []
            prior_means = []
            prior_stds = []
            posterior_means = []
            posterior_stds = []
            for time_index in range(sequence_length):
                h = world_model.rssm.step_deterministic(
                    h, z, actions[:, time_index]
                )
                _, prior_mean, prior_std = world_model.rssm.predict_prior(h, rngs)
                z, posterior_mean, posterior_std = world_model.rssm.compute_posterior(
                    h, next_embeddings[:, time_index], rngs
                )
                feature = world_model.get_feature(h, z)
                features.append(feature)
                posterior_h.append(h)
                posterior_z.append(z)
                prior_means.append(prior_mean)
                prior_stds.append(prior_std)
                posterior_means.append(posterior_mean)
                posterior_stds.append(posterior_std)

            feature_sequence = jnp.stack(features, axis=1)
            flat_feature = feature_sequence.reshape(
                batch_size * sequence_length, -1
            )
            reward_prediction = world_model.predict_reward(flat_feature).reshape(
                batch_size, sequence_length
            )
            continuation_logits = world_model.predict_continuation_logits(
                flat_feature
            ).reshape(batch_size, sequence_length)
            embedding_prediction = world_model.predict_embedding(flat_feature).reshape(
                batch_size, sequence_length, self.latent_dim
            )
            telemetry_prediction = world_model.predict_telemetry(flat_feature).reshape(
                batch_size, sequence_length, 3
            )

            reward_target = symlog(_scalar_sequence(rewards))
            reward_loss = jnp.mean(
                optax.huber_loss(reward_prediction, reward_target)
            )
            done = jnp.maximum(
                _scalar_sequence(terminated), _scalar_sequence(truncated)
            )
            continuation_target = 1.0 - done
            continuation_loss = jnp.mean(
                optax.sigmoid_binary_cross_entropy(
                    continuation_logits, continuation_target
                )
            )
            embedding_loss = jnp.mean(
                jnp.square(
                    embedding_prediction - jax.lax.stop_gradient(next_embeddings)
                )
            )
            telemetry_target = jnp.concatenate(
                (
                    next_obs[0],
                    next_obs[1],
                    next_obs[2],
                ),
                axis=-1,
            )
            telemetry_loss = jnp.mean(
                optax.huber_loss(telemetry_prediction, telemetry_target)
            )

            prior_mean = jnp.stack(prior_means, axis=1)
            prior_std = jnp.stack(prior_stds, axis=1)
            posterior_mean = jnp.stack(posterior_means, axis=1)
            posterior_std = jnp.stack(posterior_stds, axis=1)
            dynamics_kl = gaussian_kl(
                jax.lax.stop_gradient(posterior_mean),
                jax.lax.stop_gradient(posterior_std),
                prior_mean,
                prior_std,
            )
            representation_kl = gaussian_kl(
                posterior_mean,
                posterior_std,
                jax.lax.stop_gradient(prior_mean),
                jax.lax.stop_gradient(prior_std),
            )
            dynamics_kl_loss = jnp.mean(jnp.maximum(dynamics_kl, self.free_nats))
            representation_kl_loss = jnp.mean(
                jnp.maximum(representation_kl, self.free_nats)
            )
            total = (
                reward_loss
                + continuation_loss
                + self.embedding_scale * embedding_loss
                + self.telemetry_scale * telemetry_loss
                + self.dyn_kl_scale * dynamics_kl_loss
                + self.rep_kl_scale * representation_kl_loss
            )
            starts = (
                jax.lax.stop_gradient(jnp.stack(posterior_h, axis=1)),
                jax.lax.stop_gradient(jnp.stack(posterior_z, axis=1)),
            )
            metrics = {
                "loss_world_model": total,
                "loss_wm_reward": reward_loss,
                "loss_wm_continue": continuation_loss,
                "loss_wm_embedding": embedding_loss,
                "loss_wm_telemetry": telemetry_loss,
                "loss_wm_kl_dyn": dynamics_kl_loss,
                "loss_wm_kl_rep": representation_kl_loss,
            }
            return total, (starts, metrics)

        (_, (starts, metrics)), gradients = nnx.value_and_grad(
            loss_fn, has_aux=True
        )(self.actor.world_model, self.rngs)
        metrics["world_gradients_finite"] = _tree_all_finite(gradients).astype(
            jnp.float32
        )
        self.world_optimizer.update(self.actor.world_model, gradients)
        return starts, metrics

    @nnx.jit
    def _train_imagination(self, h_sequence, z_sequence):
        h_candidates = h_sequence[:, self.burn_in :].reshape(-1, self.hidden_dim)
        z_candidates = z_sequence[:, self.burn_in :].reshape(-1, self.latent_dim)
        sample_count = min(self.imagination_batch_size, h_candidates.shape[0])
        indices = jax.random.permutation(
            self.rngs.noise(), h_candidates.shape[0]
        )[:sample_count]
        start_h = h_candidates[indices]
        start_z = z_candidates[indices]

        def actor_loss_fn(policy, rngs):
            h = start_h
            z = start_z
            rewards = []
            discounts = []
            next_values = []
            log_probs = []
            state_features = []
            weights = []
            weight = jnp.ones((sample_count,), dtype=h.dtype)
            for _ in range(self.imagination_horizon):
                feature = self.actor.world_model.get_feature(h, z)
                state_features.append(feature)
                weights.append(weight)
                action, log_prob = policy(feature, rngs=rngs, test=False)
                h, z, reward, continuation = self.actor.world_model.imagine_step(
                    h, z, action, rngs
                )
                next_feature = self.actor.world_model.get_feature(h, z)
                reward = jnp.clip(reward.squeeze(-1), -10.0, 10.0)
                continuation = continuation.squeeze(-1)
                discount = self.gamma * continuation
                value = symexp(self.target_critic(next_feature))
                rewards.append(reward)
                discounts.append(discount)
                next_values.append(value)
                log_probs.append(log_prob)
                weight = weight * discount

            rewards = jnp.stack(rewards, axis=0)
            discounts = jnp.stack(discounts, axis=0)
            next_values = jnp.stack(next_values, axis=0)
            log_probs = jnp.stack(log_probs, axis=0)
            weights = jnp.stack(weights, axis=0)
            returns = lambda_returns(
                rewards, discounts, next_values, lambda_=self.lambda_
            )
            return_scale = jnp.maximum(jnp.std(returns), 1.0)
            actor_objective = returns / return_scale - self.entropy_scale * log_probs
            actor_loss = -jnp.mean(
                jax.lax.stop_gradient(weights) * actor_objective
            )
            aux = (
                jax.lax.stop_gradient(returns),
                jax.lax.stop_gradient(weights),
                jax.lax.stop_gradient(jnp.stack(state_features, axis=0)),
                jnp.mean(returns),
                jnp.mean(weights),
                return_scale,
            )
            return actor_loss, aux

        (actor_loss, aux), actor_gradients = nnx.value_and_grad(
            actor_loss_fn, has_aux=True
        )(self.actor.policy, self.rngs)
        actor_gradients_finite = _tree_all_finite(actor_gradients)
        self.actor_optimizer.update(self.actor.policy, actor_gradients)
        returns, weights, state_features, mean_return, mean_survival, scale = aux

        def critic_loss_fn(critic):
            prediction = critic(
                state_features.reshape(-1, state_features.shape[-1])
            )
            target = symlog(returns).reshape(-1)
            error = optax.huber_loss(prediction, target).reshape(weights.shape)
            return jnp.mean(weights * error)

        critic_loss, critic_gradients = nnx.value_and_grad(critic_loss_fn)(
            self.critic
        )
        critic_gradients_finite = _tree_all_finite(critic_gradients)
        self.critic_optimizer.update(self.critic, critic_gradients)
        critic_params = nnx.state(self.critic, nnx.Param)
        target_params = nnx.state(self.target_critic, nnx.Param)
        updated_target = jax.tree.map(
            lambda target, current: (
                self.target_polyak * target
                + (1.0 - self.target_polyak) * current
            ),
            target_params,
            critic_params,
        )
        nnx.update(self.target_critic, updated_target)
        return {
            "loss_actor": actor_loss,
            "loss_critic": critic_loss,
            "mean_imagined_return": mean_return,
            "mean_imagined_survival": mean_survival,
            "imagined_return_scale": scale,
            "imagined_policy_updates": jnp.asarray(1.0, dtype=jnp.float32),
            "actor_gradients_finite": actor_gradients_finite.astype(jnp.float32),
            "critic_gradients_finite": critic_gradients_finite.astype(jnp.float32),
        }

    def train(self, batch):
        (h_sequence, z_sequence), metrics = self._train_world_model(batch)
        _assert_finite_training_metrics(
            metrics,
            "world model",
            finite_flags=("world_gradients_finite",),
        )
        self.world_model_updates[...] = self.world_model_updates[...] + 1
        self.train_steps[...] = self.train_steps[...] + 1
        update_count = int(np.asarray(jax.device_get(self.world_model_updates[...])))
        ready = update_count >= self.world_model_warmup_steps
        metrics.update(
            {
                "dreamer_ready": jnp.asarray(float(ready), dtype=jnp.float32),
                "dreamer_warmup_updates": jnp.asarray(
                    float(update_count), dtype=jnp.float32
                ),
                "dreamer_warmup_remaining": jnp.asarray(
                    float(max(self.world_model_warmup_steps - update_count, 0)),
                    dtype=jnp.float32,
                ),
                "loss_actor": jnp.asarray(0.0, dtype=jnp.float32),
                "loss_critic": jnp.asarray(0.0, dtype=jnp.float32),
                "mean_imagined_return": jnp.asarray(0.0, dtype=jnp.float32),
                "mean_imagined_survival": jnp.asarray(0.0, dtype=jnp.float32),
                "imagined_return_scale": jnp.asarray(0.0, dtype=jnp.float32),
                "imagined_policy_updates": jnp.asarray(0.0, dtype=jnp.float32),
            }
        )
        if ready:
            imagination_metrics = self._train_imagination(h_sequence, z_sequence)
            _assert_finite_training_metrics(
                imagination_metrics,
                "imagination",
                finite_flags=(
                    "actor_gradients_finite",
                    "critic_gradients_finite",
                ),
            )
            metrics.update(imagination_metrics)
        return metrics
