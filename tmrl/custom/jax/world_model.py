import jax
import jax.numpy as jnp
from flax import nnx
from typing import Tuple, Optional
from tmrl.core.jax.util import get_rngs


def _configured_image_dimension(name: str, fallback: int = 96) -> int:
    """Resolve legacy config defaults only when a caller omits dimensions."""

    try:
        import tmrl.config.config_constants as cfg

        return int(getattr(cfg, name))
    except (FileNotFoundError, KeyError, RuntimeError):
        return int(fallback)


def symlog(x: jnp.ndarray) -> jnp.ndarray:
    """Symmetric logarithmic transformation for scale-invariant regression."""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x: jnp.ndarray) -> jnp.ndarray:
    """Inverse symmetric exponential transformation with numerical clamping."""
    return jnp.sign(x) * (jnp.expm1(jnp.clip(jnp.abs(x), 0.0, 15.0)))


class VisualTelemetryEncoder(nnx.Module):
    """
    Encodes visual frames and vehicle telemetry into a compact latent representation.
    """
    def __init__(self,
                 img_hist_len: int = 4,
                 img_height: int | None = None,
                 img_width: int | None = None,
                 latent_dim: int = 128,
                 channels: tuple[int, int, int, int] = (32, 64, 128, 128),
                 normalized_observations: bool = False,
                 rngs: nnx.Rngs = None):
        super().__init__()
        rngs = rngs or get_rngs()
        img_height = (
            _configured_image_dimension("IMG_HEIGHT")
            if img_height is None
            else int(img_height)
        )
        img_width = (
            _configured_image_dimension("IMG_WIDTH")
            if img_width is None
            else int(img_width)
        )
        self.img_hist_len = img_hist_len
        self.latent_dim = latent_dim
        if len(channels) != 4 or any(int(value) < 1 for value in channels):
            raise ValueError("encoder channels must contain four positive widths")
        channels = tuple(int(value) for value in channels)
        self.channels = channels
        self.normalized_observations = bool(normalized_observations)

        # Visual Conv layers (padding='VALID' to match PyTorch/TMRL parity)
        self.conv1 = nnx.Conv(img_hist_len, channels[0], kernel_size=(8, 8), strides=(2, 2), padding='VALID', rngs=rngs)
        self.conv2 = nnx.Conv(channels[0], channels[1], kernel_size=(4, 4), strides=(2, 2), padding='VALID', rngs=rngs)
        self.conv3 = nnx.Conv(channels[1], channels[2], kernel_size=(4, 4), strides=(2, 2), padding='VALID', rngs=rngs)
        self.conv4 = nnx.Conv(channels[2], channels[3], kernel_size=(4, 4), strides=(2, 2), padding='VALID', rngs=rngs)

        # Telemetry MLP
        self.telemetry_fc = nnx.Linear(3, 32, rngs=rngs)  # speed, gear, rpm

        # Fusion head. Compute the convolution output rather than assuming 96x96.
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
        self.fc_fuse = nnx.Linear(channels[3] * conv_h * conv_w + 32, latent_dim, rngs=rngs)

    def __call__(self, obs_tuple) -> jnp.ndarray:
        speed, gear, rpm, images, *rest = obs_tuple

        # Permute (B, C, H, W) to (B, H, W, C) for JAX/Flax Conv
        if images.ndim == 4 and images.shape[1] == self.img_hist_len and images.shape[-1] != self.img_hist_len:
            images = jnp.transpose(images, (0, 2, 3, 1))

        images = images.astype(jnp.float32)
        if jnp.issubdtype(obs_tuple[3].dtype, jnp.integer):
            images = images / 255.0

        x_img = nnx.relu(self.conv1(images))
        x_img = nnx.relu(self.conv2(x_img))
        x_img = nnx.relu(self.conv3(x_img))
        x_img = nnx.relu(self.conv4(x_img))
        x_img = x_img.reshape(x_img.shape[0], -1)

        if self.normalized_observations:
            # The production rollout preprocessor and sequence replay both
            # provide normalized telemetry.
            telem = jnp.concatenate([speed, gear, rpm], axis=-1)
        else:
            # Preserve the raw-observation contract used by the standalone
            # world-model/AZR experiment.
            telem = jnp.concatenate(
                [speed / 300.0, gear / 5.0, rpm / 10000.0], axis=-1
            )
        x_telem = nnx.relu(self.telemetry_fc(telem))

        fused = jnp.concatenate([x_img, x_telem], axis=-1)
        latent = nnx.relu(self.fc_fuse(fused))
        return latent


class LatentRSSM(nnx.Module):
    """
    Recurrent State-Space Model (RSSM) predicting deterministic belief (h_t)
    and stochastic latent states (z_t) under continuous control actions.
    """
    def __init__(self,
                 latent_dim: int = 128,
                 action_dim: int = 3,
                 hidden_dim: int = 256,
                 stable: bool = False,
                 rngs: nnx.Rngs = None):
        super().__init__()
        rngs = rngs or get_rngs()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.stable = bool(stable)
        self.mean_bound = 5.0
        self.std_min = 0.1
        self.std_max = 2.0

        # Sequence cell (Deterministic memory h_t)
        self.fc_h = nnx.Linear(latent_dim + action_dim + hidden_dim, hidden_dim, rngs=rngs)

        # Prior dynamics predictor p(z_t | h_t, xi_t)
        # Supports latent adversary perturbation xi_t
        self.fc_prior_mean = nnx.Linear(hidden_dim, latent_dim, rngs=rngs)
        self.fc_prior_std = nnx.Linear(hidden_dim, latent_dim, rngs=rngs)

        # Posterior representation q(z_t | h_t, embed_t)
        self.fc_post_mean = nnx.Linear(hidden_dim + latent_dim, latent_dim, rngs=rngs)
        self.fc_post_std = nnx.Linear(hidden_dim + latent_dim, latent_dim, rngs=rngs)

    def initial_state(self, batch_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        h0 = jnp.zeros((batch_size, self.hidden_dim), dtype=jnp.float32)
        z0 = jnp.zeros((batch_size, self.latent_dim), dtype=jnp.float32)
        return h0, z0

    def step_deterministic(self, prev_h: jnp.ndarray, prev_z: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        inp = jnp.concatenate([prev_h, prev_z, action], axis=-1)
        projected = self.fc_h(inp)
        if self.stable:
            # A bounded deterministic state prevents recurrent positive drift
            # from overflowing after long real-replay and imagination runs.
            return jnp.tanh(projected)
        return nnx.elu(projected)

    def distribution_parameters(
        self,
        raw_mean: jnp.ndarray,
        raw_std: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self.stable:
            mean = self.mean_bound * jnp.tanh(raw_mean / self.mean_bound)
            std = self.std_min + (self.std_max - self.std_min) * jax.nn.sigmoid(
                raw_std
            )
            return mean, std
        log_std = jnp.clip(raw_std, -5.0, 2.0)
        return raw_mean, jnp.exp(log_std)

    def predict_prior(self, h: jnp.ndarray, rngs: nnx.Rngs, latent_perturbation: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Predicts the prior distribution and samples a stochastic latent z_t."""
        mean, std = self.distribution_parameters(
            self.fc_prior_mean(h), self.fc_prior_std(h)
        )

        # Apply adversarial perturbation if present
        if latent_perturbation is not None:
            mean = mean + latent_perturbation

        noise = jax.random.normal(rngs.noise(), mean.shape) if rngs is not None else jax.random.normal(jax.random.PRNGKey(0), mean.shape)
        z = mean + std * noise
        return z, mean, std

    def compute_posterior(self, h: jnp.ndarray, embed: jnp.ndarray, rngs: nnx.Rngs) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Calculates the posterior distribution from real observations."""
        inp = jnp.concatenate([h, embed], axis=-1)
        mean, std = self.distribution_parameters(
            self.fc_post_mean(inp), self.fc_post_std(inp)
        )

        noise = jax.random.normal(rngs.noise(), mean.shape) if rngs is not None else jax.random.normal(jax.random.PRNGKey(0), mean.shape)
        z = mean + std * noise
        return z, mean, std


class LatentWorldModel(nnx.Module):
    """
    Complete TrackMania Latent World Model combining:
    1. Visual-Telemetry Encoder
    2. Recurrent State-Space Model (RSSM)
    3. Symlog Reward, Continuation, and Telemetry Prediction Heads
    """
    def __init__(self,
                 img_hist_len: int = 4,
                 img_height: int | None = None,
                 img_width: int | None = None,
                 latent_dim: int = 128,
                 action_dim: int = 3,
                 hidden_dim: int = 256,
                 encoder_channels: tuple[int, int, int, int] = (32, 64, 128, 128),
                 normalized_observations: bool = False,
                 stable_rssm: bool = False,
                 rngs: nnx.Rngs = None):
        super().__init__()
        rngs = rngs or get_rngs()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.stable_rssm = bool(stable_rssm)

        self.encoder = VisualTelemetryEncoder(
            img_hist_len=img_hist_len,
            img_height=img_height,
            img_width=img_width,
            latent_dim=latent_dim,
            channels=encoder_channels,
            normalized_observations=normalized_observations,
            rngs=rngs
        )
        self.rssm = LatentRSSM(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            stable=self.stable_rssm,
            rngs=rngs
        )

        # State feature dimension = h_dim + z_dim
        feat_dim = hidden_dim + latent_dim

        # Predictors
        self.reward_head = nnx.Linear(feat_dim, 1, kernel_init=nnx.initializers.zeros, bias_init=nnx.initializers.zeros, rngs=rngs)
        self.continue_head = nnx.Linear(feat_dim, 1, kernel_init=nnx.initializers.zeros, bias_init=nnx.initializers.zeros, rngs=rngs)
        self.telemetry_head = nnx.Linear(feat_dim, 3, rngs=rngs)  # [speed, gear, rpm]
        self.embedding_head = nnx.Linear(feat_dim, latent_dim, rngs=rngs)

    def get_feature(self, h: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([h, z], axis=-1)

    def predict_reward(self, feat: jnp.ndarray) -> jnp.ndarray:
        """Predicts step reward in symlog space."""
        return self.reward_head(feat)

    def predict_continuation(self, feat: jnp.ndarray) -> jnp.ndarray:
        """Predicts episode continuation probability (1 = continue, 0 = terminated)."""
        return nnx.sigmoid(self.predict_continuation_logits(feat))

    def predict_continuation_logits(self, feat: jnp.ndarray) -> jnp.ndarray:
        return self.continue_head(feat)

    def predict_telemetry(self, feat: jnp.ndarray) -> jnp.ndarray:
        return self.telemetry_head(feat)

    def predict_embedding(self, feat: jnp.ndarray) -> jnp.ndarray:
        """Predicts the next encoded observation as a decoder-free grounding loss."""
        return self.embedding_head(feat)

    def imagine_step(self,
                     prev_h: jnp.ndarray,
                     prev_z: jnp.ndarray,
                     action: jnp.ndarray,
                     rngs: nnx.Rngs,
                     latent_perturbation: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Rolls forward 1 step in pure latent imagination without real observations.
        Returns:
            (h_next, z_next, pred_reward, pred_continue)
        """
        h_next = self.rssm.step_deterministic(prev_h, prev_z, action)
        z_next, _, _ = self.rssm.predict_prior(h_next, rngs=rngs, latent_perturbation=latent_perturbation)
        feat = self.get_feature(h_next, z_next)

        pred_reward = symexp(self.predict_reward(feat))
        pred_continue = self.predict_continuation(feat)

        return h_next, z_next, pred_reward, pred_continue
