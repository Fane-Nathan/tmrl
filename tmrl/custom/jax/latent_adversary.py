import jax
import jax.numpy as jnp
from flax import nnx
from typing import Tuple, Optional
from tmrl.core.jax.util import get_rngs


def compute_latent_learnability_reward(solver_successes: jnp.ndarray,
                                       rollout_axis: int = 0) -> jnp.ndarray:
    """
    Compute the AZR proposer reward from repeated binary solver attempts.

    ``solver_successes`` must contain multiple attempts for the *same* task on
    ``rollout_axis``. The remaining dimensions identify tasks. Continuous
    returns from unrelated tasks are not valid input to this reward.

    AZR rewards tasks that the current solver can sometimes, but not always,
    solve: zero success and perfect success both receive zero reward, while a
    non-zero pass rate receives ``1 - pass_rate``.
    """
    successes = jnp.asarray(solver_successes, dtype=jnp.float32)
    pass_rate = jnp.mean(successes, axis=rollout_axis)
    pass_rate = jnp.clip(pass_rate, 0.0, 1.0)
    learnable = (pass_rate > 0.0) & (pass_rate < 1.0)
    return jnp.where(learnable, 1.0 - pass_rate, 0.0)


class LatentAdversaryProposer(nnx.Module):
    """
    Experimental bounded proposer operating in model latent space.

    These vectors do not by themselves correspond to valid curve, friction,
    or obstacle parameters; a grounded scenario mapper/verifier is required.
    """
    def __init__(self,
                 feature_dim: int = 384,  # 256 (h) + 128 (z)
                 latent_dim: int = 128,
                 hidden_dim: int = 256,
                 perturbation_scale: float = 0.5,
                 rngs: nnx.Rngs = None):
        super().__init__()
        rngs = rngs or get_rngs()
        self.latent_dim = latent_dim
        self.perturbation_scale = perturbation_scale

        self.fc1 = nnx.Linear(feature_dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.fc_mean = nnx.Linear(hidden_dim, latent_dim, rngs=rngs)
        self.fc_log_std = nnx.Linear(hidden_dim, latent_dim, rngs=rngs)

    def distribution(self, state_feat: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return the Gaussian mean and log standard deviation in raw space."""
        x = nnx.elu(self.fc1(state_feat))
        x = nnx.elu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = jnp.clip(self.fc_log_std(x), -4.0, 1.0)
        return mean, log_std

    def log_prob(self,
                 state_feat: jnp.ndarray,
                 raw_sample: jnp.ndarray) -> jnp.ndarray:
        """Evaluate a stored raw proposal without resampling it."""
        mean, log_std = self.distribution(state_feat)
        std = jnp.exp(log_std)
        gaussian_log_prob = -0.5 * jnp.sum(
            ((raw_sample - mean) / (std + 1e-6)) ** 2
            + 2 * log_std
            + jnp.log(2 * jnp.pi),
            axis=-1,
        )
        squashed = jnp.tanh(raw_sample)
        correction = jnp.sum(
            jnp.log(self.perturbation_scale * (1.0 - squashed ** 2) + 1e-6),
            axis=-1,
        )
        return gaussian_log_prob - correction

    def sample(self,
               state_feat: jnp.ndarray,
               rngs: Optional[nnx.Rngs] = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return ``(bounded_perturbation, raw_sample, log_probability)``."""
        mean, log_std = self.distribution(state_feat)
        std = jnp.exp(log_std)
        noise = (
            jax.random.normal(rngs.noise(), mean.shape)
            if rngs is not None
            else jax.random.normal(jax.random.PRNGKey(42), mean.shape)
        )
        raw_sample = mean + std * noise
        perturbation = jnp.tanh(raw_sample) * self.perturbation_scale
        return perturbation, raw_sample, self.log_prob(state_feat, raw_sample)

    def __call__(self, state_feat: jnp.ndarray, rngs: Optional[nnx.Rngs] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Proposes a continuous latent perturbation vector xi_t.
        Returns:
            perturbation: (B, latent_dim)
            log_prob: (B,)
        """
        perturbation, _, log_prob = self.sample(state_feat, rngs)
        return perturbation, log_prob
