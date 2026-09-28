import jax
import jax.numpy as jnp
import optax
from flax import nnx
from typing import Tuple, Dict, Optional
from tmrl.core.jax.util import get_rngs
from tmrl.custom.jax.world_model import LatentWorldModel, symlog
from tmrl.custom.jax.latent_adversary import LatentAdversaryProposer, compute_latent_learnability_reward


class LatentActor(nnx.Module):
    """
    Policy network operating on latent world model state features (h_t, z_t).
    Outputs bounded continuous actions [gas, brake, steer] in [-1, 1].
    """
    def __init__(self,
                 state_dim: int = 384,  # hidden_dim (256) + latent_dim (128)
                 act_dim: int = 3,
                 hidden_dim: int = 256,
                 rngs: nnx.Rngs = None):
        super().__init__()
        rngs = rngs or get_rngs()
        self.fc1 = nnx.Linear(state_dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.fc_mean = nnx.Linear(hidden_dim, act_dim, rngs=rngs)
        self.fc_log_std = nnx.Linear(hidden_dim, act_dim, rngs=rngs)

    def __call__(self, state_feat: jnp.ndarray, rngs: Optional[nnx.Rngs] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = nnx.elu(self.fc1(state_feat))
        x = nnx.elu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = jnp.clip(self.fc_log_std(x), -5.0, 2.0)
        std = jnp.exp(log_std)

        noise = jax.random.normal(rngs.noise(), mean.shape) if rngs is not None else jax.random.normal(jax.random.PRNGKey(0), mean.shape)

        raw_action = mean + std * noise
        action = jnp.tanh(raw_action)

        # Log prob
        log_prob = -0.5 * jnp.sum(((raw_action - mean) / (std + 1e-6)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1)
        log_prob -= jnp.sum(jnp.log(1.0 - action ** 2 + 1e-6), axis=-1)

        return action, log_prob


class LatentCritic(nnx.Module):
    """
    State-value critic network V(s) operating on latent state features.
    """
    def __init__(self,
                 state_dim: int = 384,
                 hidden_dim: int = 256,
                 rngs: nnx.Rngs = None):
        super().__init__()
        rngs = rngs or get_rngs()
        self.fc1 = nnx.Linear(state_dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.fc_v = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, state_feat: jnp.ndarray) -> jnp.ndarray:
        x = nnx.elu(self.fc1(state_feat))
        x = nnx.elu(self.fc2(x))
        return self.fc_v(x)


def compute_lambda_returns(rewards: jnp.ndarray,
                           values: jnp.ndarray,
                           continues: jnp.ndarray,
                           bootstrap_val: jnp.ndarray,
                           lambda_: float = 0.95,
                           gamma: float = 0.99) -> jnp.ndarray:
    """
    Computes generalized lambda-returns across imagined sequence horizon.
    rewards: (H, B)
    values: (H, B)
    continues: (H, B)
    bootstrap_val: (B,)
    """
    H = rewards.shape[0]
    next_values = jnp.concatenate([values[1:], bootstrap_val[None, :]], axis=0)
    discounts = continues * gamma

    # Backwards recursion for lambda returns
    def scan_fn(running_target, inputs):
        r, v_next, d = inputs
        # V_t^lambda = r + d * ((1 - lambda) * v_next + lambda * running_target)
        target = r + d * ((1.0 - lambda_) * v_next + lambda_ * running_target)
        return target, target

    inputs = (rewards, next_values, discounts)
    _, returns = jax.lax.scan(scan_fn, bootstrap_val, inputs, reverse=True)
    return returns


class LatentImaginationTrainer:
    """
    Experimental replay-grounded latent imagination engine.

    The world model is trained on real ``(obs, action, next_obs)`` transitions,
    and the latent actor receives pathwise gradients through short imagined
    rollouts. The latent proposer is deliberately disabled by default: AZR
    requires a persistent, validated task and repeated solver attempts for that
    same task, which unconstrained latent noise does not provide.
    """
    def __init__(self,
                 horizon: int = 5,
                 gamma: float = 0.99,
                 lambda_: float = 0.95,
                 lr_model: float = 3e-4,
                 lr_actor: float = 1e-4,
                 lr_critic: float = 3e-4,
                 lr_adv: float = 1e-4,
                 entropy_scale: float = 3e-4,
                 free_nats: float = 1.0,
                 max_grad_norm: float = 100.0,
                 enable_latent_adversary: bool = False,
                 rngs: nnx.Rngs = None):
        self.horizon = horizon
        self.gamma = gamma
        self.lambda_ = lambda_
        self.entropy_scale = entropy_scale
        self.free_nats = free_nats
        self.enable_latent_adversary = enable_latent_adversary
        self.rngs = rngs or get_rngs()

        # Neural Modules
        self.world_model = LatentWorldModel(rngs=self.rngs)
        self.adversary = LatentAdversaryProposer(rngs=self.rngs)
        self.actor = LatentActor(rngs=self.rngs)
        self.critic = LatentCritic(rngs=self.rngs)

        # Optimizers
        def optimizer(lr: float):
            return optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr))

        self.opt_model = nnx.Optimizer(self.world_model, optimizer(lr_model), wrt=nnx.Param)
        self.opt_actor = nnx.Optimizer(self.actor, optimizer(lr_actor), wrt=nnx.Param)
        self.opt_critic = nnx.Optimizer(self.critic, optimizer(lr_critic), wrt=nnx.Param)
        self.opt_adv = nnx.Optimizer(self.adversary, optimizer(lr_adv), wrt=nnx.Param)

    def train_world_model_step(self,
                               obs_tuple,
                               actions: jnp.ndarray,
                               rewards: jnp.ndarray,
                               continues: jnp.ndarray,
                               next_obs_tuple=None) -> Dict[str, float]:
        """
        Fit one replay-grounded transition.

        ``next_obs_tuple`` is required. Reusing the current observation as the
        posterior target does not train transition dynamics and is rejected.
        """
        if next_obs_tuple is None:
            raise ValueError("next_obs_tuple is required for world-model transition training")

        def loss_fn(model: LatentWorldModel, rngs: nnx.Rngs):
            embed = model.encoder(obs_tuple)
            next_embed = model.encoder(next_obs_tuple)
            batch_size = embed.shape[0]
            h0, _ = model.rssm.initial_state(batch_size)

            # Filter the real current observation, then predict/filter next state.
            z0, _, _ = model.rssm.compute_posterior(h0, embed, rngs)
            h1 = model.rssm.step_deterministic(h0, z0, actions)
            post_z, post_m, post_s = model.rssm.compute_posterior(h1, next_embed, rngs)
            _, prior_m, prior_s = model.rssm.predict_prior(h1, rngs)

            feat = model.get_feature(h1, post_z)
            reward_targets = jnp.asarray(rewards, dtype=jnp.float32).reshape((batch_size, 1))
            continue_targets = jnp.asarray(continues, dtype=jnp.float32).reshape((batch_size, 1))

            # Reward prediction loss (in symlog space)
            pred_rew_symlog = model.predict_reward(feat)
            rew_loss = jnp.mean((pred_rew_symlog - symlog(reward_targets)) ** 2)

            # Continuation is a Bernoulli target, so train logits with BCE.
            cont_logits = model.predict_continuation_logits(feat)
            cont_loss = jnp.mean(
                optax.sigmoid_binary_cross_entropy(cont_logits, continue_targets)
            )

            # Decoder-free next-representation consistency keeps the visual
            # latent grounded even when reward is sparse.
            pred_embed = model.predict_embedding(feat)
            embed_loss = jnp.mean(
                (pred_embed - jax.lax.stop_gradient(next_embed)) ** 2
            )

            next_speed, next_gear, next_rpm, *_ = next_obs_tuple
            telemetry_target = jnp.concatenate(
                [
                    jnp.asarray(next_speed).reshape((batch_size, -1)) / 300.0,
                    jnp.asarray(next_gear).reshape((batch_size, -1)) / 5.0,
                    jnp.asarray(next_rpm).reshape((batch_size, -1)) / 10000.0,
                ],
                axis=-1,
            )
            telemetry_loss = jnp.mean(
                (model.predict_telemetry(feat) - telemetry_target) ** 2
            )

            # Dreamer-style asymmetric KL with *lower* free-nat clipping.
            def gaussian_kl(p_m, p_s, q_m, q_s):
                var_p = p_s ** 2
                var_q = q_s ** 2
                kl_elem = jnp.log(q_s / (p_s + 1e-4)) + (var_p + (p_m - q_m) ** 2) / (2.0 * var_q + 1e-4) - 0.5
                return jnp.sum(kl_elem, axis=-1)

            kl_dyn = jnp.mean(jnp.maximum(
                self.free_nats,
                gaussian_kl(
                    jax.lax.stop_gradient(post_m),
                    jax.lax.stop_gradient(post_s),
                    prior_m,
                    prior_s,
                ),
            ))
            kl_rep = jnp.mean(jnp.maximum(
                self.free_nats,
                gaussian_kl(
                    post_m,
                    post_s,
                    jax.lax.stop_gradient(prior_m),
                    jax.lax.stop_gradient(prior_s),
                ),
            ))

            total_loss = (
                rew_loss
                + cont_loss
                + embed_loss
                + 0.1 * telemetry_loss
                + kl_dyn
                + 0.1 * kl_rep
            )
            return total_loss, {
                "model_loss": total_loss,
                "rew_loss": rew_loss,
                "cont_loss": cont_loss,
                "embed_loss": embed_loss,
                "telemetry_loss": telemetry_loss,
                "kl_dyn": kl_dyn,
                "kl_rep": kl_rep,
            }

        grads, metrics = nnx.grad(loss_fn, has_aux=True)(self.world_model, self.rngs)
        self.opt_model.update(self.world_model, grads)
        return {k: float(v) for k, v in metrics.items()}

    def _posterior_start(self, obs_tuple):
        embed = self.world_model.encoder(obs_tuple)
        batch_size = embed.shape[0]
        h0, _ = self.world_model.rssm.initial_state(batch_size)
        z0, _, _ = self.world_model.rssm.compute_posterior(h0, embed, self.rngs)
        return jax.lax.stop_gradient(h0), jax.lax.stop_gradient(z0)

    def _rollout(self, actor: LatentActor, h0, z0, rngs: nnx.Rngs):
        h_curr, z_curr = h0, z0
        states, rewards, continues, log_probs = [], [], [], []

        for _ in range(self.horizon):
            feat = self.world_model.get_feature(h_curr, z_curr)
            states.append(feat)
            action, log_prob = actor(feat, rngs)
            log_probs.append(log_prob)

            perturbation = None
            if self.enable_latent_adversary:
                perturbation, _ = self.adversary(jax.lax.stop_gradient(feat), rngs)
                perturbation = jax.lax.stop_gradient(perturbation)

            h_curr, z_curr, reward, cont = self.world_model.imagine_step(
                h_curr,
                z_curr,
                action,
                rngs,
                latent_perturbation=perturbation,
            )
            rewards.append(reward.squeeze(-1))
            continues.append(cont.squeeze(-1))

        return (
            jnp.stack(states, axis=0),
            jnp.stack(rewards, axis=0),
            jnp.stack(continues, axis=0),
            jnp.stack(log_probs, axis=0),
            self.world_model.get_feature(h_curr, z_curr),
        )

    def _survival_weights(self, continues: jnp.ndarray) -> jnp.ndarray:
        discounts = continues * self.gamma
        ones = jnp.ones_like(discounts[:1])
        return jnp.concatenate([ones, jnp.cumprod(discounts[:-1], axis=0)], axis=0)

    def train_imagination_step(self, initial_obs_tuple) -> Dict[str, float]:
        """
        Execute an H-step posterior-anchored rollout and update actor/critic.

        The proposer remains fixed here. Its update requires repeated matched
        outcomes through :meth:`train_adversary_step`.
        """
        # Start from a filtered posterior state grounded in replay observations.
        h0, z0 = self._posterior_start(initial_obs_tuple)
        states, rewards_seq, continues_seq, _, final_feat = self._rollout(
            self.actor, h0, z0, self.rngs
        )

        bootstrap_val = self.critic(final_feat).squeeze(-1)
        values_seq = jax.vmap(self.critic)(states).squeeze(-1)
        lambda_returns = compute_lambda_returns(
            rewards_seq, values_seq, continues_seq, bootstrap_val, self.lambda_, self.gamma
        )
        weights = self._survival_weights(continues_seq)

        def critic_loss_fn(critic: LatentCritic):
            all_feats = jax.lax.stop_gradient(states.reshape((-1, states.shape[-1])))
            targets = jax.lax.stop_gradient(lambda_returns.reshape(-1, 1))
            preds = critic(all_feats)
            flat_weights = jax.lax.stop_gradient(weights.reshape(-1, 1))
            return jnp.sum(flat_weights * (preds - targets) ** 2) / jnp.maximum(
                1.0, jnp.sum(flat_weights)
            )

        critic_loss, c_grads = nnx.value_and_grad(critic_loss_fn)(self.critic)
        self.opt_critic.update(self.critic, c_grads)

        # Recompute the exact rollout inside the gradient. For continuous
        # actions this provides Dreamer-style reparameterized pathwise credit;
        # no unrelated actions are re-sampled on frozen states.
        def actor_loss_fn(actor: LatentActor, rngs: nnx.Rngs):
            actor_states, actor_rewards, actor_continues, log_probs, actor_final = self._rollout(
                actor, h0, z0, rngs
            )
            actor_values = jax.vmap(self.critic)(actor_states).squeeze(-1)
            actor_bootstrap = self.critic(actor_final).squeeze(-1)
            actor_returns = compute_lambda_returns(
                actor_rewards,
                actor_values,
                actor_continues,
                actor_bootstrap,
                self.lambda_,
                self.gamma,
            )
            actor_weights = self._survival_weights(actor_continues)
            return_scale = jax.lax.stop_gradient(jnp.maximum(
                1.0,
                jnp.percentile(actor_returns, 95.0) - jnp.percentile(actor_returns, 5.0),
            ))
            denom = jnp.maximum(1.0, jnp.sum(actor_weights))
            objective = jnp.sum(actor_weights * actor_returns / return_scale) / denom
            entropy = jnp.sum(actor_weights * (-log_probs)) / denom
            return -objective - self.entropy_scale * entropy

        actor_loss, a_grads = nnx.value_and_grad(actor_loss_fn)(self.actor, self.rngs)
        self.opt_actor.update(self.actor, a_grads)

        return {
            "mean_imagined_return": float(jnp.mean(lambda_returns)),
            "mean_reward": float(jnp.mean(rewards_seq)),
            "actor_loss": float(actor_loss),
            "critic_loss": float(critic_loss),
            "adversary_enabled": float(self.enable_latent_adversary),
        }

    def propose_latent_tasks(self, initial_obs_tuple):
        """Sample persistent proposals for an external repeated-solve verifier.

        Returns the task features, bounded perturbations, and raw samples. A
        caller must validate each task and execute multiple solver attempts for
        the same proposal before calling :meth:`train_adversary_step`.
        """
        h0, z0 = self._posterior_start(initial_obs_tuple)
        features = jax.lax.stop_gradient(self.world_model.get_feature(h0, z0))
        perturbations, raw_samples, _ = self.adversary.sample(features, self.rngs)
        return features, perturbations, raw_samples

    def train_adversary_step(self,
                               task_features: jnp.ndarray,
                               raw_proposals: jnp.ndarray,
                               solver_successes: jnp.ndarray) -> Dict[str, float]:
        """Update the proposer from repeated binary outcomes of matched tasks."""
        if not self.enable_latent_adversary:
            raise RuntimeError(
                "Latent proposer training is disabled until a grounded task validator is configured."
            )

        rewards = compute_latent_learnability_reward(solver_successes)
        features = jax.lax.stop_gradient(task_features)
        proposals = jax.lax.stop_gradient(raw_proposals)
        rewards = jax.lax.stop_gradient(rewards)

        if rewards.shape[0] != features.shape[0]:
            raise ValueError("solver_successes must provide repeated attempts for each proposal")

        def loss_fn(adversary: LatentAdversaryProposer):
            log_prob = adversary.log_prob(features, proposals)
            return -jnp.mean(log_prob * rewards)

        loss, grads = nnx.value_and_grad(loss_fn)(self.adversary)
        self.opt_adv.update(self.adversary, grads)
        return {
            "adversary_loss": float(loss),
            "mean_learnability_reward": float(jnp.mean(rewards)),
        }
