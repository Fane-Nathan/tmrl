from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.optim import Adam

from tmrl.training import TrainingAgent
from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.world_model.actor import WorldModelActor
from tmrl.world_model.config import WorldModelConfig
from tmrl.world_model.models import WorldModelCore


def _obs_at(obs_seq, t):
    return tuple(x[:, t] for x in obs_seq)


@dataclass(eq=0)
class WorldModelAgent(TrainingAgent):
    observation_space: type
    action_space: type
    device: str = None
    wm_config: WorldModelConfig = None
    img_hist_len: int = 4
    action_history_len: int = 2

    def __post_init__(self):
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.wm_config = self.wm_config or WorldModelConfig()
        action_dim = int(self.action_space.shape[0])
        self.model = WorldModelCore(
            img_hist_len=self.img_hist_len,
            action_dim=action_dim,
            action_history_len=self.action_history_len,
            latent_dim=self.wm_config.latent_dim,
            hidden_dim=self.wm_config.hidden_dim,
            q_ensemble=self.wm_config.q_ensemble,
        ).to(self.device)
        self.target_qs = no_grad(deepcopy(self.model.qs)).to(self.device)

        model_params = list(self.model.encoder.parameters())
        model_params += list(self.model.dynamics.parameters())
        model_params += list(self.model.reward.parameters())
        model_params += list(self.model.qs.parameters())
        self.model_optimizer = Adam(model_params, lr=self.wm_config.lr_model)
        self.policy_optimizer = Adam(self.model.policy.parameters(), lr=self.wm_config.lr_policy)

        self.actor = WorldModelActor(
            observation_space=self.observation_space,
            action_space=self.action_space,
            wm_config=self.wm_config,
            img_hist_len=self.img_hist_len,
            action_history_len=self.action_history_len,
            device=self.device,
        ).to(self.device)
        self.actor.core = no_grad(copy_shared(self.model))

    def get_actor(self):
        return self.actor

    def _target_min_q(self, z, action):
        x = torch.cat((z, action), dim=-1)
        values = torch.stack([q(x).squeeze(-1) for q in self.target_qs], dim=0)
        return values.min(dim=0).values

    @torch.no_grad()
    def _update_targets(self):
        tau = 1.0 - self.wm_config.polyak
        for target_q, q in zip(self.target_qs, self.model.qs):
            for target_param, param in zip(target_q.parameters(), q.parameters()):
                target_param.data.lerp_(param.data, tau)

    def train(self, batch):
        obs_seq, actions, rewards, terminated, truncated = batch
        horizon = actions.shape[1]
        if horizon != self.wm_config.horizon:
            raise ValueError(f"Expected horizon {self.wm_config.horizon}, got {horizon}.")

        latent_loss = torch.zeros((), device=self.device)
        reward_loss = torch.zeros((), device=self.device)
        q_loss = torch.zeros((), device=self.device)

        pred_z = self.model.encode(_obs_at(obs_seq, 0))

        for t in range(horizon):
            action = actions[:, t].float()
            reward = rewards[:, t].float()
            term = terminated[:, t].float()
            actual_z = self.model.encode(_obs_at(obs_seq, t))
            with torch.no_grad():
                next_z_target = self.model.encode(_obs_at(obs_seq, t + 1))

            predicted_next = self.model.next(pred_z, action)
            latent_loss = latent_loss + F.smooth_l1_loss(predicted_next, next_z_target)
            reward_pred = self.model.reward_value(pred_z, action)
            reward_loss = reward_loss + F.smooth_l1_loss(reward_pred, reward)

            with torch.no_grad():
                next_action = self.model.act_prior(next_z_target)
                backup = reward + self.wm_config.gamma * (1.0 - term) * self._target_min_q(
                    next_z_target, next_action
                )
            q_values = self.model.q_values(actual_z, action)
            q_loss = q_loss + ((q_values - backup.unsqueeze(0)) ** 2).mean()
            pred_z = predicted_next

        scale = 1.0 / horizon
        latent_loss = latent_loss * scale
        reward_loss = reward_loss * scale
        q_loss = q_loss * scale
        model_loss = (
            self.wm_config.latent_loss_coef * latent_loss
            + self.wm_config.reward_loss_coef * reward_loss
            + self.wm_config.q_loss_coef * q_loss
        )

        self.model_optimizer.zero_grad(set_to_none=True)
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.encoder.parameters())
            + list(self.model.dynamics.parameters())
            + list(self.model.reward.parameters())
            + list(self.model.qs.parameters()),
            self.wm_config.grad_clip_norm,
        )
        self.model_optimizer.step()

        for q in self.model.qs:
            q.requires_grad_(False)
        z_policy = self.model.encode(_obs_at(obs_seq, 0)).detach()
        policy_action = self.model.act_prior(z_policy)
        policy_loss = -self.model.min_q(z_policy, policy_action).mean()
        self.policy_optimizer.zero_grad(set_to_none=True)
        (self.wm_config.policy_loss_coef * policy_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), self.wm_config.grad_clip_norm)
        self.policy_optimizer.step()
        for q in self.model.qs:
            q.requires_grad_(True)

        self._update_targets()

        with torch.no_grad():
            return {
                "loss_world_model": model_loss.item(),
                "loss_latent": latent_loss.item(),
                "loss_reward": reward_loss.item(),
                "loss_q": q_loss.item(),
                "loss_policy": policy_loss.item(),
                "wm_reward_mean": rewards.mean().item(),
                "wm_q_mean": self.model.min_q(z_policy, self.model.act_prior(z_policy)).mean().item(),
                "wm_sequence_horizon": float(horizon),
                "wm_positions_per_update": float(actions.shape[0] * horizon),
            }
