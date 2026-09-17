import numpy as np
import torch

from tmrl.actor import TorchActorModule
from tmrl.world_model.config import WorldModelConfig
from tmrl.world_model.models import WorldModelCore


class WorldModelActor(TorchActorModule):
    """Inference actor that plans directly in the learned latent dynamics model."""

    def __init__(
        self,
        observation_space,
        action_space,
        wm_config=None,
        img_hist_len=4,
        action_history_len=2,
        device="cpu",
    ):
        super().__init__(observation_space, action_space, device=device)
        self.wm_config = wm_config or WorldModelConfig()
        self.action_dim = int(action_space.shape[0])
        self.core = WorldModelCore(
            img_hist_len=img_hist_len,
            action_dim=self.action_dim,
            action_history_len=action_history_len,
            latent_dim=self.wm_config.latent_dim,
            hidden_dim=self.wm_config.hidden_dim,
            q_ensemble=self.wm_config.q_ensemble,
        )

    @torch.no_grad()
    def plan(self, z, test=False):
        batch = z.shape[0]
        samples = self.wm_config.planning_samples
        horizon = self.wm_config.planning_horizon
        sigma = self.wm_config.test_planning_noise if test else self.wm_config.planning_noise

        z_roll = z[:, None, :].expand(batch, samples, z.shape[-1]).reshape(batch * samples, -1)
        score = torch.zeros(batch * samples, device=z.device)
        first_action = None
        discount = 1.0

        for step in range(horizon):
            prior = self.core.act_prior(z_roll)
            action = (prior + sigma * torch.randn_like(prior)).clamp(-1.0, 1.0)
            if step == 0:
                actions = action.view(batch, samples, self.action_dim)
                priors = prior.view(batch, samples, self.action_dim)
                actions[:, 0] = priors[:, 0]
                action = actions.reshape(batch * samples, self.action_dim)
                first_action = action.view(batch, samples, self.action_dim).clone()
            score += discount * self.core.reward_value(z_roll, action)
            z_roll = self.core.next(z_roll, action)
            discount *= self.wm_config.gamma

        terminal_action = self.core.act_prior(z_roll)
        score += discount * self.core.min_q(z_roll, terminal_action)
        score = score.view(batch, samples)
        best = score.argmax(dim=1)
        return first_action[torch.arange(batch, device=z.device), best]

    def forward(self, obs, test=False):
        return self.plan(self.core.encode(obs), test=test)

    def act(self, obs, test=False):
        action = self.forward(obs, test=test)
        return action.squeeze(0).detach().cpu().numpy().astype(np.float32)
