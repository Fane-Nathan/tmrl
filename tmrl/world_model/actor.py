import math

import numpy as np
import torch

from tmrl.actor import TorchActorModule
from tmrl.world_model.actions import canonicalize_tm_action
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
        if self.action_dim != 3:
            raise ValueError(
                f"World-model TrackMania actor expects [gas, brake, steer], got {self.action_dim} actions."
            )
        self.core = WorldModelCore(
            img_hist_len=img_hist_len,
            action_dim=self.action_dim,
            action_history_len=action_history_len,
            latent_dim=self.wm_config.latent_dim,
            hidden_dim=self.wm_config.hidden_dim,
            q_ensemble=self.wm_config.q_ensemble,
        )

        # Worker-local exploration state. These are intentionally ordinary
        # Python attributes rather than state_dict buffers: TMRL loads new
        # network weights into the existing actor, so the interaction schedule
        # continues across trainer broadcasts instead of restarting.
        self._interaction_step = 0
        self._explore_steer = 0.0

    def _exploration_epsilon(self):
        steps = max(1, int(self.wm_config.bootstrap_exploration_steps))
        fraction = min(1.0, self._interaction_step / steps)
        start = float(self.wm_config.bootstrap_epsilon_start)
        end = float(self.wm_config.bootstrap_epsilon_end)
        return start + fraction * (end - start)

    def _bootstrap_action(self, obs):
        """Generate forward-biased, temporally correlated exploration."""
        reference = obs[0]
        device = reference.device
        dtype = reference.dtype if reference.is_floating_point() else torch.float32
        batch = reference.shape[0]

        rho = float(self.wm_config.bootstrap_steer_rho)
        steer_std = float(self.wm_config.bootstrap_steer_std)
        steer_limit = float(self.wm_config.bootstrap_steer_limit)
        innovation_scale = steer_std * math.sqrt(max(0.0, 1.0 - rho * rho))
        innovation = float(torch.randn((), device=device).item()) * innovation_scale
        self._explore_steer = float(
            np.clip(rho * self._explore_steer + innovation, -steer_limit, steer_limit)
        )

        gas_min = float(self.wm_config.bootstrap_gas_min)
        gas_max = float(self.wm_config.bootstrap_gas_max)
        gas = gas_min + (gas_max - gas_min) * torch.rand(batch, device=device, dtype=dtype)
        brake = torch.zeros(batch, device=device, dtype=dtype)
        steer = torch.full((batch,), self._explore_steer, device=device, dtype=dtype)
        return canonicalize_tm_action(torch.stack((gas, brake, steer), dim=-1))

    def _inject_structured_candidates(self, actions, priors):
        """Guarantee physically useful TrackMania trajectories in random shooting.

        Candidate 0 follows the learned policy prior. The following candidates
        explicitly cover acceleration, gentle steering, coasting, and braking.
        Their index is kept consistent across the planning horizon, so they form
        coherent trajectories instead of one-step action perturbations.
        """
        batch, samples, _ = actions.shape
        actions[:, 0] = priors[:, 0]
        if samples <= 1:
            return actions

        templates = actions.new_tensor([
            [1.00, 0.00, 0.00],
            [0.80, 0.00, -0.25],
            [0.80, 0.00, 0.25],
            [0.60, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [0.00, 0.50, 0.00],
        ])
        count = min(samples - 1, templates.shape[0])
        actions[:, 1:1 + count] = templates[:count].unsqueeze(0).expand(batch, -1, -1)
        return actions

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

        for _ in range(horizon):
            prior = canonicalize_tm_action(self.core.act_prior(z_roll))
            action = canonicalize_tm_action(prior + sigma * torch.randn_like(prior))

            actions = action.view(batch, samples, self.action_dim)
            priors = prior.view(batch, samples, self.action_dim)
            actions = self._inject_structured_candidates(actions, priors)
            action = actions.reshape(batch * samples, self.action_dim)

            if first_action is None:
                first_action = actions.clone()

            score += discount * self.core.reward_value(z_roll, action)
            z_roll = self.core.next(z_roll, action)
            discount *= self.wm_config.gamma

        terminal_action = canonicalize_tm_action(self.core.act_prior(z_roll))
        score += discount * self.core.min_q(z_roll, terminal_action)
        score = score.view(batch, samples)
        best = score.argmax(dim=1)
        return first_action[torch.arange(batch, device=z.device), best]

    def forward(self, obs, test=False):
        return self.plan(self.core.encode(obs), test=test)

    def act(self, obs, test=False):
        if not test:
            epsilon = self._exploration_epsilon()
            self._interaction_step += 1
            if float(torch.rand((), device=obs[0].device).item()) < epsilon:
                action = self._bootstrap_action(obs)
            else:
                action = self.forward(obs, test=False)
        else:
            action = self.forward(obs, test=True)
        return action.squeeze(0).detach().cpu().numpy().astype(np.float32)
