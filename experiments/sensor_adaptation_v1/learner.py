"""Warm-start SAC/REDQ fine-tuning; no hidden fault-label supervision."""
import copy
import math
from collections import deque

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from frozen_models import ReactiveREDQAgent, TransformerRL2Agent


def configure(model, algorithm, context):
    if algorithm == "transformer":
        if not 1 <= context <= 32:
            raise ValueError("This pilot is restricted to the pretrained positional support, 1..32")
        model.context_deque = deque(maxlen=context)
        model.lambda_task = 0.0
    # Use the same deterministic representations during collection and updates.
    # Gradients still flow in eval mode; entropy comes from the Gaussian actor.
    model.eval()
    return model


def load_agent(algorithm, checkpoint, context=32, device="cpu"):
    cls = TransformerRL2Agent if algorithm == "transformer" else ReactiveREDQAgent
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["agent"] if "agent" in payload else payload
    # Token and action sizes come from the checkpoint: HalfCheetah-v5 is 25/6, Ant-v5 is 37/8.
    in_key = "trunk.embedder.proj.0.weight" if algorithm == "transformer" else "actor.net.0.weight"
    model = cls(obs_dim=int(state[in_key].shape[1]), act_dim=int(state["actor.mu.weight"].shape[0]))
    model.load_state_dict(state, strict=True)
    return configure(model.to(device), algorithm, context)


def features(model, algorithm, observations, lengths, target=False):
    index = torch.arange(observations.shape[0], device=observations.device)
    if algorithm == "reactive":
        return observations[index, lengths - 1]
    trunk = model.target_trunk if target else model.trunk
    task = model.target_task_encoder if target else model.task_encoder
    h = trunk(observations)[index, lengths - 1]
    z, _ = task(h)
    return torch.cat((h, z), dim=-1)


def q_value(model, algorithm, index, feat, action, target=False):
    critic = (model.target_critics if target else model.critics)[index]
    return critic(torch.cat((feat, action), dim=-1)) if algorithm == "transformer" else critic(feat, action)


class Learner:
    def __init__(self, algorithm, checkpoint, config, device, seed):
        self.algorithm, self.config, self.device = algorithm, config, device
        self.context = config["context"] if algorithm == "transformer" else 1
        self.model = load_agent(algorithm, checkpoint, self.context, device)
        self.actor_params = list(self.model.actor.parameters())
        self.critic_params = list(self.model.critics.parameters())
        if algorithm == "transformer":
            self.critic_params += list(self.model.trunk.parameters()) + list(self.model.task_encoder.parameters())
        self.actor_opt = torch.optim.Adam(self.actor_params, lr=config["actor_lr"])
        self.critic_opt = torch.optim.Adam(self.critic_params, lr=config["critic_lr"])
        self.log_alpha = torch.tensor([math.log(config["initial_alpha"])], device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config["alpha_lr"])
        self.rng = np.random.default_rng(seed)
        self.updates = 0

    def rollout_model(self):
        model = copy.deepcopy(self.model).to("cpu")
        return configure(model, self.algorithm, self.context)

    def update(self, batch):
        m, alg, cfg = self.model, self.algorithm, self.config
        feat = features(m, alg, batch["obs"], batch["lengths"])
        with torch.no_grad():
            # Current policy chooses the next action; target encoders/critics value it.
            online_next = features(m, alg, batch["next_obs"], batch["next_lengths"])
            target_next = features(m, alg, batch["next_obs"], batch["next_lengths"], target=True)
            next_action, next_logp = m.actor(online_next)
            subset = self.rng.choice(m.num_critics, m.subset_m, replace=False)
            minimum = torch.stack([q_value(m, alg, int(j), target_next, next_action, True) for j in subset]).min(0).values
            target = batch["rewards"] + cfg["gamma"] * (1 - batch["terminated"]) * (minimum - self.log_alpha.exp() * next_logp)
        critic_loss = sum(F.mse_loss(q_value(m, alg, j, feat, batch["actions"]), target) for j in range(m.num_critics))
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_norm = nn.utils.clip_grad_norm_(self.critic_params, cfg["grad_clip"])
        self.critic_opt.step()

        with torch.no_grad():
            feat = features(m, alg, batch["obs"], batch["lengths"])
        for parameter in m.critics.parameters():
            parameter.requires_grad_(False)
        action, logp = m.actor(feat)
        minimum = torch.minimum(q_value(m, alg, 0, feat, action), q_value(m, alg, 1, feat, action))
        actor_loss = (self.log_alpha.exp().detach() * logp - minimum).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_norm = nn.utils.clip_grad_norm_(self.actor_params, cfg["grad_clip"])
        self.actor_opt.step()
        for parameter in m.critics.parameters():
            parameter.requires_grad_(True)

        alpha_loss = -(self.log_alpha * (logp.detach() - m.act_dim)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            pairs = [(m.critics, m.target_critics)]
            if alg == "transformer":
                pairs.extend([(m.trunk, m.target_trunk), (m.task_encoder, m.target_task_encoder)])
            for online, target_module in pairs:
                for p, target_p in zip(online.parameters(), target_module.parameters()):
                    target_p.mul_(cfg["polyak"]).add_(p, alpha=1 - cfg["polyak"])
        self.updates += 1
        metrics = {"critic_loss": float(critic_loss.detach()), "actor_loss": float(actor_loss.detach()),
                   "alpha": float(self.log_alpha.exp().detach()), "critic_gradient_norm": float(critic_norm),
                   "actor_gradient_norm": float(actor_norm)}
        if not all(math.isfinite(v) for v in metrics.values()):
            raise FloatingPointError(f"Nonfinite update: {metrics}")
        return metrics

    def save(self, path):
        torch.save({"agent": self.model.state_dict(), "actor_optimizer": self.actor_opt.state_dict(),
                    "critic_optimizer": self.critic_opt.state_dict(), "log_alpha": self.log_alpha.detach(),
                    "alpha_optimizer": self.alpha_opt.state_dict(), "updates": self.updates}, path)
