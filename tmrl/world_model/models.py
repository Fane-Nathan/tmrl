import torch
import torch.nn as nn


class ObservationEncoder(nn.Module):
    def __init__(self, img_hist_len, action_dim, action_history_len, latent_dim=256, hidden_dim=256):
        super().__init__()
        self.action_history_len = action_history_len
        self.cnn = nn.Sequential(
            nn.Conv2d(img_hist_len, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        aux_dim = 3 + action_dim * action_history_len
        self.proj = nn.Sequential(
            nn.Linear(64 * 4 * 4 + aux_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, obs):
        speed, gear, rpm, images, *action_hist = obs
        visual = self.cnn(images.float())
        aux = [speed.float(), gear.float(), rpm.float()]
        aux.extend(a.float() for a in action_hist[: self.action_history_len])
        if len(aux) != 3 + self.action_history_len:
            raise ValueError(
                f"Expected {self.action_history_len} previous actions in observation, got {len(action_hist)}."
            )
        return self.proj(torch.cat([visual, *aux], dim=-1))


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class LatentDynamics(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim):
        super().__init__()
        self.delta = MLP(latent_dim + action_dim, hidden_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z, action):
        delta = torch.tanh(self.delta(torch.cat((z, action), dim=-1)))
        return self.norm(z + 0.5 * delta)


class PolicyPrior(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = MLP(latent_dim, hidden_dim, action_dim)

    def forward(self, z):
        return torch.tanh(self.net(z))


class WorldModelCore(nn.Module):
    def __init__(
        self,
        img_hist_len,
        action_dim,
        action_history_len,
        latent_dim=256,
        hidden_dim=256,
        q_ensemble=2,
    ):
        super().__init__()
        self.encoder = ObservationEncoder(
            img_hist_len=img_hist_len,
            action_dim=action_dim,
            action_history_len=action_history_len,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )
        self.dynamics = LatentDynamics(latent_dim, action_dim, hidden_dim)
        self.reward = MLP(latent_dim + action_dim, hidden_dim, 1)
        self.qs = nn.ModuleList(
            [MLP(latent_dim + action_dim, hidden_dim, 1) for _ in range(q_ensemble)]
        )
        self.policy = PolicyPrior(latent_dim, action_dim, hidden_dim)

    def encode(self, obs):
        return self.encoder(obs)

    def next(self, z, action):
        return self.dynamics(z, action)

    def reward_value(self, z, action):
        return self.reward(torch.cat((z, action), dim=-1)).squeeze(-1)

    def q_values(self, z, action):
        x = torch.cat((z, action), dim=-1)
        return torch.stack([q(x).squeeze(-1) for q in self.qs], dim=0)

    def min_q(self, z, action):
        return self.q_values(z, action).min(dim=0).values

    def act_prior(self, z):
        return self.policy(z)
