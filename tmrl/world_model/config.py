from dataclasses import dataclass


@dataclass(frozen=True)
class WorldModelConfig:
    """Small defaults for the first Trackmania world-model baseline.

    The objective is deliberately narrow: reproduce the original TMRL image
    task with a latent dynamics model and short-horizon planning while keeping
    inference small enough for a real-time 20 Hz worker.
    """

    latent_dim: int = 256
    hidden_dim: int = 256
    horizon: int = 5
    batch_size: int = 16
    planning_horizon: int = 5
    planning_samples: int = 64
    planning_noise: float = 0.35
    test_planning_noise: float = 0.15
    gamma: float = 0.995
    polyak: float = 0.995
    lr_model: float = 3e-4
    lr_policy: float = 3e-4
    latent_loss_coef: float = 1.0
    reward_loss_coef: float = 1.0
    q_loss_coef: float = 1.0
    policy_loss_coef: float = 1.0
    grad_clip_norm: float = 10.0
    q_ensemble: int = 2

    @classmethod
    def from_mapping(cls, values):
        values = values or {}
        valid = cls.__dataclass_fields__.keys()
        return cls(**{k.lower(): v for k, v in values.items() if k.lower() in valid})
