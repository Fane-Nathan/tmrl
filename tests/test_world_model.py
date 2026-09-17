import torch
from gymnasium.spaces import Box, Tuple

from tmrl.world_model.agent import WorldModelAgent
from tmrl.world_model.config import WorldModelConfig


def test_world_model_train_step_shapes():
    action_space = Box(low=-1.0, high=1.0, shape=(3,))
    observation_space = Tuple((
        Box(low=0.0, high=1.0, shape=(1,)),
        Box(low=0.0, high=1.0, shape=(1,)),
        Box(low=0.0, high=1.0, shape=(1,)),
        Box(low=0.0, high=1.0, shape=(4, 64, 64)),
        action_space,
        action_space,
    ))
    cfg = WorldModelConfig(horizon=3, batch_size=2, planning_samples=4, planning_horizon=2)
    agent = WorldModelAgent(
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
        wm_config=cfg,
        img_hist_len=4,
        action_history_len=2,
    )

    batch_size, horizon = 2, cfg.horizon
    obs_seq = (
        torch.rand(batch_size, horizon + 1, 1),
        torch.rand(batch_size, horizon + 1, 1),
        torch.rand(batch_size, horizon + 1, 1),
        torch.rand(batch_size, horizon + 1, 4, 64, 64),
        torch.rand(batch_size, horizon + 1, 3) * 2 - 1,
        torch.rand(batch_size, horizon + 1, 3) * 2 - 1,
    )
    actions = torch.rand(batch_size, horizon, 3) * 2 - 1
    rewards = torch.rand(batch_size, horizon)
    terminated = torch.zeros(batch_size, horizon)
    truncated = torch.zeros(batch_size, horizon)

    metrics = agent.train((obs_seq, actions, rewards, terminated, truncated))
    assert metrics["wm_sequence_horizon"] == float(horizon)
    assert metrics["wm_positions_per_update"] == float(batch_size * horizon)

    actor = agent.get_actor()
    action = actor.forward(tuple(x[:, 0] for x in obs_seq), test=True)
    assert action.shape == (batch_size, 3)
    assert torch.all(action <= 1.0)
    assert torch.all(action >= -1.0)
