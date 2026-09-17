import torch
from gymnasium.spaces import Box, Tuple

from tmrl.world_model.actions import canonicalize_tm_action
from tmrl.world_model.actor import WorldModelActor
from tmrl.world_model.agent import WorldModelAgent
from tmrl.world_model.config import WorldModelConfig


def _spaces():
    action_space = Box(low=-1.0, high=1.0, shape=(3,))
    observation_space = Tuple((
        Box(low=0.0, high=1.0, shape=(1,)),
        Box(low=0.0, high=1.0, shape=(1,)),
        Box(low=0.0, high=1.0, shape=(1,)),
        Box(low=0.0, high=1.0, shape=(4, 64, 64)),
        action_space,
        action_space,
    ))
    return observation_space, action_space


def _obs(batch_size=1):
    return (
        torch.rand(batch_size, 1),
        torch.rand(batch_size, 1),
        torch.rand(batch_size, 1),
        torch.rand(batch_size, 4, 64, 64),
        torch.rand(batch_size, 3) * 2 - 1,
        torch.rand(batch_size, 3) * 2 - 1,
    )


def test_canonical_trackmania_action_semantics():
    actions = torch.tensor([
        [-0.8, -0.2, 0.4],
        [0.9, 0.3, -1.4],
        [0.2, 0.8, 1.4],
    ])
    canonical = canonicalize_tm_action(actions)

    assert torch.allclose(canonical[0], torch.tensor([0.0, 0.0, 0.4]))
    assert torch.allclose(canonical[1], torch.tensor([0.6, 0.0, -1.0]))
    assert torch.allclose(canonical[2], torch.tensor([0.0, 0.6, 1.0]))
    assert torch.all(canonical[:, :2] >= 0.0)
    assert torch.all(canonical[:, :2] <= 1.0)
    assert torch.all(canonical[:, 0] * canonical[:, 1] == 0.0)


def test_bootstrap_exploration_is_forward_biased_and_canonical():
    observation_space, action_space = _spaces()
    cfg = WorldModelConfig(
        planning_samples=8,
        planning_horizon=2,
        bootstrap_exploration_steps=10,
        bootstrap_epsilon_start=1.0,
        bootstrap_epsilon_end=1.0,
        bootstrap_gas_min=0.75,
        bootstrap_gas_max=0.75,
        bootstrap_steer_std=0.0,
    )
    actor = WorldModelActor(
        observation_space=observation_space,
        action_space=action_space,
        wm_config=cfg,
        img_hist_len=4,
        action_history_len=2,
        device="cpu",
    )

    action = actor.act(_obs(), test=False)
    assert action.shape == (3,)
    assert abs(float(action[0]) - 0.75) < 1e-6
    assert float(action[1]) == 0.0
    assert float(action[2]) == 0.0
    assert actor._interaction_step == 1


def test_world_model_train_step_shapes():
    observation_space, action_space = _spaces()
    cfg = WorldModelConfig(horizon=3, batch_size=2, planning_samples=8, planning_horizon=2)
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
    assert torch.all(action[:, 0] >= 0.0)
    assert torch.all(action[:, 0] <= 1.0)
    assert torch.all(action[:, 1] >= 0.0)
    assert torch.all(action[:, 1] <= 1.0)
    assert torch.all(action[:, 2] >= -1.0)
    assert torch.all(action[:, 2] <= 1.0)
    assert torch.all(action[:, 0] * action[:, 1] == 0.0)
