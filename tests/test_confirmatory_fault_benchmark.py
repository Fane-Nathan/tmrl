import random

import numpy as np
import torch

from scripts.train_transformer_fault_adaptation import (
    CausalSelfAttention,
    SequenceReplayBuffer,
    TransformerRL2Agent,
)


def _episode(length: int, obs_dim: int = 25, act_dim: int = 6):
    obs = np.arange((length + 1) * obs_dim, dtype=np.float32).reshape(length + 1, obs_dim)
    acts = np.zeros((length, act_dim), dtype=np.float32)
    rews = np.arange(length, dtype=np.float32)
    dones = np.zeros(length, dtype=np.float32)
    dones[-1] = 1.0
    return obs, acts, rews, dones


def test_sequence_replay_can_sample_final_terminal_window(monkeypatch):
    L = 4
    T = 7
    replay = SequenceReplayBuffer(max_transitions=100, max_seq_len=L, burn_in=1)
    obs, acts, rews, dones = _episode(T)
    replay.add_episode(obs, acts, rews, dones, fault_id=1)

    monkeypatch.setattr(random, "choice", lambda xs: xs[0])
    monkeypatch.setattr(random, "randint", lambda low, high: high)
    batch = replay.sample_batch(1)

    assert batch["rews"][0, -1, 0].item() == float(T - 1)
    assert batch["dones"][0, -1, 0].item() == 1.0
    assert batch["valid"].sum().item() == float(L)


def test_short_episode_padding_is_explicitly_masked(monkeypatch):
    L = 5
    T = 3
    replay = SequenceReplayBuffer(max_transitions=100, max_seq_len=L, burn_in=1)
    obs, acts, rews, dones = _episode(T)
    replay.add_episode(obs, acts, rews, dones, fault_id=1)
    monkeypatch.setattr(random, "choice", lambda xs: xs[0])

    batch = replay.sample_batch(1)
    np.testing.assert_array_equal(
        batch["valid"][0].cpu().numpy(),
        np.array([1, 1, 1, 0, 0], dtype=np.float32),
    )
    assert batch["valid_lengths"][0].item() == T


def test_history_blocked_final_query_cannot_read_prior_tokens():
    torch.manual_seed(7)
    attn = CausalSelfAttention(d_model=8, n_heads=2, dropout=0.0).eval()
    x1 = torch.randn(1, 4, 8)
    x2 = x1.clone()
    x2[:, :-1] += 10.0

    with torch.no_grad():
        y1 = attn(x1, block_history_for_last_token=True)
        y2 = attn(x2, block_history_for_last_token=True)
        y1_full = attn(x1, block_history_for_last_token=False)
        y2_full = attn(x2, block_history_for_last_token=False)

    torch.testing.assert_close(y1[:, -1], y2[:, -1], rtol=0.0, atol=1e-6)
    assert not torch.allclose(y1_full[:, -1], y2_full[:, -1])


def test_inference_context_never_exceeds_trained_window():
    agent = TransformerRL2Agent(context_window=4)
    obs = np.zeros(25, dtype=np.float32)
    for _ in range(9):
        agent.step_in_context(obs, adaptive=True, deterministic=True, device="cpu")
    assert len(agent.context_deque) == 4
    assert agent.context_deque.maxlen == 4
