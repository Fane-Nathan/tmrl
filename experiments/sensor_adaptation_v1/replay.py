"""Episode replay with exact rolling contexts and terminal-transition coverage."""
import numpy as np
import torch


def save_episodes(path, episodes):
    payload = {f"{i}_{key}": value for i, episode in enumerate(episodes)
               for key, value in episode.items()}
    np.savez_compressed(path, **payload)


def load_episodes(path):
    with np.load(path, allow_pickle=False) as data:
        count = len([k for k in data.files if k.endswith("_actions")])
        return [{key: data[f"{i}_{key}"].copy() for key in
                 ("observations", "actions", "rewards", "terminated")} for i in range(count)]


def extract_transition(episode, t, context):
    """Next context slides when full; it includes the actual final observation."""
    observations = episode["observations"]
    if not 0 <= t < len(episode["actions"]):
        raise IndexError(t)
    current = observations[max(0, t - context + 1):t + 1]
    following = observations[max(0, t - context + 2):t + 2]
    return current, following


def sample_batch(new_episodes, old_episodes, batch_size, context, old_fraction, rng, device):
    if not new_episodes or (old_fraction and not old_episodes):
        raise ValueError("Required replay pool is empty")
    n_old = int(round(batch_size * old_fraction))
    obs_dim = new_episodes[0]["observations"].shape[-1]
    acts_dim = new_episodes[0]["actions"].shape[-1]
    current = np.zeros((batch_size, context, obs_dim), np.float32)
    following = np.zeros_like(current)
    lengths = np.zeros(batch_size, np.int64)
    next_lengths = np.zeros_like(lengths)
    actions = np.zeros((batch_size, acts_dim), np.float32)
    rewards, terminated = np.zeros((batch_size, 1), np.float32), np.zeros((batch_size, 1), np.float32)
    origins, positions = [], []
    # With unequal episode lengths (early termination, budget-capped episodes), pick episodes in
    # proportion to length so sampling stays uniform over transitions. Equal lengths keep the
    # pilot's uniform draw and therefore its exact random stream.
    weights = {}
    for key, pool in (("old", old_episodes), ("new", new_episodes)):
        sizes = np.asarray([len(e["actions"]) for e in pool], np.float64)
        weights[key] = None if sizes.size == 0 or sizes.min() == sizes.max() else sizes / sizes.sum()
    for i in range(batch_size):
        old = i < n_old
        pool = old_episodes if old else new_episodes
        p = weights["old" if old else "new"]
        episode = pool[int(rng.integers(len(pool)))] if p is None else pool[int(rng.choice(len(pool), p=p))]
        size = len(episode["actions"])
        # Deliberately represent deployment's short initial contexts.
        limit = min(size, context) if rng.random() < 0.25 else size
        t = int(rng.integers(limit))
        cur, nxt = extract_transition(episode, t, context)
        current[i, :len(cur)], following[i, :len(nxt)] = cur, nxt
        lengths[i], next_lengths[i] = len(cur), len(nxt)
        actions[i], rewards[i, 0] = episode["actions"][t], episode["rewards"][t]
        terminated[i, 0] = episode["terminated"][t]
        origins.append("old" if old else "new")
        positions.append(t)
    # Right padding follows every queried token, so causal attention cannot see it.
    return {"obs": torch.as_tensor(current, device=device),
            "next_obs": torch.as_tensor(following, device=device),
            "lengths": torch.as_tensor(lengths, device=device),
            "next_lengths": torch.as_tensor(next_lengths, device=device),
            "actions": torch.as_tensor(actions, device=device),
            "rewards": torch.as_tensor(rewards, device=device),
            "terminated": torch.as_tensor(terminated, device=device),
            "origins": origins, "positions": positions}
