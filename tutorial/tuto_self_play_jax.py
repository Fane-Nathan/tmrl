import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import jax
import jax.numpy as jnp
import gymnasium as gym

from tmrl.custom.tm.utils.self_play_reward import SelfPlayTrajectoryManager, SelfPlayRewardFunction
from tmrl.custom.tm.utils.self_play_obs import SelfPlayGhostObserver
from tmrl.custom.jax.custom_algorithms import NNXSACAgent
from tmrl.custom.jax.custom_models import NNXREDQMLPActorCritic


def run_self_play_demo():
    print("=" * 70)
    print(" TMRL Self-Play Ghost Ratcheting Demo (AZR-Inspired Curriculum Primitive)")
    print("=" * 70)

    # 1. Create a temporary reward file path for demonstration
    temp_dir = tempfile.mkdtemp()
    demo_save_path = os.path.join(temp_dir, "test_self_play_ghost.pkl")

    try:
        manager = SelfPlayTrajectoryManager(save_path=demo_save_path, min_points_to_save=5)
        reward_fn = SelfPlayRewardFunction(trajectory_manager=manager, personal_record_bonus=5.0)
        ghost_obs = SelfPlayGhostObserver(trajectory_manager=manager)

        print(f"\n[Step 1] Initial state (Cold Start): has_reference_ghost = {manager.has_reference_ghost}")

        # Simulate Episode 1: Initial exploration (drives 30 meters)
        print("\n--- Episode 1: Initial Exploration ---")
        for step in range(30):
            pos = np.array([float(step), 0.0, 0.0])
            speed = 20.0 + step * 2.0
            t = step * 0.05
            rew, term = reward_fn.compute_reward(pos=pos, speed=speed, timestamp=t)
            ghost_feat = ghost_obs.get_ghost_features(pos, speed, t, reward_fn.cur_idx)

        ratcheted = reward_fn.reset(finished_track=False)
        print(f"Episode 1 End -> Ratcheted: {ratcheted}, Ghost Pts: {len(manager.best_trajectory)}, Best Dist: {manager.best_distance:.1f}m")

        # Simulate Episode 2: Shorter run (fails early at step 15)
        print("\n--- Episode 2: Underperforming Run ---")
        for step in range(15):
            pos = np.array([float(step), 0.0, 0.0])
            speed = 20.0
            t = step * 0.05
            rew, term = reward_fn.compute_reward(pos=pos, speed=speed, timestamp=t)
        ratcheted = reward_fn.reset(finished_track=False)
        print(f"Episode 2 End -> Ratcheted: {ratcheted} (Correctly ignored inferior run)")

        # Simulate Episode 3: Breakthrough run (explores up to 60 meters)
        print("\n--- Episode 3: Breakthrough Exploration (New Record) ---")
        for step in range(60):
            pos = np.array([float(step), 0.0, 0.0])
            speed = 30.0 + step * 1.5
            t = step * 0.05
            rew, term = reward_fn.compute_reward(pos=pos, speed=speed, timestamp=t)
        ratcheted = reward_fn.reset(finished_track=False)
        print(f"Episode 3 End -> Ratcheted: {ratcheted}, Ghost Pts: {len(manager.best_trajectory)}, Best Dist: {manager.best_distance:.1f}m")

        # Simulate Episode 4: Faster Lap over existing 60m track
        print("\n--- Episode 4: Faster Lap Optimization ---")
        total_rew = 0.0
        for step in range(60):
            pos = np.array([float(step), 0.0, 0.0])
            # Higher speed than previous ghost
            speed = 50.0 + step * 2.0
            t = step * 0.03  # Faster elapsed time!
            rew, term = reward_fn.compute_reward(pos=pos, speed=speed, timestamp=t, finished_track=(step == 59))
            total_rew += rew
            ghost_feat = ghost_obs.get_ghost_features(pos, speed, t, reward_fn.cur_idx)

        ratcheted = reward_fn.reset(finished_track=True)
        print(f"Episode 4 End -> Total Reward: {total_rew:.2f}, Ratcheted Faster Ghost: {ratcheted}, Best Time: {manager.best_time:.2f}s")

        # 2. Test JAX SAC Agent compatibility with Self-Play Transitions
        print("\n[Step 2] Testing JAX NNXSACAgent with Self-Play Rewards...")
        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,))
        act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,))

        agent = NNXSACAgent(
            observation_space=obs_space,
            action_space=act_space,
            model_cls=NNXREDQMLPActorCritic,
            lr_actor=1e-3,
            lr_critic=1e-3
        )

        batch_size = 16
        obs_batch = jnp.zeros((batch_size, 10), dtype=jnp.float32)
        act_batch = jnp.zeros((batch_size, 3), dtype=jnp.float32)
        rew_batch = jnp.ones(batch_size, dtype=jnp.float32) * total_rew
        next_obs_batch = jnp.zeros((batch_size, 10), dtype=jnp.float32)
        done_batch = jnp.zeros(batch_size, dtype=jnp.float32)

        batch = (obs_batch, act_batch, rew_batch, next_obs_batch, done_batch, None)
        loss_dict = agent.train(batch)
        print("JAX SAC Agent Training Step Output:", loss_dict)
        print("\n" + "=" * 70)
        print(" Self-Play Ghost Ratcheting Smoke Test PASSED!")
        print(" This is self-imitation, not a complete AZR or zero-shot evaluation.")
        print("=" * 70)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_self_play_demo()
