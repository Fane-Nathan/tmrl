import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import jax
import jax.numpy as jnp
from tmrl.core.util import load
import tmrl.config.config_constants as cfg
from tmrl.core.jax.util import get_rngs
from tmrl.custom.jax.imagination_trainer import LatentImaginationTrainer


def main():
    print("=" * 75)
    print("  TRAINING A REPLAY-GROUNDED LATENT WORLD MODEL AND IMAGINATION POLICY")
    print("=" * 75)

    cpt_path = r"C:\Users\felix\TmrlData\checkpoints\AbsoluteZero_Test_t.tcpt"
    if not os.path.exists(cpt_path):
        print(f"Checkpoint not found at {cpt_path}")
        return

    print(f"[1] Loading live replay buffer checkpoint: {cpt_path} ...", flush=True)
    t0 = time.time()
    cpt = load(cpt_path)
    print(f"    Loaded checkpoint in {time.time() - t0:.2f}s", flush=True)

    # Inspect checkpoint object
    memory = None
    if hasattr(cpt, 'memory'):
        memory = cpt.memory
    elif isinstance(cpt, dict) and 'memory' in cpt:
        memory = cpt['memory']
    
    if memory is None:
        print("    [!] Could not locate memory attribute on checkpoint, using cpt directly.")
        memory = cpt

    print(f"[2] Replay Buffer Type: {type(memory)}", flush=True)
    if hasattr(memory, '__len__'):
        print(f"    Total transitions in buffer: {len(memory)}", flush=True)

    # Initialize Imagination Trainer
    rngs = get_rngs(params_seed=42, noise_seed=42)
    trainer = LatentImaginationTrainer(
        horizon=5,
        gamma=0.99,
        lambda_=0.95,
        lr_model=3e-4,
        lr_actor=1e-4,
        lr_critic=3e-4,
        lr_adv=1e-4,
        rngs=rngs
    )
    print("[3] Initialized latent world model, actor, and critic (proposer disabled).", flush=True)

    # Sample batch from memory
    print("[4] Sampling a transition batch from real memory...", flush=True)

    try:
        sample = memory.sample()
    except Exception as e:
        raise RuntimeError(
            "Could not sample the real replay buffer. Synthetic fallback is intentionally disabled "
            "because unrelated random transitions cannot train a world model."
        ) from e

    if not isinstance(sample, (tuple, list)) or len(sample) != 6:
        raise RuntimeError("Expected replay.sample() to return six transition fields")

    obs_tuple, actions, rewards, next_obs_tuple, terminated, truncated = sample

    def to_jax(leaf):
        if hasattr(leaf, "detach"):
            leaf = leaf.detach().cpu().numpy()
        return jnp.asarray(np.asarray(leaf))

    obs_tuple = jax.tree.map(to_jax, obs_tuple)
    next_obs_tuple = jax.tree.map(to_jax, next_obs_tuple)
    actions = to_jax(actions)
    rewards = to_jax(rewards)
    terminated = to_jax(terminated)
    truncated = to_jax(truncated)
    continues = 1.0 - jnp.maximum(terminated, truncated)
    batch_size = int(actions.shape[0])
    print(f"    Sampled {batch_size} real replay transitions.", flush=True)

    print("\n[5] Fitting Latent World Model on Real Dynamics (20 Steps)...", flush=True)
    for step in range(20):
        metrics = trainer.train_world_model_step(
            obs_tuple, actions, rewards, continues, next_obs_tuple=next_obs_tuple
        )
        if (step + 1) % 5 == 0:
            print(
                f"    Step {step+1}/20 - Model: {metrics['model_loss']:.4f} | "
                f"Reward: {metrics['rew_loss']:.4f} | Embed: {metrics['embed_loss']:.4f} | "
                f"KL dyn/rep: {metrics['kl_dyn']:.4f}/{metrics['kl_rep']:.4f}",
                flush=True,
            )

    print("\n[6] Running short replay-anchored imagination policy updates (50 steps)...", flush=True)
    t_start = time.time()
    for step in range(50):
        stats = trainer.train_imagination_step(next_obs_tuple)
        if (step + 1) % 10 == 0:
            print(
                f"    Imagination Step {step+1}/50 - Return: {stats['mean_imagined_return']:.2f} | "
                f"Actor: {stats['actor_loss']:.3f} | Critic: {stats['critic_loss']:.3f}",
                flush=True,
            )

    dt = time.time() - t_start
    total_imagined = 50 * trainer.horizon * batch_size
    print(
        f"\n[7] Completed {total_imagined} replay-anchored imagined transitions in {dt:.2f}s "
        f"(~{total_imagined/max(dt, 1e-4):.0f} trans/sec).",
        flush=True,
    )
    print("    This is a training smoke run, not a held-out zero-shot evaluation.", flush=True)
    print("=" * 75)


if __name__ == "__main__":
    main()
