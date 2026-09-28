import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
import tmrl.config.config_constants as cfg
from tmrl.core.jax.util import get_rngs
from tmrl.custom.jax.imagination_trainer import LatentImaginationTrainer


def main():
    print("=" * 70)
    print("TRACKMANIA REPLAY-GROUNDED LATENT IMAGINATION SMOKE TEST")
    print("=" * 70)

    # 1. Initialize Trainer
    rngs = get_rngs(params_seed=42, noise_seed=42)
    trainer = LatentImaginationTrainer(
        horizon=3,
        gamma=0.99,
        lambda_=0.95,
        rngs=rngs
    )
    print("[1] Initialized LatentImaginationTrainer successfully.", flush=True)

    # 2. Synthesize Batches of TrackMania Observations (96x96x4 images + telemetry)
    batch_size = 16
    rng = np.random.default_rng(7)
    speed = jnp.array(rng.uniform(50.0, 300.0, (batch_size, 1)), dtype=jnp.float32)
    gear = jnp.array(rng.integers(1, 5, (batch_size, 1)), dtype=jnp.float32)
    rpm = jnp.array(rng.uniform(5000.0, 10000.0, (batch_size, 1)), dtype=jnp.float32)
    # (B, H, W, C)
    images = jnp.array(
        rng.uniform(
            0.0,
            1.0,
            (batch_size, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, cfg.IMG_HIST_LEN),
        ),
        dtype=jnp.float32,
    )

    obs_tuple = (speed, gear, rpm, images)
    actions = jnp.array(rng.uniform(-1.0, 1.0, (batch_size, 3)), dtype=jnp.float32)
    next_speed = jnp.clip(speed + 2.0 * actions[:, :1], 0.0, 400.0)
    next_gear = gear
    next_rpm = jnp.clip(rpm + 50.0 * actions[:, :1], 0.0, 12000.0)
    next_images = jnp.roll(images, shift=1, axis=2)
    next_obs_tuple = (next_speed, next_gear, next_rpm, next_images)
    rewards = 0.01 * next_speed.squeeze(-1) - 0.1 * jnp.abs(actions[:, 2])
    continues = jnp.ones((batch_size,), dtype=jnp.float32)

    print(f"[2] Generated synthetic transition batch with shape: images={images.shape}, telem=3d, act=3d", flush=True)

    # 3. Train World Model on Transitions
    print("\n[3] Fitting World Model Dynamics & Symlog Predictor...", flush=True)
    for step in range(5):
        metrics = trainer.train_world_model_step(
            obs_tuple, actions, rewards, continues, next_obs_tuple=next_obs_tuple
        )
        print(
            f"    Step {step+1}/5 - Model: {metrics['model_loss']:.4f} | "
            f"Reward: {metrics['rew_loss']:.4f} | Embed: {metrics['embed_loss']:.4f} | "
            f"KL dyn/rep: {metrics['kl_dyn']:.4f}/{metrics['kl_rep']:.4f}",
            flush=True,
        )

    # 4. Latent Imagination Rollout & Policy Optimization
    print("\n[4] Running replay-anchored latent imagination rollouts (H=3)...", flush=True)
    t0 = time.time()
    num_dream_steps = 10
    total_imagined_transitions = num_dream_steps * trainer.horizon * batch_size

    for step in range(num_dream_steps):
        stats = trainer.train_imagination_step(next_obs_tuple)
        print(
            f"    Step {step+1}/{num_dream_steps} - Return: {stats['mean_imagined_return']:.2f} | "
            f"Actor: {stats['actor_loss']:.3f} | Critic: {stats['critic_loss']:.3f}",
            flush=True,
        )

    t1 = time.time()
    dt = t1 - t0
    fps = total_imagined_transitions / max(dt, 1e-5)
    print(
        f"\n[5] Performance: imagined {total_imagined_transitions} transitions in {dt:.3f}s "
        f"(~{fps:.0f} transitions/sec on {jax.default_backend()}).",
        flush=True,
    )
    print("\n[6] LATENT IMAGINATION SMOKE TEST PASSED.", flush=True)
    print("    Held-out track transfer was not evaluated; this is not proof of zero-shot learning.", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
