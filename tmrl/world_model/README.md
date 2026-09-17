# TMRL World Model v1.1

This package is an experimental **non-SAC** Trackmania training path. It keeps
TMRL's real-time environment, networking, compression, reward function and
worker/server infrastructure, while replacing the SAC learner with:

- a compact visual encoder,
- a latent transition model,
- a reward model,
- a small pessimistic Q ensemble,
- a deterministic policy prior,
- vehicle-aware bootstrap exploration, and
- short-horizon latent shooting at inference time.

Trackmania is the first controlled visual continuous-control testbed. The wider
research objective is to improve sample efficiency, wall-clock time-to-skill and
final control performance, then progressively test generalization and transfer
toward real robotic control. Comparisons against TMRL SAC are an early sanity
benchmark, not the end goal.

## Run

Use the same Trackmania/OpenPlanet/TmrlData setup as normal TMRL. Start the
three components in separate terminals:

```bash
python -m tmrl.world_model.run server
python -m tmrl.world_model.run trainer --wandb
python -m tmrl.world_model.run worker
```

For standalone inference with the world-model weights:

```bash
python -m tmrl.world_model.run test
```

Use a fresh `RUN_NAME` when changing controller/replay semantics so an old
checkpoint does not contaminate the experiment.

## Optional config

Add a top-level `WORLD_MODEL` object to `~/TmrlData/config/config.json` to
override defaults. Keys are case-insensitive for model hyperparameters.

```json
"WORLD_MODEL": {
  "RUN_NAME": "WORLD_MODEL_V1_1",
  "HORIZON": 5,
  "BATCH_SIZE": 16,
  "LATENT_DIM": 256,
  "HIDDEN_DIM": 256,
  "PLANNING_HORIZON": 5,
  "PLANNING_SAMPLES": 64,
  "PLANNING_NOISE": 0.35,
  "TEST_PLANNING_NOISE": 0.15,
  "BOOTSTRAP_EXPLORATION_STEPS": 15000,
  "BOOTSTRAP_EPSILON_START": 0.70,
  "BOOTSTRAP_EPSILON_END": 0.10,
  "BOOTSTRAP_GAS_MIN": 0.60,
  "BOOTSTRAP_GAS_MAX": 1.00,
  "BOOTSTRAP_STEER_RHO": 0.90,
  "BOOTSTRAP_STEER_STD": 0.30,
  "BOOTSTRAP_STEER_LIMIT": 0.70,
  "START_TRAINING": 5000,
  "UPDATES_PER_ENV_STEP": 1.0
}
```

`HORIZON` is the contiguous replay sequence length used for world-model
training. A sequence never crosses an episode reset. `PLANNING_HORIZON` is the
number of latent steps simulated before executing only the first selected
action.

V1.1 canonicalizes controls to the physical Trackmania semantics:
`gas in [0, 1]`, `brake in [0, 1]`, and `steer in [-1, 1]`, with simultaneous
pedal commands reduced to one net longitudinal command. During training only,
the worker mixes MPC with forward-biased exploration. Steering noise is
temporally correlated so early data contains coherent trajectories rather than
independent 20 Hz steering jitter. Evaluation (`test=True`) always bypasses
bootstrap exploration.

The planner also reserves candidates for coherent full-horizon behaviors such
as full-throttle straight driving, moderate-throttle left/right arcs, coasting,
and braking. This ensures the planner can evaluate physically useful behavior
even while the learned policy prior is immature.

## Fair comparisons

Compare at least:

1. return vs real environment steps,
2. environment steps to a fixed competence threshold,
3. return vs wall-clock time,
4. GPU/compute time to the same threshold, and
5. final repeated evaluation performance.

A world-model update processes `batch_size * horizon` temporal positions, so
raw optimizer-step counts are not directly comparable to SAC optimizer steps.
