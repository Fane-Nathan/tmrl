# TMRL World Model v1

This package is an experimental **non-SAC** Trackmania training path. It keeps
TMRL's real-time environment, networking, compression, reward function and
worker/server infrastructure, while replacing the SAC learner with:

- a compact visual encoder,
- a latent transition model,
- a reward model,
- a small pessimistic Q ensemble,
- a deterministic policy prior, and
- short-horizon latent shooting at inference time.

The first research objective is deliberately narrow: **reach the same driving
competence as the original image-based TMRL SAC using fewer real environment
steps**. It is not an ADAS stack and does not yet implement uncertainty/event
modeling.

## Optional config

Add a top-level `WORLD_MODEL` object to `~/TmrlData/config/config.json` to
override defaults. Keys are case-insensitive for model hyperparameters.

```json
"WORLD_MODEL": {
  "RUN_NAME": "WORLD_MODEL_V1",
  "HORIZON": 5,
  "BATCH_SIZE": 16,
  "LATENT_DIM": 256,
  "HIDDEN_DIM": 256,
  "PLANNING_HORIZON": 5,
  "PLANNING_SAMPLES": 64,
  "PLANNING_NOISE": 0.35,
  "TEST_PLANNING_NOISE": 0.15,
  "START_TRAINING": 5000,
  "UPDATES_PER_ENV_STEP": 1.0
}
```

`HORIZON` is the contiguous replay sequence length used for world-model
training. A sequence never crosses an episode reset. `PLANNING_HORIZON` is the
number of latent steps simulated before executing only the first selected
action.

## Fair SAC comparison

Compare at least:

1. return vs real environment steps,
2. environment steps to a fixed competence threshold,
3. return vs wall-clock time, and
4. GPU training time / positions replayed.

A world-model update processes `batch_size * horizon` temporal positions, so
raw optimizer-step counts are not directly comparable to SAC optimizer steps.
