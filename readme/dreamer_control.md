# Recurrent Dreamer control for TrackMania 2020

`DREAMER` is an opt-in, image-based controller whose deployed worker actor owns
the observation encoder, recurrent state-space model (RSSM), and latent policy.
The policy optimized on imagined trajectories is therefore the same policy that
drives the car. This is separate from the default SAC pipeline.

The implementation follows the central Dreamer training flow:

1. Sample contiguous replay sequences without crossing episode boundaries.
2. Train the encoder, RSSM, reconstruction, reward, and continuation heads.
3. Start latent rollouts from detached posterior states.
4. Train a latent actor and slow-target critic on imagined lambda returns.
5. Broadcast the encoder, RSSM filter, and latent actor to the rollout worker.

It is a compact PyTorch Dreamer-style implementation sized for an 8 GB GPU,
not a bit-for-bit reproduction of the large JAX DreamerV3 reference model. In
particular, it currently uses Gaussian stochastic states and scalar symlog
value regression instead of DreamerV3's categorical states and two-hot heads.

## Configuration

Keep image observations and grayscale enabled. Add the `DREAMER` block and set
the algorithm as follows:

```json
{
  "RUN_NAME": "Dreamer_AZR_Test",
  "RESET_TRAINING": false,
  "ALG": {
    "ALGORITHM": "DREAMER",
    "GAMMA": 0.99,
    "DREAMER": {
      "BATCH_SIZE": 8,
      "SEQUENCE_LENGTH": 16,
      "BURN_IN": 5,
      "LATENT_DIM": 128,
      "HIDDEN_DIM": 256,
      "POLICY_HIDDEN_DIM": 256,
      "RECONSTRUCTION_SIZE": 24,
      "LR_WORLD_MODEL": 0.0003,
      "LR_ACTOR": 0.00008,
      "LR_CRITIC": 0.00008,
      "LAMBDA": 0.95,
      "FREE_NATS": 1.0,
      "HORIZON": 8,
      "IMAGINATION_BATCH_SIZE": 32,
      "WARMUP_STEPS": 2000,
      "ENTROPY_SCALE": 0.0003,
      "TARGET_POLYAK": 0.99,
      "GRAD_CLIP": 100.0,
      "REPLAY_MEMORY": {
        "MODE": "poincare",
        "CANDIDATES": 128,
        "UNIFORM_FRACTION": 0.25,
        "EMBED_DIM": 16,
        "CURVATURE": 1.0,
        "TANGENT_SCALE": 0.75,
        "MAX_RADIUS": 0.95,
        "SEED": 0
      }
    },
    "AZR_IMAGINATION": {
      "ENABLED": true,
      "LR_PROPOSER": 0.0001,
      "SOLVER_ATTEMPTS": 4,
      "TASKS_PER_PROPOSAL": 4,
      "TASK_BATCH_SIZE": 8,
      "TASK_BUFFER_SIZE": 1024,
      "TASK_MAX_AGE": 2000,
      "PROPOSAL_INTERVAL": 8,
      "MIN_CONTINUATION": 0.5,
      "MAX_PERTURBATION": 0.2,
      "TARGET_QUANTILE": 0.5,
      "SEED": 0
    }
  }
}
```

The nested Dreamer batch size overrides the top-level SAC batch size. If the
trainer runs out of VRAM, first reduce `BATCH_SIZE` to `4`, then reduce
`IMAGINATION_BATCH_SIZE` to `16`.

### Experimental Poincare replay index

`REPLAY_MEMORY.MODE` supports three controlled replay-selection modes:

- `uniform` preserves the original random sequence sampler.
- `euclidean` chooses a diverse batch using Euclidean distances.
- `poincare` maps the same detached descriptors into a Poincare ball and uses
  hyperbolic distance for the diversity selection.

The descriptor combines action, speed, gear, RPM, reward, and coarse image
statistics. It is a replay index only: it does not change the RSSM, actor,
world-model losses, raw replay tensors, or worker inference. A
`UNIFORM_FRACTION` of `0.25` keeps random anchors in every selected batch so
rare outliers cannot completely take over training.

Trainer statistics expose `replay_sampler_code` (`0` uniform, `1` Euclidean,
`2` Poincare), `replay_candidate_count`, `replay_mean_distance`, and
`replay_mean_radius`. Compare held-out-map retention across all three modes;
larger hyperbolic distance alone is not evidence of better driving or
zero-shot generalization.

## Reusing the current SAC replay safely

Do not run SAC and Dreamer trainers against the same server at the same time.
After stopping the current trainer and worker, copy the SAC checkpoint to the
new run name before changing the configuration:

```powershell
Copy-Item -LiteralPath "$HOME\TmrlData\checkpoints\AbsoluteZero_Test_t.tcpt" `
  -Destination "$HOME\TmrlData\checkpoints\Dreamer_AZR_Test_t.tcpt"
```

The checkpoint updater creates a fresh Dreamer agent and sequence sampler while
retaining the copied raw replay tensors. The SAC checkpoint remains available
under its original run name.

Start the components in separate terminals in this order:

```powershell
.\.venv\Scripts\python.exe -m tmrl --server
.\.venv\Scripts\python.exe -m tmrl --trainer
# Wait for the Dreamer startup/rebuild messages, then start the worker:
.\.venv\Scripts\python.exe -m tmrl --worker
```

Use a trainer before a worker for a new run name so the worker receives a
compatible recurrent actor broadcast before collecting a full episode.

## Expected metrics

During world-model warm-up, `dreamer_ready`, `azr_ready`, and
`imagined_policy_updates` are correctly zero while the world-model losses are
nonzero. At `WARMUP_STEPS`, `dreamer_ready` becomes `1` and
`imagined_policy_updates` should normally be `1` on every update. With AZR
enabled, proposals occur only on `PROPOSAL_INTERVAL`; accepted task counts can
still be zero when all candidates are unsupported, impossible, or trivial.

This does not guarantee perfect driving or automatic zero-shot generalization.
The world model must first become accurate on real replay, and evaluation on
held-out tracks is still required.
