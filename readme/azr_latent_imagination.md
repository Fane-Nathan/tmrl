# AZR, Latent Imagination, and Zero-Shot Transfer

This repository contains three connected curriculum components:

- ghost ratcheting, which is grounded self-imitation against the agent's best trajectory;
- a Torch world model grounded by real replay transitions and deployed-policy features;
- an AZR-style latent task loop that proposes, validates, repeatedly solves, prioritizes, and learns from short imagined challenges.

The Torch task loop updates the same SAC action head broadcast to workers. It is
still not a structured TrackMania map generator, and a successful smoke test is
not evidence of zero-shot transfer.

## Production behavior

The connected path is enabled under `ALG`:

```json
{
  "ALG": {
    "ALGORITHM": "SAC",
    "AZR_IMAGINATION": {
      "ENABLED": true,
      "WARMUP_STEPS": 2000,
      "HORIZON": 3,
      "SOLVER_ATTEMPTS": 8,
      "TASKS_PER_PROPOSAL": 8,
      "TASK_BATCH_SIZE": 8,
      "TASK_BUFFER_SIZE": 1024,
      "TASK_MAX_AGE": 2000,
      "PROPOSAL_INTERVAL": 4,
      "IMAGINATION_INTERVAL": 2,
      "MIN_CONTINUATION": 0.55,
      "MAX_PERTURBATION": 0.2,
      "TARGET_QUANTILE": 0.5,
      "ACTOR_LOSS_SCALE": 0.02
    }
  }
}
```

At runtime the Trainer now performs:

```text
real replay -> SAC update + world-model update
            -> posterior-anchored task proposal
            -> support/continuation validation
            -> repeated stochastic solver attempts
            -> AZR learnability priority
            -> short screened imagination
            -> deployed SAC action-head update
            -> worker broadcast
```

Real SAC training never waits for imagination. The world model must complete
`WARMUP_STEPS` updates before tasks are accepted. Latent perturbations are
bounded in posterior-standard-deviation units, tasks expire as the model
changes, and actor gradients are scaled and clipped. The SAC critics remain
anchored to real observations; imagined transitions update the shared action
head through differentiable model returns instead of fabricating raw images for
the critics.

Monitor these Trainer metrics:

- `azr_ready`: becomes 1 after world-model warm-up;
- `azr_tasks_proposed`, `azr_tasks_accepted`, `azr_task_buffer_size`;
- `azr_mean_pass_rate`, `azr_mean_learnability`, `azr_mean_survival`;
- `loss_adversary`, `loss_imagination_actor`, `mean_imagined_return`;
- `imagined_policy_updates`: proves an accepted imagined update reached the deployed policy.

When enabling the feature against an old non-world-model checkpoint, the
checkpoint updater detects the mode change, rebuilds the agent once, and keeps
the existing replay memory. Leave `RESET_TRAINING: false` for normal restarts so
the learned world model and latent task buffer resume from their checkpoint.

## Standalone JAX prototype

The JAX prototype now:

- requires `next_obs` for transition training;
- filters current and next observations through the RSSM posterior;
- normalizes TrackMania telemetry;
- uses Bernoulli continuation loss, next-embedding consistency, and lower free-nat KL balancing;
- starts imagination from replay posterior states;
- computes continuous-action actor gradients through the exact imagined rollout;
- keeps latent proposals disabled unless a caller supplies matched, repeated binary solver outcomes from a grounded validator.

## AZR boundary and zero-shot evaluation

AZR's transferable pattern is **propose -> validate -> solve repeatedly -> update**. For each persistent task, the official method estimates a binary pass rate over repeated solver attempts and uses:

```text
learnability = 0                 if pass_rate is 0 or 1
learnability = 1 - pass_rate     otherwise
```

The production bridge uses replay-anchored **model tasks**. For full environment-level AZR, a task must instead exist in an executable, semantically valid space such as a track seed, bounded Bezier control points, spawn/checkpoint selection, or supported physics parameters. A screened latent task can improve the deployed policy, but it is not evidence that a new drivable map exists.

Remaining work for an environment-level zero-shot claim:

1. Add contiguous replay-sequence sampling with episode-boundary masks.
2. Validate open-loop world-model predictions on held-out replay sequences before policy use.
3. Add a structured map/scenario generator and simulator validity checks.
4. Reset workers into the same generated scenario for repeated real solver attempts.
5. Freeze weights and evaluate on tracks excluded from policy, replay, world-model, and curriculum training.

Report completion rate, lap time, return, crash/off-track rate, worst-decile performance, and multiple random seeds. Only this held-out protocol supports a zero-shot-transfer claim.

## Primary references

- [Absolute Zero Reasoner paper](https://arxiv.org/html/2505.03335) and [official code](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner)
- [DreamerV3](https://arxiv.org/html/2301.04104) and [official code](https://github.com/danijar/dreamerv3)
- [PAIRED](https://arxiv.org/html/2012.02096)
- [Replay-Guided Adversarial Environment Design](https://proceedings.neurips.cc/paper/2021/file/0e915db6326b6fb6a3c56546980a8c93-Paper.pdf)
- [MBPO](https://arxiv.org/html/1906.08253) and [MOPO](https://arxiv.org/html/2005.13239)
