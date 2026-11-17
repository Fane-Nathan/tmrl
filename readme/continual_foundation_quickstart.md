# Continual Foundation Dreamer: M0/M1 quickstart

This repository now contains the first deployable vertical slice:

- JAX/Flax NNX Dreamer training on the WSL2 NVIDIA GPU;
- real, episode-safe TrackMania replay sequences;
- latent imagination updates to the exact policy that is broadcast;
- a portable, checksummed Torch mirror for the native-Windows worker;
- recurrent episode reset, atomic checkpoint/broadcast writes, and corrupt
  bundle rejection;
- a separate `CONTINUAL_DREAMER_JAX` algorithm and run name.

It is not yet the complete continual-learning system. Dynamic experts,
protected replay, consolidation, and AZR remain disabled until a multi-map
foundation passes the M2 evaluation gates.

## 1. Bootstrap the WSL2 trainer

From Ubuntu/WSL2:

```bash
cd <repo-root>
bash scripts/bootstrap_continual_jax_wsl2.sh
```

This creates `/home/$USER/.venvs/tmrl-jax-cuda13`, installs JAX CUDA plus a
CPU-only Torch replay dependency, and writes the GPU preflight result under
`/home/$USER/tmrl-artifacts/continual-foundation/preflight/`.

Do not install a Linux NVIDIA display driver inside WSL2. The Windows NVIDIA
driver supplies the WSL GPU interface.

## 2. Validate GPU learning and the production stack

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false

/home/$USER/.venvs/tmrl-jax-cuda13/bin/python \
  -m tmrl.tools.continual_preflight \
  --output /home/$USER/tmrl-artifacts/continual-foundation/preflight/m1_gpu_smoke.json

/home/$USER/.venvs/tmrl-jax-cuda13/bin/python \
  -m tmrl.tools.validate_continual_trainer_stack \
  --output-directory \
  /home/$USER/tmrl-artifacts/continual-foundation/preflight/trainer-stack
```

The first command verifies the JAX GPU environment and preflight checks. The second command instantiates the exact
configured trainer and round-trips its NNX/Optax checkpoint and worker bundle.

## 3. Create an isolated M1 config

Do not edit `config.json` while the current Torch trainer or worker is running.
Generate a separate file first:

```powershell
cd <repo-root>
.\.venv\Scripts\python.exe -m tmrl.tools.prepare_continual_jax_config `
  --source $HOME\TmrlData\config\config.json `
  --output $HOME\TmrlData\config\config.continual-jax-m1-stable-v2.json `
  --run-name Continual_Dreamer_JAX_M1_Stable_v2 `
  --scrub-wandb-key
```

Prepare the WSL trainer config. The helper detects the current WSL2 NAT gateway,
points only the trainer at the Windows relay, and verifies that port `55555` is
reachable:

```bash
cd <repo-root>
bash scripts/configure_continual_jax_wsl_network.sh
```

Run this helper again if WSL's gateway changes. It keeps
`LOCALHOST_WORKER=true` for the native-Windows worker but sets
`LOCALHOST_TRAINER=false` for the WSL trainer.

The preceding WSL config is backed up as
`$HOME/TmrlData/config/config.pre-continual.bak.json`. The active native
Windows `config.json` is not changed until you switch it explicitly.

## 4. Freeze the current Torch baseline

Stop the Torch trainer first. The freeze command rejects a source whose size or
modification time changes during copying:

```powershell
.\.venv\Scripts\python.exe -m tmrl.tools.freeze_continual_baseline `
  --source $HOME\TmrlData\checkpoints\AbsoluteZero_DREAMER_t.tcpt `
  --source $HOME\TmrlData\weights\AbsoluteZero_DREAMER_t.tmod `
  --output-dir $HOME\TmrlData\continual\baselines\torch-dreamer-m0
```

The destination must be empty. Every artifact is SHA-256 checksummed and the
live checkpoint is never overwritten.

## 5. Switch and run

After the old trainer and worker have stopped, back up and switch only the
native-Windows config:

```powershell
Copy-Item -LiteralPath $HOME\TmrlData\config\config.json `
  -Destination $HOME\TmrlData\config\config.pre-continual-windows.json
Copy-Item -LiteralPath $HOME\TmrlData\config\config.continual-jax-m1-stable-v2.json `
  -Destination $HOME\TmrlData\config\config.json
```

Keep the TMRL server on Windows. Run the trainer in WSL2:

```bash
cd <repo-root>
bash scripts/configure_continual_jax_wsl_network.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false
/home/$USER/.venvs/tmrl-jax-cuda13/bin/python -m tmrl --trainer
```

Run the TrackMania worker in native Windows:

```powershell
cd <repo-root>
.\.venv\Scripts\python.exe -m tmrl --worker
```

So yes: it is still the server/trainer/worker workflow, but the trainer is now
the WSL2 JAX process and the worker is the native-Windows Torch mirror. Use a
new run name; never point M1 at `AbsoluteZero_DREAMER` or the poisoned
`Continual_Dreamer_JAX_M1` checkpoint.

Expected behavior:

- the worker uses CPU inference with real-time affinity/priority tuning;
- the first JAX update includes compilation, then warmed updates are fast;
- `dreamer_ready` remains `0` during the configured 2,000 real-replay warm-up;
- `imagined_policy_updates` becomes `1` only after warm-up;
- a worker rejects corrupt, non-finite, or old Torch bundles and retains its
  last valid actor until a version-4 JAX bundle arrives;
- `world_gradients_finite`, `actor_gradients_finite`, and
  `critic_gradients_finite` remain `1`; a non-finite update stops training
  before the checkpoint or actor bundle can be overwritten.

## 6. Verify worker latency

```powershell
.\.venv\Scripts\python.exe -m tmrl.tools.benchmark_continual_actor `
  --runtime torch --device cpu --iterations 1000 `
  --realtime-cpu-tuning --max-p95-ms 2

.\.venv\Scripts\python.exe -m tmrl.tools.benchmark_continual_actor `
  --runtime torch --device cpu --iterations 1000 --stochastic `
  --realtime-cpu-tuning --max-p95-ms 2
```

Both deterministic evaluation and exploratory collection must pass the 2 ms
p95 gate on the deployed worker machine.

## 7. Lock the map split before claiming a foundation

Prepare a map manifest JSON file, replace all
placeholder IDs and hashes, set `frozen` to `true`, then validate it:

```powershell
.\.venv\Scripts\python.exe -m tmrl.tools.validate_continual_map_manifest `
  <path-to-map-manifest-json> `
  --require-frozen
```

All four splits must be present. A held-out map cannot contribute replay,
ghosts, reward geometry, normalization statistics, router prototypes, or model
updates before its zero-shot event is committed.

The next milestone is M2: provide the actual multi-map list and reward/ghost
artifacts, freeze the split, and collect foundation/validation/zero-shot
evidence. Expert expansion remains intentionally blocked until then.
