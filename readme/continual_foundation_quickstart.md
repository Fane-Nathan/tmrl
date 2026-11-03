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
cd /mnt/d/Project/tmrl
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
  scripts/smoke_continual_jax_gpu.py \
  --output /home/$USER/tmrl-artifacts/continual-foundation/preflight/m1_gpu_smoke.json

/home/$USER/.venvs/tmrl-jax-cuda13/bin/python \
  -m tmrl.tools.validate_continual_trainer_stack \
  --output-directory \
  /home/$USER/tmrl-artifacts/continual-foundation/preflight/trainer-stack
```

The first command must report `backend: gpu`, `world_model_changed: true`, and
`deployed_policy_changed: true`. The second command instantiates the exact
configured trainer and round-trips its NNX/Optax checkpoint and worker bundle.

## 3. Create an isolated M1 config

Do not edit `config.json` while the current Torch trainer or worker is running.
Generate a separate file first:

```powershell
cd D:\Project\tmrl
.\.venv\Scripts\python.exe -m tmrl.tools.prepare_continual_jax_config `
  --source C:\Users\felix\TmrlData\config\config.json `
  --output C:\Users\felix\TmrlData\config\config.continual-jax-m1-stable-v2.json `
  --run-name Continual_Dreamer_JAX_M1_Stable_v2 `
  --scrub-wandb-key
```

Prepare the WSL trainer config. The helper detects the current WSL2 NAT gateway,
points only the trainer at the Windows relay, and verifies that port `55555` is
reachable:

```bash
cd /mnt/d/Project/tmrl
bash scripts/configure_continual_jax_wsl_network.sh
```

Run this helper again if WSL's gateway changes. It keeps
`LOCALHOST_WORKER=true` for the native-Windows worker but sets
`LOCALHOST_TRAINER=false` for the WSL trainer.

On this machine the WSL config has already been prepared, and its preceding
version is recoverable as
`/home/felix/TmrlData/config/config.pre-continual.bak.json`. The active native
Windows `config.json` has not been changed.

## 4. Freeze the current Torch baseline

Stop the Torch trainer first. The freeze command rejects a source whose size or
modification time changes during copying:

```powershell
.\.venv\Scripts\python.exe -m tmrl.tools.freeze_continual_baseline `
  --source C:\Users\felix\TmrlData\checkpoints\AbsoluteZero_DREAMER_t.tcpt `
  --source C:\Users\felix\TmrlData\weights\AbsoluteZero_DREAMER_t.tmod `
  --output-dir C:\Users\felix\TmrlData\continual\baselines\torch-dreamer-m0
```

The destination must be empty. Every artifact is SHA-256 checksummed and the
live checkpoint is never overwritten.

## 5. Switch and run

After the old trainer and worker have stopped, back up and switch only the
native-Windows config:

```powershell
Copy-Item -LiteralPath C:\Users\felix\TmrlData\config\config.json `
  -Destination C:\Users\felix\TmrlData\config\config.pre-continual-windows.json
Copy-Item -LiteralPath C:\Users\felix\TmrlData\config\config.continual-jax-m1-stable-v2.json `
  -Destination C:\Users\felix\TmrlData\config\config.json
```

Keep the TMRL server on Windows. Run the trainer in WSL2:

```bash
cd /mnt/d/Project/tmrl
bash scripts/configure_continual_jax_wsl_network.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false
/home/$USER/.venvs/tmrl-jax-cuda13/bin/python -m tmrl --trainer
```

Run the TrackMania worker in native Windows:

```powershell
cd D:\Project\tmrl
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

Copy `experiments/continual_foundation/map_manifest.example.json`, replace all
placeholder IDs and hashes, set `frozen` to `true`, then validate it:

```powershell
.\.venv\Scripts\python.exe -m tmrl.tools.validate_continual_map_manifest `
  experiments\continual_foundation\map_manifest.v1.json `
  --require-frozen
```

All four splits must be present. A held-out map cannot contribute replay,
ghosts, reward geometry, normalization statistics, router prototypes, or model
updates before its zero-shot event is committed.

The next milestone is M2: provide the actual multi-map list and reward/ghost
artifacts, freeze the split, and collect foundation/validation/zero-shot
evidence. Expert expansion remains intentionally blocked until then.
