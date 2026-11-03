# Continual Foundation Dreamer (JAX): hard implementation plan

Status: proposed architecture and execution contract  
Date: 2026-08-22  
Target: TrackMania 2020, TMRL, JAX/Flax NNX trainer, Windows rollout worker

## 1. Outcome

Build a driving system with two learning timescales:

1. A slow, general foundation learns reusable perception, vehicle dynamics,
   basic control, and planning across a diverse training-map distribution.
2. A fast, append-only skill bank learns genuinely novel situations without
   overwriting mastered skills.

The system must separately measure:

- **Zero-shot performance:** the first evaluation on an unseen map, before any
  gradient update or replay insertion from that map.
- **Online adaptation:** improvement after collecting experience on that map.
- **Continual retention:** performance on all earlier maps after adaptation.

The goal is not an infinitely wide JIT-compiled network. The goal is an
unbounded host-side registry of fixed-shape expert parameter PyTrees, while the
accelerator executes one stable compiled expert function and a fixed number of
active expert slots.

## 2. Current-state audit

The implementation must start from these facts rather than assumptions:

- The active `config.json` selects `ALGORITHM = DREAMER`.
- That CLI path constructs `TorchDreamerActor`, `TorchDreamerAgent`,
  `ArrayTorchMemoryTMFullSequence`, and `TorchTrainingOffline`.
- The JAX files currently provide an NNX SAC agent, a standalone latent world
  model, and a standalone imagination trainer. They are not wired into the
  deployed Dreamer CLI, sequence replay, checkpoint updater, or recurrent
  rollout actor.
- The JAX world-model regression tests pass, but this only proves component
  behavior. It does not prove a deployed JAX Dreamer worker/trainer loop.
- Native Windows currently reports JAX `0.11.1`, Flax `0.12.9`, Optax `0.2.8`,
  and `jax.default_backend() == "cpu"`.
- The machine has an RTX 5070 Laptop GPU with approximately 8 GB VRAM, and the
  current Torch installation sees it successfully.
- Official JAX CUDA wheels do not support native Windows. NVIDIA JAX training
  must run in Linux; WSL2 support is experimental but is the practical route
  on this machine.
- Current TrackMania replay is one bounded FIFO store. Once full, early maps
  are removed regardless of sampling geometry.
- Current actor broadcasts contain model state only. The receiving actor is
  constructed ahead of time and expects a compatible fixed structure.
- Poincare replay is an optional diversity sampler. It is not a continual
  memory guarantee and is not part of the core architecture below.

These findings create two mandatory prerequisites:

1. Establish a GPU-backed JAX trainer environment under WSL2/Linux.
2. Integrate a real JAX Dreamer agent and recurrent actor into the TMRL
   worker/trainer/checkpoint lifecycle before adding continual capacity.

## 3. Non-negotiable invariants

The implementation is accepted only if all of these remain true:

1. **No hidden training before zero-shot evaluation.** A held-out map cannot
   enter replay, expert initialization, normalization statistics, router
   prototypes, or world-model updates before its zero-shot score is recorded.
2. **Frozen means frozen.** During fast adaptation, foundation and old-expert
   parameter leaves must remain byte-identical.
3. **No FIFO erasure of mastered maps.** Each admitted map or skill receives a
   protected archive quota independent of recent replay.
4. **Expansion occurs only at episode boundaries.** The worker never changes
   recurrent architecture or active expert state in the middle of a run.
5. **The compiled expert signature is constant.** Adding an expert must not
   change expert parameter shapes, input shapes, output shapes, or top-K slot
   count.
6. **Old behavior is evaluated, not inferred.** Every accepted update is
   followed by evaluation on all previously mastered validation maps.
7. **Consolidation is transactional.** Foundation consolidation starts from an
   atomic snapshot and is rolled back if any retention gate fails.
8. **Expansion is evidence-driven.** A new map alone is not sufficient reason
   to allocate an expert. Existing experts are probed first.
9. **Capacity is budgeted.** Experts can grow, but storage, active VRAM,
   inference latency, and router search are measured and bounded.
10. **No zero-shot claim from one map.** Foundation training must use diverse
    maps, surfaces, corner geometries, speeds, and failure/recovery states.

## 4. Target architecture

```text
Windows TrackMania worker                    WSL2/Linux JAX trainer
-------------------------                    ----------------------
pixels + telemetry                           balanced sequence batches
        |                                             |
        v                                             v
frozen/slow foundation encoder <---------- foundation world model/RSSM
        |                                             |
        v                                             v
foundation latent f_t ---------------------> imagination + AZR (later)
        |
        v
episode context / novelty router
        |
        +---- active slot 0: fixed-shape expert parameters
        +---- active slot 1: fixed-shape expert parameters
        |
        v
residual skill composition
        |
        v
foundation actor head -> gas / brake / steer
```

The trainer owns the full expert registry. The rollout worker receives the
foundation actor plus a fixed-size cache of the experts relevant to the current
map. The number of stored experts can grow without increasing the JIT-visible
number of active slots.

### 4.1 Parameter ownership

| State | Update rate | Owner | Retention rule |
| --- | --- | --- | --- |
| Foundation encoder | Slow | Trainer | Frozen during adaptation |
| Foundation RSSM and prediction heads | Slow | Trainer | Frozen during adaptation |
| Foundation actor and critic | Slow | Trainer | Frozen during adaptation |
| New expert parameters | Fast | Trainer | Only active new expert is trainable |
| Mastered expert parameters | None | Trainer/archive | Immutable |
| Expert optimizer state | Fast | Trainer | One state per trainable expert |
| Router prototype per expert | Append-only | Trainer and worker cache | Old prototypes immutable initially |
| Recent replay | Fast turnover | Trainer | FIFO allowed |
| Protected episodic archive | Slow | Trainer/disk | Quota per map/skill |
| Consolidation teacher | Temporary frozen snapshot | Trainer | Destroy only after acceptance |

### 4.2 Fixed-shape expert template

The first expert implementation is a residual latent feature adapter:

```text
input feature: h_t (256) concatenated with z_t (128) = 384

LayerNorm(384)
Linear(384, 64)
SiLU
Linear(64, 384)
learned residual scale initialized near zero

output: f_t + scale * adapter(f_t)
```

This is approximately 50,000 parameters per expert. It can adapt actor and
critic features without resizing the foundation or changing the worker's
feature width.

After this path is proven, a second optional expert component may add a
fixed-shape residual to the RSSM deterministic transition. It must not be added
until world-model surprise shows that feature-only adaptation is insufficient.

For fixed `TOP_K = 2`, composition is:

```text
f_adapted = f + w_0 * A(phi_0, f) + w_1 * A(phi_1, f)
```

Missing slots contain a zero adapter and zero routing weight. The shapes never
depend on total registry size.

### 4.3 Expert registry

The global registry is host-side state, not one variable-width JAX array:

```text
ExpertRecord
  expert_id
  schema_version
  parent_expert_id
  creation_reason
  creation_update
  parameter_file
  optimizer_file (only while trainable)
  prototype_mean
  prototype_variance
  maps_seen
  skills_seen
  mastery_metrics
  status: candidate | trainable | mastered | merged | retired
```

Every expert uses the same parameter PyTree structure. The same compiled
`apply_expert(params, latent)` function is reused for every expert value.

### 4.4 Router

The initial router is deliberately simple and stable:

1. Build an episode context from a rolling mean and variance of foundation
   latent features during the first configurable observation window.
2. Compare the context against independent diagonal-Gaussian prototypes stored
   with each expert.
3. Select the closest `TOP_K` experts plus routing weights.
4. Keep routing fixed for the episode in the first implementation.

This avoids a shared classifier whose output dimension grows and whose old
decision boundaries can be forgotten. A learned router is permitted only after
the prototype router establishes a measurable baseline.

The first controlled milestone may use an explicit map ID. Automatic task-free
routing is a later acceptance gate, not something to debug simultaneously with
expert expansion.

## 5. What the foundation must learn

A one-map master is not a foundation. The foundation curriculum must contain
enough variation to identify reusable causal skills:

- steering response and counter-steering;
- acceleration, braking distance, and speed control;
- curvature estimation and racing-line geometry;
- entry, apex, and exit behavior;
- recovery after oversteer, wall contact, and off-line states;
- jumps, crests, elevation changes, and landing stabilization;
- surface-dependent behavior when applicable;
- visual variation, lighting, camera motion, and partial occlusion;
- short, long, open, and technical track layouts.

The map split is immutable once an experiment begins:

- **Foundation-train maps:** available to foundation optimization.
- **Foundation-validation maps:** used for early stopping and tuning.
- **Continual-stream maps:** presented sequentially for adaptation.
- **Held-out zero-shot maps:** never observed until their scored first attempt.

Generated or hand-built segments may expand foundation-train diversity, but no
held-out geometry can be reused through ghosts, replay, reward files, or router
statistics.

## 6. Memory system

Continual memory has two stores with different jobs.

### 6.1 Recent replay

- Bounded FIFO.
- Optimized for the current map and fast adaptation.
- Stores contiguous sequence metadata.
- May use uniform, Euclidean, or Poincare candidate sampling as an ablation.

### 6.2 Protected episodic archive

- CPU/disk-backed and checkpointed independently from recent replay.
- Fixed quota per admitted map initially; later quota per discovered skill.
- Uses reservoir sampling inside each quota rather than global FIFO trimming.
- Preserves a balanced mixture of successes, recoveries, failures, rare
  transitions, high world-model-surprise sequences, and representative normal
  driving.
- Stores `map_id`, `episode_id`, `sequence_start`, `expert_id`, outcome,
  progress, finish flag, and optional skill label alongside raw replay data.
- Never samples a recurrent sequence across an episode boundary.

Initial batch composition:

```text
75% current/recent sequences
25% uniformly balanced sequences from previously admitted archive partitions
```

This ratio is a starting controlled baseline, not a permanent truth. It is
tuned only from foundation-validation and continual-stream results.

## 7. Learning lifecycle

### 7.1 Foundation phase

1. Disable expansion.
2. Train encoder, RSSM, actor, critic, and world-model heads across the
   foundation-train distribution.
3. Evaluate on foundation-validation maps.
4. Record first-attempt held-out baselines only when the foundation checkpoint
   is frozen for an evaluation campaign.
5. Publish `foundation_version` and an architecture hash.

### 7.2 First encounter with a continual-stream map

1. Snapshot all random seeds, foundation version, registry version, and replay
   manifest.
2. Run the first evaluation without learning or replay insertion.
3. Record zero-shot progress, finish, return, lap time, interventions, and
   uncertainty.
4. Build context only after the zero-shot result is committed.
5. Probe the base policy and nearest existing experts under a fixed episode
   budget.

### 7.3 Expansion decision

A candidate expert is created only when all configurable conditions hold:

- context novelty exceeds the calibrated known-context threshold;
- world-model surprise or policy uncertainty stays high for several episodes;
- progress or completion remains below the adaptation target;
- no existing expert passes its probe criterion;
- expansion cooldown and parameter budgets permit growth.

The candidate is initialized from the closest expert when transfer is useful,
or from a near-zero residual adapter when no parent is suitable.

### 7.4 Fast adaptation

1. Freeze the full foundation and all mastered experts.
2. Train only the candidate expert, its optimizer state, and its independent
   prototype statistics.
3. Mix recent and protected archive sequences.
4. Evaluate all old maps at configured intervals.
5. Reject and restore the candidate if it damages old routing or misses its new
   map learning target.
6. Mark it mastered only after all acceptance gates pass.

Because old parameters and old prototypes are immutable, structural isolation
provides the primary retention guarantee. Replay remains necessary for router
validation, evaluation, later consolidation, and any future shared updates.

### 7.5 Consolidation

Consolidation is optional and much slower than adaptation:

1. Atomically snapshot the foundation, registry, and evaluation matrix.
2. Build a balanced dataset across all admitted maps/skills.
3. Use the pre-consolidation foundation plus experts as frozen teachers.
4. Train the foundation with Dreamer losses plus policy, value, latent, reward,
   and continuation distillation losses.
5. Re-evaluate every previous map and all zero-shot validation maps.
6. Accept only if retention and forward-transfer gates pass.
7. Otherwise restore the exact snapshot.

Experts proven redundant after successful consolidation may be marked merged;
their artifacts remain recoverable until a later explicit garbage-collection
policy is validated.

## 8. JAX execution design

### 8.1 Stable compiled boundary

JIT-compiled functions operate on:

- foundation parameters of fixed structure;
- exactly `TOP_K` expert parameter PyTrees of identical fixed structure;
- fixed-shape observations, recurrent state, actions, and routing weights.

Registry search, expert allocation, checkpoint I/O, map admission, and expansion
decisions remain outside JIT.

Adding an expert means creating another value with the established expert
PyTree schema. It does not mean appending a variable-length dimension to a
jitted tensor.

### 8.2 Optimizers

- Foundation optimizer state is separate and inactive during adaptation.
- Every trainable expert has its own Optax state with the same tree structure.
- Mastered experts discard or archive optimizer state to reduce storage.
- Consolidation uses a newly created foundation optimizer state or an
  explicitly versioned resumed state; the choice is recorded in the manifest.

### 8.3 Compilation tests

Tests must demonstrate:

- expert 0 and expert 100 call the same compiled signature;
- allocating a new registry record does not compile the expert apply function;
- changing batch size, sequence length, top-K, or expert schema is detected as
  a deliberate compilation/configuration event;
- no expert count appears as a JAX array shape in training or inference.

## 9. Windows/WSL2 deployment

The required split is:

```text
Native Windows
  TrackMania 2020
  capture and gamepad worker
  reward/ghost integration
  TMRL server (initially)

WSL2/Linux
  JAX CUDA trainer
  recent replay and protected archive
  checkpoints and expert registry
  evaluation database
```

Phase 0 must prove:

- WSL2 sees the RTX GPU through JAX;
- `jax.default_backend()` reports GPU;
- the trainer can connect to the Windows TMRL server;
- worker-to-trainer buffer transfer and trainer-to-worker actor broadcasts work
  across the boundary;
- checkpoint and archive I/O live on WSL2 ext4 rather than `/mnt/c` or `/mnt/d`
  when sustained I/O performance matters;
- the Windows worker stays below the 50 ms environment step deadline.

If GPU JAX under WSL2 is not reliable, the fallback is the already deployed
Torch Dreamer on native Windows. CPU JAX is not an acceptable training backend
for the full image-based continual Dreamer.

## 10. Checkpoint and broadcast contracts

### 10.1 Durable checkpoint layout

```text
run_checkpoint/
  manifest.json
  foundation/
    graph.json
    state.msgpack
    optimizer.msgpack
  experts/
    expert_000001/
      metadata.json
      state.msgpack
      optimizer.msgpack        # only while trainable
  router/
    prototypes.msgpack
  replay/
    recent_manifest.json
    archive_manifest.json
    partitions/...
  evaluation/
    matrix.jsonl
    zero_shot_events.jsonl
```

`manifest.json` contains at least:

```json
{
  "schema_version": 1,
  "algorithm": "CONTINUAL_DREAMER_JAX",
  "foundation_version": "foundation_0001",
  "foundation_architecture_hash": "...",
  "expert_schema_hash": "...",
  "top_k": 2,
  "registry_version": 0,
  "active_candidate": null,
  "last_atomic_evaluation": "..."
}
```

Writes use temporary files plus atomic replacement. A partially written expert
must never become visible in the active manifest.

### 10.2 Worker actor bundle

The worker receives a fixed-structure bundle:

```text
schema version
foundation architecture hash
foundation recurrent actor state
exactly TOP_K expert slot states
slot expert IDs
router prototypes for the active cache
normalization state
bundle checksum
```

The worker validates the schema and checksum before swapping actors. If the
bundle is incompatible, it keeps the last valid actor and reports the rejection.

The existing state-only `NNXActorModule.save/load` implementation must be
overridden for this actor bundle. Core networking can continue transporting
opaque bytes unless chunking or size limits fail in measurement.

## 11. Configuration contract

Do not silently overload the current Torch `DREAMER` mode. Introduce a separate
algorithm name so checkpoints and workers cannot be confused:

```json
{
  "ALG": {
    "ALGORITHM": "CONTINUAL_DREAMER_JAX",
    "CONTINUAL_DREAMER": {
      "FOUNDATION": {
        "LATENT_DIM": 128,
        "HIDDEN_DIM": 256,
        "FREEZE_DURING_ADAPTATION": true
      },
      "EXPERTS": {
        "ENABLED": false,
        "BOTTLENECK_DIM": 64,
        "TOP_K": 2,
        "MAX_ACTIVE_SLOTS": 2,
        "EXPAND_ONLY_AT_EPISODE_BOUNDARY": true
      },
      "ROUTER": {
        "MODE": "explicit_map_id",
        "CONTEXT_STEPS": 40,
        "NOVELTY_QUANTILE": 0.99
      },
      "ARCHIVE": {
        "ENABLED": false,
        "CURRENT_BATCH_FRACTION": 0.75,
        "PER_MAP_SEQUENCE_QUOTA": 50000
      },
      "CONSOLIDATION": {
        "ENABLED": false
      }
    }
  }
}
```

Features are enabled milestone by milestone. Defaults remain conservative and
old checkpoints never opt in automatically.

## 12. Implementation milestones and hard exit gates

### M0 - Environment and baseline freeze

Deliverables:

- reproducible WSL2 JAX CUDA environment specification;
- backend/device/versions preflight command;
- frozen current Torch Dreamer checkpoint and metrics;
- fixed foundation/continual/held-out map manifest;
- run naming and artifact directory convention.

Exit gates:

- JAX reports an NVIDIA GPU in WSL2;
- a small NNX train step runs on GPU without fallback;
- cross-OS buffer and actor round trip passes;
- existing 30-test repository baseline remains green;
- no active production checkpoint is overwritten.

### M1 - Deployed JAX Dreamer parity

Deliverables:

- JAX recurrent Dreamer actor implementing `NNXActorModule`;
- sequence-based JAX Dreamer training agent implementing `TrainingAgent`;
- JAX-compatible episode-safe image sequence batch path;
- CLI/config selection under `CONTINUAL_DREAMER_JAX`;
- checkpoint resume and worker broadcast integration;
- AZR disabled during parity work.

Exit gates:

- real replay updates encoder, RSSM, actor, and critic;
- the exact actor optimized in imagination is broadcast and drives the worker;
- recurrent state resets at episode boundaries;
- checkpoint/resume reproduces deterministic test actions within tolerance;
- warmed p95 worker policy inference is below 2 ms on the deployed worker
  machine, leaving headroom inside the 50 ms environment step;
- no synthetic transition is used as evidence of real dynamics learning.

### M2 - Foundation curriculum and evaluation harness

Deliverables:

- immutable map split manifest;
- multi-map data collection and training scheduler;
- zero-shot event recorder that prevents pre-evaluation leakage;
- evaluation matrix and TrackMania-specific metrics;
- foundation versioning and freeze operation.

Exit gates:

- foundation trains on multiple diverse maps;
- every held-out zero-shot score is recorded before adaptation;
- three repeated seeds establish confidence intervals;
- results distinguish return, progress, finish rate, and lap time;
- the checkpoint is designated foundation only after validation.

### M3 - Protected replay and functional retention baseline

Deliverables:

- recent replay plus protected per-map archive;
- reservoir admission and balanced sequence sampler;
- explicit map metadata and archive migration tools;
- optional policy/value/world-model distillation baseline;
- sequence and quota diagnostics.

Exit gates:

- filling recent replay cannot remove protected old-map sequences;
- sampled sequences never cross episodes or map partitions;
- after sequential training on at least three maps, mean old-map progress drops
  no more than 10% relative and finish rate drops no more than 10 percentage
  points, or the milestone is explicitly recorded as insufficient;
- archive size and sampling ratios match configured values after checkpoint
  resume.

### M4 - One manually routed fixed-shape expert

Deliverables:

- expert parameter schema and pure apply function;
- one feature adapter and independent Optax state;
- manual map-to-expert routing;
- frozen parameter audit;
- worker bundle with fixed active slots.

Exit gates:

- foundation and old expert leaves are byte-identical before and after
  adaptation;
- the new expert improves its target map over frozen-foundation performance;
- all old-map metrics stay within the M3 retention gate;
- expert 0 and expert 1 reuse the same JIT compilation signature;
- adding the registry record does not alter worker inference tensor shapes.

### M5 - Registry, checkpointing, and automatic expansion

Deliverables:

- durable expert registry and atomic artifact layout;
- candidate/mastered/merged lifecycle;
- novelty and probe-based expansion controller;
- expansion cooldown, budgets, and diagnostics;
- resume from checkpoints containing multiple experts.

Exit gates:

- known-map routing accuracy is at least 95% on validation episodes;
- false expert expansion on known maps is below 5%;
- held-out contexts are detected with AUROC at least 0.90 before thresholds are
  frozen;
- an interrupted expert write restores the previous valid manifest;
- registry growth does not increase JIT-visible active slot count.

### M6 - Task-free routing and reusable skills

Deliverables:

- context router without required map IDs;
- skill-level rather than map-only expert metadata;
- fixed-top-K skill composition;
- expert reuse and transfer report.

Exit gates:

- routing meets M5 gates without map identity;
- at least one expert improves adaptation on more than one map;
- expert-enabled zero-shot performance exceeds the foundation-only baseline on
  held-out maps across repeated seeds;
- warmed worker p95 inference remains below 2 ms.

### M7 - Transactional consolidation

Deliverables:

- frozen teacher snapshot;
- balanced consolidation dataset;
- distillation losses and rollback controller;
- redundancy/merge analysis;
- before/after evaluation report.

Exit gates:

- no mastered map loses more than 5% relative progress or five percentage
  points finish rate;
- foundation-only held-out zero-shot performance does not regress outside its
  confidence interval;
- at least one redundant expert can be disabled without violating retention;
- failed consolidation restores bit-identical pre-consolidation artifacts.

### M8 - AZR and imagination integration

Deliverables:

- expert-aware imagined rollout starts;
- task proposals associated with foundation/expert coverage;
- support and survival filters per expert;
- real-versus-imagined validation metrics;
- safeguards against allocating experts from unsupported hallucinations.

Exit gates:

- AZR tasks are accepted only after repeated solver attempts;
- imagined improvements transfer to real held-out sequences;
- expansion cannot be triggered from imagination alone;
- M6 and M7 retention gates remain satisfied.

## 13. Evaluation matrix

After learning stream item `i`, evaluate every map `j` and store:

```text
R[i, j] = normalized progress or task return on map j after learning item i
```

Required aggregate metrics:

- Average performance: mean of the final row of `R`.
- Forgetting on map `j`: `max_i R[i, j] - R[final, j]`.
- Backward transfer: final old-map score minus score when each map was learned.
- Forward transfer: new-map pre-update score minus a scratch-agent baseline.
- Adaptation efficiency: environment steps to reach the mastery threshold.
- Zero-shot finish rate, progress, return, and lap time.
- Expert reuse rate, registry parameter count, disk footprint, and active VRAM.
- Router accuracy, novelty AUROC, false expansion rate, and routing latency.
- World-model error split by current map, archived maps, and held-out maps.
- Worker p50/p95/p99 inference latency and time-step timeout rate.

Reward alone is insufficient because reward scales and ghosts may differ by
map. Progress, completion, and lap time are first-class metrics.

## 14. Required tests

### Unit tests

- expert PyTree schema equality across IDs;
- zero adapter is an identity function;
- only the active expert receives nonzero gradients;
- foundation and mastered expert leaves remain unchanged;
- router prototype update affects only the candidate expert;
- reservoir quotas and partition balance;
- episode-safe and map-safe sequence sampling;
- expansion hysteresis and cooldown;
- manifest hashing, validation, and atomic replacement;
- fixed top-K padding and deterministic routing;
- zero-shot event rejects contaminated maps.

### Integration tests

- JAX Dreamer trains from real contiguous sequences;
- trainer actor and worker actor match after broadcast;
- expert allocation, training, mastering, save, reload, and routing round trip;
- worker rejects a corrupt or incompatible actor bundle and keeps the last good
  actor;
- checkpoint resume preserves registry, optimizer, router, replay, and
  evaluation states;
- compilation counter does not increase solely because a same-schema expert ID
  changes;
- current Torch checkpoint and raw replay remain recoverable.

### End-to-end gates before TrackMania

Use small synthetic or Gymnasium continual tasks to prove:

- conflicting task B can be learned without changing frozen task-A behavior;
- a third task creates or reuses an expert according to configured novelty;
- checkpoint interruption recovery works;
- automatic evaluation computes the expected forgetting matrix.

These tests validate infrastructure only. TrackMania experiments remain the
authoritative driving evidence.

## 15. File-level implementation map

Planned new files:

```text
tmrl/custom/jax/continual_dreamer.py
tmrl/custom/jax/continual_models.py
tmrl/custom/jax/expert_registry.py
tmrl/custom/jax/expert_router.py
tmrl/custom/jax/continual_replay.py
tmrl/custom/jax/continual_losses.py
tmrl/custom/jax/continual_checkpoints.py
tmrl/custom/jax/continual_evaluation.py
tmrl/tools/continual_preflight.py
tests/test_jax_dreamer_integration.py
tests/test_continual_replay.py
tests/test_expert_registry.py
tests/test_expert_routing.py
tests/test_continual_checkpoint.py
tests/test_continual_evaluation.py
```

Planned changes to existing files:

- `tmrl/config/config_objects.py`: add explicit
  `CONTINUAL_DREAMER_JAX` construction without changing Torch `DREAMER`.
- `tmrl/core/jax/actor.py`: keep generic state behavior; the continual actor
  overrides save/load with its versioned bundle.
- `tmrl/core/jax/training_offline.py`: expose episode-boundary and evaluation
  hooks without embedding continual policy into the generic trainer.
- `tmrl/core/networking.py`: remain opaque-byte transport unless measured bundle
  size or version negotiation requires a generic envelope.
- `tmrl/custom/jax/world_model.py`: reuse encoder/RSSM pieces after sequence and
  deployment parity are established.
- `tmrl/custom/jax/imagination_trainer.py`: refactor reusable losses only after
  the deployed agent exists; do not treat the standalone trainer as the agent.
- TrackMania interface/buffer metadata: add an explicit experiment map key for
  controlled milestones, then add automatic context routing later.

## 16. Risk register and rollback

| Risk | Early signal | Mitigation | Rollback |
| --- | --- | --- | --- |
| JAX remains CPU-only | Backend reports CPU | Move trainer to WSL2/Linux CUDA | Continue Torch baseline |
| JAX recompiles per expert | Compile counter rises on ID swap | Same-schema single-expert apply, fixed top-K slots | Disable dynamic routing |
| Foundation is not general | Poor held-out first attempts | Increase map/skill diversity before expansion | Revert foundation designation |
| Archive is dominated by easy data | Low rare/failure coverage | Stratified reservoir and surprise quota | Rebuild from source replay |
| Router forgets old tasks | Known routing accuracy falls | Immutable independent prototypes | Explicit map-ID fallback |
| Expert explosion | High new-expert rate | Probe existing experts, cooldown, budget, merge | Disable automatic expansion |
| No transfer between map experts | Reuse rate near zero | Move labels from maps to reusable skills | Keep map experts as retention baseline |
| World model forgets | Archived prediction loss rises | Freeze during adaptation; balanced consolidation | Restore snapshot |
| Worker misses real-time deadline | warmed p95 policy inference above 2 ms | Fixed active cache, smaller adapters, episode routing | Foundation-only actor |
| Broadcast incompatibility | Bundle rejected | Schema/hash/checksum and atomic swap | Keep last good worker actor |
| Consolidation damages skills | Evaluation matrix regresses | Full teacher distillation and hard gates | Atomic restore |
| AZR trains hallucinations | Imagined gain lacks real transfer | Replay anchors and real support filters | Disable AZR expansion influence |

## 17. First implementation slice

Do not start by coding expert growth. The first vertical slice is M0 plus the
smallest part of M1:

1. Create the WSL2 JAX CUDA environment and preflight.
2. Prove cross-OS trainer/server/worker transport with a small NNX actor.
3. Implement a recurrent JAX Dreamer actor whose optimized latent policy is the
   worker policy.
4. Train it on real episode-safe replay sequences.
5. Save, resume, broadcast, and compare trainer/worker actions.
6. Keep experts, archive changes, consolidation, AZR, and Poincare sampling
   disabled during this parity slice.

Only after that slice passes should M2 establish the foundation benchmark. Only
after the foundation benchmark exists should continual memory and dynamic
experts be enabled.

## 18. Definition of done

The continual system is complete only when all of the following are evidenced
in stored artifacts and repeatable commands:

- a deployed GPU JAX Dreamer controls TrackMania through TMRL;
- a multi-map foundation has immutable train/validation/continual/held-out
  splits;
- zero-shot scores are recorded before learning on every held-out map;
- sequential adaptation retains old maps within the hard gates;
- new same-schema experts can be allocated without changing JIT-visible shapes;
- at least one expert is reused across maps and improves held-out zero-shot
  performance;
- router, registry, replay, optimizer, and evaluation state survive restart;
- the worker safely receives fixed-structure expert bundles in real time;
- failed expansion or consolidation restores the prior working system;
- AZR, if enabled, passes real-transfer and retention gates;
- all unit, integration, interruption, and TrackMania evaluation gates pass
  across the declared seeds.

Anything less is an experiment or milestone, not a completed continual-learning
driver.

## 19. Primary references

- Progressive Neural Networks: <https://arxiv.org/abs/1606.04671>
- Dynamically Expandable Networks: <https://openreview.net/forum?id=Sk7KsfW0->
- CLEAR experience replay for continual RL:
  <https://papers.nips.cc/paper/8327-experience-replay-for-continual-learning>
- Avalanche continual-learning framework: <https://avalanche.continualai.org/>
- JAX JIT compilation constraints:
  <https://docs.jax.dev/en/latest/jit-compilation.html>
- JAX installation and supported platforms:
  <https://docs.jax.dev/en/latest/installation.html>
