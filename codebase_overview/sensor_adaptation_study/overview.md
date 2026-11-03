# Sensor Adaptation Research Suite (`experiments/`)

> ICoDiT 2026 Camera-Ready experimental protocols, MuJoCo sensor fault wrappers, SAC learners, and statistical analysis scripts.

**Total Modules Documented:** 16

---

## Table of Contents

- [`experiments\sensor_adaptation_v1\analyze_extension.py`](#experimentssensoradaptationv1analyzeextensionpy)
- [`experiments\sensor_adaptation_v1\analyze_replay_bias.py`](#experimentssensoradaptationv1analyzereplaybiaspy)
- [`experiments\sensor_adaptation_v1\analyze_results.py`](#experimentssensoradaptationv1analyzeresultspy)
- [`experiments\sensor_adaptation_v1\benchmark_latency.py`](#experimentssensoradaptationv1benchmarklatencypy)
- [`experiments\sensor_adaptation_v1\build_paper_assets.py`](#experimentssensoradaptationv1buildpaperassetspy)
- [`experiments\sensor_adaptation_v1\frozen_models.py`](#experimentssensoradaptationv1frozenmodelspy)
- [`experiments\sensor_adaptation_v1\learner.py`](#experimentssensoradaptationv1learnerpy)
- [`experiments\sensor_adaptation_v1\prepare_review_release.py`](#experimentssensoradaptationv1preparereviewreleasepy)
- [`experiments\sensor_adaptation_v1\replay.py`](#experimentssensoradaptationv1replaypy)
- [`experiments\sensor_adaptation_v1\run_study.py`](#experimentssensoradaptationv1runstudypy)
- [`experiments\sensor_adaptation_v1\sensor_env.py`](#experimentssensoradaptationv1sensorenvpy)
- [`experiments\sensor_adaptation_v1\share_wandb_review.py`](#experimentssensoradaptationv1sharewandbreviewpy)
- [`experiments\sensor_adaptation_v1\test_contracts.py`](#experimentssensoradaptationv1testcontractspy)
- [`experiments\sensor_adaptation_v1\upload_wandb.py`](#experimentssensoradaptationv1uploadwandbpy)
- [`experiments\sensor_adaptation_v1\verify_review_release.py`](#experimentssensoradaptationv1verifyreviewreleasepy)
- [`experiments\sensor_adaptation_v1\verify_wandb_upload.py`](#experimentssensoradaptationv1verifywandbuploadpy)

---
## Module Outlines

### [`experiments\sensor_adaptation_v1\analyze_extension.py`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/analyze_extension.py)

**Purpose:** Pre-registered analysis for sensor-extension-2.0 / sensor-ant-2.0 (see protocol analysis_plan).

#### Functions

* **[`read_branches(root)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/analyze_extension.py#L24)**
* **[`branch_means(branch, budgets, max_episodes)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/analyze_extension.py#L45)**: Mean return per (budget, measured condition); optionally only the first episodes.
* **[`trapezoid_mean(budgets, values)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/analyze_extension.py#L61)**
* **[`seed_metrics(branches, budgets)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/analyze_extension.py#L66)**
* **[`holm(pvalues)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/analyze_extension.py#L85)**
* **[`describe(differences)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/analyze_extension.py#L94)**
* **[`index_rows(rows, budget)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/analyze_extension.py#L112)**
* **[`paired(table, key_a, key_b, metric, seeds)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/analyze_extension.py#L120)**
* **[`analyze(rows, config, budget)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/analyze_extension.py#L127)**
* **[`reproducibility(branches, pilot_dir, pilot_budgets, episodes)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/analyze_extension.py#L188)**: Exact comparison of shared branches/episodes with the frozen pilot.
* **[`write_csv(path, rows)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/analyze_extension.py#L209)**
* **[`main()`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/analyze_extension.py#L220)**


---

### [`experiments\sensor_adaptation_v1\analyze_replay_bias.py`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/analyze_replay_bias.py)

**Purpose:** Replay-mixing diagnostic (reviewer request): critic bias on clean vs corrupted data.

#### Functions

* **[`returns_to_go(rewards, gamma)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/analyze_replay_bias.py#L33)**
* **[`pack(episode, steps, context, successor)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/analyze_replay_bias.py#L41)**
* **[`critic_on_trajectory(model, algorithm, path, context, gamma, tail, device, alpha)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/analyze_replay_bias.py#L54)**
* **[`branch_bias(branch, shard_inputs, budget, config, device, tail)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/analyze_replay_bias.py#L90)**
* **[`main()`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/analyze_replay_bias.py#L116)**


---

### [`experiments\sensor_adaptation_v1\analyze_results.py`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/analyze_results.py)

**Purpose:** Verify all completed branches and regenerate descriptive pilot outputs.

#### Functions

* **[`digest(path)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/analyze_results.py#L14)**
* **[`write_csv(path, rows)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/analyze_results.py#L18)**
* **[`analyze(root)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/analyze_results.py#L25)**
* **[`plot_curves(root, config, study, retention)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/analyze_results.py#L156)**


---

### [`experiments\sensor_adaptation_v1\benchmark_latency.py`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/benchmark_latency.py)

**Purpose:** Per-action inference latency of the pilot controllers (reviewer request: 100-500 Hz control).

#### Functions

* **[`cpu_name()`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/benchmark_latency.py#L27)**
* **[`tokens_for(trajectories, algorithm, seed)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/benchmark_latency.py#L36)**
* **[`measure(model, algorithm, tokens, device, steps, warmup, episode_length)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/benchmark_latency.py#L41)**
* **[`summarize(ms, context_full_fraction)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/benchmark_latency.py#L61)**
* **[`main()`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/benchmark_latency.py#L69)**


---

### [`experiments\sensor_adaptation_v1\build_paper_assets.py`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/build_paper_assets.py)

**Purpose:** Build manuscript tables and legible vector figures from a verified pilot.

#### Functions

* **[`build(root, output)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/build_paper_assets.py#L13)**


---

### [`experiments\sensor_adaptation_v1\frozen_models.py`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py)

**Purpose:** Checkpoint-compatible model definitions extracted without changes from the prior implementation.

#### Classes

* **[`TimestepEmbedder(nn.Module)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L14)**
  * [`__init__(self, in_dim, d_model)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L15)
  * [`forward(self, x)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L24)
* **[`CausalSelfAttention(nn.Module)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L28)**
  * [`__init__(self, d_model, n_heads, dropout)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L29)
  * [`forward(self, x)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L42)
* **[`TransformerBlock(nn.Module)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L59)**
  * [`__init__(self, d_model, n_heads, d_ff, dropout)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L60)
  * [`forward(self, x)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L72)
* **[`TransformerTrunk(nn.Module)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L78)**
  * [`__init__(self, in_dim, d_model, n_layers, n_heads, d_ff, max_len, dropout)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L79)
  * [`forward(self, x)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L99)
* **[`TaskEncoder(nn.Module)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L107)**
  * [`__init__(self, d_model, z_dim, num_classes)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L108)
  * [`forward(self, h)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L113)
* **[`GaussianActorHead(nn.Module)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L119)**
  * [`__init__(self, in_dim, act_dim, hidden_dim)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L120)
  * [`forward(self, x, deterministic)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L131)
* **[`CriticHead(nn.Module)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L151)**
  * [`__init__(self, in_dim, hidden_dim, use_layernorm)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L152)
  * [`forward(self, x)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L168)
* **[`TransformerRL2Agent(nn.Module)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L172)**
  * [`__init__(self, obs_dim, act_dim, d_model, z_dim, num_critics, subset_m, use_layernorm, lambda_task)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L173)
  * [`reset_context(self)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L228)
  * [`step_in_context(self, obs, adaptive, deterministic, device)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L231)
* **[`MLPGaussianActor(nn.Module)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L254)**
  * [`__init__(self, obs_dim, act_dim, hidden_dim)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L255)
  * [`forward(self, x, deterministic)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L266)
* **[`MLPCritic(nn.Module)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L285)**
  * [`__init__(self, obs_dim, act_dim, hidden_dim, use_layernorm)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L286)
  * [`forward(self, obs, act)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L302)
* **[`ReactiveREDQAgent(nn.Module)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L306)**
  * [`__init__(self, obs_dim, act_dim, num_critics, subset_m, use_layernorm)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L307)
  * [`act(self, obs, deterministic, device)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py#L334)


---

### [`experiments\sensor_adaptation_v1\learner.py`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/learner.py)

**Purpose:** Warm-start SAC/REDQ fine-tuning; no hidden fault-label supervision.

#### Classes

* **[`Learner`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/learner.py#L53)**
  * [`__init__(self, algorithm, checkpoint, config, device, seed)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/learner.py#L54)
  * [`rollout_model(self)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/learner.py#L69)
  * [`update(self, batch)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/learner.py#L73)
  * [`save(self, path)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/learner.py#L123)

#### Functions

* **[`configure(model, algorithm, context)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/learner.py#L14)**
* **[`load_agent(algorithm, checkpoint, context, device)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/learner.py#L26)**
* **[`features(model, algorithm, observations, lengths, target)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/learner.py#L37)**
* **[`q_value(model, algorithm, index, feat, action, target)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/learner.py#L48)**


---

### [`experiments\sensor_adaptation_v1\prepare_review_release.py`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/prepare_review_release.py)

**Purpose:** Prepare an identity-scrubbed copy; leave the executed experiment untouched.

#### Functions

* **[`redact_string(text)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/prepare_review_release.py#L21)**
* **[`write_json(path, value)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/prepare_review_release.py#L26)**
* **[`main(root, output, paper, qa)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/prepare_review_release.py#L30)**


---

### [`experiments\sensor_adaptation_v1\replay.py`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/replay.py)

**Purpose:** Episode replay with exact rolling contexts and terminal-transition coverage.

#### Functions

* **[`save_episodes(path, episodes)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/replay.py#L6)**
* **[`load_episodes(path)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/replay.py#L12)**
* **[`extract_transition(episode, t, context)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/replay.py#L19)**: Next context slides when full; it includes the actual final observation.
* **[`sample_batch(new_episodes, old_episodes, batch_size, context, old_fraction, rng, device)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/replay.py#L29)**


---

### [`experiments\sensor_adaptation_v1\run_study.py`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/run_study.py)

**Purpose:** Execute isolated A -> sensor B -> A adaptation/retention branches.

#### Functions

* **[`sha256(path)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/run_study.py#L32)**
* **[`write_json(path, value)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/run_study.py#L36)**
* **[`seed_all(seed)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/run_study.py#L40)**
* **[`preserve_training_rng()`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/run_study.py#L49)**
* **[`rollout(model, algorithm, condition, config, environment_seed, deterministic, onset, max_steps)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/run_study.py#L63)**
* **[`prepare(output, config, smoke)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/run_study.py#L104)**
* **[`evaluate(learner, config, condition, folder, budget, writer, checkpoint_hash, adaptation_seconds)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/run_study.py#L135)**
* **[`run_branch(output, manifest, config, algorithm, seed, condition, retained_fraction, old_episodes, device)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/run_study.py#L156)**
* **[`main()`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/run_study.py#L258)**


---

### [`experiments\sensor_adaptation_v1\sensor_env.py`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/sensor_env.py)

**Purpose:** Observation faults only. Actuator commands and base rewards pass through.

#### Classes

* **[`SensorFaultEnv(gym.Wrapper)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/sensor_env.py#L9)**
  * [`__init__(self, env, fault, noise_std, delay, onset)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/sensor_env.py#L12)
  * [`_observe(self, true_obs, info)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/sensor_env.py#L42)
  * [`reset(self)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/sensor_env.py#L64)
  * [`step(self, action)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/sensor_env.py#L75)

#### Functions

* **[`make_env(condition, config, onset)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/sensor_env.py#L86)**


---

### [`experiments\sensor_adaptation_v1\share_wandb_review.py`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/share_wandb_review.py)

**Purpose:** Enable reviewer access to this verified project, then test without credentials.

#### Functions

* **[`query(session, document, variables)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/share_wandb_review.py#L20)**
* **[`main()`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/share_wandb_review.py#L29)**


---

### [`experiments\sensor_adaptation_v1\test_contracts.py`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/test_contracts.py)

**Purpose:** Behavioral checks: sensing, physics isolation, replay boundaries, and learning.

#### Classes

* **[`CounterEnv(gym.Env)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/test_contracts.py#L23)**
  * [`__init__(self)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/test_contracts.py#L24)
  * [`reset(self)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/test_contracts.py#L30)
  * [`step(self, action)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/test_contracts.py#L35)
* **[`SensorTests(unittest.TestCase)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/test_contracts.py#L48)**
  * [`test_noise_changes_sensor_but_preserves_command_reward_and_bookkeeping(self)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/test_contracts.py#L49)
  * [`test_noise_repeatable_and_independent_of_environment_rng(self)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/test_contracts.py#L62)
  * [`test_delay_is_of_reading_not_of_action_or_reward(self)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/test_contracts.py#L69)
  * [`test_delayed_fault_onset_uses_actual_prefault_history(self)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/test_contracts.py#L80)
  * [`test_invalid_faults_fail_instead_of_silently_doing_nothing(self)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/test_contracts.py#L89)
  * [`test_real_mujoco_physics_identical_under_identical_commands(self)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/test_contracts.py#L95)
* **[`ReplayTests(unittest.TestCase)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/test_contracts.py#L114)**
  * [`test_short_and_final_transitions_have_correct_next_context(self)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/test_contracts.py#L115)
  * [`test_mixed_replay_counts_and_terminal_coverage(self)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/test_contracts.py#L127)
  * [`test_replay_roundtrip_without_pickle(self)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/test_contracts.py#L136)
* **[`LegacyBoundaryTests(unittest.TestCase)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/test_contracts.py#L146)**
  * [`test_actuator_wrapper_rejects_sensor_labels(self)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/test_contracts.py#L147)
* **[`LearningTests(unittest.TestCase)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/test_contracts.py#L158)**
  * [`test_right_padding_is_invisible_to_last_valid_context(self)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/test_contracts.py#L159)
  * [`test_real_update_changes_model_and_preserves_unused_positions(self)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/test_contracts.py#L169)

#### Functions

* **[`episode(length)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/test_contracts.py#L41)**


---

### [`experiments\sensor_adaptation_v1\upload_wandb.py`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/upload_wandb.py)

**Purpose:** Import the completed, verified pilot without rerunning or changing training.

#### Functions

* **[`read_json(path)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/upload_wandb.py#L18)**
* **[`read_csv(path)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/upload_wandb.py#L22)**
* **[`run_id(release_hash, branch)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/upload_wandb.py#L34)**
* **[`load_payload(root)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/upload_wandb.py#L38)**
* **[`table(wandb, rows)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/upload_wandb.py#L78)**
* **[`overview_html(payload)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/upload_wandb.py#L83)**
* **[`save_receipt(path, value)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/upload_wandb.py#L109)**
* **[`upload(root, payload, entity, project, receipt_path)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/upload_wandb.py#L113)**


---

### [`experiments\sensor_adaptation_v1\verify_review_release.py`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/verify_review_release.py)

**Purpose:** Verify the shared evidence without running training or changing any result.

#### Functions

* **[`digest(path)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/verify_review_release.py#L11)**
* **[`verify(root)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/verify_review_release.py#L16)**


---

### [`experiments\sensor_adaptation_v1\verify_wandb_upload.py`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/verify_wandb_upload.py)

**Purpose:** Read back W&B runs, tables and artifact digests; no remote mutations.

#### Functions

* **[`same(actual, expected)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/verify_wandb_upload.py#L15)**
* **[`main(root, receipt_path)`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/verify_wandb_upload.py#L22)**


---
