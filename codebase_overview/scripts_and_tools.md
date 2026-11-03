# Execution Scripts & Build Pipelines (`scripts/`)

> Source training recipes, automated unattended pipelines, latency benchmarking, and camera-ready LaTeX asset builders.

**Total Modules Documented:** 46

---

## Table of Contents

- [`scripts\add_trackmania_paper_figure.py`](#scriptsaddtrackmaniapaperfigurepy)
- [`scripts\audit_driving_dataset.py`](#scriptsauditdrivingdatasetpy)
- [`scripts\audit_foundation_deployment.py`](#scriptsauditfoundationdeploymentpy)
- [`scripts\benchmark_fault_env.py`](#scriptsbenchmarkfaultenvpy)
- [`scripts\benchmark_live_pipeline.py`](#scriptsbenchmarklivepipelinepy)
- [`scripts\build_camera_ready_assets.py`](#scriptsbuildcamerareadyassetspy)
- [`scripts\build_camera_ready_workflow_figure.py`](#scriptsbuildcamerareadyworkflowfigurepy)
- [`scripts\build_reward_from_demos.py`](#scriptsbuildrewardfromdemospy)
- [`scripts\build_trajectory_assist.py`](#scriptsbuildtrajectoryassistpy)
- [`scripts\confirmatory_logging.py`](#scriptsconfirmatoryloggingpy)
- [`scripts\cycle_map_macro.py`](#scriptscyclemapmacropy)
- [`scripts\data\bulk_tmnf_scraper.py`](#scriptsdatabulktmnfscraperpy)
- [`scripts\data\convert_raw_gbx_to_dataset.py`](#scriptsdataconvertrawgbxtodatasetpy)
- [`scripts\data\download_1k_diverse_maps.py`](#scriptsdatadownload1kdiversemapspy)
- [`scripts\data\ultra_fast_1m_scraper.py`](#scriptsdataultrafast1mscraperpy)
- [`scripts\diagnose_visual_policy.py`](#scriptsdiagnosevisualpolicypy)
- [`scripts\eval_in_context_learning.py`](#scriptsevalincontextlearningpy)
- [`scripts\evaluate_live_driving.py`](#scriptsevaluatelivedrivingpy)
- [`scripts\export_live_driving_demos.py`](#scriptsexportlivedrivingdemospy)
- [`scripts\finalize_sensor_pack.py`](#scriptsfinalizesensorpackpy)
- [`scripts\finetune_target_track.py`](#scriptsfinetunetargettrackpy)
- [`scripts\generate_confirmatory_results.py`](#scriptsgenerateconfirmatoryresultspy)
- [`scripts\generate_statistical_report.py`](#scriptsgeneratestatisticalreportpy)
- [`scripts\kaggle\train_car_brain_kaggle.py`](#scriptskaggletraincarbrainkagglepy)
- [`scripts\live_drive_car_brain.py`](#scriptslivedrivecarbrainpy)
- [`scripts\prepare_camera_ready_release.py`](#scriptspreparecamerareadyreleasepy)
- [`scripts\qa_sensor_paper.py`](#scriptsqasensorpaperpy)
- [`scripts\record_track_demonstration.py`](#scriptsrecordtrackdemonstrationpy)
- [`scripts\run_adaptation_benchmark.py`](#scriptsrunadaptationbenchmarkpy)
- [`scripts\run_ant_sources.py`](#scriptsrunantsourcespy)
- [`scripts\run_camera_ready_pipeline.py`](#scriptsruncamerareadypipelinepy)
- [`scripts\run_confirmatory_study.py`](#scriptsrunconfirmatorystudypy)
- [`scripts\run_reproducible_benchmark.py`](#scriptsrunreproduciblebenchmarkpy)
- [`scripts\smoke_continual_jax_gpu.py`](#scriptssmokecontinualjaxgpupy)
- [`scripts\test_straight.py`](#scriptsteststraightpy)
- [`scripts\test_track_b_forward.py`](#scriptstesttrackbforwardpy)
- [`scripts\train_50k_curriculum.py`](#scriptstrain50kcurriculumpy)
- [`scripts\train_dreamer_track_b.py`](#scriptstraindreamertrackbpy)
- [`scripts\train_multitrack_vision_curriculum.py`](#scriptstrainmultitrackvisioncurriculumpy)
- [`scripts\train_reactive_robust_baseline.py`](#scriptstrainreactiverobustbaselinepy)
- [`scripts\train_reactive_source_recipe_v1.py`](#scriptstrainreactivesourcerecipev1py)
- [`scripts\train_streaming_continual.py`](#scriptstrainstreamingcontinualpy)
- [`scripts\train_transformer_fault_adaptation.py`](#scriptstraintransformerfaultadaptationpy)
- [`scripts\train_transformer_source_recipe_v1.py`](#scriptstraintransformersourcerecipev1py)
- [`scripts\train_vision_online.py`](#scriptstrainvisiononlinepy)
- [`scripts\verify_adaptation_pipeline.py`](#scriptsverifyadaptationpipelinepy)

---
## Module Outlines

### [`scripts\add_trackmania_paper_figure.py`](file:///D:/Project/tmrl/scripts/add_trackmania_paper_figure.py)

**Purpose:** Build a paper workflow figure from an unchanged local development capture.

#### Functions

* **[`main()`](file:///D:/Project/tmrl/scripts/add_trackmania_paper_figure.py#L24)**


---

### [`scripts\audit_driving_dataset.py`](file:///D:/Project/tmrl/scripts/audit_driving_dataset.py)

**Purpose:** Build and execute a compact, reproducible data-quality notebook.

#### Functions

* **[`main()`](file:///D:/Project/tmrl/scripts/audit_driving_dataset.py#L12)**


---

### [`scripts\audit_foundation_deployment.py`](file:///D:/Project/tmrl/scripts/audit_foundation_deployment.py)

**Purpose:** Offline safety gate for the policy that will control the TMRL worker.

#### Functions

* **[`tensor_state_digest(state_dict, prefix)`](file:///D:/Project/tmrl/scripts/audit_foundation_deployment.py#L31)**
* **[`file_digest(path)`](file:///D:/Project/tmrl/scripts/audit_foundation_deployment.py#L40)**
* **[`worker_observation(state, image, action_history)`](file:///D:/Project/tmrl/scripts/audit_foundation_deployment.py#L48)**: Convert a recorded demonstration step to the worker's preprocessed form.
* **[`evaluate(args)`](file:///D:/Project/tmrl/scripts/audit_foundation_deployment.py#L71)**
* **[`main()`](file:///D:/Project/tmrl/scripts/audit_foundation_deployment.py#L240)**


---

### [`scripts\benchmark_fault_env.py`](file:///D:/Project/tmrl/scripts/benchmark_fault_env.py)

**Purpose:** Gymnasium Fault-Injection Wrapper for In-Context Reinforcement Learning Benchmark.

#### Classes

* **[`FaultInjectionWrapper(gym.Wrapper)`](file:///D:/Project/tmrl/scripts/benchmark_fault_env.py#L31)**: Wraps a Box continuous control environment and injects episode-level faults.
  * [`__init__(self, env, split, fixed_fault, severity, fixed_latency, seed)`](file:///D:/Project/tmrl/scripts/benchmark_fault_env.py#L46)
  * [`_sample_fault(self)`](file:///D:/Project/tmrl/scripts/benchmark_fault_env.py#L103) — *Draws one unobservable fault per episode according to the split.*
  * [`_apply_actuator_fault(self, action)`](file:///D:/Project/tmrl/scripts/benchmark_fault_env.py#L162) — *Applies active fault to commanded action.*
  * [`_pack_observation(self, base_obs)`](file:///D:/Project/tmrl/scripts/benchmark_fault_env.py#L186)
  * [`reset(self)`](file:///D:/Project/tmrl/scripts/benchmark_fault_env.py#L195)
  * [`step(self, action)`](file:///D:/Project/tmrl/scripts/benchmark_fault_env.py#L212)
  * [`get_fault_id(self)`](file:///D:/Project/tmrl/scripts/benchmark_fault_env.py#L230) — *Returns integer index for auxiliary classification head.*

#### Functions

* **[`make_fault_env(env_id, split, fixed_fault, seed)`](file:///D:/Project/tmrl/scripts/benchmark_fault_env.py#L243)**: Factory function for creating wrapped fault-injected environments.


---

### [`scripts\benchmark_live_pipeline.py`](file:///D:/Project/tmrl/scripts/benchmark_live_pipeline.py)

**Purpose:** Profile the live Trackmania vision-policy pipeline without sending controls.

#### Functions

* **[`percentile(values, q)`](file:///D:/Project/tmrl/scripts/benchmark_live_pipeline.py#L39)**
* **[`summarize_ms(samples)`](file:///D:/Project/tmrl/scripts/benchmark_live_pipeline.py#L45)**
* **[`build_actor(device)`](file:///D:/Project/tmrl/scripts/benchmark_live_pipeline.py#L57)**
* **[`main()`](file:///D:/Project/tmrl/scripts/benchmark_live_pipeline.py#L100)**


---

### [`scripts\build_camera_ready_assets.py`](file:///D:/Project/tmrl/scripts/build_camera_ready_assets.py)

**Purpose:** Build camera-ready figures, tables and number macros from saved analysis outputs.

#### Functions

* **[`num(x, digits, sign)`](file:///D:/Project/tmrl/scripts/build_camera_ready_assets.py#L30)**: Returns are reported as integers by default; rates and residuals pass digits explicitly.
* **[`pval(p)`](file:///D:/Project/tmrl/scripts/build_camera_ready_assets.py#L42)**
* **[`load_seed_rows(root)`](file:///D:/Project/tmrl/scripts/build_camera_ready_assets.py#L48)**
* **[`tci(values)`](file:///D:/Project/tmrl/scripts/build_camera_ready_assets.py#L57)**
* **[`macros_for_env(env, root, macros)`](file:///D:/Project/tmrl/scripts/build_camera_ready_assets.py#L63)**
* **[`bias_gap_macros(env, bias, macros)`](file:///D:/Project/tmrl/scripts/build_camera_ready_assets.py#L154)**: Exploratory, not pre-specified: per seed, Monte-Carlo value bias on corrupted data minus that on clean data,
* **[`pilot_equivalent_macros(env, root, macros, seeds, episodes, budget)`](file:///D:/Project/tmrl/scripts/build_camera_ready_assets.py#L187)**: The pilot's own metric (seeds 42-44, evaluation episodes 0-2, 4,000 steps) from the extension records.
* **[`behavior_macros(env, root, macros, final, horizon)`](file:///D:/Project/tmrl/scripts/build_camera_ready_assets.py#L217)**: Source forward speed and early-termination rates from saved evaluation records.
* **[`curve_figure(panels, path, row_height)`](file:///D:/Project/tmrl/scripts/build_camera_ready_assets.py#L255)**: One row per (task, fault) with sensor- and clean-condition panels, at LNCS text width (4.8 in), so fonts
* **[`task_header(env, ncols, first)`](file:///D:/Project/tmrl/scripts/build_camera_ready_assets.py#L311)**
* **[`outcomes_table(envs, macros)`](file:///D:/Project/tmrl/scripts/build_camera_ready_assets.py#L316)**
* **[`tests_table(envs, macros)`](file:///D:/Project/tmrl/scripts/build_camera_ready_assets.py#L340)**
* **[`bias_table(envs, macros)`](file:///D:/Project/tmrl/scripts/build_camera_ready_assets.py#L364)**
* **[`latency_table(path, macros)`](file:///D:/Project/tmrl/scripts/build_camera_ready_assets.py#L393)**
* **[`main()`](file:///D:/Project/tmrl/scripts/build_camera_ready_assets.py#L417)**


---

### [`scripts\build_camera_ready_workflow_figure.py`](file:///D:/Project/tmrl/scripts/build_camera_ready_workflow_figure.py)

**Purpose:** Camera-ready workflow figure: unchanged TrackMania capture plus a schematic of the executed study.

#### Functions

* **[`main()`](file:///D:/Project/tmrl/scripts/build_camera_ready_workflow_figure.py#L23)**


---

### [`scripts\build_reward_from_demos.py`](file:///D:/Project/tmrl/scripts/build_reward_from_demos.py)

**Purpose:** Converts recorded expert demonstration positions from data/target_track_demos.pt

#### Functions

* **[`main()`](file:///D:/Project/tmrl/scripts/build_reward_from_demos.py#L21)**


---

### [`scripts\build_trajectory_assist.py`](file:///D:/Project/tmrl/scripts/build_trajectory_assist.py)

**Purpose:** Build a compact deployment trajectory from a recorded demonstration.

#### Functions

* **[`sha256(path)`](file:///D:/Project/tmrl/scripts/build_trajectory_assist.py#L21)**
* **[`main()`](file:///D:/Project/tmrl/scripts/build_trajectory_assist.py#L29)**


---

### [`scripts\confirmatory_logging.py`](file:///D:/Project/tmrl/scripts/confirmatory_logging.py)

**Purpose:** Confirmatory Experiment Reproducibility & Logging Infrastructure.

#### Functions

* **[`get_git_commit(repo_root)`](file:///D:/Project/tmrl/scripts/confirmatory_logging.py#L25)**
* **[`get_git_diff(repo_root)`](file:///D:/Project/tmrl/scripts/confirmatory_logging.py#L39)**
* **[`compute_file_sha256(filepath)`](file:///D:/Project/tmrl/scripts/confirmatory_logging.py#L55)**
* **[`get_hardware_info()`](file:///D:/Project/tmrl/scripts/confirmatory_logging.py#L63)**
* **[`get_environment_info()`](file:///D:/Project/tmrl/scripts/confirmatory_logging.py#L80)**
* **[`setup_run_directory(output_dir, run_name, config, seed, repo_root)`](file:///D:/Project/tmrl/scripts/confirmatory_logging.py#L91)**
* **[`update_manifest(manifest_path, entry)`](file:///D:/Project/tmrl/scripts/confirmatory_logging.py#L130)**


---

### [`scripts\cycle_map_macro.py`](file:///D:/Project/tmrl/scripts/cycle_map_macro.py)

**Purpose:** Trackmania 2020 Map Cycler Macro for Starter Access

#### Functions

* **[`get_map_list()`](file:///D:/Project/tmrl/scripts/cycle_map_macro.py#L56)**
* **[`load_state()`](file:///D:/Project/tmrl/scripts/cycle_map_macro.py#L63)**
* **[`save_state(state)`](file:///D:/Project/tmrl/scripts/cycle_map_macro.py#L73)**
* **[`copy_active_map(index)`](file:///D:/Project/tmrl/scripts/cycle_map_macro.py#L78)**
* **[`send_keyboard_key(scancode, is_extended, duration)`](file:///D:/Project/tmrl/scripts/cycle_map_macro.py#L90)**
* **[`activate_trackmania_window()`](file:///D:/Project/tmrl/scripts/cycle_map_macro.py#L122)**
* **[`execute_cycle_sequence(strategy, gamepad)`](file:///D:/Project/tmrl/scripts/cycle_map_macro.py#L135)**: Executes the in-game menu sequence to reload / advance the track.
* **[`cycle_now(strategy, gamepad)`](file:///D:/Project/tmrl/scripts/cycle_map_macro.py#L197)**
* **[`run_auto_loop(interval_sec, strategy)`](file:///D:/Project/tmrl/scripts/cycle_map_macro.py#L218)**


---

### [`scripts\data\bulk_tmnf_scraper.py`](file:///D:/Project/tmrl/scripts/data/bulk_tmnf_scraper.py)

**Purpose:** Ultra-High-Speed Concurrent Replay Scraper & Parser for TrackMania (TMX / TMNF).

#### Functions

* **[`fetch_tracks_paginated(target_tracks, cache_file)`](file:///D:/Project/tmrl/scripts/data/bulk_tmnf_scraper.py#L32)**: Fetches track IDs using pagination or loads from local cache.
* **[`parse_and_save_replay(content, output_csv_path)`](file:///D:/Project/tmrl/scripts/data/bulk_tmnf_scraper.py#L91)**: Parses .Replay.Gbx bytes into 20 Hz state-action CSV.
* **[`process_single_track(track_id, max_replays, output_dir, completed_replays, session)`](file:///D:/Project/tmrl/scripts/data/bulk_tmnf_scraper.py#L165)**: Fetches and downloads all replays for a given track.
* **[`main()`](file:///D:/Project/tmrl/scripts/data/bulk_tmnf_scraper.py#L211)**


---

### [`scripts\data\convert_raw_gbx_to_dataset.py`](file:///D:/Project/tmrl/scripts/data/convert_raw_gbx_to_dataset.py)

**Purpose:** High-Performance Multi-Core Converter for Raw TrackMania .Replay.Gbx Files.

#### Functions

* **[`process_single_gbx(args_tuple)`](file:///D:/Project/tmrl/scripts/data/convert_raw_gbx_to_dataset.py#L27)**
* **[`main()`](file:///D:/Project/tmrl/scripts/data/convert_raw_gbx_to_dataset.py#L107)**


---

### [`scripts\data\download_1k_diverse_maps.py`](file:///D:/Project/tmrl/scripts/data/download_1k_diverse_maps.py)

**Purpose:** High-Speed Downloader for 1,000 Diverse TrackMania 2020 Maps from TMX.

#### Functions

* **[`sanitize_filename(name)`](file:///D:/Project/tmrl/scripts/data/download_1k_diverse_maps.py#L50)**: Removes non-ascii or filesystem-illegal characters.
* **[`fetch_category_maps(category, tag_id, seen_ids)`](file:///D:/Project/tmrl/scripts/data/download_1k_diverse_maps.py#L58)**: Fetches up to 100 top-rated map metadata for a specific tag using TMX API.
* **[`download_single_map(item, category, output_dirs)`](file:///D:/Project/tmrl/scripts/data/download_1k_diverse_maps.py#L81)**: Downloads one map from TMX and saves it to all target directories.
* **[`generate_matchsettings_xml(map_filenames, time_limit_sec)`](file:///D:/Project/tmrl/scripts/data/download_1k_diverse_maps.py#L119)**: Generates TrackMania MatchSettings playlist XML for arcade/local rotation.
* **[`main()`](file:///D:/Project/tmrl/scripts/data/download_1k_diverse_maps.py#L142)**


---

### [`scripts\data\ultra_fast_1m_scraper.py`](file:///D:/Project/tmrl/scripts/data/ultra_fast_1m_scraper.py)

**Purpose:** Pipelined 150 Mbps Replay Harvest Engine for 1,000,000 TrackMania Replays.

#### Classes

* **[`ReplayStreamingHarvester`](file:///D:/Project/tmrl/scripts/data/ultra_fast_1m_scraper.py#L24)**
  * [`__init__(self, output_dir, concurrency, target_replays)`](file:///D:/Project/tmrl/scripts/data/ultra_fast_1m_scraper.py#L25)
  * [`producer(self, session)`](file:///D:/Project/tmrl/scripts/data/ultra_fast_1m_scraper.py#L57) — *Continuously pages tracks from ManiaExchange and pushes to download queue.*
  * [`download_replay_binary(self, session, track_id, replay_id)`](file:///D:/Project/tmrl/scripts/data/ultra_fast_1m_scraper.py#L93) — *Streams raw .Replay.Gbx directly into the single flat output directory.*
  * [`worker(self, session)`](file:///D:/Project/tmrl/scripts/data/ultra_fast_1m_scraper.py#L130) — *Worker task processing tracks from the queue.*
  * [`monitor(self)`](file:///D:/Project/tmrl/scripts/data/ultra_fast_1m_scraper.py#L162) — *Displays live throughput statistics (MB/s, maps/sec, and replays/sec).*
  * [`run(self)`](file:///D:/Project/tmrl/scripts/data/ultra_fast_1m_scraper.py#L207)

#### Functions

* **[`main()`](file:///D:/Project/tmrl/scripts/data/ultra_fast_1m_scraper.py#L238)**


---

### [`scripts\diagnose_visual_policy.py`](file:///D:/Project/tmrl/scripts/diagnose_visual_policy.py)

**Purpose:** Recompute held-out action metrics and inspect saturation, without game input.

#### Functions

* **[`collect(model, data, device, amp)`](file:///D:/Project/tmrl/scripts/diagnose_visual_policy.py#L21)**
* **[`main()`](file:///D:/Project/tmrl/scripts/diagnose_visual_policy.py#L40)**


---

### [`scripts\eval_in_context_learning.py`](file:///D:/Project/tmrl/scripts/eval_in_context_learning.py)

**Purpose:** Evaluation Harness for In-Context Learning (ICL) in TrackMania.

#### Functions

* **[`compute_95_ci(data)`](file:///D:/Project/tmrl/scripts/eval_in_context_learning.py#L32)**: Computes mean and half-width of 95% confidence interval using Student's t distribution.
* **[`run_synthetic_eval(model, num_episodes, episode_length, state_dim, action_dim, device, inject_latency)`](file:///D:/Project/tmrl/scripts/eval_in_context_learning.py#L44)**: Simulates evaluation episodes to compute the adaptation gap.
* **[`main()`](file:///D:/Project/tmrl/scripts/eval_in_context_learning.py#L127)**


---

### [`scripts\evaluate_live_driving.py`](file:///D:/Project/tmrl/scripts/evaluate_live_driving.py)

**Purpose:** Run a bounded local driving evaluation, without a trainer or replay server.

#### Classes

* **[`MotionStallGuard`](file:///D:/Project/tmrl/scripts/evaluate_live_driving.py#L68)**: Detect low displacement without consulting a demonstrated route.
  * [`__init__(self, window_seconds, min_displacement_m, grace_seconds)`](file:///D:/Project/tmrl/scripts/evaluate_live_driving.py#L73)
  * [`update(self, elapsed, position)`](file:///D:/Project/tmrl/scripts/evaluate_live_driving.py#L80)
* **[`FrameFreshnessGuard`](file:///D:/Project/tmrl/scripts/evaluate_live_driving.py#L93)**: Stop on identical camera pixels despite material vehicle movement.
  * [`__init__(self, max_repeat_seconds, min_motion_m)`](file:///D:/Project/tmrl/scripts/evaluate_live_driving.py#L99)
  * [`update(self, elapsed, frame, position)`](file:///D:/Project/tmrl/scripts/evaluate_live_driving.py#L107)
* **[`CycleDeadlineGuard`](file:///D:/Project/tmrl/scripts/evaluate_live_driving.py#L122)**: Reject repeated gross deadline overruns, allowing isolated startup jitter.
  * [`__init__(self, max_cycle_seconds, consecutive_limit)`](file:///D:/Project/tmrl/scripts/evaluate_live_driving.py#L124)
  * [`update(self, duration)`](file:///D:/Project/tmrl/scripts/evaluate_live_driving.py#L129)

#### Functions

* **[`extract_latest_replay_info(started_epoch)`](file:///D:/Project/tmrl/scripts/evaluate_live_driving.py#L27)**: Scan TrackMania replay directories for replays saved during this run.
* **[`reference_config_for_run(args, safety_config)`](file:///D:/Project/tmrl/scripts/evaluate_live_driving.py#L136)**
* **[`parse_args(argv)`](file:///D:/Project/tmrl/scripts/evaluate_live_driving.py#L152)**
* **[`main()`](file:///D:/Project/tmrl/scripts/evaluate_live_driving.py#L196)**


---

### [`scripts\export_live_driving_demos.py`](file:///D:/Project/tmrl/scripts/export_live_driving_demos.py)

**Purpose:** Export verified completed live runs as explicitly assisted BC demonstrations.

#### Functions

* **[`sha256(path)`](file:///D:/Project/tmrl/scripts/export_live_driving_demos.py#L19)**
* **[`load_verified_run(result_path)`](file:///D:/Project/tmrl/scripts/export_live_driving_demos.py#L27)**
* **[`main()`](file:///D:/Project/tmrl/scripts/export_live_driving_demos.py#L101)**


---

### [`scripts\finalize_sensor_pack.py`](file:///D:/Project/tmrl/scripts/finalize_sensor_pack.py)

**Purpose:** Finalize private preparation notes and a portable paper/results review package.

#### Functions

* **[`add_file(path, arc)`](file:///D:/Project/tmrl/scripts/finalize_sensor_pack.py#L183)**


---

### [`scripts\finetune_target_track.py`](file:///D:/Project/tmrl/scripts/finetune_target_track.py)

**Purpose:** Target-Track Fast Adaptation for TrackMania Foundation Model (GEN-1.5 Paradigm).

#### Functions

* **[`extract_causal_token_windows(episodes, window_len, stride, has_vision)`](file:///D:/Project/tmrl/scripts/finetune_target_track.py#L41)**: Extracts rolling causal windows over pre-encoded visual tokens and kinematics.
* **[`main()`](file:///D:/Project/tmrl/scripts/finetune_target_track.py#L94)**


---

### [`scripts\generate_confirmatory_results.py`](file:///D:/Project/tmrl/scripts/generate_confirmatory_results.py)

**Purpose:** Confirmatory Statistical Analysis and Publication Artifact Generator.

#### Functions

* **[`seed_bootstrap_ci(data, n_boot, ci, seed)`](file:///D:/Project/tmrl/scripts/generate_confirmatory_results.py#L31)**: Computes mean and percentile bootstrap confidence interval over seed-level values.
* **[`compute_distribution_metrics(values)`](file:///D:/Project/tmrl/scripts/generate_confirmatory_results.py#L54)**
* **[`evaluate_decision_rule(g_ci, g_values)`](file:///D:/Project/tmrl/scripts/generate_confirmatory_results.py#L70)**
* **[`load_confirmatory_data(results_dir)`](file:///D:/Project/tmrl/scripts/generate_confirmatory_results.py#L112)**
* **[`generate_confirmatory_artifacts(results_dir, output_dir, protocol_path)`](file:///D:/Project/tmrl/scripts/generate_confirmatory_results.py#L131)**


---

### [`scripts\generate_statistical_report.py`](file:///D:/Project/tmrl/scripts/generate_statistical_report.py)

**Purpose:** Statistical Reliability and Automatic Figure Generation for RL Fault Adaptation Benchmark.

#### Functions

* **[`bootstrap_ci(data, n_boot, ci)`](file:///D:/Project/tmrl/scripts/generate_statistical_report.py#L41)**: Computes mean and percentile bootstrap confidence interval.
* **[`compute_iqm(data)`](file:///D:/Project/tmrl/scripts/generate_statistical_report.py#L62)**: Interquartile Mean (trimmed mean of middle 50%).
* **[`load_all_reports(results_dir)`](file:///D:/Project/tmrl/scripts/generate_statistical_report.py#L73)**: Loads all transformer and reactive benchmark reports.
* **[`analyze_benchmark(data)`](file:///D:/Project/tmrl/scripts/generate_statistical_report.py#L94)**
* **[`plot_fault_adaptation(results, output_png, output_pdf)`](file:///D:/Project/tmrl/scripts/generate_statistical_report.py#L164)**: Plots clean, publication-grade academic comparison figure without distracting scatter dots.
* **[`print_comparison_markdown_table(results)`](file:///D:/Project/tmrl/scripts/generate_statistical_report.py#L286)**


---

### [`scripts\kaggle\train_car_brain_kaggle.py`](file:///D:/Project/tmrl/scripts/kaggle/train_car_brain_kaggle.py)

**Purpose:** Pretraining TrackMania Causal Transformer "Car Brain" on Kaggle.

#### Classes

* **[`TrackManiaReplayDataset(Dataset)`](file:///D:/Project/tmrl/scripts/kaggle/train_car_brain_kaggle.py#L186)**: Loads TrackMania replay CSVs, sub-samples to 20 Hz, extracts sliding windows of length L.
  * [`__init__(self, csv_files, window_len, downsample_stride, norm_stats)`](file:///D:/Project/tmrl/scripts/kaggle/train_car_brain_kaggle.py#L190)
  * [`get_norm_stats(self)`](file:///D:/Project/tmrl/scripts/kaggle/train_car_brain_kaggle.py#L275)
  * [`__len__(self)`](file:///D:/Project/tmrl/scripts/kaggle/train_car_brain_kaggle.py#L278)
  * [`__getitem__(self, idx)`](file:///D:/Project/tmrl/scripts/kaggle/train_car_brain_kaggle.py#L281)

#### Functions

* **[`evaluate_in_context_adaptation(model, val_loader, device)`](file:///D:/Project/tmrl/scripts/kaggle/train_car_brain_kaggle.py#L310)**: Evaluates next-action prediction error across context lengths:
* **[`main()`](file:///D:/Project/tmrl/scripts/kaggle/train_car_brain_kaggle.py#L363)**


---

### [`scripts\live_drive_car_brain.py`](file:///D:/Project/tmrl/scripts/live_drive_car_brain.py)

**Purpose:** Multi-Modal In-Game Inference Driver for TrackMania 2020.

#### Functions

* **[`send_control(act, speed, steer_gain, deadband)`](file:///D:/Project/tmrl/scripts/live_drive_car_brain.py#L50)**: Applies continuous actions to TrackMania using TMRL standard mappings.
* **[`release_controls()`](file:///D:/Project/tmrl/scripts/live_drive_car_brain.py#L102)**: Resets all control inputs upon exit.
* **[`main()`](file:///D:/Project/tmrl/scripts/live_drive_car_brain.py#L119)**


---

### [`scripts\prepare_camera_ready_release.py`](file:///D:/Project/tmrl/scripts/prepare_camera_ready_release.py)

**Purpose:** Stage the camera-ready code and data release locally. Nothing is uploaded or published.

#### Functions

* **[`sha256(path)`](file:///D:/Project/tmrl/scripts/prepare_camera_ready_release.py#L45)**
* **[`plan()`](file:///D:/Project/tmrl/scripts/prepare_camera_ready_release.py#L53)**
* **[`failed_attempts_note()`](file:///D:/Project/tmrl/scripts/prepare_camera_ready_release.py#L86)**
* **[`main()`](file:///D:/Project/tmrl/scripts/prepare_camera_ready_release.py#L99)**


---

### [`scripts\qa_sensor_paper.py`](file:///D:/Project/tmrl/scripts/qa_sensor_paper.py)

**Purpose:** Read-only structural checks for the anonymous sensor pilot manuscript.

#### Functions

* **[`inspect_font(font)`](file:///D:/Project/tmrl/scripts/qa_sensor_paper.py#L31)**
* **[`inspect_resources(resources)`](file:///D:/Project/tmrl/scripts/qa_sensor_paper.py#L44)**


---

### [`scripts\record_track_demonstration.py`](file:///D:/Project/tmrl/scripts/record_track_demonstration.py)

**Purpose:** Records human expert driving demonstration directly from TrackMania 2020 via OpenPlanet

#### Functions

* **[`sha256_file(path)`](file:///D:/Project/tmrl/scripts/record_track_demonstration.py#L32)**
* **[`main()`](file:///D:/Project/tmrl/scripts/record_track_demonstration.py#L40)**


---

### [`scripts\run_adaptation_benchmark.py`](file:///D:/Project/tmrl/scripts/run_adaptation_benchmark.py)

**Purpose:** Clean TrackMania 2020 Adaptation & World-Record Benchmark Orchestrator

#### Functions

* **[`sha256_file(path)`](file:///D:/Project/tmrl/scripts/run_adaptation_benchmark.py#L46)**
* **[`evaluate_arm_checkpoint(arm_name, step, track, model_weights, output_dir, attempts, max_steps, max_seconds, map_uid, map_file, world_record_time)`](file:///D:/Project/tmrl/scripts/run_adaptation_benchmark.py#L54)**: Runs N evaluation attempts for a given arm, checkpoint, and track.
* **[`train_adaptation_steps(base_model_path, target_steps, output_path, track_b_demo_path, track_a_replay_path, track_a_fraction, from_scratch, device, lr, batch_size, window_len)`](file:///D:/Project/tmrl/scripts/run_adaptation_benchmark.py#L187)**: Trains MultiModalCarBrain for target_steps, saving to output_path.
* **[`main()`](file:///D:/Project/tmrl/scripts/run_adaptation_benchmark.py#L348)**


---

### [`scripts\run_ant_sources.py`](file:///D:/Project/tmrl/scripts/run_ant_sources.py)

**Purpose:** Train Ant-v5 source policies with the HalfCheetah source recipe, gated on free RAM.

#### Classes

* **[`MemoryStatus(ctypes.Structure)`](file:///D:/Project/tmrl/scripts/run_ant_sources.py#L22)**

#### Functions

* **[`free_gb()`](file:///D:/Project/tmrl/scripts/run_ant_sources.py#L30)**
* **[`main()`](file:///D:/Project/tmrl/scripts/run_ant_sources.py#L37)**


---

### [`scripts\run_camera_ready_pipeline.py`](file:///D:/Project/tmrl/scripts/run_camera_ready_pipeline.py)

**Purpose:** Unattended camera-ready pipeline (31 Oct 2026, v3.2).

#### Classes

* **[`PerformanceInfo(ctypes.Structure)`](file:///D:/Project/tmrl/scripts/run_camera_ready_pipeline.py#L65)**
* **[`Job`](file:///D:/Project/tmrl/scripts/run_camera_ready_pipeline.py#L113)**
  * [`__init__(self, name, kind, args, deps)`](file:///D:/Project/tmrl/scripts/run_camera_ready_pipeline.py#L114)
  * [`est_gb(self)`](file:///D:/Project/tmrl/scripts/run_camera_ready_pipeline.py#L119)
  * [`root(self)`](file:///D:/Project/tmrl/scripts/run_camera_ready_pipeline.py#L125)
  * [`source_log(self)`](file:///D:/Project/tmrl/scripts/run_camera_ready_pipeline.py#L128)
  * [`source_dir(self)`](file:///D:/Project/tmrl/scripts/run_camera_ready_pipeline.py#L131)
  * [`step_log(self)`](file:///D:/Project/tmrl/scripts/run_camera_ready_pipeline.py#L134)
  * [`done(self)`](file:///D:/Project/tmrl/scripts/run_camera_ready_pipeline.py#L137)
  * [`matches(self, cmd)`](file:///D:/Project/tmrl/scripts/run_camera_ready_pipeline.py#L153)
  * [`alive(self)`](file:///D:/Project/tmrl/scripts/run_camera_ready_pipeline.py#L161)
  * [`failed(self)`](file:///D:/Project/tmrl/scripts/run_camera_ready_pipeline.py#L172)
  * [`running(self)`](file:///D:/Project/tmrl/scripts/run_camera_ready_pipeline.py#L177)
  * [`has_leftovers(self)`](file:///D:/Project/tmrl/scripts/run_camera_ready_pipeline.py#L181)
  * [`archive_failure(self)`](file:///D:/Project/tmrl/scripts/run_camera_ready_pipeline.py#L188)
  * [`launch(self)`](file:///D:/Project/tmrl/scripts/run_camera_ready_pipeline.py#L211)

#### Functions

* **[`commit_headroom_gb()`](file:///D:/Project/tmrl/scripts/run_camera_ready_pipeline.py#L74)**
* **[`free_vram_gb()`](file:///D:/Project/tmrl/scripts/run_camera_ready_pipeline.py#L81)**
* **[`log(message)`](file:///D:/Project/tmrl/scripts/run_camera_ready_pipeline.py#L91)**
* **[`read_status(root)`](file:///D:/Project/tmrl/scripts/run_camera_ready_pipeline.py#L98)**
* **[`value_after(cmd, flag)`](file:///D:/Project/tmrl/scripts/run_camera_ready_pipeline.py#L106)**
* **[`build_jobs()`](file:///D:/Project/tmrl/scripts/run_camera_ready_pipeline.py#L241)**
* **[`adopt_running(jobs)`](file:///D:/Project/tmrl/scripts/run_camera_ready_pipeline.py#L270)**
* **[`verify_budgets(root, expected)`](file:///D:/Project/tmrl/scripts/run_camera_ready_pipeline.py#L287)**
* **[`write_status(stage, jobs, headroom, vram)`](file:///D:/Project/tmrl/scripts/run_camera_ready_pipeline.py#L297)**
* **[`main()`](file:///D:/Project/tmrl/scripts/run_camera_ready_pipeline.py#L308)**


---

### [`scripts\run_confirmatory_study.py`](file:///D:/Project/tmrl/scripts/run_confirmatory_study.py)

**Purpose:** Confirmatory Study Parallel Batch Orchestrator.

#### Functions

* **[`get_job_list(algorithms, seeds, output_dir, expected_protocol_sha256, skip_completed)`](file:///D:/Project/tmrl/scripts/run_confirmatory_study.py#L29)**
* **[`run_batch(jobs, max_parallel, num_threads, iterations, config_path, output_dir, smoke_test)`](file:///D:/Project/tmrl/scripts/run_confirmatory_study.py#L77)**
* **[`main()`](file:///D:/Project/tmrl/scripts/run_confirmatory_study.py#L203)**


---

### [`scripts\run_reproducible_benchmark.py`](file:///D:/Project/tmrl/scripts/run_reproducible_benchmark.py)

**Purpose:** Unified runner script to execute the complete reproducible fault-adaptation benchmark.

#### Functions

* **[`run_cmd(cmd, cwd)`](file:///D:/Project/tmrl/scripts/run_reproducible_benchmark.py#L15)**
* **[`main()`](file:///D:/Project/tmrl/scripts/run_reproducible_benchmark.py#L23)**


---

### [`scripts\smoke_continual_jax_gpu.py`](file:///D:/Project/tmrl/scripts/smoke_continual_jax_gpu.py)

**Purpose:** Compile and execute one real-replay plus imagination update on JAX GPU.

#### Functions

* **[`_spaces(image_size)`](file:///D:/Project/tmrl/scripts/smoke_continual_jax_gpu.py#L29)**
* **[`_batch(batch_size, sequence_length, image_size)`](file:///D:/Project/tmrl/scripts/smoke_continual_jax_gpu.py#L43)**
* **[`_parameters(module)`](file:///D:/Project/tmrl/scripts/smoke_continual_jax_gpu.py#L72)**
* **[`_changed(before, module)`](file:///D:/Project/tmrl/scripts/smoke_continual_jax_gpu.py#L79)**
* **[`_write_json_atomic(path, payload)`](file:///D:/Project/tmrl/scripts/smoke_continual_jax_gpu.py#L84)**
* **[`run_smoke(image_size, require_gpu)`](file:///D:/Project/tmrl/scripts/smoke_continual_jax_gpu.py#L107)**
* **[`main(argv)`](file:///D:/Project/tmrl/scripts/smoke_continual_jax_gpu.py#L170)**


---

### [`scripts\test_straight.py`](file:///D:/Project/tmrl/scripts/test_straight.py)

**Role:** Source module in `scripts`.


---

### [`scripts\test_track_b_forward.py`](file:///D:/Project/tmrl/scripts/test_track_b_forward.py)

**Role:** Source module in `scripts`.


---

### [`scripts\train_50k_curriculum.py`](file:///D:/Project/tmrl/scripts/train_50k_curriculum.py)

**Purpose:** Chunked 50k Curriculum Foundation Model Trainer for TrackMania.

#### Classes

* **[`ChunkedReplayBuffer`](file:///D:/Project/tmrl/scripts/train_50k_curriculum.py#L113)**
  * [`__init__(self, window_len, device)`](file:///D:/Project/tmrl/scripts/train_50k_curriculum.py#L114)
  * [`clear(self)`](file:///D:/Project/tmrl/scripts/train_50k_curriculum.py#L126)
  * [`clear_vram(self)`](file:///D:/Project/tmrl/scripts/train_50k_curriculum.py#L130)
  * [`load_chunk_files(self, file_paths, cache_file, max_workers)`](file:///D:/Project/tmrl/scripts/train_50k_curriculum.py#L137) — *Loads and parses a discrete chunk of 50,000 files with instant disk caching.*
  * [`add_historical_reservoir(self, output_dir, current_chunk, total_historical)`](file:///D:/Project/tmrl/scripts/train_50k_curriculum.py#L190) — *Blends a balanced sample of past replays from disk caches so the AI never forgets.*
  * [`stage_to_vram(self, stride, max_windows)`](file:///D:/Project/tmrl/scripts/train_50k_curriculum.py#L218) — *Stages up to 280k sliding windows (~1.4 GB) directly onto GPU VRAM for rock-solid stability.*

#### Functions

* **[`parse_gbx_to_arrays(content)`](file:///D:/Project/tmrl/scripts/train_50k_curriculum.py#L59)**: Fast-parses raw .Replay.Gbx bytes into 20 Hz state and action arrays.
* **[`main()`](file:///D:/Project/tmrl/scripts/train_50k_curriculum.py#L262)**


---

### [`scripts\train_dreamer_track_b.py`](file:///D:/Project/tmrl/scripts/train_dreamer_track_b.py)

**Purpose:** Orchestration and Real-Time World Record Tracking for Dreamer on Track B (Summer 2020 - 01)

#### Classes

* **[`DreamerTrainingManager`](file:///D:/Project/tmrl/scripts/train_dreamer_track_b.py#L86)**
  * [`__init__(self, target_wr, max_training_minutes)`](file:///D:/Project/tmrl/scripts/train_dreamer_track_b.py#L87)
  * [`log_status(self, msg)`](file:///D:/Project/tmrl/scripts/train_dreamer_track_b.py#L101)
  * [`start_processes(self)`](file:///D:/Project/tmrl/scripts/train_dreamer_track_b.py#L107)
  * [`_stream_logs(self, proc, prefix)`](file:///D:/Project/tmrl/scripts/train_dreamer_track_b.py#L150)
  * [`_stream_worker(self, proc)`](file:///D:/Project/tmrl/scripts/train_dreamer_track_b.py#L159)
  * [`_record_worker_line(self, line)`](file:///D:/Project/tmrl/scripts/train_dreamer_track_b.py#L170)
  * [`_save_progress(self)`](file:///D:/Project/tmrl/scripts/train_dreamer_track_b.py#L186)
  * [`check_milestone(self, lap_time)`](file:///D:/Project/tmrl/scripts/train_dreamer_track_b.py#L203)
  * [`run(self)`](file:///D:/Project/tmrl/scripts/train_dreamer_track_b.py#L237)
  * [`shutdown(self)`](file:///D:/Project/tmrl/scripts/train_dreamer_track_b.py#L263)

#### Functions

* **[`get_latest_replay(start_epoch)`](file:///D:/Project/tmrl/scripts/train_dreamer_track_b.py#L58)**
* **[`main()`](file:///D:/Project/tmrl/scripts/train_dreamer_track_b.py#L280)**


---

### [`scripts\train_multitrack_vision_curriculum.py`](file:///D:/Project/tmrl/scripts/train_multitrack_vision_curriculum.py)

**Purpose:** Offline Visual Robustness Fine-Tuner.

#### Classes

* **[`MultiTrackVisionSynthesizer`](file:///D:/Project/tmrl/scripts/train_multitrack_vision_curriculum.py#L52)**: Applies appearance perturbations to recorded camera observations.
  * [`__init__(self, device)`](file:///D:/Project/tmrl/scripts/train_multitrack_vision_curriculum.py#L61)
  * [`augment_batch(self, imgs, domain)`](file:///D:/Project/tmrl/scripts/train_multitrack_vision_curriculum.py#L71) — *imgs: (B, T, 4, 96, 96) in [0.0, 1.0]*
* **[`CandidateCheckpoints`](file:///D:/Project/tmrl/scripts/train_multitrack_vision_curriculum.py#L419)**: Clean held-out objective selects best; the last update is saved separately.
  * [`__init__(self, path)`](file:///D:/Project/tmrl/scripts/train_multitrack_vision_curriculum.py#L421)
  * [`consider(self, model, step, metrics)`](file:///D:/Project/tmrl/scripts/train_multitrack_vision_curriculum.py#L427)
  * [`save_latest(self, model, step, snapshot)`](file:///D:/Project/tmrl/scripts/train_multitrack_vision_curriculum.py#L439)

#### Functions

* **[`extract_causal_sequence_dataset(demo_file, window_len, stride, split, return_mask)`](file:///D:/Project/tmrl/scripts/train_multitrack_vision_curriculum.py#L141)**: Extracts rolling causal windows of images, states, previous actions, and target actions.
* **[`select_episode_split(episodes, split)`](file:///D:/Project/tmrl/scripts/train_multitrack_vision_curriculum.py#L256)**: Split before extracting windows; windows from a lap cannot cross splits.
* **[`save_candidate(model, output_path)`](file:///D:/Project/tmrl/scripts/train_multitrack_vision_curriculum.py#L273)**: Never hot-reload untested training weights into the active driver.
* **[`optimizer_parameter_groups(model, lr_vision, lr_backbone)`](file:///D:/Project/tmrl/scripts/train_multitrack_vision_curriculum.py#L290)**: Fail closed if a new trainable parameter has no deliberate LR assignment.
* **[`action_loss(predictions, targets, mask)`](file:///D:/Project/tmrl/scripts/train_multitrack_vision_curriculum.py#L315)**: Per-valid-token mean Huber over actions plus 2.5 times steering Huber.
* **[`trigger_logit_loss(logits, targets, mask)`](file:///D:/Project/tmrl/scripts/train_multitrack_vision_curriculum.py#L325)**: Signed trigger pressure is (tanh(z)+1)/2 == sigmoid(2*z).
* **[`evaluate_windows(model, data, device, batch_size, image_mode, seed, zero_previous_actions)`](file:///D:/Project/tmrl/scripts/train_multitrack_vision_curriculum.py#L340)**: All held-out windows, teacher-forced actions, no random augmentation.
* **[`file_sha256(path)`](file:///D:/Project/tmrl/scripts/train_multitrack_vision_curriculum.py#L403)**
* **[`write_metrics(path, values)`](file:///D:/Project/tmrl/scripts/train_multitrack_vision_curriculum.py#L411)**
* **[`gradient_group_norms(groups)`](file:///D:/Project/tmrl/scripts/train_multitrack_vision_curriculum.py#L445)**
* **[`parse_args(argv)`](file:///D:/Project/tmrl/scripts/train_multitrack_vision_curriculum.py#L459)**
* **[`main()`](file:///D:/Project/tmrl/scripts/train_multitrack_vision_curriculum.py#L507)**


---

### [`scripts\train_reactive_robust_baseline.py`](file:///D:/Project/tmrl/scripts/train_reactive_robust_baseline.py)

**Purpose:** Independent Reactive Robust Baseline (MLP REDQ-SAC without Transformer/History).

#### Classes

* **[`MLPGaussianActor(nn.Module)`](file:///D:/Project/tmrl/scripts/train_reactive_robust_baseline.py#L55)**
  * [`__init__(self, obs_dim, act_dim, hidden_dim)`](file:///D:/Project/tmrl/scripts/train_reactive_robust_baseline.py#L56)
  * [`forward(self, x, deterministic)`](file:///D:/Project/tmrl/scripts/train_reactive_robust_baseline.py#L67)
* **[`MLPCritic(nn.Module)`](file:///D:/Project/tmrl/scripts/train_reactive_robust_baseline.py#L86)**
  * [`__init__(self, obs_dim, act_dim, hidden_dim, use_layernorm)`](file:///D:/Project/tmrl/scripts/train_reactive_robust_baseline.py#L87)
  * [`forward(self, obs, act)`](file:///D:/Project/tmrl/scripts/train_reactive_robust_baseline.py#L103)
* **[`ReactiveREDQAgent(nn.Module)`](file:///D:/Project/tmrl/scripts/train_reactive_robust_baseline.py#L107)**
  * [`__init__(self, obs_dim, act_dim, num_critics, subset_m, use_layernorm)`](file:///D:/Project/tmrl/scripts/train_reactive_robust_baseline.py#L108)
  * [`act(self, obs, deterministic, device)`](file:///D:/Project/tmrl/scripts/train_reactive_robust_baseline.py#L135)
* **[`TransitionReplayBuffer`](file:///D:/Project/tmrl/scripts/train_reactive_robust_baseline.py#L142)**
  * [`__init__(self, capacity, obs_dim, act_dim)`](file:///D:/Project/tmrl/scripts/train_reactive_robust_baseline.py#L143)
  * [`add(self, obs, act, rew, next_obs, done)`](file:///D:/Project/tmrl/scripts/train_reactive_robust_baseline.py#L154)
  * [`sample(self, batch_size, device)`](file:///D:/Project/tmrl/scripts/train_reactive_robust_baseline.py#L164)

#### Functions

* **[`evaluate_reactive_policy(agent, split, episodes, device, base_seed)`](file:///D:/Project/tmrl/scripts/train_reactive_robust_baseline.py#L175)**
* **[`train_reactive_baseline(seed, iterations, collection_episodes_per_iter, gradient_steps_per_iter, utd, batch_size, use_layernorm, val_interval, val_episodes, test_episodes, base_eval_seed, num_threads, device, output_dir, config_path, smoke_test)`](file:///D:/Project/tmrl/scripts/train_reactive_robust_baseline.py#L206)**


---

### [`scripts\train_reactive_source_recipe_v1.py`](file:///D:/Project/tmrl/scripts/train_reactive_source_recipe_v1.py)

**Purpose:** Independent Reactive Robust Baseline (MLP REDQ-SAC without Transformer/History).

#### Classes

* **[`MLPGaussianActor(nn.Module)`](file:///D:/Project/tmrl/scripts/train_reactive_source_recipe_v1.py#L57)**
  * [`__init__(self, obs_dim, act_dim, hidden_dim)`](file:///D:/Project/tmrl/scripts/train_reactive_source_recipe_v1.py#L58)
  * [`forward(self, x, deterministic)`](file:///D:/Project/tmrl/scripts/train_reactive_source_recipe_v1.py#L69)
* **[`MLPCritic(nn.Module)`](file:///D:/Project/tmrl/scripts/train_reactive_source_recipe_v1.py#L88)**
  * [`__init__(self, obs_dim, act_dim, hidden_dim, use_layernorm)`](file:///D:/Project/tmrl/scripts/train_reactive_source_recipe_v1.py#L89)
  * [`forward(self, obs, act)`](file:///D:/Project/tmrl/scripts/train_reactive_source_recipe_v1.py#L105)
* **[`ReactiveREDQAgent(nn.Module)`](file:///D:/Project/tmrl/scripts/train_reactive_source_recipe_v1.py#L109)**
  * [`__init__(self, obs_dim, act_dim, num_critics, subset_m, use_layernorm)`](file:///D:/Project/tmrl/scripts/train_reactive_source_recipe_v1.py#L110)
  * [`act(self, obs, deterministic, device)`](file:///D:/Project/tmrl/scripts/train_reactive_source_recipe_v1.py#L137)
* **[`TransitionReplayBuffer`](file:///D:/Project/tmrl/scripts/train_reactive_source_recipe_v1.py#L144)**
  * [`__init__(self, capacity, obs_dim, act_dim)`](file:///D:/Project/tmrl/scripts/train_reactive_source_recipe_v1.py#L145)
  * [`add(self, obs, act, rew, next_obs, done)`](file:///D:/Project/tmrl/scripts/train_reactive_source_recipe_v1.py#L156)
  * [`sample(self, batch_size, device)`](file:///D:/Project/tmrl/scripts/train_reactive_source_recipe_v1.py#L166)

#### Functions

* **[`evaluate_reactive_policy(agent, split, episodes, device, base_seed)`](file:///D:/Project/tmrl/scripts/train_reactive_source_recipe_v1.py#L177)**
* **[`train_reactive_baseline(seed, iterations, collection_episodes_per_iter, gradient_steps_per_iter, utd, batch_size, use_layernorm, val_interval, val_episodes, test_episodes, base_eval_seed, num_threads, device, output_dir, config_path, smoke_test)`](file:///D:/Project/tmrl/scripts/train_reactive_source_recipe_v1.py#L208)**


---

### [`scripts\train_streaming_continual.py`](file:///D:/Project/tmrl/scripts/train_streaming_continual.py)

**Purpose:** SOTA High-VRAM Streaming Continual Pretraining Engine for TrackMania Foundation Model.

#### Classes

* **[`HighVRAMContinualBuffer`](file:///D:/Project/tmrl/scripts/train_streaming_continual.py#L109)**: Pre-stages window tensors directly in GPU VRAM for maximum bandwidth.
  * [`__init__(self, window_len, device)`](file:///D:/Project/tmrl/scripts/train_streaming_continual.py#L112)
  * [`ingest_new_gbx_files(self, gbx_files, max_batch, max_workers)`](file:///D:/Project/tmrl/scripts/train_streaming_continual.py#L127)
  * [`save_cache(self, cache_file)`](file:///D:/Project/tmrl/scripts/train_streaming_continual.py#L163) — *Saves parsed episodes and file registry to disk for instant restart.*
  * [`load_cache(self, cache_file)`](file:///D:/Project/tmrl/scripts/train_streaming_continual.py#L174) — *Restores parsed buffer in < 1 second.*
  * [`clear_vram(self)`](file:///D:/Project/tmrl/scripts/train_streaming_continual.py#L189) — *Frees GPU memory between epochs to allow clean cycling of next batch.*
  * [`stage_to_vram(self, stride, max_vram_gb)`](file:///D:/Project/tmrl/scripts/train_streaming_continual.py#L197) — *Constructs and stages tensors directly onto GPU VRAM up to a hard max ceiling.*

#### Functions

* **[`parse_gbx_to_arrays(content)`](file:///D:/Project/tmrl/scripts/train_streaming_continual.py#L55)**: Fast-parses raw .Replay.Gbx bytes into 20 Hz state and action arrays.
* **[`main()`](file:///D:/Project/tmrl/scripts/train_streaming_continual.py#L249)**


---

### [`scripts\train_transformer_fault_adaptation.py`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py)

**Purpose:** Sequence-Level Transformer RL^2 Agent for In-Context Fault Adaptation.

#### Classes

* **[`TimestepEmbedder(nn.Module)`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py#L63)**
  * [`__init__(self, in_dim, d_model)`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py#L64)
  * [`forward(self, x)`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py#L73)
* **[`CausalSelfAttention(nn.Module)`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py#L77)**
  * [`__init__(self, d_model, n_heads, dropout)`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py#L78)
  * [`forward(self, x, block_history_for_last_token)`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py#L91)
* **[`TransformerBlock(nn.Module)`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py#L121)**
  * [`__init__(self, d_model, n_heads, d_ff, dropout)`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py#L122)
  * [`forward(self, x, block_history_for_last_token)`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py#L134)
* **[`TransformerTrunk(nn.Module)`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py#L143)**
  * [`__init__(self, in_dim, d_model, n_layers, n_heads, d_ff, max_len, dropout)`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py#L144)
  * [`forward(self, x, block_history_for_last_token)`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py#L164)
* **[`TaskEncoder(nn.Module)`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py#L179)**
  * [`__init__(self, d_model, z_dim, num_classes)`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py#L180)
  * [`forward(self, h)`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py#L185)
* **[`GaussianActorHead(nn.Module)`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py#L191)**
  * [`__init__(self, in_dim, act_dim, hidden_dim)`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py#L192)
  * [`forward(self, x, deterministic)`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py#L203)
* **[`CriticHead(nn.Module)`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py#L223)**
  * [`__init__(self, in_dim, hidden_dim, use_layernorm)`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py#L224)
  * [`forward(self, x)`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py#L240)
* **[`TransformerRL2Agent(nn.Module)`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py#L244)**
  * [`__init__(self, obs_dim, act_dim, d_model, z_dim, num_critics, subset_m, use_layernorm, lambda_task, context_window)`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py#L245)
  * [`reset_context(self)`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py#L313)
  * [`step_in_context(self, obs, adaptive, history_blocked, deterministic, device)`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py#L316)
* **[`SequenceReplayBuffer`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py#L347)**
  * [`__init__(self, max_transitions, max_seq_len, burn_in)`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py#L348)
  * [`add_episode(self, obs, acts, rews, dones, fault_id)`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py#L356)
  * [`sample_batch(self, batch_size, device)`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py#L375)

#### Functions

* **[`evaluate_policy_on_split(agent, split, episodes, adaptive, mode, device, base_seed)`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py#L439)**: Evaluates the frozen agent on a specific fault split without parameter updates.
* **[`train_transformer_agent(seed, iterations, collection_episodes_per_iter, gradient_steps_per_iter, utd, batch_size, burn_in, train_seq_len, lambda_task, use_layernorm, val_interval, val_episodes, test_episodes, base_eval_seed, num_threads, device, output_dir, config_path, smoke_test)`](file:///D:/Project/tmrl/scripts/train_transformer_fault_adaptation.py#L493)**


---

### [`scripts\train_transformer_source_recipe_v1.py`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py)

**Purpose:** Sequence-Level Transformer RL^2 Agent for In-Context Fault Adaptation.

#### Classes

* **[`TimestepEmbedder(nn.Module)`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py#L67)**
  * [`__init__(self, in_dim, d_model)`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py#L68)
  * [`forward(self, x)`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py#L77)
* **[`CausalSelfAttention(nn.Module)`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py#L81)**
  * [`__init__(self, d_model, n_heads, dropout)`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py#L82)
  * [`forward(self, x)`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py#L95)
* **[`TransformerBlock(nn.Module)`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py#L112)**
  * [`__init__(self, d_model, n_heads, d_ff, dropout)`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py#L113)
  * [`forward(self, x)`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py#L125)
* **[`TransformerTrunk(nn.Module)`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py#L131)**
  * [`__init__(self, in_dim, d_model, n_layers, n_heads, d_ff, max_len, dropout)`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py#L132)
  * [`forward(self, x)`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py#L152)
* **[`TaskEncoder(nn.Module)`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py#L160)**
  * [`__init__(self, d_model, z_dim, num_classes)`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py#L161)
  * [`forward(self, h)`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py#L166)
* **[`GaussianActorHead(nn.Module)`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py#L172)**
  * [`__init__(self, in_dim, act_dim, hidden_dim)`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py#L173)
  * [`forward(self, x, deterministic)`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py#L184)
* **[`CriticHead(nn.Module)`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py#L204)**
  * [`__init__(self, in_dim, hidden_dim, use_layernorm)`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py#L205)
  * [`forward(self, x)`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py#L221)
* **[`TransformerRL2Agent(nn.Module)`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py#L225)**
  * [`__init__(self, obs_dim, act_dim, d_model, z_dim, num_critics, subset_m, use_layernorm, lambda_task)`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py#L226)
  * [`reset_context(self)`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py#L281)
  * [`step_in_context(self, obs, adaptive, deterministic, device)`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py#L284)
* **[`SequenceReplayBuffer`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py#L311)**
  * [`__init__(self, max_transitions, max_seq_len, burn_in)`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py#L312)
  * [`add_episode(self, obs, acts, rews, dones, fault_id)`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py#L320)
  * [`sample_batch(self, batch_size, device)`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py#L339)

#### Functions

* **[`evaluate_policy_on_split(agent, split, episodes, adaptive, device, base_seed)`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py#L385)**: Evaluates the frozen agent on a specific fault split without parameter updates.
* **[`train_transformer_agent(seed, iterations, collection_episodes_per_iter, gradient_steps_per_iter, utd, batch_size, burn_in, train_seq_len, lambda_task, use_layernorm, val_interval, val_episodes, test_episodes, base_eval_seed, num_threads, device, output_dir, config_path, smoke_test)`](file:///D:/Project/tmrl/scripts/train_transformer_source_recipe_v1.py#L427)**


---

### [`scripts\train_vision_online.py`](file:///D:/Project/tmrl/scripts/train_vision_online.py)

**Purpose:** Real-Time Online Vision Training with 1,000,000 Foundation Backbone for TrackMania 2020.

#### Classes

* **[`CompactReplayBuffer`](file:///D:/Project/tmrl/scripts/train_vision_online.py#L111)**: Stores transitions in CPU RAM with uint8 images (<150 MB for 10,000 steps).
  * [`__init__(self, capacity)`](file:///D:/Project/tmrl/scripts/train_vision_online.py#L113)
  * [`push(self, img_uint8, state, action, reward)`](file:///D:/Project/tmrl/scripts/train_vision_online.py#L121)
  * [`sample_batch(self, batch_size, seq_len, device)`](file:///D:/Project/tmrl/scripts/train_vision_online.py#L128)
  * [`__len__(self)`](file:///D:/Project/tmrl/scripts/train_vision_online.py#L158)

#### Functions

* **[`send_control(act, speed, steer_gain, deadband)`](file:///D:/Project/tmrl/scripts/train_vision_online.py#L46)**
* **[`release_controls()`](file:///D:/Project/tmrl/scripts/train_vision_online.py#L95)**
* **[`main()`](file:///D:/Project/tmrl/scripts/train_vision_online.py#L163)**


---

### [`scripts\verify_adaptation_pipeline.py`](file:///D:/Project/tmrl/scripts/verify_adaptation_pipeline.py)

**Purpose:** Verify that training updates actively alter the deployed policy and worker actions.

#### Functions

* **[`tensor_state_digest(state_dict, prefixes)`](file:///D:/Project/tmrl/scripts/verify_adaptation_pipeline.py#L33)**
* **[`make_dummy_observation()`](file:///D:/Project/tmrl/scripts/verify_adaptation_pipeline.py#L42)**
* **[`main()`](file:///D:/Project/tmrl/scripts/verify_adaptation_pipeline.py#L52)**


---
