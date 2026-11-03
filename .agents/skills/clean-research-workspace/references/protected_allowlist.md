# Protected Research Allowlist Reference

Every file and directory in this reference must **NEVER be deleted** during workspace cleanup operations.

---

## 1. Research Implementation (`experiments/sensor_adaptation_v1/`)
* `sensor_env.py`: Environment fault wrappers (Gaussian noise, temporal delay FIFO queue).
* `learner.py`: Soft Actor-Critic (SAC) implementation with Transformer (32 tokens) and Reactive backbones.
* `replay.py`: The replay buffer supporting retained clean replay mixing ($\rho=0.0$ vs $\rho=0.5$).
* `frozen_models.py`: Architecture definitions and deterministic checkpoint loader.
* `run_study.py`: Core study runner executing adaptation branches, checkpoints, and evaluations.
* `protocol.json`: Original pilot protocol.
* `protocol_extension.json`: Camera-ready extension protocol (HalfCheetah-v5, 8 seeds).
* `protocol_ant.json`: Camera-ready extension protocol (Ant-v5, 8 seeds).
* `model_provenance.json`: Checkpoint hashes and origin metadata.
* `test_contracts.py`: Research integrity test contracts.
* `analyze_extension.py`: Pre-registered statistical analysis (Wilcoxon, Holm, $d_z$, bootstrap).
* `analyze_replay_bias.py`: Critic held-out Bellman error and value-bias gap diagnostics.
* `analyze_results.py`: Pilot analysis script.
* `benchmark_latency.py`: 10,000-step per-action deployment inference latency benchmark.
* `build_paper_assets.py`: Pilot asset generator.
* `README.md`: Research module documentation.

---

## 2. Production Scripts & Orchestrators (`scripts/`)
* `train_transformer_source_recipe_v1.py`: Source model training recipe for Transformer.
* `train_reactive_source_recipe_v1.py`: Source model training recipe for Reactive.
* `run_camera_ready_pipeline.py`: Master unattended memory-gated pipeline runner.
* `run_ant_sources.py`: Ant-v5 memory-gated pretraining runner.
* `build_camera_ready_assets.py`: Automatic data extraction, LaTeX number macros (`\R{...}`), tables, and vector curves.
* `prepare_camera_ready_release.py`: Code and data release packager.
* `benchmark_fault_env.py`: Benchmark verification helper.
* `confirmatory_logging.py`: Checkpoint hashing and audit logger.
* `add_trackmania_paper_figure.py`: TrackMania paper figure helper.
* `build_camera_ready_workflow_figure.py`: Workflow figure builder.

---

## 3. Publication Deliverables & Experimental Records
* `output/icodit_2026/camera_ready/*`: Complete camera-ready manuscript, LaTeX sources, rebuttal (`RESPONSE_TO_REVIEWERS.md`), and figures.
* `output/sensor_adaptation/extension_20261031_v1/*`: HalfCheetah 80-branch results, metrics, trajectories.
* `output/sensor_adaptation/ant_20261031_v1/*`: Ant-v5 80-branch results, metrics, trajectories.
* `output/sensor_adaptation/budget_verification.json`: Exact step/update verification proof.
* `output/sensor_adaptation/latency_20261031.json`: Inference latency benchmark numbers.
* `fault_benchmark_results/*`: Pretrained HalfCheetah source checkpoints.
* `fault_benchmark_results_ant/*`: Pretrained Ant-v5 source checkpoints.

---

## 4. Framework & Configuration
* `tmrl/*`: Complete base TrackMania RL package.
* `codebase_overview/*`: Hierarchical repository documentation tree.
* `pyproject.toml`, `setup.py`, `README.md`, `LICENSE`, `requirements.txt`.

