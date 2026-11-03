# Workspace Architecture & Codebase Overview

> **Generated Documentation Catalog** for `D:\Project\tmrl` across all directory depths.
> **Total Source Files Analyzed:** 172

---

## Subsystem Navigation Directory

This repository contains two major interconnected codebases: the **TrackMania Reinforcement Learning (`tmrl`)** system and the **ICoDiT 2026 Continual Sensor-Adaptation Research Suite**.

### [TrackMania RL Core Framework (`tmrl/`)](tmrl_core/overview.md) (78 files)
Networking, actor-critic architectures, Gym environments, memory buffers, and TrackMania interface wrappers.

👉 **[Explore TrackMania RL Core Framework (`tmrl/`) Documentation &raquo;](tmrl_core/overview.md)**

### [Sensor Adaptation Research Suite (`experiments/`)](sensor_adaptation_study/overview.md) (16 files)
ICoDiT 2026 Camera-Ready experimental protocols, MuJoCo sensor fault wrappers, SAC learners, and statistical analysis scripts.

👉 **[Explore Sensor Adaptation Research Suite (`experiments/`) Documentation &raquo;](sensor_adaptation_study/overview.md)**

### [Execution Scripts & Build Pipelines (`scripts/`)](scripts_and_tools/overview.md) (46 files)
Source training recipes, automated unattended pipelines, latency benchmarking, and camera-ready LaTeX asset builders.

👉 **[Explore Execution Scripts & Build Pipelines (`scripts/`) Documentation &raquo;](scripts_and_tools/overview.md)**

### [Diagnostic & Scratch Utilities (`scratch/`)](scratch_and_experiments/overview.md) (5 files)
Prototyping scripts, vision encoder checks, and empirical validation tools.

👉 **[Explore Diagnostic & Scratch Utilities (`scratch/`) Documentation &raquo;](scratch_and_experiments/overview.md)**

### [Top-Level Root Modules](top_level/overview.md) (27 files)
Root package configuration, entry points, and setup scripts.

👉 **[Explore Top-Level Root Modules Documentation &raquo;](top_level/overview.md)**

---
## High-Level Architecture

```mermaid
graph TD
    subgraph TMRL ['TrackMania RL Core (tmrl/)']
        GymEnv['Gym Environments<br/>(TrackMania / Real-Time)'] --> Actor['Actor Client<br/>(Live Rollouts)']
        Actor --> Buffer['Memory Buffers<br/>(Replay / Samples)']
        Buffer --> Server['Training Server<br/>(Networked SAC)']
    end
    subgraph RESEARCH ['ICoDiT Sensor Adaptation Research']
        FaultWrap['MuJoCo Fault Wrappers<br/>(Noise & Delay 3-step)'] --> SACLearner['History-Conditioned SAC<br/>(Transformer & Reactive)']
        SACLearner --> ReplayMix['Replay Retention Buffer<br/>(rho=0.0 vs rho=0.5)']
        ReplayMix --> Analysis['Statistical Verification<br/>(Wilcoxon, Holm, d_z)']
    end
    subgraph PIPELINE ['Automated Pipelines (scripts/)']
        Pipeline['Master Pipeline Runner<br/>(Memory-Gated Scheduler)'] --> CameraReady['Camera-Ready Asset Builder<br/>(LaTeX Tables, Curves & Manuscript)']
    end
    RESEARCH --> PIPELINE
```

---
## File Tree Summary

### `top_level/`

- [`docs\source\conf.py`](file:///D:/Project/tmrl/docs/source/conf.py)
- [`setup.py`](file:///D:/Project/tmrl/setup.py)
- [`test_output_brain\reliability_20260912_v3\candidate.model.py`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.model.py)
- [`test_output_brain\reliability_20260912_v3\candidate.trainer.py`](file:///D:/Project/tmrl/test_output_brain/reliability_20260912_v3/candidate.trainer.py)
- [`test_output_brain\visual_policy_1k_20260911_v2\candidate.model.py`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.model.py)
- [`test_output_brain\visual_policy_1k_20260911_v2\candidate.trainer.py`](file:///D:/Project/tmrl/test_output_brain/visual_policy_1k_20260911_v2/candidate.trainer.py)
- [`tests\test_camera_preprocessing.py`](file:///D:/Project/tmrl/tests/test_camera_preprocessing.py)
- [`tests\test_confirmatory_fault_benchmark.py`](file:///D:/Project/tmrl/tests/test_confirmatory_fault_benchmark.py)
- [`tests\test_driving_data.py`](file:///D:/Project/tmrl/tests/test_driving_data.py)
- [`tests\test_driving_safety.py`](file:///D:/Project/tmrl/tests/test_driving_safety.py)
- [`tests\test_foundation_observation_contract.py`](file:///D:/Project/tmrl/tests/test_foundation_observation_contract.py)
- [`tests\test_gamepad_action_contract.py`](file:///D:/Project/tmrl/tests/test_gamepad_action_contract.py)
- [`tests\test_unseen_map_evaluation.py`](file:///D:/Project/tmrl/tests/test_unseen_map_evaluation.py)
- [`tests\test_visual_training.py`](file:///D:/Project/tmrl/tests/test_visual_training.py)
- [`tmp\finalize_sensor_pack.py`](file:///D:/Project/tmrl/tmp/finalize_sensor_pack.py)
- *...and 12 more files (see [top_level/overview.md](top_level/overview.md))*

### `sensor_adaptation_study/`

- [`experiments\sensor_adaptation_v1\analyze_extension.py`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/analyze_extension.py)
- [`experiments\sensor_adaptation_v1\analyze_replay_bias.py`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/analyze_replay_bias.py)
- [`experiments\sensor_adaptation_v1\analyze_results.py`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/analyze_results.py)
- [`experiments\sensor_adaptation_v1\benchmark_latency.py`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/benchmark_latency.py)
- [`experiments\sensor_adaptation_v1\build_paper_assets.py`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/build_paper_assets.py)
- [`experiments\sensor_adaptation_v1\frozen_models.py`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/frozen_models.py)
- [`experiments\sensor_adaptation_v1\learner.py`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/learner.py)
- [`experiments\sensor_adaptation_v1\prepare_review_release.py`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/prepare_review_release.py)
- [`experiments\sensor_adaptation_v1\replay.py`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/replay.py)
- [`experiments\sensor_adaptation_v1\run_study.py`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/run_study.py)
- [`experiments\sensor_adaptation_v1\sensor_env.py`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/sensor_env.py)
- [`experiments\sensor_adaptation_v1\share_wandb_review.py`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/share_wandb_review.py)
- [`experiments\sensor_adaptation_v1\test_contracts.py`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/test_contracts.py)
- [`experiments\sensor_adaptation_v1\upload_wandb.py`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/upload_wandb.py)
- [`experiments\sensor_adaptation_v1\verify_review_release.py`](file:///D:/Project/tmrl/experiments/sensor_adaptation_v1/verify_review_release.py)
- *...and 1 more files (see [sensor_adaptation_study/overview.md](sensor_adaptation_study/overview.md))*

### `scratch_and_experiments/`

- [`scratch\build_workspace_overview.py`](file:///D:/Project/tmrl/scratch/build_workspace_overview.py)
- [`scratch\collect_experiment_data.py`](file:///D:/Project/tmrl/scratch/collect_experiment_data.py)
- [`scratch\plot_live_experiments.py`](file:///D:/Project/tmrl/scratch/plot_live_experiments.py)
- [`scratch\test_pure_vision_training.py`](file:///D:/Project/tmrl/scratch/test_pure_vision_training.py)
- [`scratch\test_timing_benchmark.py`](file:///D:/Project/tmrl/scratch/test_timing_benchmark.py)

### `scripts_and_tools/`

- [`scripts\add_trackmania_paper_figure.py`](file:///D:/Project/tmrl/scripts/add_trackmania_paper_figure.py)
- [`scripts\audit_driving_dataset.py`](file:///D:/Project/tmrl/scripts/audit_driving_dataset.py)
- [`scripts\audit_foundation_deployment.py`](file:///D:/Project/tmrl/scripts/audit_foundation_deployment.py)
- [`scripts\benchmark_fault_env.py`](file:///D:/Project/tmrl/scripts/benchmark_fault_env.py)
- [`scripts\benchmark_live_pipeline.py`](file:///D:/Project/tmrl/scripts/benchmark_live_pipeline.py)
- [`scripts\build_camera_ready_assets.py`](file:///D:/Project/tmrl/scripts/build_camera_ready_assets.py)
- [`scripts\build_camera_ready_workflow_figure.py`](file:///D:/Project/tmrl/scripts/build_camera_ready_workflow_figure.py)
- [`scripts\build_reward_from_demos.py`](file:///D:/Project/tmrl/scripts/build_reward_from_demos.py)
- [`scripts\build_trajectory_assist.py`](file:///D:/Project/tmrl/scripts/build_trajectory_assist.py)
- [`scripts\confirmatory_logging.py`](file:///D:/Project/tmrl/scripts/confirmatory_logging.py)
- [`scripts\cycle_map_macro.py`](file:///D:/Project/tmrl/scripts/cycle_map_macro.py)
- [`scripts\data\bulk_tmnf_scraper.py`](file:///D:/Project/tmrl/scripts/data/bulk_tmnf_scraper.py)
- [`scripts\data\convert_raw_gbx_to_dataset.py`](file:///D:/Project/tmrl/scripts/data/convert_raw_gbx_to_dataset.py)
- [`scripts\data\download_1k_diverse_maps.py`](file:///D:/Project/tmrl/scripts/data/download_1k_diverse_maps.py)
- [`scripts\data\ultra_fast_1m_scraper.py`](file:///D:/Project/tmrl/scripts/data/ultra_fast_1m_scraper.py)
- *...and 31 more files (see [scripts_and_tools/overview.md](scripts_and_tools/overview.md))*

### `tmrl_core/`

- [`tmrl\__init__.py`](file:///D:/Project/tmrl/tmrl/__init__.py)
- [`tmrl\__main__.py`](file:///D:/Project/tmrl/tmrl/__main__.py)
- [`tmrl\config\__init__.py`](file:///D:/Project/tmrl/tmrl/config/__init__.py)
- [`tmrl\config\config_constants.py`](file:///D:/Project/tmrl/tmrl/config/config_constants.py)
- [`tmrl\config\config_objects.py`](file:///D:/Project/tmrl/tmrl/config/config_objects.py)
- [`tmrl\core\__init__.py`](file:///D:/Project/tmrl/tmrl/core/__init__.py)
- [`tmrl\core\actor.py`](file:///D:/Project/tmrl/tmrl/core/actor.py)
- [`tmrl\core\envs.py`](file:///D:/Project/tmrl/tmrl/core/envs.py)
- [`tmrl\core\jax\__init__.py`](file:///D:/Project/tmrl/tmrl/core/jax/__init__.py)
- [`tmrl\core\jax\actor.py`](file:///D:/Project/tmrl/tmrl/core/jax/actor.py)
- [`tmrl\core\jax\training_offline.py`](file:///D:/Project/tmrl/tmrl/core/jax/training_offline.py)
- [`tmrl\core\jax\util.py`](file:///D:/Project/tmrl/tmrl/core/jax/util.py)
- [`tmrl\core\memory.py`](file:///D:/Project/tmrl/tmrl/core/memory.py)
- [`tmrl\core\networking.py`](file:///D:/Project/tmrl/tmrl/core/networking.py)
- [`tmrl\core\torch\__init__.py`](file:///D:/Project/tmrl/tmrl/core/torch/__init__.py)
- *...and 63 more files (see [tmrl_core/overview.md](tmrl_core/overview.md))*
