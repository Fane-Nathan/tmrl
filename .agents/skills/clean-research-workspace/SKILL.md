---
name: clean-research-workspace
description: >-
  Identifies and purges temporary prototyping directories (e.g., `scratch/`), unreferenced one-off test scripts, and ephemeral caches, while safeguarding core research algorithms, source recipes, and camera-ready publication assets.
---

# Clean Research Workspace

## Overview
This skill safely tidies the research workspace by identifying and removing obsolete prototyping scratchpads, temporary test scripts, and build caches, while strictly protecting all core research code, replication protocols, training recipes, and publication assets.

## Dependencies
None. Uses standard agent filesystem inspection and deletion commands.

## Quick Start
Trigger this skill whenever you want to tidy the repository for release, archive old exploratory scripts, or clean up disk space:

> *"Clean up the workspace and remove all scratch and one-off testing files"*
> *"Purge obsolete prototype scripts but keep all research algorithms and paper assets"*
> *"Tidy this repository to production / publication standard"*

---

## Protected Core vs. Junk Classifications

### 1. Protected Core (NEVER Delete)
Before taking any action, ensure the following directories and files are shielded:
* **Research Suite (`experiments/sensor_adaptation_v1/`):**
  - All learner models (`learner.py`, `frozen_models.py`, `replay.py`, `sensor_env.py`)
  - Experiment runners & protocols (`run_study.py`, `protocol_extension.json`, `protocol_ant.json`)
  - Statistical analysis & diagnostics (`analyze_extension.py`, `analyze_replay_bias.py`, `benchmark_latency.py`)
* **Pipeline Orchestrators (`scripts/`):**
  - `train_transformer_source_recipe_v1.py`
  - `train_reactive_source_recipe_v1.py`
  - `run_camera_ready_pipeline.py`
  - `build_camera_ready_assets.py`
  - `prepare_camera_ready_release.py`
  - `benchmark_fault_env.py`
  - `confirmatory_logging.py`
* **Base Package (`tmrl/`):** Complete core library.
* **Camera-Ready & Experimental Data:**
  - `paper/` (`manuscript.pdf`, LaTeX files, figures, slides, notes, reporting)
  - `output/sensor_adaptation/` (evaluation trajectories, latency JSON, budget verification)
  - `fault_benchmark_results/` & `fault_benchmark_results_ant/`
* **Repository Meta:** `.git/`, `pyproject.toml`, `setup.py`, `README.md`, `requirements.txt`.

For the complete itemized allowlist, consult [references/protected_allowlist.md](references/protected_allowlist.md).

---

### 2. Targeted for Immediate Deletion (Junk)
The following paths and patterns are classified as junk:
* **Prototyping Folders:**
  - `scratch/` (entire directory and all contents)
* **One-Off & Standalone Verification Scripts:**
  - Root one-off tests: `test_*.py` (e.g., `test_straight.py`)
  - Standalone exploratory/prototype scripts in `scripts/`:
    - `scripts/smoke_continual_jax_gpu.py`
    - `scripts/test_straight.py`
    - `scripts/test_track_b_forward.py`
    - `scripts/cycle_map_macro.py`
    - `scripts/diagnose_visual_policy.py`
    - `scripts/train_50k_curriculum.py`
    - `scripts/train_dreamer_track_b.py`
    - `scripts/train_multitrack_vision_curriculum.py`
    - `scripts/train_streaming_continual.py`
    - `scripts/train_vision_online.py`
    - `scripts/audit_driving_dataset.py`
    - `scripts/audit_foundation_deployment.py`
    - `scripts/build_reward_from_demos.py`
    - `scripts/build_trajectory_assist.py`
    - `scripts/evaluate_live_driving.py`
    - `scripts/eval_in_context_learning.py`
    - `scripts/export_live_driving_demos.py`
    - `scripts/live_drive_car_brain.py`
    - `scripts/record_track_demonstration.py`
* **Caches & Temporary Artifacts:**
  - `__pycache__/`, `*.pyc`, `*.tmp`, `*.bak`, `.pytest_cache/`

---

## Workflow

### 1. Workspace Audit
Scan the repository and identify all candidate files matching the junk patterns.

### 2. Allowlist Verification
Cross-reference every candidate path against [references/protected_allowlist.md](references/protected_allowlist.md). If any candidate matches a protected path, immediately drop it from the deletion set.

### 3. Immediate Deletion
Execute removal using agent filesystem tools or shell commands:
* Remove `scratch/` recursively.
* Remove verified one-off scripts in `scripts/` and root.
* Clean up orphan `__pycache__` and temporary files.

### 4. Summary & Verification
Print a concise summary table of deleted files, confirming that all protected research assets remain intact and intact git status.

---

## Common Mistakes
1. **Blind Wildcard Deletion:** Running `rm scripts/*.py` or `rm test_*.py` without checking whether a test script is part of the research contracts (`experiments/sensor_adaptation_v1/test_contracts.py` is protected!).
2. **Deleting Source Model Checkpoints:** Deleting `fault_benchmark_results/` or `fault_benchmark_results_ant/` which are required for replicating evaluation results.
3. **Deleting Build Pipeline Helpers:** Deleting `scripts/build_camera_ready_assets.py` or `scripts/prepare_camera_ready_release.py`.

