# TMRL & Paper Replication Scripts Directory

This directory contains executable tooling for replicating the ICoDiT 2026 paper, running the sensor-fault benchmarks, and operating live TrackMania environments.

---

## 1. Camera-Ready & Paper Reporting Artifacts
* **`build_camera_ready_assets.py`**: Generates LaTeX macros (`results_macros.tex`), statistical tables, effect size computations, and critic diagnostic summaries.
* **`build_camera_ready_workflow_figure.py`**: Compiles the study workflow schematic and TrackMania development capture into `paper/trackmania_pilot_workflow.pdf` and `.png`.
* **`build_presentation_deck.py`**: Automates PowerPoint generation and PDF export for the official conference presentation deck (`193 - Presentation.pptx` / `.pdf`).
* **`build_rebuttal_letter.py`**: Populates the official IAIC Author Response / Rebuttal Letter template (`193 - Rebuttal Letter.docx` / `.pdf`).
* **`run_camera_ready_pipeline.py`**: End-to-end orchestration runner that validates seeds, executes analysis, updates macros, and verifies reproduction.
* **`prepare_camera_ready_release.py`**: Packages code, models, and protocols for the open reproducibility release package.

---

## 2. Research Pilot & Benchmark Experiments
* **`train_reactive_source_recipe_v1.py`**: Trains the nominal reactive SAC baseline policies (MLP, 400k steps).
* **`train_transformer_source_recipe_v1.py`**: Trains the nominal history-conditioned SAC baseline policies (GPT-style causal transformer, 400k steps).
* **`run_adaptation_benchmark.py`**: Executes the 5 continued-training branches (Clean, Noise, Noise+Replay, Delay, Delay+Replay).
* **`run_ant_sources.py`**: Orchestrates Ant-v5 source training across seeds.
* **`benchmark_fault_env.py`**: Smoke tests and unit verifies the Gymnasium sensor-fault observation wrappers.
* **`verify_adaptation_pipeline.py`**: Validates data schemas, step budgets, and evaluation logs.
* **`confirmatory_logging.py`**: Helper harness ensuring cryptographic reproducibility logging across branches.

---

## 3. TrackMania Live Control & Environment
* **`evaluate_live_driving.py`**: Real-time closed-loop evaluation harness on live TrackMania 2020 game instances.
* **`benchmark_live_pipeline.py`**: End-to-end wall-clock latency measurement for camera frame capture, preprocessing, and inference.
* **`finetune_target_track.py`**: Fine-tunes trained policies to specific map geometries.
* **`cycle_map_macro.py`** & **`map_cycle_state.json`**: Automated game UI map cycling macro.

---

## 4. Launchers & System Setup
* **`launch_all.bat`**: Windows batch script to launch the TMRL server, trainer, and roll-out worker simultaneously.
* **`run_server.bat`**, **`run_trainer.bat`**, **`run_worker.bat`**: Modular process launchers.
* **`run_map_cycler.bat`**: Launches the map cycling background task.
* **`run_tier4_vision_training.bat`**: Specialized launcher for vision-in-the-loop training.
* **`bootstrap_continual_jax_wsl2.sh`**: WSL2 environment bootstrapping script for accelerated training.
* **`configure_continual_jax_wsl_network.sh`**: Configures virtual ethernet bridges between Windows and WSL2.

---

## 5. Tooling Subdirectories
* **`dataset_tools/`**: Map scraping and track dataset conversion utilities.
* **`kaggle/`**: Cloud-based GPU training notebook integration scripts.
