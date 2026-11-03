# History-Conditioned SAC under Sensor Faults: A Pilot Study of Adaptation and Retention

[![Conference: ICoDiT 2026](https://img.shields.io/badge/Conference-ICoDiT%202026-001E61.svg)](https://socs.binus.ac.id/icodit)
[![Paper ID: 193](https://img.shields.io/badge/Paper%20ID-193-0284C7.svg)]()
[![Status: Accepted](https://img.shields.io/badge/Status-Accepted%20(Oral)-10B981.svg)]()
[![Publisher: Springer CCIS](https://img.shields.io/badge/Publisher-Springer%20Nature%20CCIS-E11D48.svg)]()
[![Indexing: Scopus](https://img.shields.io/badge/Indexing-Scopus-F59E0B.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-gray.svg)](LICENSE)

Official replication repository for **ICoDiT 2026 Paper #193**:  
*"History-Conditioned SAC under Sensor Faults: A Pilot Study of Adaptation and Retention"*.

**Authors:** Felix Surjodinoto\*, Keandre Rafael, Delroy, Samuel Philip  
**Affiliation:** School of Computer Science, Bina Nusantara University (BINUS), Jakarta, Indonesia  
**Conference:** The 1st International Conference on Computing and Digital Technology (ICoDiT 2026)

---

## 📌 Overview & Key Contributions

When continuous-control robotic agents encounter in-field sensor faults (e.g. uncalibrated Gaussian noise or multi-step observation delays), standard reactive policies exhibit catastrophic performance collapse. This repository implements and evaluates:

1. **History-Conditioned Soft Actor-Critic (SAC-HC):** Incorporates a causal transformer history encoder over previous observation-action tuples.
2. **Sensor Fault Wrappers:** Clean, modular Gym wrappers for stochastic Gaussian sensor noise, FIFO observation delay queues, and sensor dropout.
3. **Adaptation vs. Retention Trade-off:** Empirical evaluation across 160 branches (8 source seeds × 5 training conditions) on **HalfCheetah-v5** and **Ant-v5**.
4. **Replay Buffer Transition Mixing Diagnostic:** Formal diagnostic of soft Bellman target distortions and value bias gaps under mixed clean/corrupted data.
5. **Real-Time 500 Hz Latency Verification:** Sub-millisecond (0.58 ms mean CPU latency) inference latency profile ensuring deployment feasibility.

---

## 📂 Repository File Tree (Core Files Only)

This repository is strictly organized to contain only the 20 core replication files:

```text
├── sensor_env.py                        # Gym environment wrappers (Gaussian noise, delay buffer, dropout)
├── learner.py                           # History-Conditioned SAC (SAC-HC) training & update loop
├── frozen_models.py                     # Causal Transformer history encoder & twin Q-critics
├── replay.py                            # Replay buffer managing clean vs. fault transition mixing
├── train_transformer_source_recipe.py   # Source policy training recipe for SAC-HC (400k steps)
├── train_reactive_source_recipe.py      # Source policy training recipe for reactive baseline
├── run_study.py                         # Multi-seed continued-training & fault adaptation runner
├── benchmark_latency.py                 # 500 Hz inference latency benchmark script (<2 ms threshold)
├── latency_20261031.json                # Empirical latency dataset (10,000 timed per-step measurements)
├── protocol.json                        # Pre-registered HalfCheetah 8-seed experiment protocol
├── protocol_ant.json                    # Pre-registered Ant-v5 8-seed experiment protocol
├── protocol_extension.json              # Pre-registered extended budget protocol
├── model_provenance.json                # Pre-trained checkpoint hashes & metadata
├── analyze_results.py                   # Outcome evaluation and return metrics extraction
├── analyze_extension.py                 # Statistical testing (Wilcoxon signed-rank + Holm correction, d_z)
├── analyze_replay_bias.py               # Soft Bellman target residual and value bias gap analysis
├── requirements.txt                     # Minimal Python dependencies
├── .gitignore                           # Git ignore rules
├── LICENSE                              # MIT License
└── README.md                            # Documentation and quickstart instructions
```

---

## 🚀 Quickstart & Installation

### 1. Requirements

* Python 3.10+
* PyTorch >= 2.2.0
* Gymnasium with MuJoCo (`gymnasium[mujoco] >= 0.29.1`)

```bash
git clone https://github.com/Fane-Nathan/tmrl.git
cd tmrl
pip install -r requirements.txt
```

---

## 🔬 Reproducing Paper Results

### Step 1: Verify 500 Hz Inference Latency
Verify that per-action deployment latency satisfies the 2.0 ms deadline for 500 Hz real-time robotics:
```bash
python benchmark_latency.py
```
*(Expected: ~0.58 ms mean per-step latency on a single CPU thread, ~0.09 ms for reactive control).*

### Step 2: Train Source Policies
Train clean source controllers across pre-registered seeds:
```bash
# Train Transformer SAC-HC source policy
python train_transformer_source_recipe.py --seed 42 --task halfcheetah

# Train Reactive SAC baseline source policy
python train_reactive_source_recipe.py --seed 42 --task halfcheetah
```

### Step 3: Run Fault Adaptation & Retention Study
Run continued training under sensor faults (e.g. observation delay or Gaussian noise) with and without replay buffer mixing:
```bash
# HalfCheetah-v5 continued-training study
python run_study.py --protocol protocol.json --task halfcheetah

# Ant-v5 continued-training study
python run_study.py --protocol protocol_ant.json --task ant
```

### Step 4: Extract Results & Statistical Tests
Compute evaluation returns, exact Wilcoxon signed-rank test statistics with Holm-Bonferroni correction, and critic diagnostic metrics:
```bash
# 1. Performance outcomes table
python analyze_results.py

# 2. Hypothesis testing & effect sizes (Tables 3 & 4 in paper)
python analyze_extension.py

# 3. Soft Bellman residual & value bias diagnostic (Table 5 in paper)
python analyze_replay_bias.py
```

---

## 📊 Citation & Attribution

If you use this code or study design in your research, please cite our ICoDiT 2026 paper:

```bibtex
@inproceedings{surjodinoto2026history,
  author    = {Surjodinoto, Felix and Rafael, Keandre and Delroy and Philip, Samuel},
  title     = {History-Conditioned SAC under Sensor Faults: A Pilot Study of Adaptation and Retention},
  booktitle = {Proceedings of The 1st International Conference on Computing and Digital Technology (ICoDiT 2026)},
  series    = {Communications in Computer and Information Science (CCIS)},
  publisher = {Springer Nature},
  year      = {2026},
  note      = {Scopus Indexed}
}
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
