# Sensor adaptation and retention experiment

This isolated experiment tests the author's intended **observation** faults and continued-training retention. It preserves the prior actuator benchmark, source checkpoints, and deployed TrackMania models.

## Run

From this directory, using the research Python environment:

```powershell
& 'D:\Project\tmrl\.venv\Scripts\python.exe' test_contracts.py
& 'D:\Project\tmrl\.venv\Scripts\python.exe' -u run_study.py --output 'D:\Project\tmrl\output\sensor_adaptation\UNUSED_RUN_NAME' --smoke
& 'D:\Project\tmrl\.venv\Scripts\python.exe' -u run_study.py --output 'D:\Project\tmrl\output\sensor_adaptation\ANOTHER_UNUSED_RUN_NAME'
& 'D:\Project\tmrl\.venv\Scripts\python.exe' analyze_results.py 'D:\Project\tmrl\output\sensor_adaptation\ANOTHER_UNUSED_RUN_NAME'
```

Output directories must be new. The runner refuses to overwrite an experiment. Each run freezes its protocol, exact source files, input checkpoints, hashes, and package/hardware versions before collecting results. A full pilot contains 24 branches: three pretrained seeds, two policy types, two sensor faults, and two retained-replay fractions.

## What is implemented

- Incorrect observations: independently seeded Gaussian sensor noise. Delayed observations: a three-step FIFO sensor queue. Applied control commands and the base reward function are unchanged.
- Previous command and reward are appended **after** the sensor transform and retain their current timing. True observations and fault labels go only to audit logs.
- A clean-condition evaluation before and after learning on the sensor condition, using the exact same post-training checkpoint for both conditions.
- Paired fine-tuning with 0% versus 50% fixed old-condition replay, matched in new-environment interactions and update counts.
- A 32-token transformer context in both collection and replay, with exact rolling successor histories, short-context support, and final transitions included. Each sequence trains one final valid transition.
- Every evaluation trajectory, checkpoint hash, training log, and collected replay saved for verification.

## Method choices and limits

`protocol.json` specifies the full pilot. Its source models were trained previously on actuator perturbations; this is warm-start adaptation, not clean-only training from scratch. The fixed old replay is collected from those frozen models on clean observations. Both replay arms incur the same old-data collection; only the 50% arm uses that pool for updates.

The checkpoint-compatible model definitions in `frozen_models.py` are extracted without class changes from the old source; `model_provenance.json` records origin hashes. The new learner initializes optimizer states and entropy temperature afresh, uses one critic-ensemble update per actor update, disables dropout consistently, and uses no auxiliary fault-class labels. The reactive controller receives the same packed token but has no long history. These are controller-level comparisons, not capacity-matched architecture ablations.

Four thousand new-environment steps, three training seeds, and three evaluation episodes per condition are an engineering pilot. Report adaptation curves and old-condition performance changes honestly. A higher new-condition return with lower old-condition performance is a tradeoff, not proof that forgetting is solved. There is no declared recovery-time threshold or retention tolerance. The pilot does not establish arbitrary new-map transfer or diagnose the cause of the earlier SAC training plateau.

The small `--smoke` run verifies execution, not learning quality. Main results are generated only after all branches exit successfully and saved artifacts are read back.

## Camera-ready extension (October 2026)

The camera-ready study keeps the pilot's code path and adds seeds, budget, a control and a second task. Its analysis was fixed in the protocol files before any extension run.

| Protocol | Task | Sources | Branches per source | Budget |
|---|---|---|---|---|
| `protocol_extension.json` (sensor-extension-2.0) | HalfCheetah-v5 | seeds 42-49 per controller, `fault_benchmark_results/` | noise and delay with retained-replay fraction 0 or 0.5, plus a clean-continuation control (rho = 0) | 20,000 steps, 5,000 updates, evaluations at 0/2k/4k/8k/12k/16k/20k |
| `protocol_ant.json` (sensor-ant-2.0) | Ant-v5, 27-dim observation (contact forces excluded) | seeds 42-49 per controller, `fault_benchmark_results_ant/` | same | same; collection is step-budgeted because Ant episodes can terminate early |

Seeds 42-44 of the HalfCheetah extension use the pilot's branch and evaluation seeds; `analyze_extension.py --pilot` compares their returns with the pilot exactly.

### Source models

HalfCheetah sources were trained in September 2026. Ant sources use the same trainers, preserved as `scripts/train_transformer_source_recipe_v1.py` and `scripts/train_reactive_source_recipe_v1.py`. These are git `54224e48f` versions whose file hashes match the ones in `model_provenance.json`; the only change is an `--env` option. Their recorded run configurations equal the HalfCheetah sources' apart from the environment label. From the repository root:

```powershell
python -u scripts/train_transformer_source_recipe_v1.py --env Ant-v5 --seed 42 --num_threads 2 --output_dir fault_benchmark_results_ant
python -u scripts/train_reactive_source_recipe_v1.py --env Ant-v5 --seed 42 --num_threads 2 --output_dir fault_benchmark_results_ant
```

### Study, analyses and paper assets

From this directory (shards can run in parallel; sharding does not change any branch):

```powershell
python -u run_study.py --protocol protocol_extension.json --algorithms transformer --seeds 42 43 44 --output <ROOT>\transformer_a
python analyze_extension.py <ROOT> --protocol protocol_extension.json --pilot <PILOT_ROOT>
python analyze_replay_bias.py <ROOT> --protocol protocol_extension.json
python benchmark_latency.py --trajectories <PILOT_ROOT> --out latency.json
```

`analyze_extension.py` refuses to analyse unless every shard is COMPLETE and the root contains exactly the protocol's seeds, conditions and controllers. `scripts/build_camera_ready_assets.py` turns the saved analysis files into the paper's figures, tables and `\R{...}` number macros. `scripts/run_camera_ready_pipeline.py` is the unattended scheduler used for the camera-ready runs. It gates launches on the Windows commit charge, adopts running jobs, retries a failed job once and moves failed outputs to `failed_attempts/`.

### Statistics

The replication unit is the source seed (n = 8). The primary family has eight exact two-sided Wilcoxon signed-rank tests per task with Holm correction. With n = 8 the smallest exact p is 0.0078, so no Holm-adjusted p in that family can fall below 0.0625. The four-test fault-specific forgetting family can reach 0.031. Interval estimates and sign counts are reported alongside every test.
