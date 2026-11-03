# Paper 193: summary of camera-ready changes

History-Conditioned SAC under Sensor Faults: A Pilot Study of Adaptation and Retention. Felix Surjodinoto, Keandre Rafael, Delroy, Samuel Philip (Bina Nusantara University).

We thank both reviewers. The camera-ready version adds:
- an extended experiment with eight source seeds per controller and task (previously three);
- a five-times-longer continued-training budget;
- a clean-continuation control;
- a second task, Ant-v5;
- pre-specified significance tests;
- a critic-level analysis of replay mixing;
- inference-latency measurements;
- a public code and data release.

The statistical analysis was fixed in protocol files before the new runs. The new runs reproduce every branch of the earlier pilot exactly. Section, table and figure numbers refer to the camera-ready PDF (15 pages).

## Reviewer 4

| Comment | Change | Where |
|---|---|---|
| Benchmark diversity (Humanoid-v5, Ant-v5) | Ant-v5 added with the same design: 8 source seeds per controller, 5 branches each, 20,000 steps (80 branches). The 27-dimensional observation excludes contact forces. Collection is step-budgeted because Ant episodes terminate early. Under the same source recipe, every Ant source learned to stand rather than walk (mean forward velocity within ±0.007 m/s), and the faults cost the sources little. We report this rather than tune the recipe after the fact. Humanoid-v5 was not run, for two reasons. SAC's reported Humanoid learning curves span millions of steps, far beyond the fixed 400,000-step source recipe. Our workstation was also at its memory limit. Both reasons are stated as limitations. | Sec. 2.1, 3.3, 4; Tables 3–5; Fig. 2 |
| Off-policy bias and soft Bellman target distortion from mixing clean and corrupted transitions | New critic diagnostic (Sec. 2.5) explaining both distortions. For every branch, the critics are evaluated on held-out clean and corrupted trajectories, at the source checkpoint and after 20,000 steps. We report the held-out soft Bellman residual (computed with the training target and the saved temperature) and a value-bias gap. The gap is the Monte-Carlo value bias on corrupted data minus that on clean data of the same checkpoint, which cancels the soft-value offset. **Finding (HalfCheetah, delay):** the source critics overvalued delayed histories relative to clean ones (+171 and +268). Training on delayed data alone shrank the gap to +27 and +114; with replay it stayed at +89 and +204, a paired effect of +62 [16, 109] and +90 [41, 139], with 7 of 8 seeds positive in each. Replay also raised the reactive controller's residual on corrupted data under both faults, with all 8 seeds agreeing. This is the one condition where replay cost adaptation. Under noise, and throughout Ant, the gaps stayed small. The diagnostic lies outside the pre-specified analysis, and the gap is labelled exploratory. | Sec. 2.5, 3.4, 4; Table 5 |
| Inference latency for 100–500 Hz control | Per-action latency of the deployment call at batch size one with real model inputs (simulation excluded), 10,000 timed steps each, on CPU with 1 and 4 threads and on GPU. On one CPU thread the transformer averages 0.58 ms (99th percentile 0.66 ms, maximum 1.56 ms), so every timed call fits the 2 ms period of 500 Hz control. The reactive controller averages 0.09 ms. The GPU is slower at batch size one (1.64 ms mean). | Sec. 2.6, 3.5; Table 6 |
| Reproducibility (seed scripts, delay/noise wrappers, checkpoints) | Public release containing the fault wrappers, source-training recipes, study runner, frozen protocols, launch and analysis scripts, per-branch evaluation records, and source and adapted checkpoints. Exactness is checked twice. The 24 HalfCheetah branches that share seeds with the pilot reproduce its returns exactly (largest difference 0.000). Nine branches rerun after a resource failure match the first run bit for bit, final checkpoints included. All 160 branches ran exactly 20,000 steps and 5,000 updates. | Code availability; Sec. 2.6, 3.1 |

## Reviewer 5

| Comment | Change | Where |
|---|---|---|
| More setup detail: transformer architecture, history conditioning, noise and delay implementation | Table 1 lists components and parameter counts. Sec. 2.2 describes the token, context handling, embedding, causal blocks, feature head and the reset of history at each episode. Sec. 2.1 gives the noise model, the delay queue, the wrapper's step logic and the input vector. Table 2 gives the full continued-training configuration, and the source recipe is stated in numbers (Sec. 2.2). | Sec. 2.1–2.3; Tables 1–2 |
| More seeds | 8 source seeds per controller and task (previously 3). | Sec. 2.3; Table 2 |
| Longer budgets | 20,000 steps and 5,000 updates per branch, with 7 evaluation budgets (previously 4,000 steps). | Table 2; Fig. 2 |
| More environments | Ant-v5 added (see Reviewer 4). | Sec. 3.3 |
| Significance testing | Pre-specified exact Wilcoxon signed-rank tests with Holm correction, 95% t intervals, seed-bootstrap intervals and d_z. The source seed is the replication unit. With eight seeds, the Holm-adjusted floor is 0.0625 in the eight-test family and 0.031 in the four-test family. We state this rather than change the correction after the runs. | Sec. 2.4; Tables 3–4 |
| Clearer explanation of transformer vs reactive differences | New discussion paragraph. It first sets out what each controller can observe under each fault, then compares this with the results: under noise they agree, under delay they contradict it. It then gives four candidate explanations, each stated as untested, and the ablations needed to separate them. | Sec. 4 |

## Other changes

- **Clean-continuation control.** This control separates fault-specific forgetting from changes caused by continued optimization alone. In HalfCheetah, the sources kept improving under clean training (+782 and +963), so every fault branch without replay fell behind the control. In Ant, clean training did not improve the standing sources.
- **Pilot result revised.** The three-seed pilot showed a small positive transformer gain under delay. With eight seeds this gain does not hold: −87 at 4,000 steps, and negative through 20,000 steps. The paper reports this explicitly.
- **Ant warm-start transient.** Every Ant arm, including the clean controls, lost return in the first 2,000 steps and then recovered partly or fully. Because the clean controls dipped too, the transient comes from restarting training, not from the faults. This is reported in Sec. 3.3.
- **Front matter.** Authors and affiliation, a disclosure-of-interests statement and an updated AI-usage declaration have been added.
