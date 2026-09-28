> Historical preparation note: use [REVISION_STATUS.md](REVISION_STATUS.md) for the current sensor-focused manuscript and [CMT_FIELDS_AND_DECLARATIONS.md](CMT_FIELDS_AND_DECLARATIONS.md) for the matching submission fields. The earlier actuator results below remain a separate result set.

# Manuscript review for ICoDiT 2026

Reviewed: complete paper.pdf (8 pages), corresponding live Prism source, local confirmatory results, and the W&B reactive run. Page references below refer to the supplied PDF, before any reformatting.

## Highest-priority revisions

### 1. Reconcile the main claim with the later experiment

**Locations:** title and abstract (p. 1), contributions (pp. 1-2), Table II (p. 4), Figure 1 and results (p. 5), limitations (pp. 6-7), conclusion (p. 7).

The manuscript's positive cross-mechanism claim rests on three seeds and test-condition-selected checkpoints. Local files contain a later eight-seed study with validation-selected checkpoints and an independently trained reactive baseline. This later evidence needs to be addressed in the submission.

| Item | PDF's exploratory experiment | Later local experiment |
| --- | --- | --- |
| Training seeds | 3 | 8: 42-49 |
| Training budget | Up to 1,000 iterations | 100 iterations, 400,000 environment steps per run in the manifest |
| Checkpoint selection | Best performance on the held-out fault conditions | Best score on unseen parameters of training mechanisms only |
| Training faults | action_scale, action_noise | action_scale, action_noise |
| Test faults | Latency; pooled dead/sign-flip result | Latency; dead actuator and sign flip reported separately |
| Reactive baseline | Described as future work | 8 trained reactive runs available |
| Latency adaptation gap | +652.3 +/- 224.7 | -61.8, 95% CI [-231.3, +141.6] |

The later study does not establish a positive latency adaptation effect. It also does not prove that memory is universally harmful or that the protocols are equivalent. Different budgets and selection rules must remain visible in the paper.

Recommended revision strategy:

1. Treat the original test-selected result as exploratory evidence with an explicit selection-bias limitation.
2. Present the later evaluation as a separate protocol with its actual budget and validation split.
3. Report its reactive baseline and separate secondary fault conditions.
4. Make the title, abstract and conclusion reflect uncertainty and protocol sensitivity.
5. Update every repeated number and Figure 1 together; do not replace a number while retaining the old interpretation or caption.

The current title's claim that memory buys adaptation should not be retained as an unconditional result on the present evidence. A neutral title direction is **in-context reinforcement learning under actuator faults and real-time constraints**, with the final wording chosen after the authors settle the analysis.

### 2. Explain the checkpoint-selection problem plainly

**Location:** IV-A, p. 4. The paper explicitly says it evaluates held-out faults during training and retains the best held-out checkpoint per seed.

Those faults influenced model selection, so their final score is not an untouched test estimate. A fresh set of evaluation episodes does not remove the selection performed on the same fault conditions. The later protocol provides a cleaner split: training faults for learning, unseen parameters of those mechanisms for validation, and separate faults for final testing.

Retain exact provenance for both protocols, including checkpoint hashes and selection rules. Avoid describing a retrospectively written protocol as preregistered without independent evidence of when it was fixed.

### 3. Calibrate what the no-history ablation demonstrates

**Locations:** III-H (p. 4), IV-A (p. 5).

Resetting context each step measures the effect of that intervention on this trained policy. It also changes the input/context distribution seen by a history-trained agent. A positive gap alone does not uniquely establish online system identification or superiority over a trained reactive controller. The local reactive baseline is therefore directly relevant.

Additional history controls and adaptation-over-time curves would strengthen the interpretation, but should not be rushed into a final test set to chase a favorable result. Clearly separate evidence already collected from proposed future experiments.

## Independent verification of the later results

Recomputed from each seed's final_test_raw_episodes.csv: 8 transformer runs x 150 rows + 8 reactive runs x 75 rows = **1,800 episode records**. Each mode/condition has 25 episodes per seed. Adaptive and no-history episode IDs and environment seeds match. All 16 checkpoint file SHA-256 hashes match the recorded metadata and episode records.

The independent unit is the training seed, not each evaluation episode. The intervals below use 2,000 percentile-bootstrap resamples of eight seed-level gaps with RNG seed 12345, reproducing the existing summary to displayed precision. CSV returns are rounded to four decimal places.

| Condition | Adaptive mean | No-history mean | Reactive mean | G: adaptive minus no-history, 95% CI | D: adaptive minus reactive, 95% CI |
| --- | ---: | ---: | ---: | --- | --- |
| Latency (primary) | 469.2 | 531.0 | 447.3 | -61.8 [-231.3, 141.6] | 21.9 [-73.6, 111.2] |
| Dead actuator (secondary) | 1929.3 | 2059.4 | 2470.8 | -130.1 [-354.3, 58.9] | -541.4 [-997.6, -201.0] |
| Sign flip (secondary) | 1126.3 | 1289.7 | 1309.0 | -163.5 [-303.4, -24.7] | -182.8 [-431.6, 3.8] |

The latency gap is positive for 2/8 seeds; the repository labels it seed-sensitive/unstable. Secondary intervals are pointwise, with no multiplicity correction established in this audit. Do not present every secondary interval as a separately confirmed discovery.

This review checked stored numerical consistency and provenance links. It did not rerun training, validate all simulator/fault code, or establish that a metric is causally attributable to one implementation choice. Machine-readable checks are in evidence/independent_results_audit.json.

## TrackMania results

### Run B: direct matching evidence found

The [SAC_RLPD_BC_v1 overview](https://wandb.ai/trackmania-rl/trackmania-rl/runs/SAC_RLPD_BC_v1/overview) shows return_test = **279.7099914551** and episode_length_test = **1372**, matching p. 6. At the stated 20 Hz, 1372 steps correspond to **68.6 seconds of nominal control intervals**.

The overview records a start on 11 June 2026 at 21:54:53 (timezone not established from the displayed timestamp), runtime 10 h 58 m 18 s, code commit ee286f72c58a7d4a6e909a093542990deddd568d, and an NVIDIA GeForce RTX 5070 Laptop GPU. Its state is Killed; that does not invalidate metrics recorded before the run stopped.

This supports the stored return and step count. It does not by itself verify finish-event detection, full end-to-end wall-clock time since demonstration collection, the first completion time, or a multi-episode success rate. Attach the completion flag/replay, evaluation checkpoint and timestamps if available. Match the claimed peak to the appropriate history row rather than assuming a summary field is a maximum.

### Runs A/C and broader causal claims

**Locations:** IV-B (pp. 5-6), limitations and conclusion (pp. 6-7).

- Identify exact W&B run IDs and code revisions for Run A and each sequential Run C intervention.
- Freeze Run C's reported evaluation date and endpoint. Its actual status may have changed since the PDF dated 7 September.
- Explain that Run A had no demonstrations; the abstract/conclusion currently compress the story in a way that can imply both transformer runs used identical demonstrations. Only B/C share demonstrations.
- Replace categorical claims that replay alone decided the outcome or exonerated the architecture with observations supported by the runs. Algorithms, observation histories, critic counts, training budgets and sequential interventions differ.
- One run per configuration does not support general superiority claims. Preserve this limitation.
- Retain Neural ODE and RESeL as implementation details unless there are isolated component ablations supporting stronger claims.
- Name hardware, environment/software versions, training/evaluation time, map identifier, finish detection and demonstration provenance for reproducibility.

## Conference compliance revisions

| Location | Finding | Required correction |
| --- | --- | --- |
| p. 1 | Four names, institutional affiliations and emails | Remove from the review manuscript; enter separately in CMT |
| Entire document | IEEE two-column layout; source class IEEEtran | Migrate to the prescribed template and recheck pagination |
| pp. 3-7 | Methods and evaluation sections present, but no explicit Discussion | Use clear Introduction, Method, Results, Discussion sections |
| Before references, p. 7 | No AI Usage Declaration | Add accurate tool names and purposes; include implementation assistance |
| pp. 7-8 | 30 references, insufficient recent share | See REFERENCE_AUDIT.md |
| p. 5 Table III | Very small table text | Reflow in the final template; preserve readable font size |
| p. 8 | Only four bibliography entries on an otherwise mostly blank page | Recheck flow after the template change; do not change margins to force length |

All eight original PDF pages were rendered and inspected. The document is readable overall; the main problems are compliance and evidence rather than corrupt rendering. PDF author/title metadata fields were empty, but visible author details still break anonymity. The Creator field identifies OpenAI Prism; that alone does not establish what AI assistance was used.

## Writing and reporting checks

- Remove the assertion that three seeds is a universal minimum for a confidence interval (p. 6). Just state the sample size and its limitations.
- Specify the CI construction for the old three-seed result; do not silently apply the later bootstrap method to it.
- Distinguish critic-updates-per-actor-update from gradient-updates-per-environment-step. The manuscript uses UTD for the former; report both counts/ratios so readers can reproduce the budget.
- Verify the broad literature claim that prior work only tested unseen parameters. A novelty claim needs a current primary-source search, not just a familiar set of older citations.
- Verify metadata of all references against their primary publication/repository pages, including the exact RESeL title and authors and software citation versions. The present recency audit uses the years printed in the draft; it is not a completed bibliographic verification.
- Keep borrowed framework descriptions distinct from the authors' implementation contribution and cite TMRL/rtgym accurately.

## Files retained for the next revision

- source/complete-paper-original.tex: source backup read from the supplied Prism project; original research content retained.
- evidence/independent_results_audit.json: independent arithmetic and checkpoint-hash checks.
- REFERENCE_AUDIT.md: per-reference publication-year audit.
- AUTHOR_PACKET.md: author metadata, AI disclosure draft and organizer questions.

The live Prism manuscript has not been rewritten, and no revised PDF is represented as ready for submission. The main research claim and policy details need resolution before producing the final review version.
