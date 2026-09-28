# Driving recovery status - updated 12 September 2026

The revised **known-track trajectory controller completed three consecutive full-lap attempts**.
The learned foundation policy still fails in closed-loop driving. These are different results.

Latest update: the corrected visual candidate completed a **30-second, policy-only probe on the
user-loaded different map**, after repairing fullscreen capture and preprocessing latency.
This is not a completed lap or a one-shot pass. It still **fails the offline deployment gate**.
The active weights/config remain unchanged. Training and live evidence are separated below.

## Measured results

All successful runs were local, bounded evaluations on `tmrl-train`, with no trainer or replay server.
Completion means OpenPlanet's finish flag, not a training loss, reward estimate, or reference index.

| Run (UTC prefix) | Controller | Finish | Elapsed | Maximum distance from reference |
| --- | --- | --- | --- | --- |
| `20260911T010745_213858Z` | Existing foundation policy | No; no progress | 4.23 s | 10.67 m |
| `20260911T011438_186099Z` | Revised tracker, 22 m/s cap | Yes | 136.56 s | 3.13 m |
| `20260911T062920_480199Z` | Revised tracker, 22 m/s cap | Yes | 136.61 s | 2.91 m |
| `20260911T063227_948945Z` | Revised tracker, 20 m/s cap | Yes | 149.13 s | 3.06 m |

Raw JSON/NPZ evidence is in `test_output_brain/live_runs`. The first successful run predates
the detailed timing/camera provenance schema and is excluded from the exported training set.
Elapsed time is the evaluator wall clock, not the official race timer. There is no collision
sensor in these logs, so finishing does not establish a collision-free lap.

The reference includes post-finish movement: the successful runs finish near reference index
1048 of 1105. Reaching the last reference sample is deliberately not required for a pass.

The existing foundation passed a saved-frame audit (steering MAE 0.0323) but failed the live
test. Saved observations cannot measure the accumulating distribution shift of closed-loop driving.
The direct evaluator is paced at 20 Hz; it does not reproduce RTGym's asynchronous delay exactly.

## Implemented changes

- Lower-gain continuous trajectory steering with a more responsive motion-heading estimate.
  This uses privileged, map-specific reference positions; it is a teacher/baseline, not learned navigation.
- Latest prior-action input for the foundation encoder, with a regression test using distinct history entries.
- Fresh telemetry with bounded timeout, disconnect handling, and clean socket shutdown.
  Capture/telemetry failures release the last controls. Every evaluation releases controls in `finally`.
- DXcam capture-region rounding fix, retaining checks against window movement.
- Explicit evaluation logs containing observations, commands, timestamps, optional frames, finish reason,
  map-file identity checks, camera declaration, and controller configuration.
- Asynchronous action replacement in the regular RTGym interface is blocked. It could race replay
  serialization. `DRIVING_SAFETY.TRAJECTORY_ASSIST.ENABLED` is now false in the active config.
- Training holds out complete episodes before creating windows, includes the final driving window,
  and raises the share of launch windows to at least 25%. Checkpoints are candidate-only; automatic
  deployment is blocked. Relative optimizer learning rates remain proportional during decay.

## New training data

`data/verified_tracking_demos_20260911.pt` contains **5,681 aligned observation-command samples**:

- Training: one completed 2,716-step lap at a 22 m/s speed cap.
- Validation: one separate completed 2,965-step lap at a 20 m/s speed cap.
- Images: causal, episode-local `4 x 96 x 96` grayscale uint8 history.
- Labels: the actual commands sent after each observation, including continuous steering.
- Maximum observation intervals: 55.26 ms and 51.66 ms; nominal period 50 ms.
- Finish labels, timestamps, input hashes, camera declaration, and map-file SHA-256 are retained.

The export rejects incomplete runs, missing frames, non-finite/out-of-range actions, non-causal
timing, large sampling gaps, duplicate source archives, and unknown camera/map metadata.
Its sidecar is `data/verified_tracking_demos_20260911.metadata.json`.

Dataset SHA-256: `a6cf8546f2eea829b5203bc64a7bc5c9fa528c677ae41872a2444d314f655b89`.

The companion notebook `test_output_brain/driving_recovery_quality_20260911.ipynb` executed
top-to-bottom and reconstructed both episodes from the source logs, checking labels, frame history,
hashes and split isolation. This follows the data-quality skill's inspectable-evidence workflow.

Limitations: one known map, one user-declared camera, two teacher-generated runs, no recovery
demonstrations, and no independently captured live map UID. Diagnostic GPS/current-control fields
must remain masked from visual-policy training. These are not independent human demonstrations.

## Run the working known-track baseline

Use the same first-person camera, with `tmrl-train` open. From `D:\Project\tmrl`:

```powershell
& D:\miniconda3\envs\rcdream\python.exe scripts/evaluate_live_driving.py --mode tracking --steps 5000 --max-deviation 4.5 --record-frames --camera-label first-person
```

Do not enable the old asynchronous assist hook. The explicit evaluator logs the actual chosen
commands synchronously and stops on finish, departure, stall, error, or step budget.

## Verification and next step

29 offline regression tests passed, including six unseen-map isolation/stall checks. A two-gradient-step CPU smoke test loaded the actual new
dataset, trained, evaluated the held-out lap, and saved an isolated checkpoint under
`test_output_brain/training_smoke_20260911`. It is **only a plumbing test**, not a trained driving model.

The deployed foundation file was not replaced; its SHA-256 remains
`3af96c021affd17a111a3fdeda491c95bd66364821936d54d5926e2cfd263f03`.
No training or driving worker is left running.

Next: train an isolated continuous-steering candidate on the verified teacher data, compare it
on the held-out lap, then test it closed-loop without trajectory assistance. The evaluator accepts
`--mode foundation --foundation-weights PATH --continuous-actions` without replacing deployed weights.
The smoke-test checkpoint is not suitable for deployment.

Gen 1.5 / one-shot ability remains unestablished. It requires a defined held-out-map protocol,
many more synchronized driving demonstrations, and reliable recovery/control before broadening
the claim. Downloaded `.Map.Gbx` geometry and appearance augmentation are not additional labeled laps.

## Unseen-map probes - 11 September 2026

The user loaded a different map. Its observed start was approximately `[773.4, 58.0, 400.0]`,
different from `tmrl-train`. The current telemetry cannot identify the live map UID, so these
runs are labeled `user-loaded-unseen-map` with no invented map-file hash.

Two bounded probes used the unchanged foundation checkpoint and **no reference control,
reference scoring, or map-based reward**:

| Run (UTC prefix) | Decoder | Result |
| --- | --- | --- |
| `20260911T065210_016131Z` | Ternary | Barrier near start; low-displacement stop at 3.93 s; no finish |
| `20260911T065353_764414Z` | Continuous | Low-displacement stop at 17.72 s; no finish |

Both released controls and saved frames, telemetry, commands and timestamps under
`test_output_brain/live_runs`. No training or driving worker remains running. One trial per
decoder is not a statistical comparison, and displacement is not verified route progress.
The current policy has not demonstrated reliable driving on this new map.

The evaluator's explicit `--unseen-map` option is allowed only with `--mode foundation` and
caps probes at 30 seconds. It does not load the old trajectory in the evaluation path. The
environment's map identity check is replaced only on that test instance; the saved configuration
remains unchanged. Failure detection uses a two-second low-displacement window after a
three-second launch grace period, not distance from an unrelated reference.

For another bounded policy-only probe on a user-loaded map:

```powershell
& D:\miniconda3\envs\rcdream\python.exe scripts/evaluate_live_driving.py --mode foundation --unseen-map --steps 400 --continuous-actions --record-frames --camera-label first-person
```

Do not use `--mode tracking` with the old trajectory on this new map. A new demonstration is
needed for its own route-following baseline; improving the learned visual policy requires
candidate training and independent closed-loop validation, not reusing the old map's route.

## Visual-policy training beyond the smoke test - 11 September 2026

Assessment: **share with caveats; not ready for promotion or a Gen 1.5 claim**.
This experiment answers whether the verified demonstrations can improve the camera policy,
not whether it can finish a map or adapt from one demonstration.

### Trainer fixes and verification

- The optimizer omitted `ln_f.weight` and `ln_f.bias` (512 trainable values). They now belong
  to the conservative backbone group. Unknown, missing or duplicate group assignments fail.
- All vision/fusion/backbone groups had finite, nonzero gradients. The final LayerNorm now
  updates and its gradients are cleared by the optimizer.
- At least 25% of the training-window pool starts strictly at time zero; the final window is
  included. Short episodes are right-padded and excluded from loss/metrics by a validity mask;
  causal attention prevents valid tokens from attending to that future padding.
- Full clean held-out validation runs before training and every 250 successful updates.
  Best, latest and numbered checkpoints are separate; only clean validation selects best.
- AMP overflow attempts are counted separately and do not advance the update count or scheduler.
- Training seeds, data/checkpoint hashes, parameter-group membership, gradients, validation
  history and image ablations are saved in JSON. V2 also snapshots the exact trainer/model source.
- `MultiModalCarBrain` can optionally return pre-tanh logits for training without changing
  checkpoint keys or the default deployed forward path.

The first run exposed another concrete bug: FP16 `tanh` rounded throttle to exact endpoints
on 98.99% of held-out window tokens. On 67.88% of tokens that saturated throttle was wrong by
more than 0.5 in signed-action units, so the saturated action-loss derivative could not correct it.
V2 computes the training output tanh in FP32 and adds a 0.1-weight pre-tanh trigger loss:
`BCEWithLogits(2 * logits, (signed_trigger_target + 1) / 2)`.
This supplies a corrective gradient for confidently wrong gas/brake outputs. A regression test
reproduces a zero FP16 action-loss gradient and a nonzero correcting logit-loss gradient.

Final verification: all 41 offline regression tests passed, and the changed Python modules
compiled successfully. These software checks do not count as driving evaluations.

Both runs started from the same unchanged canonical checkpoint, used seed 42, CUDA, batch 8,
context 16, vision LR `1e-4`, fusion/head LR `3.5e-5`, backbone LR `1e-6`, and 1,000 successful
updates. Each had four skipped AMP attempts. These are two separate runs, not a 2,000-update
checkpoint. V1 took 127.2 s; V2 took 95.0 s including validation/diagnostics. Peak PyTorch
allocated training VRAM was about 286 MiB, excluding game/driver allocations.

### Measured results

The windowed comparison uses the same complete held-out lap: 186 windows, 2,976 valid window
tokens, including 11 repeated tokens from terminal-window overlap. It is teacher-forced and
uses no random appearance augmentation. All action errors below are on the signed [-1, 1] scale.

| Checkpoint | Steering MAE | Gas MAE | Weighted action/steer Huber |
| --- | ---: | ---: | ---: |
| Original canonical model | 0.549054 | 1.371929 | 1.864616 |
| V1: optimizer/validation fixes | 0.110155 | 1.372001 | 0.649134 |
| V2: saturation-resistant trigger supervision | 0.099124 | 0.754628 | 0.407003 |

For V2, blanking images raises steering MAE to 0.251094; swapping image windows raises it
to 0.256864. Zeroing previous actions gives 0.101759. These interventions support useful
image dependence on this dataset, not exclusive visual reasoning or cross-map competence.
An independent NumPy recomputation agrees with the trainer objective within `3e-8` for both runs.

V2's gas no longer hits exact AMP endpoints on this held-out set. At observed speeds >=20 m/s,
its mean physical throttle is 18.68%, down from V1's 99.86%; the held-out teacher used 0% there.
This is not a measured live speed reduction. The training teacher's cap was 22 m/s while the
held-out teacher's cap was 20 m/s, so command error partly reflects different controller targets.
The model is not explicitly conditioned on that cap. Matching a smoother throttle to binary
teacher commands is also not equivalent to measuring safe speed control.

The exact deployment path was then evaluated on **2,965 unique held-out frames**, continuous
actions, rolling context, and its **own predicted prior actions** (fixed recorded observations):

| Metric | Original | V2 | Offline gate |
| --- | ---: | ---: | ---: |
| Steering MAE | 0.612386 | 0.128562 | <=0.10 |
| Gas MAE | 1.372418 | 0.754688 | <=0.30 |
| Mean absolute steer on straight labels | 0.426504 | 0.152039 | <=0.10 |
| Turn-sign accuracy | 53.66% | 92.32% | >=90% |
| P95 CPU action latency | 17.22 ms | 10.45 ms | <=40 ms |

Both fail the offline gate. Timing is a local diagnostic under the observed system load,
not a controlled latency benchmark; V2 had no 50 ms misses in this saved-observation run.
The audit now includes gas/brake error gates and an explicit episode-split option, so strong
steering alone cannot pass a permanently accelerating policy. The older 0.0323 saved-frame
score used a different dataset/protocol and is not comparable to this new held-out experiment.

### Artifacts and reproduction

- V1 artifacts: `test_output_brain/visual_policy_1k_20260911_v1/`.
- Corrected candidate: `test_output_brain/visual_policy_1k_20260911_v2/candidate.best.pt`.
- Training history: `test_output_brain/visual_policy_1k_20260911_v2/candidate.metrics.json`.
- Independent saturation/metric checks: `test_output_brain/visual_policy_1k_20260911_v2/diagnostic.json`.
- Deployment-path audit: `test_output_brain/visual_policy_1k_20260911_v2/deployment_audit.json`.
- V2 SHA-256: `74f7797a0b13560afa44c9f8d3bd3636c4cf17cda576a1a072ae07fd052e43a8`.

Reproduce into a fresh output directory (the trainer refuses accidental overwrite):

```powershell
& D:\miniconda3\envs\rcdream\python.exe -u scripts/train_multitrack_vision_curriculum.py --demo_file data/verified_tracking_demos_20260911.pt --existing_multimodal weights/car_brain_1m_curriculum/car_brain_multimodal.pt --output_path test_output_brain/visual_policy_1k_repeat/candidate.pt --steps 1000 --batch_size 8 --seq_len 16 --lr_vision 1e-4 --lr_backbone 1e-6 --trigger_loss_weight 0.1 --eval_every 250 --save_every 250 --eval_batch_size 8 --seed 42 --device cuda
```

After loading `tmrl-train` with the matching first-person camera, a **bounded diagnostic** run
can test V2 without promoting it or enabling trajectory assistance:

```powershell
& D:\miniconda3\envs\rcdream\python.exe -u scripts/evaluate_live_driving.py --mode foundation --foundation-weights test_output_brain/visual_policy_1k_20260911_v2/candidate.best.pt --continuous-actions --steps 5000 --max-deviation 4.5 --record-frames --camera-label first-person
```

### Remaining blockers and limits

- The known-map live test is unrun. The user chose to test a different map instead; the bounded
  probe is documented below. Keep the map/start gate for known-map runs and never apply the
  old route controller on this different map.
- Offline gate failures remain: throttle-command error, steering error, and straight-line steering.
  Better saved-observation scores do not establish fewer live wall impacts.
- Only one training lap and one validation lap exist here, on one map. Reusing validation for
  checkpoint selection means it is not an independent test set. There are no recovery examples
  and no demonstrated brake use in the held-out lap. More genuine driving/recovery data and
  held-out-map evaluation are needed before any one-shot/generalization conclusion.
- Neither candidate has been promoted. The original deployed checkpoint and saved config hashes
  remain `3af96c...263f03` and `37ec02...48cab`, respectively. No training/rollout worker is left running.

The data-validation skill guided the separate metrics/denominators, saturation check, independent
metric recomputation, and the decision to withhold promotion despite offline improvement.

## Different-map candidate probe and capture repair - 12 September 2026

The user explicitly selected the loaded different map. No map-specific demonstration, route
controller, route scoring, or map reward was used. Candidate V2 remained isolated; neither its
weights nor the canonical weights nor the saved configuration changed during these tests.

### Invalid first probe: frozen camera and slow loop

`live_runs/20260911T163049_433724Z_foundation_unseen.json` (under `test_output_brain`) recorded
50 commands over 13.13 seconds, then stopped for low displacement. All **51 recorded camera
frames were identical despite moving telemetry**. Compute p95 was 287.75 ms, far outside the
50 ms period. This is a runtime/capture failure, not a clean measurement of learned driving.

The existing capture setup forced the fullscreen game into a 512x256 client rectangle, leaving
a cropped/frozen rendered image. A new explicit `--preserve-window-size` option captures the
existing client without resizing it; the saved config is not modified. This also avoids asking
the user to change maps or display settings just for an exploratory probe.

At the preserved 2560x1600 size, the no-control profiler then found another bottleneck:
discarding BGRA alpha before resizing created a non-contiguous BGR view and forced a full-frame
copy. Shared preprocessing now resizes the contiguous capture first, then converts at policy
resolution. Tests and a captured-frame comparison verify **exact pixel equality** against the
old conversion given the same source frame. The recorded-frame microbenchmark changed from
39.19 ms to 0.108 ms median. Live preprocessing p95 changed from 46.40 ms to 0.588 ms.

The follow-up no-control preflight observed 60 distinct images out of 60 samples, with end-to-end
p95 12.82 ms. This stationary preflight may exercise the launch safeguard rather than the full
rolling transformer; the actual driving timings below are the relevant full-loop measurement.
The game pause menu was resumed using the computer-use skill before this preflight and retry.

Preflight evidence is in `test_output_brain/visual_policy_1k_20260911_v2/`:
`capture_preflight_20260912_v1.json/.png` and `capture_preflight_20260912_v2.json/.png`.

The evaluator now releases controls if:

- Identical camera pixels persist for at least 0.5 seconds while the car moves at least 0.5 m.
  Static frames while the car is stationary do not trigger this check.
- Three consecutive observation/inference cycles exceed twice the 50 ms period.
  Isolated startup jitter does not automatically stop the test.

These conservative checks detect obvious bad inputs/timing, not all possible synchronization,
collision, or route-following problems. The 30-second unseen-map cap remains in force.
Replaying the saved observations through the guards flags both frozen-camera and repeated-timing
failures at index 2 in the invalid probe; neither guard flags the corrected 30-second recording.

### Guarded retry: valid runtime, driving competence still unestablished

Run: `test_output_brain/live_runs/20260912T020035_604989Z_foundation_unseen.json` with matching NPZ.

- Candidate: V2, SHA-256 `74f7797a0b13560afa44c9f8d3bd3636c4cf17cda576a1a072ae07fd052e43a8`.
- Result: 30.018 seconds, **time-limit stop**, 596 continuous commands, 597 observations.
- Camera: all 597 policy images were distinct; no runtime guard triggered.
- Compute p95: **20.48 ms**; zero compute durations exceeded 50 ms.
- Observation interval p50/p95/max: 50.31 / 52.14 / 53.75 ms.
- Sampled 3D path length: 302.75 m; start-to-end displacement: 251.29 m.
  Neither number measures verified route progress, and neither implies collision-free driving.
- Speed: median 5.64 m/s, peak 48.80 m/s. Much of the latter half stayed near 5 m/s.
  Low speed and occasional launch-safeguard actions remain concerns, not a claim of reliable control.
- No finish flag, no claimed completed lap, and all controls released at exit.

The map start `[773.4, 58.0015, 400.0]` differs from `tmrl-train`; live map UID is unavailable.
"Unseen" here means different from the known visual fine-tuning map, not proven absent from every
earlier pretraining source. The user-declared first-person camera and fullscreen aspect ratio are
also not a controlled match to the 512x256 demonstration capture. The single short probe is not
enough to claim generalization or fewer collisions. Previous probes had different runtime/display
conditions and must not be presented as a controlled policy comparison.

Reproduce a bounded diagnostic on the currently loaded different map:

```powershell
& D:\miniconda3\envs\rcdream\python.exe -u scripts/evaluate_live_driving.py --mode foundation --unseen-map --preserve-window-size --foundation-weights test_output_brain/visual_policy_1k_20260911_v2/candidate.best.pt --continuous-actions --steps 600 --record-frames --camera-label first-person
```

Final software checks: **50 regression tests pass**, including pixel parity and the new camera/
timing guards. No trainer or driving worker remains running. Known-map completion, reliable recovery,
controlled held-out-map evaluation, and one-shot adaptation remain unproven.
