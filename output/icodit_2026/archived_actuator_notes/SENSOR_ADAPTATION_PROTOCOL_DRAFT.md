# Sensor adaptation and retention: proposed evaluation

**Status: experiment design draft, not an implemented or completed study.** Prepared 16 September 2026 after the author confirmed that the intended faults are incorrect or delayed observations. This proposal restores the original objective: fast adaptation while training, with retention of earlier skills.

## Research question

Can history-conditioned SAC learn to cope quickly with a changed observation process while preserving its previously learned control performance?

History conditioning describes the information supplied to the policy. A transformer can encode that history, while SAC supplies the actor-critic learning objective. These are compatible components of the proposed agent.

## Intervention to implement

Use a controlled observation-fault benchmark first. A change in sensing within one locomotion environment is a task change; it does not by itself establish generalization to arbitrary new maps or physical environments.

- **Incorrect observations:** alter specified observation channels using a recorded, seeded corruption process. Define the affected channels, units, severity, and whether the error is persistent bias or independently sampled noise. Calibrate any normalization only from permitted training/calibration data.
- **Delayed observations:** supply an earlier observation from a timestamped queue. Define the delay, initialization/padding rule, reset behavior, and fault-onset time.
- Apply the sensor transformation to the base observation before appending the previous command/reward fields. Specify separately whether those fields are current or delayed; do not accidentally delay the whole packed token.
- Keep the requested actuator behavior unchanged for the primary sensor study. Log the true observation for audit only; the policy must not receive that audit channel or hidden fault identity at evaluation.
- Preserve deterministic seeds and fault schedules so policies can be evaluated on matched cases.

Gymnasium's official [ObservationWrapper](https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.ObservationWrapper) transforms observations returned by reset and step. Its [DelayObservation](https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.wrappers.DelayObservation) provides a fixed observation delay and zero padding before the queue has sufficient history; it does not support random delays in the documented interface. These semantics are reference points, not evidence that the present workspace implements them. Sources checked through BrowserOS on 16 September 2026.

## Measure the two adaptation modes separately

1. **Within-episode context response:** freeze the policy weights, introduce a sensor change at a recorded point, and measure behavior as additional history arrives. Reset history consistently between independent trials. This asks what the trained context mechanism can do.
2. **Adaptation during further training:** start from a recorded model trained on condition A, allow a fixed interaction/update budget on condition B, and measure B performance across that budget. This is the mode needed to directly address the author's training-time goal.

For both modes, align training and inference context lengths and position handling. Do not reuse the existing out-of-support single-token intervention as a clean causal estimate of memory's benefit.

## Sequential retention measurement

- **A before B:** record performance on the previously learned condition using a fixed evaluation set.
- **B adaptation:** train on the changed observation condition, logging every evaluation checkpoint, environment-interaction count, update count, and elapsed training time.
- **A after B:** evaluate the exact post-B model on the same old-condition evaluation set with a fresh, standardized context. Do not reload the pre-B checkpoint for this measurement.
- Report both B improvement and the A before/after difference. Predefine an acceptable retention-loss margin before inspecting final test results if the paper will claim retention within a tolerance. A statistically uncertain difference alone does not establish no forgetting.

Report performance curves and performance at fixed budgets. A time-to-recovery metric requires a threshold defined before final test inspection and must record cases that never reach the threshold within the budget.

## Essential comparisons and evidence limits

Use a current-observation SAC baseline with compatible training support, the history-conditioned candidate, and a continued-training control that makes the retention intervention explicit. If retained experience is the proposed anti-forgetting mechanism, compare the same update rule with and without that retained experience under matched budgets. Report the replay memory and computation used. Choose the final comparison set and budgets before running the main study.

Record independent training seeds, validation-only selection, separate test schedules, environment and package versions, exact source revision, checkpoint hashes, and raw per-episode/per-checkpoint metrics. Preliminary engineering checks and exploratory pilots remain separate from final evaluation.

The existing actuator-fault data remain their original result set. They must not be relabeled as sensor-fault data. No mechanism for preventing forgetting has yet been validated by this proposed protocol.
