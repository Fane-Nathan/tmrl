# Research framing: rapid adaptation with skill retention

Working note based on the author's clarification on 16 September 2026. This records the intended research direction and separates it from the measurements currently available. The compiled paper and Prism manuscript have not yet incorporated this framing note.

## Intended goal and development path

**Long-term research goal:** adapt quickly while training in a new environment and retain the skills learned in earlier environments.

The author reports the following progression:

1. SAC-based experiments using transformer or world-model-style components stopped improving beyond a level of performance.
2. These training plateaus motivated a move toward history-conditioned control: provide recent observations, commands, and rewards so the policy can respond to changed conditions.
3. Fault injection was chosen to create a controlled adaptation problem. The author explicitly confirmed incorrect or delayed observations: sensor faults. The saved experiment instead implements actuator faults, so it does not implement the intended intervention.

The earlier plateau observations are author-reported development history. They are not yet a verified, matched comparison demonstrating that transformers or world models caused the plateaus. A transformer is also used in the current history-conditioned agent, so the transition should be described in terms of the training setup and research question, rather than as abandoning transformers for history.

## Three different questions

| Question | What would count as evidence | Current evidence |
| --- | --- | --- |
| Does training continue to improve? | Evaluation performance plotted against environment interactions and elapsed training time, with the run configuration and checkpoint recorded. | Earlier plateaus are reported by the author; their exact runs and causes still need mapping. |
| Does history enable rapid adaptation? | Performance or recovery measured as interaction accumulates after a controlled change; compare context conditions with compatible training and inference support. | The saved study compares aggregate episode returns with history and a single token. It does not separately estimate time or interactions to recovery. |
| Are previously learned skills retained during new training? | Measure environment A, train/adapt on B, and evaluate A again using the resulting updated model under the same evaluation conditions. | This A-to-B-to-A measurement is absent from the saved fault protocol. Its test policy weights are frozen. |

A plateau and forgetting require different measurements. The absence of parameter updates during the current evaluation cannot establish retention during subsequent training.

## Checked implementation facts

The current `scripts/benchmark_fault_env.py` and the version packaged with the revision have identical SHA-256 hashes: `67f791591137037db36d76eed93f9415ca02d593139f993cc4439506f3a55751`.

- `sensor_noise` appears in `FAULT_NAMES` (line 40), but there is no observation-corruption branch.
- Training and validation sample action scaling or action noise in `_sample_fault` (line 96 onward).
- `_apply_actuator_fault` (line 155 onward) implements action scaling, additive action noise, a disabled actuator, sign reversal, and action latency.
- `_pack_observation` (line 179 onward) includes the returned base observation unchanged, alongside the previous commanded action, reward, and terminal flag.
- `step` (line 205 onward) applies the corrupted action to the environment. The state can therefore change because the executed control changed; that is different from corrupting a sensor reading.
- A `fixed_fault='sensor_noise'` setting in this wrapper would currently label an episode without injecting sensor noise. No sensor-fault result should be inferred from that label.
- `evaluate_policy_on_split` in the packaged transformer script (line 381 onward) evaluates frozen weights, resetting context each episode.
- The existing checkpoint audit finds 32-token training support versus a history queue up to 64 tokens. This is a verified mismatch in the inspected implementation and saved positional embeddings, but its causal contribution to the earlier training plateau has not been measured.

These findings apply to the inspected benchmark, not to every historical experimental implementation in the workspace.

## Proposed motivation paragraph

> Our long-term objective is to develop reinforcement-learning controllers that adapt quickly to new environments while retaining previously learned skills. During development of an SAC-based real-time controller, we encountered training plateaus when experimenting with transformer and world-model-style components. These development observations motivated a narrower question: can a policy use recent observations, commands, and rewards to cope with incorrect or delayed sensor readings? We therefore seek to evaluate history-conditioned control under observation faults, alongside adaptation speed and retention of previously learned behavior. The currently archived experiment changes actuator behavior instead, so it provides related diagnostic evidence but does not yet answer this intended sensor-fault and continual-learning question.

This is proposed author wording, not a new experimental result. Identify the exact world-model algorithm and relevant runs before making architecture-specific claims in the manuscript. Sensor corruption is now confirmed as the intended experiment. The paragraph above is research framing for the revised study, not a completed-results claim to paste unchanged into the existing actuator paper. Implement and verify observation corruption, then collect separately identified results before claiming a sensor-fault evaluation.

## How the current paper connects to the goal

- Open the Introduction with rapid adaptation and skill retention as the research motivation.
- Explain the reported plateau as the reason for choosing a smaller, controlled diagnostic question; do not claim an established cause without comparative evidence.
- Explain precisely what information history supplies and what the fault changes.
- Keep the verified eight-seed results and their uncertainty. Those results limit the tested configuration; they do not settle the broader continual-learning goal.
- Discuss the train/inference context mismatch as an implementation finding that restricts interpretation.
- State separately that adaptation speed and retention during further training remain unevaluated.

The existing title accurately names the measured actuator benchmark, but that benchmark differs from the confirmed sensor-fault intention. Restoring the original objective now requires experimental alignment as well as narrative revision. A title or conclusion claiming fast sensor adaptation without forgetting requires new observation-fault, adaptation-speed, and retention measurements.

## Next evidence to obtain

1. Implement and verify the confirmed sensor faults: incorrect or delayed observations. See `SENSOR_ADAPTATION_PROTOCOL_DRAFT.md`.
2. Map the reported training plateaus to identifiable runs and configurations, including the exact world-model method. Treat suspected causes as hypotheses until checked.
3. Align context and position support in a separately recorded diagnostic; retain the archived results as their original result set.
4. For a direct test of the original goal, define an A-to-B-to-A experiment with a recorded new-environment interaction/time budget and unchanged old-environment evaluation conditions. Record adaptation and retention separately. Preserve the same post-B model for the retention test; restoring an old checkpoint would not test what the updated model retained.

No new sensor-fault result, recovery-time estimate, sequential-retention result, or verified cause of the training plateau has been produced by this framing work.
