# Research framing: adaptation during training with retention

Updated 16 September 2026 after completing the separately identified sensor pilot.

The original goal is rapid adaptation to new conditions while preserving earlier skills. The author reported plateaus after adding transformer/world-model-style components to SAC, which motivated history conditioning and faulty sensing. Those reports remain motivation, not a verified diagnosis of the plateau's cause.

## What now matches that goal

The new experiment in `experiments/sensor_adaptation_v1` changes observations using seeded noise or a three-step delay. It permits parameter updates in condition B and measures the same updated model on the earlier clean condition A. The 0%/50% retained-replay comparison asks whether this particular rehearsal strategy helps retention within a fixed interaction budget. The manuscript now describes this experiment and the author's original motivation.

The completed 24-branch pilot supplies actual adaptation and retention measurements, but does not establish general fast adaptation without forgetting. It uses three existing actuator-trained models per controller, one environment, two fixed sensing changes and 4,000 new-condition steps per branch. Initial A is a clean evaluation setting, not new clean-only pretraining. See the verified pilot report for every outcome.

## Historical implementation finding

Before the new guard, the working and archived actuator wrapper matched SHA-256 `67f791591137037db36d76eed93f9415ca02d593139f993cc4439506f3a55751`. It listed `sensor_noise` but changed actuators rather than observations; requesting the sensor label could silently do nothing. The current working wrapper has since been changed to reject unsupported sensor-noise/delay requests and point to the actual sensor implementation. The archived wrapper remains unchanged as the historical record.

The earlier 32-token training versus 64-token inference mismatch is also preserved as an audit finding. The new pilot caps actual histories at 32 and reconstructs corresponding current and successor histories in replay. This correction does not establish why the earlier learning plateau occurred.

## Keep these claims separate

- A training plateau means observed performance stopped improving; it is not by itself forgetting.
- Adaptation means improvement under a new condition as additional interactions and updates occur.
- Retention is measured on the old condition with the post-adaptation model, not by restoring the original checkpoint.
- Three evaluation budgets do not identify recovery time without a predeclared recovery threshold.
- A small or uncertain old-condition difference does not prove no forgetting without a meaningful predefined tolerance.

The earlier actuator paper is a separate result set. The current sensor pilot does not relabel its trajectories, reproduce its pretraining, diagnose plasticity or primacy bias, or establish unseen TrackMania-map transfer.
