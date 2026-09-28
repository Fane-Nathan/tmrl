# Sensor adaptation protocol: implementation and completion status

The earlier design draft is preserved at `archived_actuator_notes/SENSOR_ADAPTATION_PROTOCOL_DRAFT.md`. A bounded continued-training pilot has now been implemented and completed; this file's historical name is retained for existing links.

The frozen executed specification is `../sensor_adaptation/pilot_20260916_v1/config.json`; exact executed code is in that run's `source/` folder. Working implementation and usage instructions are in `../../experiments/sensor_adaptation_v1/`.

The pilot contains 24 branches from three source seeds per architecture, two architectures, two sensor conditions and two replay fractions. Each branch uses 4,000 new-condition steps, 1,000 update cycles and evaluations at 0/2,000/4,000 steps on A and B. Each evaluation has three deterministic episodes. A is clean sensing; B is either independent Gaussian sensor noise (0.1 for eight positions, 0.5 for nine velocities) or a three-step observation delay with zero padding. Previous command/reward channels remain current. Two clean episodes form the fixed old replay pool; the retained arm samples exactly half its batch from that pool.

The process exited successfully and independent readback verified 432 evaluation episodes, 48 checkpoint hashes, identical paired starts, actual observation corruption and unchanged original weights. The report and seed tables are under the same run folder. This is a warm-start engineering pilot, not a confirmatory study or a demonstration that forgetting has been solved. Recovery thresholds, retention tolerance, longer budgets, multiple task orders and a matched continued-A-training control remain future experimental work.
