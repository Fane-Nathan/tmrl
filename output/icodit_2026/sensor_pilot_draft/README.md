# Sensor-focused author-review manuscript

This revision restores the author's research question: adaptation during continued training under faulty observations, measured alongside retention of earlier clean-condition behavior. It supersedes the actuator-only framing as the intended paper direction. The older actuator experiment and draft remain preserved as separate artifacts.

## Source and evidence

- `manuscript.tex` contains the paper structure and implemented method.
- All 24 pilot branches completed. Saved-data verification passed for 432 evaluation episodes and 48 candidate checkpoint hashes; the original six checkpoints remain unchanged. The text, tables and figures report those completed results.
- `references.bib` contains 15 primary-source references, with 11 published in 2022-2026 (73.3%). See `REFERENCE_VERIFICATION.md`.
- The supplied class and bibliography style are unchanged from the conference-linked LNCS 2.20 template.
- The experiment implementation is in `D:/Project/tmrl/experiments/sensor_adaptation_v1`.
- Full local raw experiment records are in `D:/Project/tmrl/output/sensor_adaptation/pilot_20260916_v1`.

## Build

From this folder:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex
```

## Scope

The new study is a short, three-seed warm-start pilot. It tests sensor corruption and continued parameter learning; it does not claim successful fast adaptation without forgetting. All outcome claims must be tied to the completed result files. The source models were originally trained on actuator perturbations, and their optimizer/temperature states were not archived. The new experiment resets those states explicitly and preserves its own source snapshot and checkpoints.

Author approval, accurate competing-interest declarations, conference AI-policy clarification, and required screening remain necessary. No CMT submission or email has been sent.
