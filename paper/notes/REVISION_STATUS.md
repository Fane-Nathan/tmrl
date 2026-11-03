# Sensor-focused revision: completed work and remaining decisions

Updated 16 September 2026. The current artifact is an anonymous **author-review draft**, not a submitted or policy-cleared paper. It follows the author's clarified goal: adapt during training while retaining earlier behavior, with incorrect or delayed observations.

## Current files

- PDF: [sensor adaptation pilot](D:/Project/tmrl/output/pdf/icodit-2026-sensor-adaptation-pilot.pdf), 14 pages.
- Editable project: [ICoDiT Sensor Adaptation Pilot](https://prism.openai.com/?u=03c884e3-d1d4-4e05-9719-ca220c0e668b&pg=1&m=manuscript.tex).
- Local LaTeX: [manuscript](D:/Project/tmrl/output/icodit_2026/sensor_pilot_draft/manuscript.tex).
- Results: [verified pilot report](D:/Project/tmrl/output/sensor_adaptation/pilot_20260916_v1/REPORT.md).
- Source and review package: `../icodit-2026-sensor-pilot-package.zip`.
- Prism import: `../icodit-2026-sensor-pilot-latex.zip`.
- Private author/CMT worksheet: `CMT_FIELDS_AND_DECLARATIONS.md`.
- Technical checks: `SENSOR_PILOT_QA.json` and the pilot's `verification.json`.

## Completed

1. Implemented actual observation noise and a three-step observation delay. Saved audits verify the exact sensor vector received by the policy. Commands and base reward are unchanged by the sensor wrapper.
2. Aligned transformer collection, evaluation and replay to at most 32 tokens, including short and sliding successor histories. The old benchmark now rejects unsupported sensor requests instead of silently labeling a no-op.
3. Completed all 24 specified branches: three pretrained seeds, two controllers, two sensor faults, and 0%/50% retained replay. Each branch used 4,000 new-condition interactions and 1,000 actor/critic-ensemble updates.
4. Verified 432 saved evaluation episodes, all 48 candidate checkpoint hashes, paired starts and same-model retention evaluations. All six source checkpoints remain unchanged. The process exited successfully; the small smoke run is excluded from scientific results.
5. Measured adaptation and retention separately. The report retains every specified branch and seed, including declines. This pilot does not establish fast adaptation without forgetting, arbitrary new-environment transfer, or the cause of the earlier SAC plateau.
6. Rewrote the manuscript around the sensor experiment and original research motivation, with exact methods, measured outcomes, limits, two vector figures and an accurate expanded AI declaration.
7. Compiled the unmodified conference-linked LNCS template: 14 pages, all fonts embedded, all 15 references cited, 11/15 (73.3%) dated 2022-2026, author identities absent from PDF text and author metadata. Every final page was visually inspected.

The prior actuator data and draft remain a separate historical result set under `revised/`; the original paper and Prism project remain available. Earlier status/worksheet/framing files are preserved under `archived_actuator_notes/`.

The new Prism project compiled to 14 pages. Its exported source archive matches all 12 imported files byte for byte, and the exported PDF has the same text as the local PDF on every page.

## Remaining author and conference requirements

- All four authors must approve the substantially revised study, factual declarations and manuscript. Felix Surjodinoto remains the selected contact and presenter.
- Competing interests, funding and final publication names/order require factual author confirmation. No unsupported no-interests statement has been inserted.
- The conference [manuscript preparation](https://socs.binus.ac.id/icodit/post/preparing-your-manuscript/) and [AI tools](https://socs.binus.ac.id/icodit/post/ai-tools/) pages have differing wording. The present declaration openly includes AI-assisted implementation, substantive writing and experiment work. An accurate clarification email is drafted but unsent; permission has not been inferred from disclosure.
- Required similarity (published limit 20%) and AI screening (published limit 50%) have not been run. No detector score or acceptance outcome is claimed.
- Confirm the exact CMT deadline and authenticated fields. BrowserOS reached the CMT login screen; the submission fields were unavailable without signing in. No submission, payment, email or author invitation has been performed.

The [published deadline](https://socs.binus.ac.id/icodit/post/deadline-policies/) is 16 September 2026 AoE. If interpreted as 23:59 UTC-12, that is 17 September 18:59 UTC+7; the portal cutoff is not verified. Do not rely on that conversion as a confirmed deadline.
