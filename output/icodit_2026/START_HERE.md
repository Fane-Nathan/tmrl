# Current preparation package

**Camera-ready (current, 31 October 2026):** `camera_ready/manuscript.tex`, compiled to `camera_ready/manuscript.pdf` (copy: `../pdf/icodit-2026-paper193-camera-ready.pdf`), 15 pages. It answers both reviewers with 8 seeds per controller, 20,000-step budgets, a clean-continuation control, Ant-v5, pre-specified tests, a critic diagnostic and latency measurements. Start with `camera_ready/CAMERA_READY_STATUS.md` (decisions, results, remaining author actions) and `camera_ready/RESPONSE_TO_REVIEWERS.md`.

## Earlier versions

Use `REVISION_STATUS.md` for the completed sensor-focused revision and its remaining author/conference requirements. Use `CMT_FIELDS_AND_DECLARATIONS.md` for the matching title, abstract, author details and unsent policy inquiry.

The submitted paper was the **15-page figure revision** (Figure 1 adds a real TrackMania development capture and a diagram of the executed HalfCheetah procedure; numerical results unchanged). Its compiled PDF is `../pdf/icodit-2026-sensor-pilot-with-trackmania.pdf`.

The sources of earlier drafts (the 14-page sensor draft `sensor_pilot_draft/`, the first actuator-fault paper `revised/` and `archived_actuator_notes/`, `source/`, and `paper_research/`) were removed from the working tree on 3 November 2026. They remain in git history up to commit `14f37d004`; restore with `git restore --source=14f37d004 -- <path>`. The earlier actuator paper and its audit are historical evidence, not sensor results.
