# ICoDiT 2026 Paper 193: ready paper

History-Conditioned SAC under Sensor Faults: A Pilot Study of Adaptation and Retention (camera-ready, 31 October 2026, 15 pages).

| Path | Content |
|---|---|
| `manuscript.tex`, `manuscript.pdf`, `manuscript.bbl` | The paper (LNCS). `manuscript.pdf` is the file to upload |
| `*_content.tex`, `table_*.tex`, `results_macros.tex`, `fig_curves*.pdf`, `trackmania_*` | Section text, tables, number macros and figures that `manuscript.tex` includes |
| `references.bib`, `llncs.cls`, `splncs04.bst` | Bibliography and Springer class/style |
| `slides.pdf` | Presentation slides |
| `CAMERA_READY_STATUS.md` | Decisions, key results and the author actions still open |
| `RESPONSE_TO_REVIEWERS.md` | Change summary per reviewer comment |
| `FIGURE_PROVENANCE.json` | Provenance of Figure 1 |
| `notes/` | Process records from earlier stages (declarations text, audits, submission plan, pilot-era files) |
| `reporting/` | Evidence for the BINUS ePDP report (acceptance letter, invoice, Scopus evidence, proof cards) |

Build (from this folder): `pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript`.

Tables, figures and every number in the results text are generated from the saved study outputs; do not edit them by hand. From the repository root:

```
python scripts/build_camera_ready_assets.py --env hc=output/sensor_adaptation/extension_20261031_v1 --env ant=output/sensor_adaptation/ant_20261031_v1 --latency output/sensor_adaptation/latency_20261031.json
```

The code, protocols and tests behind the results are in `experiments/sensor_adaptation_v1/` (see its README).

Earlier drafts (the 14-page sensor draft, the first actuator-fault paper and their data) were removed from the working tree on 3 November 2026 and remain in git history up to commit `14f37d004`; restore with `git restore --source=14f37d004 -- <path>`.
