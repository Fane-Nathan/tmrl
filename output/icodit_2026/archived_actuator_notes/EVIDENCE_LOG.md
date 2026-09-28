# Evidence and unresolved provenance

Checked on 15 September 2026. Public conference pages and the authenticated Prism/W&B pages were read with BrowserOS neo. No messages were sent to other people.

## Conference sources

- [Homepage: event, venue, hybrid format, conditional CCIS publication](https://socs.binus.ac.id/icodit).
- [Submission guidelines: double-blind review, CMT, screening thresholds](https://socs.binus.ac.id/icodit/post/submission-guidelines-3/).
- [Manuscript requirements: IMRaD, 6-15 pages, reference recency, AI declaration](https://socs.binus.ac.id/icodit/post/preparing-your-manuscript/).
- [Scope: ISAI track](https://socs.binus.ac.id/icodit/post/scope-of-topics-12/).
- [Deadlines: September submission and November registration/final submission/presentation](https://socs.binus.ac.id/icodit/post/deadline-policies/).
- [Fees: internal/external/student categories and one paper per registration](https://socs.binus.ac.id/icodit/post/registration-fees-2/).
- [Templates and linked publisher instructions](https://socs.binus.ac.id/icodit/post/templates-2/).
- [Detailed AI policy, including inconsistent image wording](https://socs.binus.ac.id/icodit/post/ai-tools/).
- [Official author login link](https://socs.binus.ac.id/icodit/post/authors-reviewers-login/).
- [Secretariat contact](https://socs.binus.ac.id/icodit/post/contact-us-15/).

Read-only page observations are retained in evidence/web-observations.json. Live dates and fees should be rechecked before submitting or paying. The CMT submission form was not accessible without sign-in. Template downloads failed; no downloaded archive was validated.

## Manuscript sources

- Supplied PDF: D:/Project/tmrl/complete paper.pdf, 8 pages, file modified 7 September 2026.
- [Supplied Prism project and complete paper.tex](https://prism.openai.com/?u=410f9b0a-6b6e-47f5-8410-442fa4bb39f9&pg=1&m=complete+paper.tex).
- The source was read from the editor and saved as source/complete-paper-original.tex. A project ZIP export did not complete; this source backup was recovered separately.
- Original PNG assets were found in C:/Users/felix/Documents/Machine_Learning_Research/paper_research and copied unchanged into source/. Their hashes and original paths are in source/asset-provenance.json. The old adaptation plot is evidence of the existing manuscript figure, not a validated replacement for the later analysis.

## Run mapping status

| Paper item | Source checked | What is verified | Remaining uncertainty |
| --- | --- | --- | --- |
| TrackMania Run B | [SAC_RLPD_BC_v1](https://wandb.ai/trackmania-rl/trackmania-rl/runs/SAC_RLPD_BC_v1/overview) | return_test 279.7099914551 and episode_length_test 1372 match the draft | Finish-event/replay, first-completion timing, full evaluation history and checkpoint link |
| TrackMania Run A | [Final_Test_Experiment_REAL_v2](https://wandb.ai/trackmania-rl/trackmania-rl/runs/Final_Test_Experiment_REAL_v2/overview), [v3](https://wandb.ai/trackmania-rl/trackmania-rl/runs/Final_Test_Experiment_REAL_v3/overview) | Runs exist; summaries and code revisions recorded | No verified match to Table III or the claimed 122.49 peak; summaries are not full histories |
| TrackMania Run C | [SEQSAC_v6](https://wandb.ai/trackmania-rl/trackmania-rl/runs/SEQSAC_v6/overview), [v7](https://wandb.ai/trackmania-rl/trackmania-rl/runs/SEQSAC_v7/overview), [v8](https://wandb.ai/trackmania-rl/trackmania-rl/runs/SEQSAC_v8/overview), [v9_speed](https://wandb.ai/trackmania-rl/trackmania-rl/runs/SEQSAC_v9_speed/overview) | Candidate run family exists; v6 has BC-related metrics; v7/v8 show no summary metrics | Exact phase-to-run/checkpoint mapping and the claimed 86.1 peak remain unverified |
| Table II / Figure 1: old three-seed study | Current workspace result folders, scratch reports, earlier research directory and manuscript assets | The claim and original figure exist in manuscript materials | Original per-seed raw experiment records, checkpoint hashes, full CI computation and run IDs were not recovered |
| Later eight-seed study | D:/Project/tmrl/fault_benchmark_results; confirmatory_protocol.json | All 16 checkpoint hashes and 1,800 episode records checked; seed means and CIs recomputed | Experimental implementation and simulator execution were not rerun/audited end to end |

The author does not remember the old run mapping. Do not fill these gaps from memory or equate similarly named runs with the manuscript's evidence. The overview checks above are a bounded search, not an exhaustive reconstruction of all W&B histories or projects.

## Verification boundaries

- A W&B final summary value need not be the historical maximum.
- Matching return and episode length supports the reported metric; finish flags or replays provide stronger evidence of a completed lap.
- A checkpoint hash confirms which stored file the metadata references; it does not independently establish the correctness of the code that produced it.
- The eight-seed study has a different budget and checkpoint-selection rule from the draft. Avoid attributing their different outcomes to only one change.
- Reference years were counted as printed. Author/title/venue/DOI accuracy and recent literature completeness remain to be checked against primary sources.
- No plagiarism/AI screening service was run, and no passing score is claimed.

## Recommended evidence decision

Use traceable results as the basis for the final manuscript. If the older three-seed result cannot be reconstructed, it should not serve as the unqualified headline result. Keep any exploratory discussion accurately labeled and limit claims to the evidence the authors can inspect and defend.
