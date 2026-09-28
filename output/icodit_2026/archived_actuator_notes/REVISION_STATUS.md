# ICoDiT 2026 revision: what is fixed and what remains

Prepared 15 September 2026; final checks updated 16 September 2026. This file supersedes the action lists in the earlier preparation pack. **The manuscript is an anonymous author-review draft. It has not been submitted.**

## Research motivation update

The author clarified that the original goal is rapid adaptation during training while retaining earlier skills. Reported training plateaus with transformer/world-model-style SAC experiments led to the history-conditioned fault study. The author has now confirmed sensor faults (incorrect/delayed observations) as the intended intervention; the saved benchmark instead alters actuators. See `RESEARCH_FRAMING.md` and `SENSOR_ADAPTATION_PROTOCOL_DRAFT.md` for the corrected research direction and required measurements. The present PDF remains a technically checked, narrower benchmark draft; its motivation is under revision in light of this clarification. The existing data do not measure adaptation speed or retention during sequential training.

## Files to use

- Final local review PDF: `../pdf/icodit-2026-anonymous-review-draft.pdf`
- Editable manuscript: `revised/manuscript.tex`
- Separate editable Prism project: [ICoDiT 2026 Review Draft](https://prism.openai.com/?u=7722d861-deb7-4604-ac24-51ac72ba8be0&pg=1&m=manuscript.tex)
- Final technical checks: `FINAL_QA.json` (13 pages, embedded fonts, all references resolved, all pages visually reviewed).
- Bibliography: `revised/references.bib`
- Lightweight LaTeX import ZIP: `../icodit-2026-latex-import.zip`
- Complete revision package: `../icodit-2026-revision-pack.zip`
- CMT title, abstract, author worksheet, and declarations: `CMT_FIELDS_AND_DECLARATIONS.md`
- Reproduction instructions and data: `revised/README.md`
- Primary-source verification ledger: `revised/REFERENCE_VERIFICATION.md`

The original PDF and `source/complete-paper-original.tex` remain unchanged. Do not submit the original IEEE-formatted paper or the earlier preparation ZIP as the final manuscript. The source ZIP is for editing/import, not an instruction to upload all private preparation materials to CMT.

## Completed revisions

1. **Scientific claims:** replaced the untraceable three-seed positive headline with the traceable eight-seed result. The primary history effect is **-61.8**, with 95% interval **[-231.3, 141.6]**. Positive adaptation is not established for this implementation. The original title and conclusion would overstate the evidence.
2. **Results:** rebuilt the tables and vector plot from all **1,800** saved evaluation records. Preserved all eight seeds and used seed-level, paired percentile bootstrap intervals; secondary intervals are explicitly uncorrected for multiple comparisons.
3. **Provenance:** verified all **16** checkpoint hashes, selected validation iterations, episode counts, and common episode seeds. The aggregation script is included. This is an archive consistency audit, not a new training run or full simulator replication.
4. **Meaning of the comparison:** renamed the old “no-history” mode to “single-token.” Both it and the reactive baseline still receive the previous commanded action and reward.
5. **Context mismatch:** documented 32-token training versus up-to-64-token inference. In all eight checkpoints, positional embeddings at indices 32–127 exactly match fresh seeded initialization of the current code. This prevents interpreting the measured difference as a clean causal estimate of memory's value.
6. **Optimization reporting:** distinguished ten critic updates per actor update from the **0.25 critic-ensemble updates per newly collected environment step** implied by the recorded training loop and budget. Sequence and transition batches are not computationally matched.
7. **TrackMania:** retained the traceable run's stored return and length as a limited observation. Removed unverifiable A/C quantitative comparisons and claims that demonstrations, replay, or architecture alone caused the outcomes. A separate finish flag and repeated completion rate have not been verified.
8. **Format:** converted to the unmodified LNCS 2.20 class and bibliography style from the conference-linked archive, using its default Computer Modern fonts and page geometry, numbered references, and explicit Introduction / Method / Results / Discussion sections. The compiled draft is **13 pages**.
9. **Anonymity:** removed author identities and institutional details from the PDF and its author metadata. The author worksheet is separate.
10. **References:** 14 entries, all cited; **10/14 = 71.4%** have publication years 2022–2026. The undated TMRL repository is conservatively counted as nonrecent. Bibliographic metadata and relevant abstracts were checked against primary sources. AMAGO's coauthor is correctly recorded as Jim Fan.
11. **AI disclosure:** added an explicit declaration immediately before the references, including Gemini implementation assistance and Codex revision, source lookup, analysis-script, and plotting assistance. It does not claim that assistance was language-only.
12. **Submission contact:** Felix Surjodinoto is the user-designated corresponding contact and registered presenter. The four listed authors share a BINUS affiliation.

The Prism import compiled successfully. Its manuscript source hash matches the local source, and its exported PDF is also 13 pages. The original Prism project remains separate.

## Items that prose editing cannot resolve

### A. Author decisions and declarations

- All four authors must review and approve the changed scientific conclusion and the final paper. This revision substantially reframes the original manuscript.
- Confirm the final author order and full names. The worksheet preserves the original PDF's order. The conversation used both Felix Nathaniel and Felix Surjodinoto; the latest explicit contact/presenter choice is Felix Surjodinoto.
- **Competing interests remain unconfirmed.** The authors identified their shared BINUS affiliation when asked. That information alone does not establish either the presence or absence of a relevant financial/personal interest. No unsupported “no competing interests” declaration has been inserted. Use the wording options in the declarations worksheet after checking with the authors.
- Confirm that the AI declaration covers all actual use, including any additional earlier assistance with prose or figures. Exact Gemini model/version was not provided.
- Confirm originality, no simultaneous submission, funding information if requested, author contributions, and institutional approval requirements, where applicable. These are factual author declarations, not matters that can be inferred from a PDF.

### B. Conference policy and screening

The [manuscript-preparation page](https://socs.binus.ac.id/icodit/post/preparing-your-manuscript/) restricts AI to limited manuscript preparation and prohibits AI-generated data, references, figures, and images. The [AI Tools page](https://socs.binus.ac.id/icodit/post/ai-tools/) uses different wording, distinguishing fabrication from certain disclosed assistance. The present work uses actual recorded observations and deterministically computed statistics, but it also includes AI-assisted implementation and substantive revision. **Organizer clarification is needed; do not describe this as confirmed policy compliance.** An accurate inquiry is drafted in the worksheet; it has not been sent.

The site lists similarity at or below 20% (with exclusions) and AI screening at or below 50%. Neither check has been run. No score or acceptance outcome is claimed. If using Turnitin, follow the site's instruction to avoid depositing the draft in a repository. Manual author verification of sources and content remains required by the conference.

Both the publisher and conference template archives were downloaded through BrowserOS. The final manuscript uses the unmodified class and bibliography style from the conference-linked archive, which contains LNCS 2.20. Archive and file hashes are recorded in `revised/data/template_manifest.json`. The newer publisher template was used only during preparation. No publisher indexing or acceptance is guaranteed.

### C. Experimental limits

The archived data can support this qualified empirical report. They cannot support the original broad “memory buys adaptation” claim. Repairing context support requires a **new, separately labeled evaluation**; changing text cannot repair old trajectories. A capped-context diagnostic on existing checkpoints and a fresh matched-context training study answer different questions. Do not silently replace existing results, select the most favorable seeds, or tune using the present test faults and then call them unseen.

The benchmark scripts were untracked at the saved commit and the historical package capture is incomplete. A current source snapshot is included with hashes, but this does not reconstruct missing historical provenance. The manuscript states this limit.

## Deadline and upload

The [published deadline policy](https://socs.binus.ac.id/icodit/post/deadline-policies/) lists **16 September 2026, AoE** for full papers. If this means 23:59 UTC-12, it corresponds to 17 September at 18:59 in UTC+7; the exact CMT cutoff has not been verified. Submit earlier rather than relying on that interpretation.

Use the [ICoDiT 2026 CMT portal](https://cmt3.research.microsoft.com/ICoDiT2026/Submission/Index). Its authenticated submission fields and file limits have not been inspected. The proposed subject area is ISAI / Intelligent Systems & Artificial Intelligence. After resolving the items above, paste the title and abstract from the worksheet, enter the authors, upload the agreed anonymous PDF, inspect the uploaded preview, and retain the confirmation and submitted-file hash.

No email, payment, author invitation, final submission, or public publication has been performed.
