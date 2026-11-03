> Historical preparation note: use [REVISION_STATUS.md](REVISION_STATUS.md) for the current sensor-focused manuscript and [CMT_FIELDS_AND_DECLARATIONS.md](CMT_FIELDS_AND_DECLARATIONS.md) for the matching submission fields. The earlier actuator results below remain a separate result set.

# ICoDiT 2026 submission preparation

Checked on 15 September 2026 using BrowserOS neo, the supplied PDF, the live Prism project, W&B, and local experiment files.

## Decision

**The topic fits the conference, but the current manuscript is not ready to submit.** Resolve the evidence and claims first, then finish anonymity, references, disclosure, and formatting. The deadline is close enough that these items should be handled today.

The draft is *Memory Buys Adaptation, Demonstrations Buy Competence: In-Context Reinforcement Learning Under Real-Time Constraints*. The recommended track is **ISAI - Intelligent Systems & Artificial Intelligence**, particularly autonomous/adaptive systems and robust AI. This is a fit assessment based on the [official scope](https://socs.binus.ac.id/icodit/post/scope-of-topics-12/).

## 1. Dates and submission route

| Milestone | Official date |
| --- | --- |
| Full-paper submission | **16 September 2026** |
| Acceptance notification | 28-30 October 2026 |
| Early-bird registration | 4 November 2026 |
| Regular registration | 8 November 2026 |
| Camera-ready manuscript | 11 November 2026 |
| Conference | 19 November 2026 |

The website specifies **Anywhere on Earth (AoE)** for all deadlines. If the submission deadline ends at 23:59:59 AoE (UTC-12), it corresponds to **17 September, 18:59:59 in Bangkok/Jakarta/WIB (UTC+7)**. The site does not explicitly state the clock time, and the authenticated CMT cutoff has not been checked. Set an internal target of **16 September, 18:00 UTC+7** or earlier, rather than relying on the final hour.

Sources: [Deadlines and policies](https://socs.binus.ac.id/icodit/post/deadline-policies/), [official CMT entry](https://cmt3.research.microsoft.com/ICoDiT2026/Submission/Index). CMT opens the correct conference's login page; its submission form requires sign-in.

## 2. Mandatory rules and present status

| Requirement | Current status | Action |
| --- | --- | --- |
| Original, unpublished complete research paper in English | English complete draft supplied; publication status unconfirmed | All authors confirm originality and any prior/concurrent submission status |
| Computer science contribution within conference scope | Fits ISAI | Make the evaluated computational contribution explicit |
| 6-15 pages, including figures, tables and references | **8 pages** in the supplied PDF | Recheck after template conversion and reference revision |
| Introduction, Method, Results, Discussion (IMRaD) | Introduction and methods present; results and interpretation combined; no separate Discussion section | Make the required structure explicit |
| Fully anonymous review manuscript | **Fails:** four named authors, affiliations and emails on page 1 | Remove identity from manuscript, acknowledgments, self-identifying links and file metadata; keep identities in CMT |
| Conference template | Current source uses **IEEEtran** | Use the conference-prescribed Word/LaTeX template; do not assume the IEEE layout is appropriate |
| At least 70% of references from last five years | **Fails:** 7/30 dated 2022-2026; even counting all 2021 references gives only 13/30 | Revise related work and bibliography using relevant, verified recent sources |
| Similarity at most 20% | Unchecked | Use an authorized institutional check with references/template material excluded and repository deposit disabled |
| AI-generated-content screening at most 50% | Unchecked | Do not treat an estimate from reading as a screening result |
| Required AI disclosure | **Missing** | Add an accurate AI Usage Declaration immediately before references |
| Accurate CMT author information | Prepared for review in AUTHOR_PACKET.md | Confirm order, spellings, affiliation and corresponding author |
| Scientific claims consistent with evidence | **Major revision needed** | Reconcile the three-seed exploratory study and later eight-seed confirmatory study |

Sources: [Preparing your manuscript](https://socs.binus.ac.id/icodit/post/preparing-your-manuscript/), [submission guidelines](https://socs.binus.ac.id/icodit/post/submission-guidelines-3/), [templates](https://socs.binus.ac.id/icodit/post/templates-2/).

The template page describes template use as strongly recommended at review and mandatory at camera-ready, while its detailed paragraph says authors must use it. Use it now to avoid ambiguity. The linked archive download failed in BrowserOS; its contents were not inspected. Links for obtaining the files are retained in AUTHOR_PACKET.md.

## 3. The scientific issue to resolve first

The PDF reports a positive latency adaptation gap of **+652 +/- 225** over three seeds. Page 4 also says the best checkpoint was chosen using the same held-out fault conditions later reported as test results. This exposes the evaluation to test-set selection bias.

The later local study uses eight seeds, a separate validation split for checkpoint selection, and a trained reactive baseline. Independent recomputation from its episode CSVs gives **G = -61.8, 95% seed-bootstrap CI [-231.3, +141.6]** on latency; only 2/8 seed gaps are positive. Positive adaptation is **not established under that protocol**.

These are different protocols: the PDF allows up to 1,000 training iterations; the later study uses 100. The later result therefore cannot be used to claim that checkpoint selection alone caused the difference. Report the protocol differences and outcomes transparently. A stronger preparation route is to frame the contribution around the limits of transfer and the observed real-time deployment behavior, with the three-seed study clearly identified as exploratory.

See MANUSCRIPT_REVIEW.md and evidence/independent_results_audit.json for the detailed checks. No new training was run during this review.

## 4. Priority work plan

### Today: establish the version of the science being submitted

- [ ] Map each reported result, table and figure to its run IDs, code revision, checkpoint-selection rule and raw data.
- [ ] Confirm which W&B runs correspond to TrackMania Runs A/C and the original three-seed experiment.
- [ ] Reconcile the eight-seed confirmatory results with the title, abstract, contributions, Table II, Figure 1, results discussion and conclusion.
- [ ] Give Run C a dated status and fixed evaluation endpoint; replace open-ended statements about ongoing training.
- [ ] Document each author's actual use of AI tools, including Gemini-assisted implementation.
- [ ] Obtain a working template archive and make a separate anonymous review version.

### Before the internal submission target

- [ ] Make IMRaD sections explicit and revise unsupported generalizations.
- [ ] Complete the reference recency and metadata audit. Do not add irrelevant citations to reach a percentage.
- [ ] Check equations, units, tables, figure captions, cross-references and citation resolution.
- [ ] Recompile and visually inspect every final page; confirm 6-15 pages at the prescribed settings.
- [ ] Check anonymization in both visible content and PDF properties.
- [ ] Complete institutional similarity screening and resolve the conference's AI policy questions if relevant.
- [ ] Have all coauthors review the final claims, disclosure, author order and submission file.
- [ ] Sign in to CMT, verify the actual deadline, subject areas, abstract/file limits and any required declarations.
- [ ] Enter the final title/abstract/keywords and all authors; upload the agreed anonymous PDF.
- [ ] Save the submission identifier, confirmation and an exact local copy of the submitted file.

### After acceptance

- [ ] Address reviewer comments and maintain a response-to-reviewers record.
- [ ] Restore authors and any permitted acknowledgments in the camera-ready source.
- [ ] Complete copyright/publishing forms requested by the organizers/publisher.
- [ ] Register and pay under the confirmed presenter category.
- [ ] Submit the camera-ready source/PDF and other requested documents by 11 November.
- [ ] Prepare and deliver the conference presentation. Presentation is required for accepted papers.

## 5. Budget and attendance

| Presenter category | Early bird | Regular |
| --- | --- | --- |
| Internal BINUS | IDR 4,800,000 listed as a single rate | Same single listed rate; confirm eligibility |
| Local external student | IDR 7,770,000 | IDR 8,695,000 |
| Local external non-student | IDR 9,250,000 | IDR 10,175,000 |
| International external student | USD 420 | USD 470 |
| International external non-student | USD 500 | USD 550 |

All four listed authors have BINUS affiliations, so **internal BINUS is the likely category to confirm**, rather than assuming an external student price. Each registration covers one 6-15 page paper and one speaker. The student category requires an active undergraduate/graduate student ID. The website does not publish a separate online presenter discount. Non-presenter tickets are different categories and should not be used to budget a paper presentation. Source: [registration fees](https://socs.binus.ac.id/icodit/post/registration-fees-2/).

The event is hybrid at BINUS @Alam Sutera. Budget travel/accommodation only if presenting on site. The [conference homepage](https://socs.binus.ac.id/icodit) says accepted papers will be submitted for publication in Springer Nature **CCIS**, subject to publisher scope, editorial and quality requirements. This is a conditional publication pathway, not a guarantee that an individual submission will be published or indexed.

## 6. Items still requiring author or organizer input

- Exact run mapping for the old three-seed study and TrackMania A/C.
- Whether additional AI assistance was used for manuscript prose, references or figures; exact Gemini model if known.
- Confirmation of internal BINUS fee eligibility, corresponding author and presenter.
- Conference clarification of inconsistent AI image rules and the five-year reference window if 2021 papers matter.
- Authenticated CMT field/file requirements and precise cutoff.

**No conference submission, payment or email to the organizers was made.** This folder is a preparation and review package, not an accepted or submission-ready manuscript.
