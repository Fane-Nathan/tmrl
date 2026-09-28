# CMT fields and author declarations

Prepared 15 September 2026; updated 16 September 2026. This worksheet is private preparation material and must not be included in the anonymous review PDF. Read `REVISION_STATUS.md` before submitting.

## Title

History-Conditioned Control under Unseen Actuator Faults: An Eight-Seed Evaluation

## Abstract

History-conditioned policies can use past observations, actions, and rewards to respond to hidden changes in dynamics. Whether that capability improves control under an unseen actuator fault requires an explicit empirical test. We evaluate a transformer actor–critic trained on action scaling and action noise in HalfCheetah-v5. Eight independent training seeds are compared with the same checkpoints evaluated using a single observation token and with eight separately trained reactive baselines. Checkpoints are selected using validation perturbations from the training mechanisms; held-out tests contain action latency, a disabled actuator, or a sign reversal. Across 25 deterministic episodes per seed and condition, the primary latency history effect is -61.8 return units, with a 95% seed-level bootstrap interval of [-231.3, 141.6]. The corresponding advantage over the reactive baseline is 21.9 with interval [-73.6, 111.2]. Thus, positive history-enabled adaptation is not established for this configuration. A code and checkpoint audit identifies a material limitation: training uses 32-token windows while history-enabled inference can use 64 tokens, including positional embeddings outside the training window. These findings concern the evaluated implementation and do not establish an intrinsic disadvantage of memory. A separately documented TrackMania run illustrates the narrower evidence available from a single real-time deployment. The study provides reproducible result aggregation and identifies controls required for stronger adaptation claims.

## Keywords and track

Reinforcement learning; actuator faults; history conditioning; empirical evaluation; real-time control.

Proposed track: **ISAI — Intelligent Systems & Artificial Intelligence**. Confirm the exact available CMT subject-area labels when signed in.

## Authors and contact

The original PDF's order is preserved below. All four are authors; “coauthor” does not make someone a separate class of contributor. Confirm final order and full publication names with everyone before submission.

| Original order | Publication name as originally printed | Email as originally printed |
| --- | --- | --- |
| 1 | Felix Surjodinoto | felix.surjodinoto@binus.ac.id |
| 2 | Keandre Rafael | keandre.rafael@binus.ac.id |
| 3 | Delroy | delroy@binus.ac.id |
| 4 | Samuel Philip | samuel.philip@binus.ac.id |

**Corresponding contact and registered presenter: Felix Surjodinoto**, explicitly selected by the user. The conversation also used “Felix Nathaniel”; the manuscript's original name and the latest explicit contact selection are Felix Surjodinoto. Confirm the intended publication name rather than silently substituting the account display name.

All four authors are from BINUS, confirmed by the user. Original printed affiliation: Computer Science Department, School of Computer Science, Bina Nusantara University, Jakarta 11530, Indonesia. Confirm its official publication wording. Use the CMT-supported handling of Delroy's single-part name without inventing a surname.

## Competing-interests declaration — factual confirmation outstanding

The user described the authors' shared BINUS affiliation. This alone does not tell us whether there is relevant funding, paid employment, ownership, a patent, or another interest that needs disclosure. The paper does not contain an invented “no interests” statement.

If all authors confirm that they have no relevant competing interests, insert the following publisher-style statement before the AI Usage Declaration:

> The authors have no competing interests to declare that are relevant to the content of this article.

If a relevant interest exists, describe the actual relationship and affected author accurately. For the anonymous review copy, omit identifying details only as permitted by the conference and supply the full information in the designated confidential submission fields. Confirm the specific treatment with the organizers if anonymization would obscure a required disclosure.

Funding and institutional affiliation are distinct from authorship. Do not list the four names as the interests declaration, and do not infer a funding source from a university affiliation.

## AI Usage Declaration — included in the manuscript

Google Antigravity with a Gemini model was used to assist implementation of the research architecture and algorithms. OpenAI Codex was used to assist manuscript revision and restructuring, literature-source lookup, formatting, code inspection, and preparation of scripts that recompute statistics and plot recorded returns. The reported episode observations are taken from saved experiment records; the table values and quantitative figure are calculated from those records. Generative AI was not used to invent experimental observations. The authors remain responsible for the implementation, source verification, analysis, interpretation, and final manuscript.

All authors should check completeness and accuracy. The exact Gemini model/version was not supplied. Add any additional tools or uses that actually occurred. The declaration does not certify compliance with the conference's differing AI-policy pages or certify a detector score.

## Organizer clarification email — draft, not sent

To: icodit@binus.edu

Subject: ICoDiT 2026: clarification of disclosed AI assistance and review submission requirements

Dear ICoDiT 2026 Secretariat,

We are preparing an ISAI-track manuscript and would appreciate clarification before submission.

Google Antigravity with a Gemini model assisted implementation of our research architecture and algorithms. Experimental observations come from executed research runs. OpenAI Codex has also assisted manuscript revision and restructuring, primary-source lookup, formatting, code inspection, and scripts that calculate statistics and generate a quantitative plot from saved episode records. We disclose these uses explicitly; no experimental observations were invented by generative AI.

The Preparing Your Manuscript page limits AI assistance to grammar/paraphrasing and prohibits AI-generated data, references, figures, and images. The AI Tools page distinguishes fabrication from disclosed assistance and permits some illustrative uses. Could you confirm whether our described implementation, revision, source-lookup, and deterministic analysis/plotting assistance is permitted, and what additional declarations or author verification are required? We are not treating a disclosure alone as proof of permission.

We have used the LaTeX template supplied in the conference-linked archive, with an anonymous 13-page IMRaD manuscript and an AI Usage Declaration immediately before the references.

Please also confirm the exact CMT cutoff for the 16 September AoE deadline and any confidential-field procedure needed for competing-interest disclosures during double-blind review. All four authors are affiliated with BINUS; please confirm the applicable internal presenter registration category for this paper.

Thank you for your guidance.

Best regards,
Felix Surjodinoto

Contact address source: [ICoDiT contact page](https://socs.binus.ac.id/icodit/post/contact-us-15/). This message has not been sent.

## Submission steps after author/policy checks

1. Sign in at [ICoDiT 2026 CMT](https://cmt3.research.microsoft.com/ICoDiT2026/Submission/Index) and confirm the conference.
2. Enter the agreed title, identical abstract, subject area, and all four authors in the approved order; select Felix Surjodinoto as contact where appropriate.
3. Complete truthful declarations for originality, concurrent submission, conflicts, funding, and AI use wherever requested. Actual CMT fields/file limits remain uninspected.
4. Run the required similarity/AI screening through an appropriate authorized service. For Turnitin, follow the conference's nonrepository instruction.
5. Upload only the agreed anonymous review manuscript and any expressly requested, anonymized supplements. Do not upload this private worksheet or the whole preparation package.
6. Open the uploaded file, check page count and author anonymity, complete submission, and save the confirmation plus the exact submitted PDF.

The [published full-paper deadline](https://socs.binus.ac.id/icodit/post/deadline-policies/) is 16 September 2026 AoE. The precise portal closing time is unverified. The fee page lists IDR 4,800,000 for the internal BINUS presenter category; eligibility still needs confirmation. No payment has been made.
