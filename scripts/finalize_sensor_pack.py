"""Finalize private preparation notes and a portable paper/results review package."""
import argparse
import csv
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(r"D:\Project\tmrl")
PREP = ROOT / "output/icodit_2026"
PAPER = PREP / "sensor_pilot_draft"
PILOT = ROOT / "output/sensor_adaptation/pilot_20260916_v1"
CODE = ROOT / "experiments/sensor_adaptation_v1"

parser = argparse.ArgumentParser()
parser.add_argument("--prism", required=True)
args = parser.parse_args()
qa = json.loads((PREP / "SENSOR_PILOT_QA.json").read_text(encoding="utf-8"))
verified = json.loads((PILOT / "verification.json").read_text(encoding="utf-8"))
assert verified["status"] == "PASS" and verified["completed_branches"] == 24
assert qa["visual_review"] == "PASS: all final pages inspected"
with (PILOT / "summary.csv").open(newline="", encoding="utf-8") as f:
    summary = list(csv.DictReader(f))
title = "History-Conditioned SAC under Sensor Faults: A Pilot Study of Adaptation and Retention"
abstract = (PAPER / "abstract_content.tex").read_text(encoding="utf-8").strip().replace(r"\%", "%").replace("~", " ").replace("--", "\u2013")
manuscript = (PAPER / "manuscript.tex").read_text(encoding="utf-8")
ai = manuscript.split(r"\section*{AI Usage Declaration}", 1)[1].split(r"\par\end{samepage}", 1)[0].strip()
pages = qa["pages"]
old_fields = (PREP / "CMT_FIELDS_AND_DECLARATIONS.md").read_text(encoding="utf-8")
authors = old_fields.split("## Authors and contact", 1)[1].split("## AI Usage Declaration", 1)[0].strip()
archive = PREP / "archived_actuator_notes"
archive.mkdir(exist_ok=True)
for name in ["REVISION_STATUS.md", "START_HERE.md", "CMT_FIELDS_AND_DECLARATIONS.md", "RESEARCH_FRAMING.md", "SENSOR_ADAPTATION_PROTOCOL_DRAFT.md"]:
    if not (archive / name).exists():
        shutil.copy2(PREP / name, archive / name)

fields = f"""# Sensor pilot: CMT fields and author declarations

Updated 16 September 2026. Private preparation only; do not upload this worksheet with the anonymous PDF. This is the sensor-focused revision. The actuator-focused worksheet is preserved under `archived_actuator_notes/`.

## Title

{title}

## Abstract

{abstract}

## Keywords and track

Reinforcement learning; sensor faults; history conditioning; retention; experience replay.

Proposed track: ISAI - Intelligent Systems & Artificial Intelligence. Verify the actual subject-area choices in CMT.

## Authors and contact

{authors}

## AI Usage Declaration - included in the manuscript

{ai}

All authors must check completeness. The exact Gemini version was not supplied. This declaration does not certify a screening score or resolve the differing conference policy pages.

## Organizer inquiry - draft, not sent

To: icodit@binus.edu

Subject: ICoDiT 2026: clarification of disclosed AI assistance and review requirements

Dear ICoDiT 2026 Secretariat,

We are preparing an ISAI manuscript on adaptation and retention under sensor faults. Google Antigravity with a Gemini model assisted our earlier research implementation. OpenAI Codex assisted substantive manuscript revision, primary-source lookup, formatting, implementation and testing of a new simulator pilot, experiment orchestration, verification, and scripts that compute tables and figures from saved observations. The experimental observations were produced by executed simulator runs. We disclose these uses and do not claim that assistance was limited to language editing.

The Preparing Your Manuscript page limits AI assistance to grammar/paraphrasing and prohibits AI-generated data, references, figures, and images. The AI Tools page distinguishes fabrication from some disclosed assistance. Could you confirm whether the described implementation, experiment, revision, and deterministic analysis/plotting assistance is permitted, and what additional author verification or declarations are required?

We use the conference-linked LaTeX template for an anonymous {pages}-page manuscript, with the AI Usage Declaration immediately before the references. Please also confirm the exact CMT cutoff for the 16 September AoE deadline and the procedure for any confidential competing-interest disclosure during double-blind review. All four authors are affiliated with BINUS; please confirm eligibility for the internal presenter registration category.

Thank you for your guidance.

Best regards,
Felix Surjodinoto

Contact source: [ICoDiT contact page](https://socs.binus.ac.id/icodit/post/contact-us-15/). This inquiry has not been sent.

## Before final submission

1. Obtain all four authors' approval of the new experiment, interpretation, author order, publication names, affiliation, funding, originality and concurrent-submission declarations.
2. Resolve the competing-interests statement from actual relationships; shared BINUS affiliation alone does not determine it.
3. Resolve the AI-policy question and complete the required similarity/AI screening. Neither screening has been run. Use a nonrepository Turnitin setting as instructed by the conference.
4. Confirm the deadline and actual fields in [ICoDiT CMT](https://cmt3.research.microsoft.com/ICoDiT2026/Submission/Index).
5. Upload only the agreed anonymous PDF and any specifically requested anonymous supplement; inspect it and retain the submitted-file hash and confirmation.

The published full-paper deadline is [16 September 2026 AoE](https://socs.binus.ac.id/icodit/post/deadline-policies/); the precise portal closing time remains unverified. The internal BINUS presenter fee was listed as IDR 4,800,000, with eligibility unconfirmed. No email, invitation, payment, or CMT submission has been made.
"""
(PREP / "CMT_FIELDS_AND_DECLARATIONS.md").write_text(fields, encoding="utf-8")

status = f"""# Sensor-focused revision: completed work and remaining decisions

Updated 16 September 2026. The current artifact is an anonymous **author-review draft**, not a submitted or policy-cleared paper. It follows the author's clarified goal: adapt during training while retaining earlier behavior, with incorrect or delayed observations.

## Current files

- PDF: [sensor adaptation pilot](/D:/Project/tmrl/output/pdf/icodit-2026-sensor-adaptation-pilot.pdf), {pages} pages.
- Editable project: [ICoDiT Sensor Adaptation Pilot]({args.prism}).
- Local LaTeX: [manuscript](/D:/Project/tmrl/output/icodit_2026/sensor_pilot_draft/manuscript.tex).
- Results: [verified pilot report](/D:/Project/tmrl/output/sensor_adaptation/pilot_20260916_v1/REPORT.md).
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
7. Compiled the unmodified conference-linked LNCS template: {pages} pages, all fonts embedded, all 15 references cited, 11/15 (73.3%) dated 2022-2026, author identities absent from PDF text and author metadata. Every final page was visually inspected.

The prior actuator data and draft remain a separate historical result set under `revised/`; the original paper and Prism project remain available. Earlier status/worksheet/framing files are preserved under `archived_actuator_notes/`.

The new Prism project compiled to 14 pages. Its exported source archive matches all 12 imported files byte for byte, and the exported PDF has the same text as the local PDF on every page.

## Remaining author and conference requirements

- All four authors must approve the substantially revised study, factual declarations and manuscript. Felix Surjodinoto remains the selected contact and presenter.
- Competing interests, funding and final publication names/order require factual author confirmation. No unsupported no-interests statement has been inserted.
- The conference [manuscript preparation](https://socs.binus.ac.id/icodit/post/preparing-your-manuscript/) and [AI tools](https://socs.binus.ac.id/icodit/post/ai-tools/) pages have differing wording. The present declaration openly includes AI-assisted implementation, substantive writing and experiment work. An accurate clarification email is drafted but unsent; permission has not been inferred from disclosure.
- Required similarity (published limit 20%) and AI screening (published limit 50%) have not been run. No detector score or acceptance outcome is claimed.
- Confirm the exact CMT deadline and authenticated fields. BrowserOS reached the CMT login screen; the submission fields were unavailable without signing in. No submission, payment, email or author invitation has been performed.

The [published deadline](https://socs.binus.ac.id/icodit/post/deadline-policies/) is 16 September 2026 AoE. If interpreted as 23:59 UTC-12, that is 17 September 18:59 UTC+7; the portal cutoff is not verified. Do not rely on that conversion as a confirmed deadline.
"""
(PREP / "REVISION_STATUS.md").write_text(status.replace("(/D:/", "(D:/"), encoding="utf-8")
(PREP / "START_HERE.md").write_text("# Current preparation package\n\nUse `REVISION_STATUS.md` for the completed sensor-focused revision and its remaining author/conference requirements. Use `CMT_FIELDS_AND_DECLARATIONS.md` for the matching title, abstract, author details and unsent policy inquiry.\n\nThe current paper is `sensor_pilot_draft/manuscript.tex`, with its compiled PDF in `../pdf/icodit-2026-sensor-adaptation-pilot.pdf`. The earlier actuator paper and its audit remain historical evidence, not sensor results.\n", encoding="utf-8")
(PAPER / "DRAFT_STATUS.md").write_text(f"# Completed sensor pilot author-review draft\n\nAll 24 pilot branches completed and were independently read back. The {pages}-page PDF has passed structural and visual checks. See `../REVISION_STATUS.md` for author declarations, AI-policy clarification and screening that remain outstanding. This paper has not been submitted.\n\nEditable Prism copy: {args.prism}\n", encoding="utf-8")
(PREP / "RESEARCH_FRAMING.md").write_text("""# Research framing: adaptation during training with retention

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
""", encoding="utf-8")
protocol_note = """# Sensor adaptation protocol: implementation and completion status

The earlier design draft is preserved at `archived_actuator_notes/SENSOR_ADAPTATION_PROTOCOL_DRAFT.md`. A bounded continued-training pilot has now been implemented and completed; this file's historical name is retained for existing links.

The frozen executed specification is `../sensor_adaptation/pilot_20260916_v1/config.json`; exact executed code is in that run's `source/` folder. Working implementation and usage instructions are in `../../experiments/sensor_adaptation_v1/`.

The pilot contains 24 branches from three source seeds per architecture, two architectures, two sensor conditions and two replay fractions. Each branch uses 4,000 new-condition steps, 1,000 update cycles and evaluations at 0/2,000/4,000 steps on A and B. Each evaluation has three deterministic episodes. A is clean sensing; B is either independent Gaussian sensor noise (0.1 for eight positions, 0.5 for nine velocities) or a three-step observation delay with zero padding. Previous command/reward channels remain current. Two clean episodes form the fixed old replay pool; the retained arm samples exactly half its batch from that pool.

The process exited successfully and independent readback verified 432 evaluation episodes, 48 checkpoint hashes, identical paired starts, actual observation corruption and unchanged original weights. The report and seed tables are under the same run folder. This is a warm-start engineering pilot, not a confirmatory study or a demonstration that forgetting has been solved. Recovery thresholds, retention tolerance, longer budgets, multiple task orders and a matched continued-A-training control remain future experimental work.
"""
(PREP / "SENSOR_ADAPTATION_PROTOCOL_DRAFT.md").write_text(protocol_note, encoding="utf-8")

files = {}
def add_file(path, arc):
    assert arc not in files
    files[arc] = path

for path in PAPER.iterdir():
    if path.is_file():
        add_file(path, "paper/" + path.name)
add_file(ROOT / "output/pdf/icodit-2026-sensor-adaptation-pilot.pdf", "paper-review.pdf")
for name in ["START_HERE.md", "REVISION_STATUS.md", "CMT_FIELDS_AND_DECLARATIONS.md", "RESEARCH_FRAMING.md", "SENSOR_ADAPTATION_PROTOCOL_DRAFT.md", "SENSOR_PILOT_QA.json"]:
    add_file(PREP / name, "private-preparation/" + name)
for path in (PILOT / "source").iterdir():
    if path.is_file():
        add_file(path, "experiment/executed-source/" + path.name)
for name in ["build_paper_assets.py", "test_contracts.py"]:
    add_file(CODE / name, "experiment/post-run-tools/" + name)
add_file(ROOT / "scripts/benchmark_fault_env.py", "experiment/legacy-wrapper-with-sensor-guard.py")
for name in ["config.json", "manifest.json", "study_result.json", "status.json", "verification.json", "REPORT.md", "all_evaluations.csv", "seed_metrics.csv", "summary.csv", "paired_replay_effects.csv", "learning_curves.png", "retention_curves.png"]:
    add_file(PILOT / name, "results/" + name)
for branch in sorted(PILOT.glob("*_seed*_sensor_*_replay*")):
    for name in ["result.json", "specification.json", "training.jsonl", "evaluations.csv"]:
        if (branch / name).exists():
            add_file(branch / name, "results/branches/" + branch.name + "/" + name)
package = ROOT / "output/icodit-2026-sensor-pilot-package.zip"
records = {name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size} for name, path in sorted(files.items())}
manifest = {"purpose": "Private paper/results review package; not an anonymous CMT supplement", "full_raw_run": str(PILOT),
            "omitted_large_artifacts": "Source/candidate checkpoint binaries and raw replay/trajectory NPZs remain in the full raw run directory. Input and candidate hashes are included in result manifests.",
            "prism": args.prism, "files": records}
with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as z:
    for name, path in sorted(files.items()):
        z.write(path, name)
    z.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2))
    z.writestr("README.md", "# Sensor pilot review package\n\nRead private-preparation/REVISION_STATUS.md first. This ZIP includes private author details and must not be uploaded wholesale to CMT. The complete raw evidence and model binaries remain in the local run directory listed in PACKAGE_MANIFEST.json. The exact executed code snapshot is distinct from post-run plotting tools and the additional legacy guard test.\n")
with zipfile.ZipFile(package) as z:
    assert z.testzip() is None
    for name, record in records.items():
        assert hashlib.sha256(z.read(name)).hexdigest() == record["sha256"]
(PREP / "SENSOR_PACKAGE_MANIFEST.json").write_text(json.dumps({"zip": str(package), "sha256": hashlib.sha256(package.read_bytes()).hexdigest(), **manifest}, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"package": str(package), "files": len(records), "bytes": package.stat().st_size, "pages": pages, "prism": args.prism}, indent=2))
