"""Build a paper workflow figure from an unchanged local development capture.

This is a schematic of the executed HalfCheetah pilot, not new experimental data.
The TrackMania screenshot illustrates the motivating application only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "output/icodit_2026/sensor_pilot_draft"
DEST = ROOT / "output/icodit_2026/sensor_pilot_with_trackmania_figure"
CAPTURE = ROOT / "test_output_brain/visual_policy_1k_20260911_v2/capture_preflight_20260912_v2.png"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for path in ORIGINAL.iterdir():
        if path.suffix in {".tex", ".bib", ".cls", ".bst", ".pdf"}:
            shutil.copy2(path, DEST / path.name)
    shutil.copy2(CAPTURE, DEST / "trackmania_development_capture.png")

    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(6.4, 6.0), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    ink, blue, teal = "#182b3a", "#254d77", "#146b63"

    def label(x, y, text, size=11, weight="normal", ha="left", color=ink):
        return ax.text(x, y, text, ha=ha, va="center", fontsize=size,
                       fontweight=weight, color=color, linespacing=1.4)

    def box(x, y, w, h, title, body, color=blue, fill="#f0f4f8"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.009",
                                   linewidth=1.0, edgecolor=color, facecolor=fill))
        label(x+w/2, y+h*0.72, title, size=10.8, weight="bold", ha="center", color=color)
        label(x+w/2, y+h*0.31, body, size=10.2, ha="center")

    def arrow(start, end, color=blue, points=None):
        if points:
            xs, ys = zip(start, *points, end)
            ax.plot(xs[:-1], ys[:-1], color=color, lw=1.05)
            start = (xs[-2], ys[-2])
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                    mutation_scale=10, lw=1.05, color=color))

    label(.02, .970, "(a) Motivating application: TrackMania 2020", 12, "bold")
    photo = fig.add_axes([.02, .704, .52, .243])
    photo.imshow(plt.imread(DEST / "trackmania_development_capture.png"))
    photo.axis("off")
    label(.575, .90, "Real-time visual control", 11.2, "bold", color=blue)
    label(.575, .813, "Goal: learn new conditions\nwhile retaining earlier\ncontrol behavior.", 10.7)
    label(.575, .73, "Saved development capture", 9.3)
    ax.plot([.02, .98], [.679, .679], color="#acb9c2", lw=.8)

    label(.02, .644, "(b) Executed pilot: HalfCheetah-v5", 12, "bold")
    box(.03, .506, .44, .103, "Six source policies", "2 controllers x 3 source seeds")
    box(.55, .506, .42, .103, "Old condition A: clean", "Collect 2,000 replay transitions")
    arrow((.475, .557), (.543, .557))

    box(.03, .332, .44, .116, "New condition B", "Noise or 3-step sensor delay\nCollect 4,000 new transitions")
    arrow((.25, .5), (.25, .454))
    box(.55, .332, .42, .116, "Continued SAC updates", "Old replay fraction: 0 or 0.5\n1,000 update cycles per branch")
    arrow((.475, .39), (.543, .39))
    arrow((.76, .5), (.76, .454))

    box(.17, .205, .66, .078, "Freeze the updated checkpoint", "At 0, 2,000 and 4,000 new B steps", color=teal, fill="#eef6f3")
    arrow((.76, .326), (.76, .289), color=teal)
    box(.03, .042, .44, .11, "Evaluate A: retention", "Clean-condition return change", color=teal, fill="#eef6f3")
    box(.55, .042, .42, .11, "Evaluate B: adaptation", "New-condition return change", color=teal, fill="#eef6f3")
    arrow((.35, .199), (.25, .158), color=teal)
    arrow((.65, .199), (.76, .158), color=teal)
    label(.5, .014, "Same checkpoint for A and B; evaluation episodes stay out of replay.", 9.8, ha="center")

    fig.savefig(DEST / "trackmania_pilot_workflow.pdf", metadata={"Creator": "Matplotlib", "Author": "", "Title": "Motivating application and executed pilot workflow"})
    fig.savefig(DEST / "trackmania_pilot_workflow.png", dpi=220)
    plt.close(fig)

    source = (ORIGINAL / "manuscript.tex").read_text(encoding="utf-8")
    insertion = r"""Figure~\ref{fig:workflow} links the motivating visual-control application to the controlled pilot. The TrackMania panel is a saved development capture that illustrates the real-time setting; it is not an adaptation measurement. The training and evaluation workflow in panel~(b) summarizes the HalfCheetah experiment reported here. It makes explicit that the updated controller is tested on both the earlier clean condition and the new sensor condition.

\begin{figure}[!htbp]
\centering
\includegraphics[width=\textwidth]{trackmania_pilot_workflow.pdf}
\caption{Research context and pilot workflow. (a) An actual TrackMania~2020 development capture from a camera-pipeline preflight; the displayed game clock is not a benchmark lap time. (b) The executed HalfCheetah-v5 procedure. Each source policy branches across two sensor conditions and two replay fractions. Both conditions are evaluated using the same updated weights, with learning disabled. Quantitative results in this paper come from panel~(b).}
\label{fig:workflow}
\end{figure}

"""
    source = source.replace(r"\subsection{Related Work}", insertion + r"\subsection{Related Work}", 1)
    source = source.replace("tables and figures are computed from saved run records.",
                            "tables and result plots are computed from saved run records. The workflow diagram summarizes the implemented pilot, and its TrackMania panel is a saved development screenshot.")
    (DEST / "manuscript.tex").write_text(source, encoding="utf-8")
    provenance = {
        "source_capture": str(CAPTURE.relative_to(ROOT)),
        "source_capture_sha256": hashlib.sha256(CAPTURE.read_bytes()).hexdigest(),
        "embedded_capture_sha256": hashlib.sha256((DEST / "trackmania_development_capture.png").read_bytes()).hexdigest(),
        "capture_context": "Camera-pipeline preflight, recorded 2026-09-12; no controls sent by the profiler.",
        "capture_modified": False,
        "figure_kind": "Schematic with an existing screenshot; no new experimental observations.",
        "experimental_panel": "Executed HalfCheetah sensor-adaptation pilot, 24 branches from six source policies.",
        "original_manuscript_sha256": hashlib.sha256((ORIGINAL / "manuscript.tex").read_bytes()).hexdigest(),
        "new_data_or_result_claims_added": False,
    }
    (DEST / "FIGURE_PROVENANCE.json").write_text(json.dumps(provenance, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"source": str(DEST / "manuscript.tex"), "figure": str(DEST / "trackmania_pilot_workflow.pdf")}, indent=2))


if __name__ == "__main__":
    main()
