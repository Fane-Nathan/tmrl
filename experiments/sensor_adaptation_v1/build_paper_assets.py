"""Build manuscript tables and legible vector figures from a verified pilot."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def build(root, output):
    verification = json.loads((root / "verification.json").read_text())
    assert verification["status"] == "PASS"
    study = json.loads((root / "study_result.json").read_text())
    config = json.loads((root / "config.json").read_text())
    assert not json.loads((root / "manifest.json").read_text())["smoke"]
    output.mkdir(exist_ok=True, parents=True)
    with (root / "summary.csv").open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    rows = []
    for row in summary:
        algorithm = row["algorithm"].capitalize()
        fault = "Noise" if row["condition"] == "sensor_noise" else "Delay"
        fraction = float(row["retained_fraction"])
        rows.append(f"{algorithm} & {fault} & {fraction:g} & {float(row['B_before_mean']):.1f} & {float(row['B_after_mean']):.1f} & ${float(row['A_change_mean']):+.1f}$ \\\\")
    table = r"""\begin{table}[t]
\caption{Fixed-budget pilot outcomes, averaged equally across three source seeds. $B(0)$ and $B(4000)$ are new-condition returns before and after adaptation; $\Delta A$ is the old clean-condition return change. Returns are not success percentages.}
\label{tab:outcomes}
\centering\small
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrr@{}}
\toprule
Controller & Fault & $\rho$ & $B(0)$ & $B(4000)$ & $\Delta A$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table}
"""
    (output / "outcomes_table.tex").write_text(table, encoding="utf-8")
    colors = {0.0: "#c56616", 0.5: "#166b8a"}
    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    for retention in [False, True]:
        fig, axes = plt.subplots(2, 2, figsize=(4.8, 4.55))
        for i, algorithm in enumerate(config["algorithms"]):
            for j, condition in enumerate(config["conditions"]):
                ax = axes[i, j]
                for fraction in config["retained_replay_fractions"]:
                    branches = [r for r in study["results"] if r["algorithm"] == algorithm and r["condition"] == condition and r["retained_fraction"] == fraction]
                    values = []
                    for branch in branches:
                        baseline = branch["curves"][0]["clean"] if retention else 0
                        channel = "clean" if retention else condition
                        values.append([p[channel] - baseline for p in branch["curves"]])
                    values = np.asarray(values)
                    x = np.asarray(config["evaluation_steps"]) / 1000
                    ax.fill_between(x, values.min(0), values.max(0), color=colors[fraction], alpha=0.13)
                    style = "o-" if fraction == 0.0 else "s--"
                    ax.plot(x, values.mean(0), style, markersize=3, linewidth=1.4, color=colors[fraction], label=f"{fraction:.0%} retained")
                    ax.scatter([x[-1]] * len(values), values[:, -1], s=9, color=colors[fraction], alpha=0.7)
                if retention:
                    ax.axhline(0, color="#555555", linewidth=0.6, linestyle="--")
                fault = "noise" if condition == "sensor_noise" else "delay"
                ax.set_title(f"{algorithm.capitalize()} / {fault}")
                ax.set_xlabel("B training steps (thousands)")
                ax.set_ylabel("A return change" if retention else "B return")
                ax.set_xticks([0, 2, 4])
                ax.tick_params(pad=2)
                ax.grid(alpha=0.14)
                ax.spines[["top", "right"]].set_visible(False)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8.5)
        fig.tight_layout(pad=0.4, h_pad=1.5, w_pad=0.9, rect=[0, 0.065, 1, 1])
        name = "sensor_retention" if retention else "sensor_adaptation"
        fig.savefig(output / f"{name}.pdf")
        fig.savefig(output / f"{name}.png", dpi=180)
        plt.close(fig)
    print("Generated eight outcome rows and two publication-size vector figures.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.root, args.output)
