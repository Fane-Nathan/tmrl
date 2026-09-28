#!/usr/bin/env python3
"""
Confirmatory Statistical Analysis and Publication Artifact Generator.

Strictly adheres to empirical reinforcement learning methodology:
1. The independent unit of analysis is the INDEPENDENT TRAINING SEED i.
2. For each seed i, R_adaptive_i and R_history_blocked_i are the means across deterministic evaluation episodes.
3. Adaptation gap G_i = R_adaptive_i - R_history_blocked_i.
4. Comparative gap D_i = R_adaptive_i - R_reactive_i.
5. NO PSEUDOREPLICATION: Episode pooling is forbidden for seed-level inference.
6. Computes seed-level 95% bootstrap confidence intervals (B=2000).
7. Evaluates the predetermined decision rules (Cases A, B, C, D).
8. Generates publication-quality LaTeX table, CSV summaries, and figures.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def seed_bootstrap_ci(data: List[float], n_boot: int = 2000, ci: float = 0.95, seed: int = 12345) -> Tuple[float, float, float]:
    """Computes mean and percentile bootstrap confidence interval over seed-level values."""
    arr = np.asarray(data, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        v = float(arr[0])
        return v, v, v

    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(np.mean(sample))

    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, 100.0 * alpha))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    mean = float(np.mean(arr))
    return mean, low, high


def compute_distribution_metrics(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "iqr": 0.0, "min": 0.0, "max": 0.0}
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "iqr": float(q75 - q25),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def evaluate_decision_rule(g_ci: Tuple[float, float, float], g_values: List[float]) -> Dict[str, str]:
    mean_g, low_g, high_g = g_ci
    n = len(g_values)
    signs = [np.sign(v) for v in g_values if abs(v) > 1e-4]
    has_positive = any(s > 0 for s in signs)
    has_negative = any(s < 0 for s in signs)

    if low_g > 0.0 and mean_g > 0.0:
        decision = "CASE_A: SUPPORTED POSITIVE ADAPTATION"
        explanation = (
            f"Mean adaptation gap G = {mean_g:+.3f} is positive and the 95% seed-level bootstrap CI "
            f"[{low_g:+.3f}, {high_g:+.3f}] strictly excludes zero on the positive side. "
            "Evidence supports reproducible memory-enabled cross-mechanism adaptation."
        )
    elif low_g <= 0.0 <= high_g:
        if has_positive and has_negative and (max(g_values) - min(g_values) > 3.0):
            decision = "CASE_D: SEED-SENSITIVE / UNSTABLE"
            explanation = (
                f"Seed adaptation gaps vary strongly with inconsistent signs ({g_values}). "
                f"The 95% bootstrap CI [{low_g:+.3f}, {high_g:+.3f}] spans zero. "
                "Cross-mechanism adaptation is highly seed-sensitive and unreliable under this training regime."
            )
        else:
            decision = "CASE_B: ADAPTATION NOT ESTABLISHED"
            explanation = (
                f"The 95% seed-level bootstrap CI [{low_g:+.3f}, {high_g:+.3f}] includes zero (mean G = {mean_g:+.3f}). "
                "Cross-mechanism adaptation to unseen actuator latency is not established."
            )
    elif high_g < 0.0 and mean_g < 0.0:
        decision = "CASE_C: NEGATIVE HISTORY EFFECT"
        explanation = (
            f"Mean adaptation gap G = {mean_g:+.3f} is negative and the 95% bootstrap CI "
            f"[{low_g:+.3f}, {high_g:+.3f}] excludes zero on the negative side. "
            "History conditioning reduces performance under unseen action latency in this training regime."
        )
    else:
        decision = "CASE_B: ADAPTATION NOT ESTABLISHED"
        explanation = f"Confidence interval [{low_g:+.3f}, {high_g:+.3f}] does not satisfy positive support criteria."

    return {"decision": decision, "explanation": explanation}


def load_confirmatory_data(results_dir: Path) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    transformer_reports: Dict[int, Dict[str, Any]] = {}
    reactive_reports: Dict[int, Dict[str, Any]] = {}

    for path in results_dir.glob("report_seed_*.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            seed = int(data["seed"])
            transformer_reports[seed] = data

    for path in results_dir.glob("reactive_report_seed_*.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            seed = int(data["seed"])
            reactive_reports[seed] = data

    return transformer_reports, reactive_reports


def generate_confirmatory_artifacts(
    results_dir: str | Path,
    output_dir: str | Path,
    protocol_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol = None
    protocol_sha256 = None
    if protocol_path is not None:
        import hashlib
        protocol_path = Path(protocol_path)
        protocol_bytes = protocol_path.read_bytes()
        protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
        protocol = json.loads(protocol_bytes.decode("utf-8"))

    t_data, r_data = load_confirmatory_data(results_dir)
    all_t_seeds = sorted(t_data.keys())
    all_r_seeds = sorted(r_data.keys())
    common_seeds = sorted(set(all_t_seeds).intersection(set(all_r_seeds)))

    if protocol is not None:
        expected_seeds = {int(x) for x in protocol["seed_list"]}
        minimum_seeds = int(protocol.get("minimum_acceptable_seeds", len(expected_seeds)))
        unexpected = sorted((set(all_t_seeds) | set(all_r_seeds)) - expected_seeds)
        if unexpected:
            raise RuntimeError(f"Results contain seeds outside protocol seed_list: {unexpected}")
        if len(all_t_seeds) < minimum_seeds:
            raise RuntimeError(
                f"Only {len(all_t_seeds)} transformer seeds are complete; "
                f"protocol requires at least {minimum_seeds}."
            )
        for label, reports in (("transformer", t_data), ("reactive", r_data)):
            for seed, report in reports.items():
                if report.get("protocol_sha256") != protocol_sha256:
                    raise RuntimeError(
                        f"{label} seed {seed} does not match the supplied protocol hash"
                    )

    conditions = ["test_latency", "test_dead", "test_sign_flip"]
    condition_labels = {
        "test_latency": "Cross-mechanism (action\\_latency)",
        "test_dead": "Within-mechanism (dead\\_actuator)",
        "test_sign_flip": "Within-mechanism (sign\\_flip)",
    }

    stats_output: Dict[str, Any] = {
        "num_transformer_seeds": len(all_t_seeds),
        "num_reactive_seeds": len(all_r_seeds),
        "transformer_seeds": all_t_seeds,
        "reactive_seeds": all_r_seeds,
        "matched_seeds": common_seeds,
        "protocol_version": protocol.get("protocol_version") if protocol else None,
        "protocol_sha256": protocol_sha256,
        "primary_ablation": "history_blocked_context_matched",
        "endpoints": {},
    }

    # Seed-level results CSV
    seed_csv_path = results_dir / "confirmatory_seed_results.csv"
    seed_rows = []

    for cond in conditions:
        ad_seed_vals: List[float] = []
        nh_seed_vals: List[float] = []
        gap_seed_vals: List[float] = []
        react_seed_vals: List[float] = []
        comp_d_vals: List[float] = []

        for s in all_t_seeds:
            c_dict = t_data[s]["test_results"].get(cond, {})
            ad_m = float(c_dict["adaptive_mean"])
            nh_m = float(c_dict["no_history_mean"])
            g_m = float(c_dict["adaptation_gap"])
            ad_seed_vals.append(ad_m)
            nh_seed_vals.append(nh_m)
            gap_seed_vals.append(g_m)

            r_m = None
            d_m = None
            if s in r_data and cond in r_data[s]["test_results"]:
                r_m = float(r_data[s]["test_results"][cond]["mean_return"])
                d_m = ad_m - r_m
                comp_d_vals.append(d_m)

            seed_rows.append({
                "condition": cond,
                "seed": s,
                "adaptive_return": f"{ad_m:.4f}",
                "no_history_return": f"{nh_m:.4f}",
                "adaptation_gap_G": f"{g_m:.4f}",
                "reactive_return": f"{r_m:.4f}" if r_m is not None else "",
                "control_advantage_D": f"{d_m:.4f}" if d_m is not None else "",
            })

        for s in all_r_seeds:
            if s not in all_t_seeds:
                r_m = float(r_data[s]["test_results"][cond]["mean_return"])
                seed_rows.append({
                    "condition": cond,
                    "seed": s,
                    "adaptive_return": "",
                    "no_history_return": "",
                    "adaptation_gap_G": "",
                    "reactive_return": f"{r_m:.4f}",
                    "control_advantage_D": "",
                })

        for s in all_r_seeds:
            if cond in r_data[s]["test_results"]:
                react_seed_vals.append(float(r_data[s]["test_results"][cond]["mean_return"]))

        ad_ci = seed_bootstrap_ci(ad_seed_vals)
        nh_ci = seed_bootstrap_ci(nh_seed_vals)
        gap_ci = seed_bootstrap_ci(gap_seed_vals)
        react_ci = seed_bootstrap_ci(react_seed_vals)
        d_ci = seed_bootstrap_ci(comp_d_vals) if comp_d_vals else (0.0, 0.0, 0.0)

        decision_info = evaluate_decision_rule(gap_ci, gap_seed_vals) if cond == "test_latency" else None

        stats_output["endpoints"][cond] = {
            "adaptive": {
                "ci_95": ad_ci,
                "metrics": compute_distribution_metrics(ad_seed_vals),
                "seed_values": ad_seed_vals,
            },
            "no_history": {
                "ci_95": nh_ci,
                "metrics": compute_distribution_metrics(nh_seed_vals),
                "seed_values": nh_seed_vals,
            },
            "adaptation_gap_G": {
                "ci_95": gap_ci,
                "metrics": compute_distribution_metrics(gap_seed_vals),
                "seed_values": gap_seed_vals,
            },
            "reactive_baseline": {
                "ci_95": react_ci,
                "metrics": compute_distribution_metrics(react_seed_vals),
                "seed_values": react_seed_vals,
            },
            "control_advantage_D": {
                "ci_95": d_ci,
                "metrics": compute_distribution_metrics(comp_d_vals),
                "seed_values": comp_d_vals,
            },
            "decision": decision_info,
        }

    # Write seed results CSV
    with open(seed_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["condition", "seed", "adaptive_return", "no_history_return", "adaptation_gap_G", "reactive_return", "control_advantage_D"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(seed_rows)

    # Write summary CSV
    summary_csv_path = results_dir / "confirmatory_summary.csv"
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Condition", "N_Seeds", "Adaptive_Mean", "Adaptive_CI95_Low", "Adaptive_CI95_High",
                         "HistoryBlocked_Mean", "HistoryBlocked_CI95_Low", "HistoryBlocked_CI95_High",
                         "Gap_G_Mean", "Gap_G_CI95_Low", "Gap_G_CI95_High",
                         "Reactive_Mean", "Reactive_CI95_Low", "Reactive_CI95_High",
                         "Advantage_D_Mean", "Advantage_D_CI95_Low", "Advantage_D_CI95_High"])
        for cond in conditions:
            ep = stats_output["endpoints"][cond]
            writer.writerow([
                cond,
                len(all_t_seeds),
                f"{ep['adaptive']['ci_95'][0]:.4f}", f"{ep['adaptive']['ci_95'][1]:.4f}", f"{ep['adaptive']['ci_95'][2]:.4f}",
                f"{ep['no_history']['ci_95'][0]:.4f}", f"{ep['no_history']['ci_95'][1]:.4f}", f"{ep['no_history']['ci_95'][2]:.4f}",
                f"{ep['adaptation_gap_G']['ci_95'][0]:.4f}", f"{ep['adaptation_gap_G']['ci_95'][1]:.4f}", f"{ep['adaptation_gap_G']['ci_95'][2]:.4f}",
                f"{ep['reactive_baseline']['ci_95'][0]:.4f}", f"{ep['reactive_baseline']['ci_95'][1]:.4f}", f"{ep['reactive_baseline']['ci_95'][2]:.4f}",
                f"{ep['control_advantage_D']['ci_95'][0]:.4f}", f"{ep['control_advantage_D']['ci_95'][1]:.4f}", f"{ep['control_advantage_D']['ci_95'][2]:.4f}",
            ])

    # Write statistics JSON
    stats_json_path = results_dir / "confirmatory_statistics.json"
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(stats_output, f, indent=2)

    # Generate LaTeX Table
    latex_table_path = results_dir / "confirmatory_results_table.tex"
    paper_latex_table_path = output_dir / "confirmatory_results_table.tex"
    n_s = len(all_t_seeds)
    latex_str = (
        "% Auto-generated by generate_confirmatory_results.py - DO NOT EDIT MANUALLY\n"
        "\\begin{table*}[!t]\n"
        "\\centering\n"
        f"\\caption{{Confirmatory fault evaluation across {n_s} independent training seeds under strict validation-selected checkpointing (Protocol v2, 95\\% seed-level bootstrap confidence intervals over {n_s} seeds $\\times$ 25 deterministic episodes).}}\n"
        "\\label{tab:confirmatory_results}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lccccc}\n"
        "\\toprule\n"
        "\\textbf{Held-Out Condition} & \\textbf{Transformer} & \\textbf{Transformer} & \\textbf{Adaptation Gap} & \\textbf{Reactive Baseline} & \\textbf{Control Advantage} \\\\\n"
        " & \\textbf{(Adaptive History)} & \\textbf{(History-Blocked)} & ($G = R_{\\text{adapt}} - R_{\\text{blocked}}$) & \\textbf{(MLP REDQ)} & ($D = R_{\\text{adapt}} - R_{\\text{reactive}}$) \\\\\n"
        "\\midrule\n"
    )

    for cond in conditions:
        ep = stats_output["endpoints"][cond]
        c_label = condition_labels[cond]
        ad_m, ad_l, ad_h = ep["adaptive"]["ci_95"]
        nh_m, nh_l, nh_h = ep["no_history"]["ci_95"]
        g_m, g_l, g_h = ep["adaptation_gap_G"]["ci_95"]
        r_m, r_l, r_h = ep["reactive_baseline"]["ci_95"]
        d_m, d_l, d_h = ep["control_advantage_D"]["ci_95"]

        latex_str += f"{c_label} & ${ad_m:+.1f} \\; [{ad_l:+.1f}, {ad_h:+.1f}]$ & ${nh_m:+.1f} \\; [{nh_l:+.1f}, {nh_h:+.1f}]$ & ${g_m:+.1f} \\; [{g_l:+.1f}, {g_h:+.1f}]$ & ${r_m:+.1f} \\; [{r_l:+.1f}, {r_h:+.1f}]$ & ${d_m:+.1f} \\; [{d_l:+.1f}, {d_h:+.1f}]$ \\\\\n"

    latex_str += (
        "\\bottomrule\n"
        f"\\multicolumn{{6}}{{l}}{{\\footnotesize All metrics reflect seed-level distributions ($n={n_s}$ independent training runs). Brackets denote 95\\% bootstrap confidence intervals derived strictly across training seeds.}}\\\\\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\end{table*}\n"
    )

    with open(latex_table_path, "w", encoding="utf-8") as f:
        f.write(latex_str)

    try:
        with open(paper_latex_table_path, "w", encoding="utf-8") as f:
            f.write(latex_str)
    except Exception as e:
        print(f"Warning: could not write to {paper_latex_table_path}: {e}")

    # Generate Publication Figures
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "axes.edgecolor": "#2c3e50",
        "axes.linewidth": 0.8,
        "grid.color": "#eaeaea",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
    })

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), dpi=300)

    cond_labels_short = ["Latency\n(Temporal)", "Dead Actuator\n(Extreme)", "Sign Flip\n(Extreme)"]
    x = np.arange(len(conditions))
    width = 0.26

    # Colors
    c_adapt = "#1f4e79"
    c_nohist = "#595959"
    c_react = "#2e7d32"
    c_gap = "#c00000"

    ax1 = axes[0]
    ax1.grid(True, axis="y", zorder=0)

    ad_means = [stats_output["endpoints"][c]["adaptive"]["ci_95"][0] for c in conditions]
    ad_err_low = [ad_means[i] - stats_output["endpoints"][c]["adaptive"]["ci_95"][1] for i, c in enumerate(conditions)]
    ad_err_high = [stats_output["endpoints"][c]["adaptive"]["ci_95"][2] - ad_means[i] for i, c in enumerate(conditions)]

    nh_means = [stats_output["endpoints"][c]["no_history"]["ci_95"][0] for c in conditions]
    nh_err_low = [nh_means[i] - stats_output["endpoints"][c]["no_history"]["ci_95"][1] for i, c in enumerate(conditions)]
    nh_err_high = [stats_output["endpoints"][c]["no_history"]["ci_95"][2] - nh_means[i] for i, c in enumerate(conditions)]

    r_means = [stats_output["endpoints"][c]["reactive_baseline"]["ci_95"][0] for c in conditions]
    r_err_low = [r_means[i] - stats_output["endpoints"][c]["reactive_baseline"]["ci_95"][1] for i, c in enumerate(conditions)]
    r_err_high = [stats_output["endpoints"][c]["reactive_baseline"]["ci_95"][2] - r_means[i] for i, c in enumerate(conditions)]

    ax1.bar(x - width, ad_means, width, yerr=[ad_err_low, ad_err_high], capsize=3.5, label="Transformer (Adaptive)", color=c_adapt, alpha=0.9, zorder=3)
    ax1.bar(x, nh_means, width, yerr=[nh_err_low, nh_err_high], capsize=3.5, label="Transformer (History-Blocked)", color=c_nohist, alpha=0.85, zorder=3)
    ax1.bar(x + width, r_means, width, yerr=[r_err_low, r_err_high], capsize=3.5, label="Reactive Baseline (MLP)", color=c_react, alpha=0.85, zorder=3)

    ax1.set_ylabel("Expected Return ($R$)", fontsize=10, fontweight="bold")
    ax1.set_title("Held-Out Fault Policy Return", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cond_labels_short, fontsize=9)
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.legend(loc="best", framealpha=0.9, fontsize=8.5)

    ax2 = axes[1]
    ax2.grid(True, axis="y", zorder=0)

    g_means = [stats_output["endpoints"][c]["adaptation_gap_G"]["ci_95"][0] for c in conditions]
    g_err_low = [g_means[i] - stats_output["endpoints"][c]["adaptation_gap_G"]["ci_95"][1] for i, c in enumerate(conditions)]
    g_err_high = [stats_output["endpoints"][c]["adaptation_gap_G"]["ci_95"][2] - g_means[i] for i, c in enumerate(conditions)]

    bars = ax2.bar(x, g_means, width * 1.5, yerr=[g_err_low, g_err_high], capsize=4, color=c_gap, alpha=0.85, zorder=3)
    ax2.set_ylabel("Adaptation Gap ($G = R_{\\mathrm{adapt}} - R_{\\mathrm{blocked}}$)", fontsize=10, fontweight="bold")
    ax2.set_title("In-Context Adaptation Gap", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cond_labels_short, fontsize=9)
    ax2.axhline(0, color="black", linestyle="--", linewidth=1.0)

    for i, bar in enumerate(bars):
        c = conditions[i]
        m = g_means[i]
        offset = 0.15 if m >= 0 else -0.35
        ax2.annotate(f"{m:+.2f}", (bar.get_x() + bar.get_width() / 2, m + offset), ha="center", va="center", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    fig_png_path = output_dir / "confirmatory_figure.png"
    fig_pdf_path = output_dir / "confirmatory_figure.pdf"
    plt.savefig(fig_png_path, dpi=300, bbox_inches="tight")
    plt.savefig(fig_pdf_path, bbox_inches="tight")
    plt.close()

    print(f"\nConfirmatory analysis complete across {n_s} seeds:")
    print(f"  Summary CSV:    {summary_csv_path}")
    print(f"  Seed CSV:       {seed_csv_path}")
    print(f"  LaTeX Table:    {latex_table_path}")
    print(f"  Statistics:     {stats_json_path}")
    print(f"  Figure PNG:     {fig_png_path}")
    print(f"  Figure PDF:     {fig_pdf_path}")

    primary_dec = stats_output["endpoints"]["test_latency"].get("decision", {})
    if primary_dec:
        print("\n" + "=" * 60)
        print("PREDETERMINED PRIMARY SCIENTIFIC DECISION:")
        print(f"  {primary_dec['decision']}")
        print(f"  {primary_dec['explanation']}")
        print("=" * 60 + "\n")

    return stats_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate confirmatory benchmark statistical report and figures")
    parser.add_argument("--results_dir", type=str, default="./fault_benchmark_results_v2")
    parser.add_argument("--output_dir", type=str, default="../paper_research")
    parser.add_argument("--protocol", type=str, default=None, help="Optional path to protocol JSON")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    res_dir = repo_root / args.results_dir
    out_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else (repo_root / args.output_dir)

    generate_confirmatory_artifacts(res_dir, out_dir, args.protocol)
