"""Build camera-ready figures, tables and number macros from saved analysis outputs.

Every number in the results text is a macro \\R{key} defined here from analysis files, so no value is
transcribed by hand. A missing key renders as a bold ??key in the PDF.

  python scripts/build_camera_ready_assets.py --env hc=output/sensor_adaptation/extension_20261031_v1 \
      --env ant=output/sensor_adaptation/ant_20261031_v1 --latency output/sensor_adaptation/latency_20261031.json
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "output/icodit_2026/camera_ready"
ENV_NAMES = {"hc": "HalfCheetah-v5", "ant": "Ant-v5"}
FAULT_KEYS = {"sensor_noise": "noise", "sensor_delay": "delay", "clean": "clean"}
ALG_LABEL = {"transformer": "Transformer", "reactive": "Reactive"}
COLORS = {("transformer", 0.0): "#1f5f9e", ("transformer", 0.5): "#7fb1e0",
          ("reactive", 0.0): "#b5541c", ("reactive", 0.5): "#efa774"}


def num(x, digits=0, sign=False):
    """Returns are reported as integers by default; rates and residuals pass digits explicitly."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    text = f"{abs(x):,.{digits}f}".replace(",", "{,}")
    if float(f"{abs(x):.{digits}f}") == 0:
        return text
    if x < 0:
        return r"\ensuremath{-}" + text
    return ("+" if sign else "") + text


def pval(p):
    if p < 0.001:
        return r"\ensuremath{<}0.001"
    return f"{p:.4f}" if p < 0.1 else f"{p:.2f}"


def load_seed_rows(root):
    rows = list(csv.DictReader((root / "analysis/seed_metrics.csv").open(encoding="utf-8")))
    for r in rows:
        for k in ("A_before", "A_after", "A_change", "B_before", "B_after", "B_change", "B_curve_mean"):
            r[k] = float(r[k])
        r["seed"], r["budget"], r["retained_fraction"] = int(r["seed"]), int(r["budget"]), float(r["retained_fraction"])
    return rows


def tci(values):
    v = np.asarray(values, float)
    half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), v.mean() - half, v.mean() + half


def macros_for_env(env, root, macros):
    summary = json.loads((root / "analysis/summary.json").read_text(encoding="utf-8"))
    final = max(int(b) for b in summary["results"])
    rows = load_seed_rows(root)
    macros[f"{env}-seeds"] = str(len(summary["seeds"]))
    macros[f"{env}-branches"] = str(summary["branches"])
    results = [json.loads(p.read_text(encoding="utf-8")) for p in root.glob("*/*/result.json")]
    exact = sum(r.get("status") == "COMPLETE" and r.get("environment_steps") == 20000 and r.get("updates") == 5000
                for r in results)
    macros[f"{env}-budget-exact"] = str(exact)
    groups = defaultdict(list)
    for r in rows:
        if r["budget"] == final:
            groups[(r["algorithm"], r["condition"], r["retained_fraction"])].append(r)
    for (alg, cond, rho), items in groups.items():
        key = f"{env}-{alg}-{FAULT_KEYS[cond]}-{int(rho * 100)}"
        for metric, short in (("A_change", "dA"), ("B_change", "dB"), ("A_before", "A0"), ("B_before", "B0"),
                              ("A_after", "A1"), ("B_after", "B1"),
                              ("B_curve_mean", "curve")):
            mean, lo, hi = tci([r[metric] for r in items])
            macros[f"{key}-{short}"] = num(mean, sign=short in ("dA", "dB"))
            macros[f"{key}-{short}-ci"] = f"[{num(lo)}, {num(hi)}]"
            macros[f"{key}-{short}-pos"] = str(sum(r[metric] > 0 for r in items))
            macros[f"{key}-{short}-neg"] = str(sum(r[metric] < 0 for r in items))
    # Early budgets: 2,000 steps (warm-start transient) and the pilot's 4,000 steps, for all eight seeds and for the
    # pilot's seeds 42-44 (which reproduce it exactly).
    for budget, tag in ((2000, "2k"), (4000, "4k")):
        early = defaultdict(list)
        for r in rows:
            if r["budget"] == budget:
                early[(r["algorithm"], r["condition"], r["retained_fraction"])].append(r)
        for (alg, cond, rho), items in early.items():
            key = f"{env}-{alg}-{FAULT_KEYS[cond]}-{int(rho * 100)}"
            for metric, short in (("A_change", "dA"), ("B_change", "dB")):
                macros[f"{key}-{short}-{tag}"] = num(float(np.mean([r[metric] for r in items])), sign=True)
                pilot = [r[metric] for r in items if r["seed"] in (42, 43, 44)]
                if len(pilot) == 3:
                    macros[f"{key}-{short}-{tag}-pilot"] = num(float(np.mean(pilot)), sign=True)
                macros[f"{key}-{short}-{tag}-neg"] = str(sum(r[metric] < 0 for r in items))
    # Initial clean return per controller (identical across branches of one source model).
    for alg in ALG_LABEL:
        initial = {r["seed"]: r["A_before"] for r in rows if r["algorithm"] == alg and r["budget"] == final}
        if initial:
            macros[f"{env}-{alg}-A0"] = num(float(np.mean(list(initial.values()))))
    result = summary["results"][str(final)]
    for row in result["primary_replay_effect"]:
        key = f"{env}-D-{row['algorithm']}-{FAULT_KEYS[row['fault']]}-{'dA' if row['metric'] == 'A_change' else 'dB'}"
        macros[key] = num(row["mean"], sign=True)
        macros[f"{key}-ci"] = f"[{num(row['t_ci_low'])}, {num(row['t_ci_high'])}]"
        macros[f"{key}-boot"] = f"[{num(row['boot_ci_low'])}, {num(row['boot_ci_high'])}]"
        macros[f"{key}-p"] = pval(row["wilcoxon_p"])
        macros[f"{key}-holm"] = pval(row["holm_p"])
        macros[f"{key}-pos"], macros[f"{key}-neg"] = str(row["n_positive"]), str(row["n_negative"])
        macros[f"{key}-dz"] = num(row["d_z"], 2)
    for row in result.get("fault_specific_forgetting", []):
        key = f"{env}-FS-{row['algorithm']}-{FAULT_KEYS[row['fault']]}"
        macros[key] = num(row["mean"], sign=True)
        macros[f"{key}-ci"] = f"[{num(row['t_ci_low'])}, {num(row['t_ci_high'])}]"
        macros[f"{key}-holm"] = pval(row["holm_p"])
        macros[f"{key}-pos"], macros[f"{key}-neg"] = str(row["n_positive"]), str(row["n_negative"])
    for row in result["arm_changes"]:
        key = f"{env}-arm-{row['algorithm']}-{FAULT_KEYS[row['condition']]}-{int(float(row['retained_fraction']) * 100)}-{'dA' if row['metric'] == 'A_change' else 'dB'}"
        macros[f"{key}-holm"] = pval(row["holm_p"])
    if "pilot_reproducibility" in summary:
        rep = summary["pilot_reproducibility"]
        macros[f"{env}-repro-branches"] = str(rep["branches_compared"])
        macros[f"{env}-repro-identical"] = "identical" if rep["all_identical"] else "not identical"
        macros[f"{env}-repro-maxdiff"] = num(max(c["max_abs_difference"] or 0 for c in rep["checks"]), 3)
    bias_path = root / "analysis/critic_bias_rows.csv"
    if bias_path.exists():
        bias = list(csv.DictReader(bias_path.open(encoding="utf-8")))
        agg = defaultdict(list)
        for r in bias:
            agg[(r["algorithm"], r["condition"], float(r["retained_fraction"]), int(r["budget"]), r["data"])].append(r)
        for (alg, cond, rho, budget, data), items in agg.items():
            stage = "src" if budget == 0 else "fin"
            key = f"{env}-bel-{alg}-{FAULT_KEYS[cond]}-{int(rho * 100)}-{stage}-{data[0]}"
            macros[key] = num(float(np.mean([float(r["bellman_rmse"]) for r in items])), 2)
            macros[key.replace("-bel-", "-bias-")] = num(float(np.mean([float(r["bias"]) for r in items])))
        summ = list(csv.DictReader((root / "analysis/critic_bias_summary.csv").open(encoding="utf-8")))
        for r in summ:
            if r["contrast"] == "final rho0.5 - rho0" and r["metric"] == "bellman_rmse":
                key = f"{env}-belD-{r['algorithm']}-{FAULT_KEYS[r['condition']]}-{r['data'][0]}"
                macros[key] = num(float(r["mean"]), 2, sign=True)
                macros[f"{key}-ci"] = f"[{num(float(r['t_ci_low']), 2)}, {num(float(r['t_ci_high']), 2)}]"
                macros[f"{key}-holm"] = pval(float(r["holm_p"])) if r.get("holm_p") else "n/a"
                macros[f"{key}-pos"], macros[f"{key}-neg"] = r["n_positive"], r["n_negative"]
        bias_gap_macros(env, bias, macros)
    return rows, final


def bias_gap_macros(env, bias, macros):
    """Exploratory, not pre-specified: per seed, Monte-Carlo value bias on corrupted data minus that on clean data,
    computed within one checkpoint so the shared soft-value offset cancels. Positive: corrupted-data values are
    overestimated relative to clean-data values."""
    value = {(r["algorithm"], r["condition"], float(r["retained_fraction"]), int(r["budget"]), r["data"],
              int(r["seed"])): float(r["bias"]) for r in bias}
    final = max(k[3] for k in value)
    seeds = sorted({k[5] for k in value})
    largest = 0.0
    for alg in ALG_LABEL:
        for cond in ("sensor_noise", "sensor_delay"):
            gaps = {}
            for rho, budget, stage in ((0.0, 0, "src"), (0.0, final, "0"), (0.5, final, "50")):
                try:
                    g = np.array([value[(alg, cond, rho, budget, "B_sensor", s)] - value[(alg, cond, rho, budget, "A_clean", s)]
                                  for s in seeds])
                except KeyError:
                    break
                gaps[stage] = g
                largest = max(largest, abs(float(g.mean())))
                key = f"{env}-gap-{alg}-{FAULT_KEYS[cond]}-{stage}"
                macros[key] = num(float(g.mean()), sign=True)
                macros[f"{key}-pos"] = str(int((g > 0).sum()))
            if len(gaps) == 3:
                diff = gaps["50"] - gaps["0"]
                mean, lo, hi = tci(diff)
                key = f"{env}-gapD-{alg}-{FAULT_KEYS[cond]}"
                macros[key] = num(mean, sign=True)
                macros[f"{key}-ci"] = f"[{num(lo)}, {num(hi)}]"
                macros[f"{key}-pos"], macros[f"{key}-neg"] = str(int((diff > 0).sum())), str(int((diff < 0).sum()))
    macros[f"{env}-gap-absmax"] = str(int(np.ceil(largest)))  # bound on |mean gap| over all arms and stages


def pilot_equivalent_macros(env, root, macros, seeds=(42, 43, 44), episodes=3, budget=4000):
    """The pilot's own metric (seeds 42-44, evaluation episodes 0-2, 4,000 steps) from the extension records.

    These branches reproduce the pilot's episodes exactly, so the values equal the pilot's published means."""
    per_arm = defaultdict(list)
    for shard in sorted(p.parent for p in root.glob("*/status.json")) or [root]:
        for spec_path in sorted(shard.glob("*/branch.json")):
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            if int(spec["seed"]) not in seeds or spec["condition"] == "clean":
                continue
            returns = defaultdict(list)
            for row in csv.DictReader((spec_path.parent / "evaluations.csv").open(encoding="utf-8")):
                if int(row["episode"]) < episodes:
                    returns[(int(row["adaptation_steps"]), row["condition"])].append(float(row["return"]))
            if (budget, "clean") not in returns:
                return
            cond = spec["condition"]
            per_arm[(spec["algorithm"], cond, float(spec["retained_fraction"]))].append((
                np.mean(returns[(budget, "clean")]) - np.mean(returns[(0, "clean")]),
                np.mean(returns[(budget, cond)]) - np.mean(returns[(0, cond)])))
    for (alg, cond, rho), values in per_arm.items():
        if len(values) == len(seeds):
            key = f"{env}-{alg}-{FAULT_KEYS[cond]}-{int(rho * 100)}"
            macros[f"{key}-dA-pilot4k"] = num(float(np.mean([v[0] for v in values])), sign=True)
            macros[f"{key}-dB-pilot4k"] = num(float(np.mean([v[1] for v in values])), sign=True)


VEL_INDEX = {"hc": 8, "ant": 13}  # torso/root forward velocity in the true (audit-only) sensor vector


def behavior_macros(env, root, macros, final, horizon=1000):
    """Source forward speed and early-termination rates from saved evaluation records."""
    velocity = defaultdict(dict)
    term0 = defaultdict(dict)      # (alg, measured) -> {seed: flags}; budget-0 evaluations repeat across branches
    term_final = defaultdict(list)  # (alg, condition, rho, measured) -> flags
    for shard in sorted(p.parent for p in root.glob("*/status.json")) or [root]:
        for spec_path in sorted(shard.glob("*/branch.json")):
            folder = spec_path.parent
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            alg, seed, cond = spec["algorithm"], int(spec["seed"]), spec["condition"]
            rho = float(spec["retained_fraction"])
            if seed not in velocity[alg]:
                files = sorted(folder.glob("trajectory_0_clean_*.npz"))
                if files:
                    velocity[alg][seed] = float(np.mean([np.load(f)["true_sensor"][:, VEL_INDEX[env]].mean()
                                                         for f in files]))
            flags = defaultdict(list)
            for row in csv.DictReader((folder / "evaluations.csv").open(encoding="utf-8")):
                flags[(int(row["adaptation_steps"]), row["condition"])].append(int(row["length"]) < horizon)
            for (budget, measured), values in flags.items():
                if budget == 0 and seed not in term0[(alg, measured)]:
                    term0[(alg, measured)][seed] = values
                elif budget == final:
                    term_final[(alg, cond, rho, measured)].extend(values)
    if velocity:
        macros[f"{env}-xvel-absmax"] = num(max(abs(v) for per_seed in velocity.values() for v in per_seed.values()), 3)
    for alg, per_seed in velocity.items():
        macros[f"{env}-{alg}-xvel"] = num(float(np.mean(list(per_seed.values()))), 2)
        macros[f"{env}-{alg}-xvel-min"] = num(min(per_seed.values()), 2)
        macros[f"{env}-{alg}-xvel-max"] = num(max(per_seed.values()), 2)
    for (alg, measured), per_seed in term0.items():
        pooled = [f for values in per_seed.values() for f in values]
        macros[f"{env}-{alg}-term0-{FAULT_KEYS[measured]}"] = num(100 * float(np.mean(pooled)), 0)
    for (alg, cond, rho, measured), values in term_final.items():
        side = "A" if measured == "clean" else "B"
        macros[f"{env}-{alg}-{FAULT_KEYS[cond]}-{int(rho * 100)}-termfin-{side}"] = num(100 * float(np.mean(values)), 0)


def curve_figure(panels, path, row_height=1.6):
    """One row per (task, fault) with sensor- and clean-condition panels, at LNCS text width (4.8 in), so fonts
    print at their nominal size. `panels` is a list of (env, seed rows)."""
    spec = [(env, rows, fault) for env, rows in panels for fault in ("sensor_noise", "sensor_delay")
            if any(r["condition"] == fault for r in rows)]
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "pdf.fonttype": 42})
    height = row_height * len(spec) + 0.62
    fig, axes = plt.subplots(len(spec), 2, figsize=(4.8, height), sharex=True, squeeze=False)
    for i, (env, rows, fault) in enumerate(spec):
        budgets = sorted({0} | {r["budget"] for r in rows})
        for j, (metric, title) in enumerate((("B_change", "Sensor condition $B$"),
                                             ("A_change", "Clean condition $A$ (retention)"))):
            ax = axes[i, j]
            series = [(alg, fault, rho) for alg in ALG_LABEL for rho in (0.0, 0.5)]
            if metric == "A_change":
                series += [(alg, "clean", 0.0) for alg in ALG_LABEL]
            for alg, cond, rho in series:
                means, los, his = [0.0], [0.0], [0.0]
                for b in budgets[1:]:
                    vals = [r[metric] for r in rows if r["algorithm"] == alg and r["condition"] == cond
                            and r["retained_fraction"] == rho and r["budget"] == b]
                    if len(vals) < 2:
                        break
                    m, lo, hi = tci(vals)
                    means.append(m); los.append(lo); his.append(hi)
                if len(means) < 2:
                    continue
                xs = np.asarray(budgets[:len(means)]) / 1000
                if cond == "clean":
                    ax.plot(xs, means, color=COLORS[(alg, 0.0)], ls=":", lw=1.1, label=f"{ALG_LABEL[alg]}, clean control")
                else:
                    color = COLORS[(alg, rho)]
                    ax.plot(xs, means, color=color, lw=1.1, marker="o" if rho == 0 else "s", ms=2.2,
                            label=f"{ALG_LABEL[alg]}, $\\rho$={rho:g}")
                    ax.fill_between(xs, los, his, color=color, alpha=0.14, lw=0)
            ax.axhline(0, color="#777", lw=0.6)
            if i == 0:
                ax.set_title(title)
            if j == 0:
                word = "Noise" if fault == "sensor_noise" else "Delay"
                ax.set_ylabel(f"{ENV_NAMES[env].split('-')[0]}, {word.lower()}\nreturn change" if len(panels) > 1
                              else f"{word}: return change")
            if i == len(spec) - 1:
                ax.set_xlabel("New-condition steps (thousands)")
            ax.set_xticks([b / 1000 for b in budgets if b % 4000 == 0])
            ax.grid(alpha=0.25, lw=0.4)
            ax.tick_params(length=2, pad=1.5)
    handles, labels = axes[-1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, handlelength=2.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.40 / height, 1, 1), h_pad=0.6, w_pad=0.8)
    title = " and ".join(ENV_NAMES[env] for env, _ in panels)
    fig.savefig(path, metadata={"Creator": "Matplotlib", "Author": "", "Title": f"{title} adaptation and retention"})
    plt.close(fig)


def task_header(env, ncols, first):
    rule = "" if first else r"\addlinespace[2pt]" + "\n"
    return rule + r"\multicolumn{" + str(ncols) + r"}{@{}l}{\emph{" + ENV_NAMES[env] + r"}} \\"


def outcomes_table(envs, macros):
    lines = [r"\begin{table}[t]",
             r"\caption{Outcomes at 20,000 steps: mean over eight source seeds with 95\% $t$ intervals. $B(0)$ is the zero-step sensor-condition return; $\Delta B$ and $\Delta A$ are the changes in sensor- and clean-condition return. For the clean control, $B=A$.}",
             r"\label{tab:outcomes}", r"\centering\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrr@{}}", r"\toprule",
             r"Controller & Fault & $\rho$ & $B(0)$ & $\Delta B$ [95\% CI] & $\Delta A$ [95\% CI] \\", r"\midrule"]
    for index, env in enumerate(envs):
        rows = []
        for alg in ALG_LABEL:
            for fault, label in (("noise", "Noise"), ("delay", "Delay"), ("clean", "Clean control")):
                for rho in ((0, 50) if fault != "clean" else (0,)):
                    k = f"{env}-{alg}-{fault}-{rho}"
                    if f"{k}-dA" not in macros:
                        continue
                    dB = "---" if fault == "clean" else f"{macros[k + '-dB']} {macros[k + '-dB-ci']}"
                    rows.append(f"{ALG_LABEL[alg]} & {label} & {rho / 100:g} & {macros[k + '-B0']} & {dB} & "
                                f"{macros[k + '-dA']} {macros[k + '-dA-ci']} \\\\")
        if rows:
            lines.append(task_header(env, 6, index == 0))
            lines += rows
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def tests_table(envs, macros):
    lines = [r"\begin{table}[t]",
             r"\caption{Pre-specified paired replay effect $D=\text{metric}(\rho{=}0.5)-\text{metric}(\rho{=}0)$ at 20,000 steps (eight seeds). Holm-adjusted exact Wilcoxon $p$ within each task's eight tests; $+/-$ counts seeds with positive and negative $D$.}",
             r"\label{tab:tests}", r"\centering\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrrr@{}}", r"\toprule",
             r"Controller & Fault & $D_{\Delta A}$ [95\% CI] & $p_\text{Holm}$ & $+/-$ & $D_{\Delta B}$ [95\% CI] & $p_\text{Holm}$ & $+/-$ \\",
             r"\midrule"]
    for index, env in enumerate(envs):
        rows = []
        for alg in ALG_LABEL:
            for fault, label in (("noise", "Noise"), ("delay", "Delay")):
                a, b = f"{env}-D-{alg}-{fault}-dA", f"{env}-D-{alg}-{fault}-dB"
                if a not in macros:
                    continue
                rows.append(f"{ALG_LABEL[alg]} & {label} & {macros[a]} {macros[a + '-ci']} & {macros[a + '-holm']} & "
                            f"{macros[a + '-pos']}/{macros[a + '-neg']} & {macros[b]} {macros[b + '-ci']} & "
                            f"{macros[b + '-holm']} & {macros[b + '-pos']}/{macros[b + '-neg']} \\\\")
        if rows:
            lines.append(task_header(env, 8, index == 0))
            lines += rows
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def bias_table(envs, macros):
    lines = [r"\begin{table}[t]",
             r"\caption{Critic diagnostic (mean over eight seeds) at the source checkpoint and after 20,000 steps with $\rho=0$ or $0.5$: root-mean-square held-out soft Bellman residual on clean ($A$) and corrupted ($B$) evaluation data, and the value-bias gap, the Monte-Carlo bias on $B$ data minus that on $A$ data of the same checkpoint (positive: values on corrupted data are overestimated relative to clean data; exploratory).}",
             r"\label{tab:bias}", r"\centering\scriptsize", r"\setlength{\tabcolsep}{2pt}",
             r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrrrrrr@{}}", r"\toprule",
             r" & & \multicolumn{3}{c}{Residual, clean data $A$} & \multicolumn{3}{c}{Residual, corrupted data $B$} & \multicolumn{3}{c}{Value-bias gap $B-A$} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(l){9-11}",
             r"Controller & Fault & Source & $\rho{=}0$ & $\rho{=}0.5$ & Source & $\rho{=}0$ & $\rho{=}0.5$ & Source & $\rho{=}0$ & $\rho{=}0.5$ \\",
             r"\midrule"]
    any_row = False
    for index, env in enumerate(envs):
        rows = []
        for alg in ALG_LABEL:
            for fault, label in (("noise", "Noise"), ("delay", "Delay")):
                k = f"{env}-bel-{alg}-{fault}"
                cells = [macros.get(f"{k}-{rho}-{stage}-{d}") for d in ("A", "B")
                         for rho, stage in ((0, "src"), (0, "fin"), (50, "fin"))]
                cells += [macros.get(f"{env}-gap-{alg}-{fault}-{stage}") for stage in ("src", "0", "50")]
                if any(c is None for c in cells):
                    continue
                rows.append(f"{ALG_LABEL[alg]} & {label} & " + " & ".join(cells) + r" \\")
        if rows:
            any_row = True
            lines.append(task_header(env, 11, index == 0))
            lines += rows
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    return "\n".join(lines) + "\n" if any_row else None


def latency_table(path, macros):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    cpu = report["cpu"].replace("(R)", "").replace("(TM)", "").replace(" with Radeon Graphics", "").strip()
    gpu = (report.get("gpu") or "GPU").replace("NVIDIA ", "")
    lines = [r"\begin{table}[t]",
             r"\caption{Per-action inference latency (ms) of the HalfCheetah source controllers (seed "
             + str(report.get("source_checkpoints_seed", "")) + r"), batch size one, real model inputs, simulation "
             r"excluded; " + f"{report.get('steps', 10000):,}".replace(",", "{,}") + r" timed steps after warm-up on an "
             + cpu + " CPU and a " + gpu + r". The transformer recomputes its full 32-token context at every step.}",
             r"\label{tab:latency}", r"\centering\small",
             r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"Controller & Device & Mean & p95 & p99 & Max \\", r"\midrule"]
    for row in report["results"]:
        device = "GPU" if row["device"] == "cuda" else f"CPU, {row['cpu_threads']} thread{'s' if row['cpu_threads'] > 1 else ''}"
        key = f"lat-{row['algorithm']}-{'gpu' if row['device'] == 'cuda' else 'cpu' + str(row['cpu_threads'])}"
        for stat in ("mean", "p95", "p99", "max"):
            macros[f"{key}-{stat}"] = num(row[f"{stat}_ms"], 2)
        lines.append(f"{ALG_LABEL[row['algorithm']]} & {device} & {num(row['mean_ms'], 2)} & {num(row['p95_ms'], 2)} & "
                     f"{num(row['p99_ms'], 2)} & {num(row['max_ms'], 2)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    macros["lat-cpu"] = report["cpu"].replace("(R)", "").replace("(TM)", "").strip()
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", action="append", required=True, help="key=root, e.g. hc=output/...")
    parser.add_argument("--latency", type=Path)
    parser.add_argument("--dest", type=Path, default=DEST)
    args = parser.parse_args()
    macros, envs, panels = {}, [], []
    rerun = ROOT / "output/sensor_adaptation/rerun_reproducibility_hc_reactive.json"
    if rerun.exists():
        record = json.loads(rerun.read_text(encoding="utf-8"))
        macros["rerun-branches"] = str(record["branches_compared"])
        macros["rerun-identical"] = "identical" if record["all_identical"] else "not identical"
        macros["rerun-ckpt-identical"] = str(sum(bool(c["final_checkpoint_identical"]) for c in record["checks"]))
    for item in args.env:
        env, root = item.split("=", 1)
        root = Path(root) if Path(root).is_absolute() else ROOT / root
        rows, final = macros_for_env(env, root, macros)
        behavior_macros(env, root, macros, final)
        pilot_equivalent_macros(env, root, macros)
        curve_figure([(env, rows)], args.dest / f"fig_curves_{env}.pdf")
        panels.append((env, rows))
        envs.append(env)
    if len(panels) > 1:
        curve_figure(panels, args.dest / "fig_curves.pdf", row_height=1.08)
    (args.dest / "table_outcomes.tex").write_text(outcomes_table(envs, macros), encoding="utf-8")
    (args.dest / "table_tests.tex").write_text(tests_table(envs, macros), encoding="utf-8")
    bias = bias_table(envs, macros)
    if bias:
        (args.dest / "table_bias.tex").write_text(bias, encoding="utf-8")
    if args.latency:
        (args.dest / "table_latency.tex").write_text(latency_table(args.latency, macros), encoding="utf-8")
    lines = ["% Generated by scripts/build_camera_ready_assets.py; do not edit by hand.",
             r"\makeatletter",
             r"\newcommand{\R}[1]{\ifcsname r@#1\endcsname\csname r@#1\endcsname\else\textbf{??#1}\fi}"]
    lines += [f"\\expandafter\\def\\csname r@{k}\\endcsname{{{v}}}" for k, v in sorted(macros.items())]
    lines.append(r"\makeatother")
    (args.dest / "results_macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.dest / "results_macros.json").write_text(json.dumps(macros, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(macros)} macros; envs {envs}")


if __name__ == "__main__":
    main()
