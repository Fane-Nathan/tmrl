"""Pre-registered analysis for sensor-extension-2.0 / sensor-ant-2.0 (see protocol analysis_plan).

Usage:
  python analyze_extension.py ROOT --protocol protocol_extension.json [--pilot PILOT_DIR] [--out OUT]

ROOT holds one or more shard directories produced by run_study.py (each with status.json and
branch folders). Every shard must be COMPLETE; branches are never dropped.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
BOOTSTRAP_SEED = 20261031
BOOTSTRAP_RESAMPLES = 10_000


def read_branches(root):
    shards = sorted(p.parent for p in Path(root).glob("*/status.json"))
    if not shards and (Path(root) / "status.json").exists():
        shards = [Path(root)]
    if not shards:
        raise FileNotFoundError(f"No shard status.json under {root}")
    branches = []
    for shard in shards:
        status = json.loads((shard / "status.json").read_text(encoding="utf-8"))
        if status.get("status") != "COMPLETE":
            raise RuntimeError(f"Shard {shard} is not COMPLETE: {status}")
        for spec_path in sorted(shard.glob("*/branch.json")):
            folder = spec_path.parent
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            if not (folder / "result.json").exists():
                raise RuntimeError(f"Branch {folder} has no result.json")
            rows = list(csv.DictReader((folder / "evaluations.csv").open(encoding="utf-8")))
            branches.append({**spec, "folder": folder, "rows": rows})
    return branches


def branch_means(branch, budgets, max_episodes=None):
    """Mean return per (budget, measured condition); optionally only the first episodes."""
    out = defaultdict(dict)
    grouped = defaultdict(list)
    for row in branch["rows"]:
        if max_episodes is not None and int(row["episode"]) >= max_episodes:
            continue
        grouped[(int(row["adaptation_steps"]), row["condition"])].append(float(row["return"]))
    for (budget, measured), values in grouped.items():
        out[budget][measured] = float(np.mean(values))
    missing = [b for b in budgets if b not in out]
    if missing:
        raise RuntimeError(f"{branch['folder']} lacks budgets {missing}")
    return out


def trapezoid_mean(budgets, values):
    budgets = np.asarray(budgets, float)
    return float(np.trapezoid(values, budgets) / (budgets[-1] - budgets[0]))


def seed_metrics(branches, budgets):
    rows = []
    for branch in branches:
        means = branch_means(branch, budgets)
        condition = branch["condition"]
        for budget in budgets[1:]:
            curve_budgets = [b for b in budgets if b <= budget]
            rows.append({
                "algorithm": branch["algorithm"], "seed": int(branch["seed"]), "condition": condition,
                "retained_fraction": float(branch["retained_fraction"]), "budget": budget,
                "A_before": means[0]["clean"], "A_after": means[budget]["clean"],
                "A_change": means[budget]["clean"] - means[0]["clean"],
                "B_before": means[0][condition], "B_after": means[budget][condition],
                "B_change": means[budget][condition] - means[0][condition],
                "B_curve_mean": trapezoid_mean(curve_budgets, [means[b][condition] for b in curve_budgets]),
            })
    return rows


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def describe(differences):
    d = np.asarray(differences, float)
    n = d.size
    mean, sd = float(d.mean()), float(d.std(ddof=1)) if n > 1 else float("nan")
    half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = d[rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))].mean(axis=1)
    wilcoxon = stats.wilcoxon(d, method="exact", alternative="two-sided") if n > 1 and np.any(d != 0) else None
    ttest = stats.ttest_1samp(d, 0.0) if n > 1 else None
    return {"n": n, "mean": mean, "sd": sd, "t_ci_low": mean - half, "t_ci_high": mean + half,
            "boot_ci_low": float(np.percentile(boot, 2.5)), "boot_ci_high": float(np.percentile(boot, 97.5)),
            "d_z": mean / sd if sd and sd > 0 else float("nan"),
            "n_positive": int((d > 0).sum()), "n_negative": int((d < 0).sum()),
            "wilcoxon_p": float(wilcoxon.pvalue) if wilcoxon else 1.0,
            "paired_t_p": float(ttest.pvalue) if ttest else float("nan"),
            "values": [float(x) for x in d]}


def index_rows(rows, budget):
    table = {}
    for row in rows:
        if row["budget"] == budget:
            table[(row["algorithm"], row["condition"], row["retained_fraction"], row["seed"])] = row
    return table


def paired(table, key_a, key_b, metric, seeds):
    missing = [s for s in seeds if (*key_a, s) not in table or (*key_b, s) not in table]
    if missing:
        raise RuntimeError(f"Missing paired seeds {missing} for {key_a} vs {key_b}")
    return [table[(*key_b, s)][metric] - table[(*key_a, s)][metric] for s in seeds]


def analyze(rows, config, budget):
    table = index_rows(rows, budget)
    seeds = sorted(config["seeds"])
    algorithms = config["algorithms"]
    faults = [c for c in config["conditions"] if c != "clean"]
    results = {"budget": budget}

    primary = []
    for algorithm in algorithms:
        for fault in faults:
            for metric in ("A_change", "B_change"):
                stats_row = describe(paired(table, (algorithm, fault, 0.0), (algorithm, fault, 0.5), metric, seeds))
                primary.append({"algorithm": algorithm, "fault": fault, "metric": metric,
                                "contrast": "rho0.5 - rho0", **stats_row})
    for row, adjusted in zip(primary, holm([r["wilcoxon_p"] for r in primary])):
        row["holm_p"] = adjusted
    results["primary_replay_effect"] = primary

    arms = []
    for metric in ("A_change", "B_change"):
        family = []
        for algorithm in algorithms:
            for condition in config["conditions"]:
                if metric == "B_change" and condition == "clean":
                    continue  # A and B coincide for the clean-continuation control
                for fraction in config.get("branch_fractions", {}).get(condition, config["retained_replay_fractions"]):
                    values = [table[(algorithm, condition, float(fraction), s)][metric] for s in seeds]
                    family.append({"algorithm": algorithm, "condition": condition, "retained_fraction": fraction,
                                   "metric": metric, "contrast": "change vs 0", **describe(values)})
        for row, adjusted in zip(family, holm([r["wilcoxon_p"] for r in family])):
            row["holm_p"] = adjusted
        arms.extend(family)
    results["arm_changes"] = arms

    if "clean" in config["conditions"]:
        specific = []
        for algorithm in algorithms:
            for fault in faults:
                values = paired(table, (algorithm, "clean", 0.0), (algorithm, fault, 0.0), "A_change", seeds)
                specific.append({"algorithm": algorithm, "fault": fault, "metric": "A_change",
                                 "contrast": "fault rho0 - clean continuation", **describe(values)})
        for row, adjusted in zip(specific, holm([r["wilcoxon_p"] for r in specific])):
            row["holm_p"] = adjusted
        results["fault_specific_forgetting"] = specific

    if len(algorithms) == 2:
        comparison = []
        for fault in faults:
            for fraction in config["retained_replay_fractions"]:
                for metric in ("A_change", "B_change"):
                    a = [table[(algorithms[0], fault, fraction, s)][metric] for s in seeds]
                    b = [table[(algorithms[1], fault, fraction, s)][metric] for s in seeds]
                    test = stats.mannwhitneyu(a, b, alternative="two-sided", method="exact")
                    comparison.append({"fault": fault, "retained_fraction": fraction, "metric": metric,
                                       "contrast": f"{algorithms[0]} - {algorithms[1]} (descriptive)",
                                       "mean_difference": float(np.mean(a) - np.mean(b)),
                                       "mannwhitney_p": float(test.pvalue)})
        results["controller_comparison_descriptive"] = comparison
    return results


def reproducibility(branches, pilot_dir, pilot_budgets=(0, 2000, 4000), episodes=3):
    """Exact comparison of shared branches/episodes with the frozen pilot."""
    checks = []
    for branch in branches:
        name = f"{branch['algorithm']}_seed{branch['seed']}_{branch['condition']}_replay{int(float(branch['retained_fraction']) * 100)}"
        pilot_csv = Path(pilot_dir) / name / "evaluations.csv"
        if not pilot_csv.exists():
            continue
        pilot = {(int(r["adaptation_steps"]), r["condition"], int(r["episode"])): float(r["return"])
                 for r in csv.DictReader(pilot_csv.open(encoding="utf-8"))}
        mine = {(int(r["adaptation_steps"]), r["condition"], int(r["episode"])): float(r["return"])
                for r in branch["rows"]}
        keys = [k for k in pilot if k[0] in pilot_budgets and k[2] < episodes]
        diffs = [abs(mine[k] - pilot[k]) for k in keys if k in mine]
        checks.append({"branch": name, "compared": len(diffs), "missing": len(keys) - len(diffs),
                       "max_abs_difference": max(diffs) if diffs else None,
                       "identical": bool(diffs) and max(diffs) == 0.0 and len(diffs) == len(keys)})
    return {"branches_compared": len(checks), "all_identical": bool(checks) and all(c["identical"] for c in checks),
            "checks": checks}


def write_csv(path, rows):
    if not rows:
        return
    keys = [k for k in rows[0] if k != "values"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys + (["values"] if "values" in rows[0] else []))
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **({"values": json.dumps(row["values"])} if "values" in row else {})})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--allow-subset", action="store_true",
                        help="Testing only: analyze whatever seeds the root contains (e.g. the 3-seed pilot)")
    args = parser.parse_args()
    protocol = args.protocol if args.protocol.is_absolute() else HERE / args.protocol
    config = json.loads(protocol.read_text(encoding="utf-8"))
    branches = read_branches(args.root)
    present_seeds = sorted({int(b["seed"]) for b in branches})
    present_conditions = sorted({b["condition"] for b in branches})
    present_algorithms = sorted({b["algorithm"] for b in branches})
    if (present_seeds != sorted(config["seeds"]) or set(present_conditions) != set(config["conditions"])
            or set(present_algorithms) != set(config["algorithms"])):
        if not args.allow_subset:
            raise RuntimeError(f"Root has seeds {present_seeds}, conditions {present_conditions}, algorithms "
                               f"{present_algorithms}; protocol requires {config['seeds']}, {config['conditions']}, "
                               f"{config['algorithms']}")
        config = {**config, "seeds": present_seeds,
                  "conditions": [c for c in config["conditions"] if c in present_conditions],
                  "algorithms": [a for a in config["algorithms"] if a in present_algorithms]}
    budgets = sorted(config["evaluation_steps"])
    if args.allow_subset:
        budgets = sorted({int(r["adaptation_steps"]) for b in branches for r in b["rows"]})
    rows = seed_metrics(branches, budgets)
    out = args.out or args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "seed_metrics.csv", rows)
    analysis_budgets = sorted({config["analysis_plan"]["primary_budget"], *config["analysis_plan"]["secondary_budgets"]} & set(budgets[1:]))
    if args.allow_subset:
        analysis_budgets = [budgets[-1]]
    summary = {"protocol": config.get("version"), "seeds": config["seeds"], "branches": len(branches),
               "budgets_analyzed": analysis_budgets, "results": {}}
    for budget in analysis_budgets:
        result = analyze(rows, config, budget)
        summary["results"][str(budget)] = result
        for key, value in result.items():
            if isinstance(value, list):
                write_csv(out / f"{key}_{budget}.csv", value)
    if args.pilot:
        summary["pilot_reproducibility"] = reproducibility(branches, args.pilot)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    primary_budget = str(analysis_budgets[-1]) if args.allow_subset else str(config["analysis_plan"]["primary_budget"])
    print(f"Analyzed {len(branches)} branches, seeds {config['seeds']}")
    for row in summary["results"][primary_budget]["primary_replay_effect"]:
        print(f"  {row['algorithm']:11s} {row['fault']:12s} {row['metric']:8s} D={row['mean']:+8.1f} "
              f"95%CI[{row['t_ci_low']:+.1f},{row['t_ci_high']:+.1f}] p={row['wilcoxon_p']:.4f} holm={row['holm_p']:.4f} "
              f"(+{row['n_positive']}/-{row['n_negative']})")
    if "pilot_reproducibility" in summary:
        rep = summary["pilot_reproducibility"]
        print(f"Pilot reproducibility: {rep['branches_compared']} branches, all identical = {rep['all_identical']}")


if __name__ == "__main__":
    main()
