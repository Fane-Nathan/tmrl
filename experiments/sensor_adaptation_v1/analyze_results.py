"""Verify all completed branches and regenerate descriptive pilot outputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(root):
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    study = json.loads((root / "study_result.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected = len(config["algorithms"]) * len(config["seeds"]) * len(config["conditions"]) * len(config["retained_replay_fractions"])
    assert study["status"] == "COMPLETE" and study["branches"] == expected
    for name, source_hash in manifest["source_files_sha256"].items():
        assert digest(root / "source" / name) == source_hash, name
    for source in manifest["inputs"].values():
        assert digest(source["original"]) == source["sha256"]
        assert digest(source["copy"]) == source["sha256"]
    rows, per_seed, initial_pairs = [], [], {}
    trajectories_verified, checkpoints_verified = 0, 0
    for folder in sorted(root.glob("*_seed*_sensor_*_replay*")):
        branch = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        assert branch["status"] == "COMPLETE" and branch["environment_steps"] == config["adaptation_steps"]
        assert branch["updates"] == config["updates_per_episode"] * config["adaptation_steps"] // config["episode_horizon"]
        checkpoint_hashes = {0: branch["input_sha256"]}
        for checkpoint in branch["checkpoints"]:
            assert digest(folder / checkpoint["file"]) == checkpoint["sha256"]
            checkpoint_hashes[checkpoint["steps"]] = checkpoint["sha256"]
            checkpoints_verified += 1
        if branch["algorithm"] == "transformer":
            assert branch["positional_change_outside_support"] == 0
        with (folder / "evaluations.csv").open(newline="", encoding="utf-8") as handle:
            evaluations = list(csv.DictReader(handle))
        expected_rows = len(config["evaluation_steps"]) * config["evaluation_episodes"] * 2
        assert len(evaluations) == expected_rows
        groups = {}
        for row in evaluations:
            step, ep = int(row["adaptation_steps"]), int(row["episode"])
            condition, reward = row["condition"], float(row["return"])
            assert row["checkpoint_sha256"] == checkpoint_hashes[step]
            assert int(row["environment_seed"]) == config["evaluation_seed_base"] + ep * 100
            with np.load(folder / f"trajectory_{step}_{condition}_{ep}.npz", allow_pickle=False) as trajectory:
                assert len(trajectory["actions"]) == config["episode_horizon"]
                assert abs(float(trajectory["step_rewards_float64"].sum()) - reward) < 1e-8
                np.testing.assert_array_equal(trajectory["observations"][:-1, :17], trajectory["observed_sensor"])
                np.testing.assert_array_equal(trajectory["observations"][1:, 17:23], trajectory["actions"])
                if condition == "sensor_delay":
                    delay = config["sensor_delay_steps"]
                    np.testing.assert_array_equal(trajectory["observed_sensor"][:delay], 0)
                    np.testing.assert_array_equal(trajectory["observed_sensor"][delay:], trajectory["true_sensor"][:-delay])
                elif condition == "clean":
                    np.testing.assert_array_equal(trajectory["observed_sensor"], trajectory["true_sensor"])
                else:
                    assert not np.array_equal(trajectory["observed_sensor"], trajectory["true_sensor"])
            trajectories_verified += 1
            groups.setdefault((step, condition), []).append(reward)
            rows.append({"algorithm": branch["algorithm"], "training_seed": branch["seed"],
                         "adaptation_condition": branch["condition"], "retained_fraction": branch["retained_fraction"],
                         **row})
            if step == 0:
                key = (branch["algorithm"], branch["seed"], branch["condition"], condition, ep)
                if key in initial_pairs:
                    assert abs(initial_pairs[key] - reward) < 1e-8, "Paired arms started from different performance"
                initial_pairs[key] = reward
        curve = []
        for step in config["evaluation_steps"]:
            clean = float(np.mean(groups[(step, "clean")]))
            new = float(np.mean(groups[(step, branch["condition"])]))
            reported = next(point for point in branch["curves"] if point["steps"] == step)
            assert abs(reported["clean"] - clean) < 1e-8
            assert abs(reported[branch["condition"]] - new) < 1e-8
            curve.append((step, clean, new))
        auc = float(np.trapezoid([point[2] for point in curve], [point[0] for point in curve]) / config["adaptation_steps"])
        per_seed.append({"algorithm": branch["algorithm"], "training_seed": branch["seed"],
                         "condition": branch["condition"], "retained_fraction": branch["retained_fraction"],
                         "A_before": curve[0][1], "A_after": curve[-1][1], "A_change": curve[-1][1] - curve[0][1],
                         "B_before": curve[0][2], "B_after": curve[-1][2], "B_change": curve[-1][2] - curve[0][2],
                         "B_curve_mean": auc, "training_seconds": branch["curves"][-1]["training_seconds"]})
    assert len(per_seed) == expected
    summary, replay_effects = [], []
    for algorithm in config["algorithms"]:
        for condition in config["conditions"]:
            for fraction in config["retained_replay_fractions"]:
                selected = [row for row in per_seed if row["algorithm"] == algorithm and row["condition"] == condition and row["retained_fraction"] == fraction]
                assert len(selected) == len(config["seeds"])
                result = {"algorithm": algorithm, "condition": condition, "retained_fraction": fraction, "seeds": len(selected)}
                for metric in ["A_before", "A_after", "A_change", "B_before", "B_after", "B_change", "B_curve_mean", "training_seconds"]:
                    values = [row[metric] for row in selected]
                    result.update({metric + "_mean": float(np.mean(values)), metric + "_min": min(values), metric + "_max": max(values)})
                summary.append(result)
            for seed in config["seeds"]:
                by_fraction = {row["retained_fraction"]: row for row in per_seed if row["algorithm"] == algorithm and row["condition"] == condition and row["training_seed"] == seed}
                replay_effects.append({"algorithm": algorithm, "condition": condition, "training_seed": seed,
                                       "retained_minus_none_A_after": by_fraction[0.5]["A_after"] - by_fraction[0.0]["A_after"],
                                       "retained_minus_none_B_after": by_fraction[0.5]["B_after"] - by_fraction[0.0]["B_after"]})
    write_csv(root / "all_evaluations.csv", rows)
    write_csv(root / "seed_metrics.csv", per_seed)
    write_csv(root / "summary.csv", summary)
    write_csv(root / "paired_replay_effects.csv", replay_effects)
    verification = {"status": "PASS", "completed_branches": expected, "raw_episodes_verified": trajectories_verified,
                    "candidate_checkpoint_hashes_verified": checkpoints_verified, "source_checkpoints_unchanged": True,
                    "paired_arms_start_identically": True, "observation_faults_verified_in_saved_trajectories": True,
                    "old_and_new_condition_share_the_same_checkpoint_at_each_budget": True,
                    "source_snapshot_hashes_verified": True}
    (root / "verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    plot_curves(root, config, study, retention=False)
    plot_curves(root, config, study, retention=True)
    lines = ["# Sensor adaptation and retention: completed engineering pilot", "",
             f"{expected} completed branches; {len(config['seeds'])} independent pretrained seeds per architecture; {config['adaptation_steps']:,} new-condition steps per branch.", "",
             "## Interpretation", "",
             "These are short-budget warm-start results, not evidence of general fast adaptation without forgetting. Initial models came from the actuator-fault study. The new experiment correctly changes observations and measures the exact updated model on both old and new conditions. No post hoc best-checkpoint selection is used.", "",
             "Positive B change means improved mean return on the sensor condition. Negative A change means the old clean-condition mean return declined. Values below are means over training seeds; the CSVs retain every seed and every evaluation episode. Three evaluation episodes and three training seeds provide limited precision; no equivalence or no-forgetting claim is made.", "",
             "## Fixed-budget outcomes", "",
             "| Policy | Sensor fault | Retained replay | B before | B after | B change | A change |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary:
        lines.append(f"| {row['algorithm']} | {row['condition']} | {row['retained_fraction']:.0%} | {row['B_before_mean']:.1f} | {row['B_after_mean']:.1f} | {row['B_change_mean']:+.1f} | {row['A_change_mean']:+.1f} |")
    lines += ["", "## Design and limits", "",
              "- A: clean observations. B: either independent Gaussian observation noise or a three-step observation delay. Actions are not corrupted.",
              "- Noise uses standard deviations 0.1 for eight position observations and 0.5 for nine velocities, in native observation units. Delay starts at episode reset with zero padding.",
              "- Each retention arm starts from the identical source model. Two fresh deterministic A episodes form the fixed old replay pool; the 50% arm samples half its batch from that pool. Both arms receive the same number of B interactions and optimizer updates.",
              "- Transformer collection and training use rolling contexts up to 32 positions. Each replay sample supervises its final valid transition. Short episode-start contexts and terminal transitions are supported. No fault labels enter the model input or auxiliary loss.",
              "- The reactive controller still receives the current sensor reading, previous command, reward, and terminal flag. It has no accumulated sequence. Architecture comparisons are controller comparisons, not parameter-matched causal ablations.",
              "- Fine-tuning uses reset optimizers/entropy temperature because those were not saved with the source models. Dropout is disabled consistently. Ten critics are updated once per actor update, with 0.25 ensemble updates per new environment step.",
              "- CPU action inference and GPU training are included in adaptation elapsed time; evaluation and checkpoint I/O are excluded. Budgets and hardware are recorded. Wall-clock comparisons are descriptive and can reflect machine load.",
              "- This tests continued learning after a change in sensing within HalfCheetah. It does not establish transfer to a new TrackMania map, sensory modalities, arbitrary physical environments, or the cause of the earlier training plateau.",
              "- The protocol does not set a recovery threshold or retention tolerance. Accordingly, it reports fixed-budget curves and absolute retention changes rather than claiming a recovery time or zero forgetting.", "",
              "## Artifacts", "",
              "- `learning_curves.png`: B returns, with the range across seeds shaded (not a confidence interval).",
              "- `retention_curves.png`: A return change from its own initial model.",
              "- `all_evaluations.csv`, `seed_metrics.csv`, `summary.csv`, `paired_replay_effects.csv`: raw and derived numbers.",
              "- `verification.json`: independent readback of saved trajectories, returns, checkpoint hashes, and paired starts.",
              "- `source/`, `manifest.json`, `config.json`: the exact code snapshot, environment versions, inputs, and frozen pilot configuration.",
              "- Branch folders contain actual collected replay, per-step sensor audits, fixed-budget model checkpoints, and training logs.", ""]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"verification": verification, "summary": summary}, indent=2))


def plot_curves(root, config, study, retention):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), squeeze=False)
    palette = {0.0: "#d97706", 0.5: "#176b8a"}
    for i, algorithm in enumerate(config["algorithms"]):
        for j, condition in enumerate(config["conditions"]):
            ax = axes[i, j]
            for fraction in config["retained_replay_fractions"]:
                selected = [r for r in study["results"] if r["algorithm"] == algorithm and r["condition"] == condition and r["retained_fraction"] == fraction]
                values = []
                for result in selected:
                    if retention:
                        initial = result["curves"][0]["clean"]
                        values.append([p["clean"] - initial for p in result["curves"]])
                    else:
                        values.append([p[condition] for p in result["curves"]])
                values = np.asarray(values)
                x = config["evaluation_steps"]
                ax.fill_between(x, values.min(0), values.max(0), color=palette[fraction], alpha=0.12)
                ax.plot(x, values.mean(0), "o-", color=palette[fraction], label=f"{fraction:.0%} retained replay", linewidth=1.8, markersize=4)
                ax.scatter([x[-1]] * len(values), values[:, -1], color=palette[fraction], s=14, alpha=0.65)
            if retention:
                ax.axhline(0, color="#6b7280", linestyle="--", linewidth=0.8)
            ax.set_title(f"{algorithm.capitalize()} / {'sensor noise' if condition == 'sensor_noise' else '3-step sensor delay'}", fontsize=10)
            ax.set_xlabel("New-condition training steps", fontsize=9)
            ax.set_ylabel("Old-condition return change" if retention else "New-condition return", fontsize=9)
            ax.grid(alpha=0.15)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Retention during sensor adaptation" if retention else "Learning under observation faults", fontsize=14)
    fig.text(0.5, 0.91, f"Pilot: means and ranges across {len(config['seeds'])} pretrained seeds; ranges are not confidence intervals", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.07, 1, 0.90])
    name = "retention_curves" if retention else "learning_curves"
    fig.savefig(root / (name + ".png"), dpi=170)
    fig.savefig(root / (name + ".svg"))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    analyze(parser.parse_args().root)
