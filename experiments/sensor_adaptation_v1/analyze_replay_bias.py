"""Replay-mixing diagnostic (reviewer request): critic bias on clean vs corrupted data.

For each branch, the saved critic ensemble at budget 0 (the source model) and at the final budget
values the held-out evaluation trajectories recorded with that same checkpoint, on the clean
condition A and on the branch's sensor condition B. Bias is Q(h_t, a_t) - G_t, where G_t is the
discounted Monte-Carlo return actually obtained. Histories are rebuilt exactly as in training
(replay.extract_transition, <= 32 tokens). Time steps are used only where G_t is complete: the
episode terminated, or at least `tail` steps remain (gamma^500 ~ 0.0066).

Caveats reported with the results: the critics estimate the soft value of the stochastic policy
(including entropy bonuses), whereas G_t is the return of the deterministic evaluation policy; this
offset is shared by A and B data, so differences between data sources and replay arms are the
quantities of interest. Evaluation episodes never entered replay.

  python analyze_replay_bias.py ROOT --protocol protocol_extension.json [--out OUT]
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from analyze_extension import describe, holm, read_branches
from learner import features, load_agent, q_value
from replay import extract_transition

HERE = Path(__file__).resolve().parent


def returns_to_go(rewards, gamma):
    out, running = np.zeros(len(rewards), np.float64), 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = float(rewards[t]) + gamma * running
        out[t] = running
    return out


def pack(episode, steps, context, successor):
    obs_dim = episode["observations"].shape[-1]
    histories = np.zeros((len(steps), context, obs_dim), np.float32)
    lengths = np.zeros(len(steps), np.int64)
    for i, t in enumerate(steps):
        current, following = extract_transition(episode, t, context)
        chosen = following if successor else current
        histories[i, :len(chosen)] = chosen
        lengths[i] = len(chosen)
    return histories, lengths


@torch.inference_mode()
def critic_on_trajectory(model, algorithm, path, context, gamma, tail, device, alpha):
    data = np.load(path)
    episode = {k: data[k] for k in ("observations", "actions", "rewards", "terminated")}
    T = len(episode["actions"])
    if T == 0:
        return None
    terminated = bool(episode["terminated"][-1])
    as_t = lambda x: torch.as_tensor(x, device=device)
    actions_all = as_t(episode["actions"])

    # Soft Bellman residual on every held-out transition, using the training target exactly:
    # r + gamma (1 - d) (min over a critic pair of target Q(h', a') - alpha log pi(a'|h')), a' ~ pi.
    # The random training pair is replaced by its expectation over all pairs.
    steps = list(range(T))
    cur, cur_len = pack(episode, steps, context, successor=False)
    nxt, nxt_len = pack(episode, steps, context, successor=True)
    feat = features(model, algorithm, as_t(cur), as_t(cur_len))
    online_next = features(model, algorithm, as_t(nxt), as_t(nxt_len))
    target_next = features(model, algorithm, as_t(nxt), as_t(nxt_len), target=True)
    torch.manual_seed(0)
    next_action, next_logp = model.actor(online_next)
    q = torch.stack([q_value(model, algorithm, j, feat, actions_all) for j in range(model.num_critics)])[..., 0]
    qt = torch.stack([q_value(model, algorithm, j, target_next, next_action, True)
                      for j in range(model.num_critics)])[..., 0]
    pair_min = torch.stack([torch.minimum(qt[i], qt[j]) for i in range(len(qt)) for j in range(i + 1, len(qt))]).mean(0)
    rewards, done = as_t(episode["rewards"]).double(), as_t(episode["terminated"]).double()
    target = rewards + gamma * (1 - done) * (pair_min.double() - alpha * next_logp[..., 0].double())
    residual = (target - q.double().mean(0)).cpu().numpy()

    # Monte-Carlo bias where the return-to-go is complete.
    usable = [t for t in range(T) if terminated or t <= T - tail]
    G = returns_to_go(episode["rewards"], gamma)[usable]
    qd = q.double().cpu().numpy()[:, usable]  # (critics, steps)
    return {"q_mean": qd.mean(0), "q_std": qd.std(0), "G": G, "residual": residual}


def branch_bias(branch, shard_inputs, budget, config, device, tail):
    folder, algorithm, condition = branch["folder"], branch["algorithm"], branch["condition"]
    context = config["context"] if algorithm == "transformer" else 1
    checkpoint = shard_inputs[f"{algorithm}_{branch['seed']}"]["copy"] if budget == 0 else folder / f"checkpoint_{budget}.pt"
    model = load_agent(algorithm, checkpoint, context, device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    # Source checkpoints carry no temperature; branches start from initial_alpha.
    alpha = float(payload["log_alpha"].exp()) if isinstance(payload, dict) and "log_alpha" in payload else config["initial_alpha"]
    if budget != 0 and "log_alpha" not in payload:
        raise RuntimeError(f"{checkpoint} lacks the entropy temperature needed for the soft target")
    rows = []
    for measured in dict.fromkeys(["clean", condition]):
        parts = [critic_on_trajectory(model, algorithm, folder / f"trajectory_{budget}_{measured}_{ep}.npz",
                                      context, config["gamma"], tail, device, alpha)
                 for ep in range(config["evaluation_episodes"])]
        parts = [p for p in parts if p is not None]
        q, s, G, res = (np.concatenate([p[k] for p in parts]) for k in ("q_mean", "q_std", "G", "residual"))
        rows.append({"algorithm": algorithm, "seed": int(branch["seed"]), "condition": condition,
                     "retained_fraction": float(branch["retained_fraction"]), "budget": budget,
                     "data": "A_clean" if measured == "clean" else "B_sensor", "steps_used": int(G.size),
                     "alpha": alpha, "mean_return_to_go": float(G.mean()), "bias": float((q - G).mean()),
                     "abs_error": float(np.abs(q - G).mean()), "ensemble_std": float(s.mean()),
                     "bellman_rmse": float(np.sqrt(np.mean(res ** 2))), "bellman_mean": float(res.mean())})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--budget", type=int, help="Final budget (default: protocol adaptation_steps)")
    parser.add_argument("--tail", type=int, default=500)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    protocol = args.protocol if args.protocol.is_absolute() else HERE / args.protocol
    config = json.loads(protocol.read_text(encoding="utf-8"))
    branches = read_branches(args.root)
    final = args.budget or max(int(r["adaptation_steps"]) for r in branches[0]["rows"])
    config["evaluation_episodes"] = len({r["episode"] for r in branches[0]["rows"]})
    inputs = {}
    for manifest in Path(args.root).glob("**/manifest.json"):
        for key, value in json.loads(manifest.read_text(encoding="utf-8"))["inputs"].items():
            inputs.setdefault(manifest.parent, {})[key] = value
    rows = []
    for branch in branches:
        shard_inputs = inputs[branch["folder"].parent]
        for budget in (0, final):
            rows.extend(branch_bias(branch, shard_inputs, budget, config, args.device, args.tail))
        print(f"bias done {branch['folder'].name}", flush=True)
    out = args.out or args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "critic_bias_rows.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Seed-level summaries: replay effect on post-adaptation bias, per data source.
    table = {(r["algorithm"], r["condition"], r["retained_fraction"], r["seed"], r["budget"], r["data"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    summary = []
    for algorithm in sorted({r["algorithm"] for r in rows}):
        for condition in sorted({r["condition"] for r in rows if r["condition"] != "clean"}):
            for data in ("A_clean", "B_sensor"):
                for metric in ("bellman_rmse", "bias", "abs_error", "ensemble_std"):
                    for fraction in (0.0, 0.5):
                        before = [table[(algorithm, condition, fraction, s, 0, data)][metric] for s in seeds]
                        after = [table[(algorithm, condition, fraction, s, final, data)][metric] for s in seeds]
                        summary.append({"algorithm": algorithm, "condition": condition, "data": data, "metric": metric,
                                        "contrast": f"rho{fraction}: final - source", "final_mean": float(np.mean(after)),
                                        **describe(np.subtract(after, before))})
                    diff = [table[(algorithm, condition, 0.5, s, final, data)][metric]
                            - table[(algorithm, condition, 0.0, s, final, data)][metric] for s in seeds]
                    summary.append({"algorithm": algorithm, "condition": condition, "data": data, "metric": metric,
                                    "contrast": "final rho0.5 - rho0", "final_mean": float("nan"), **describe(diff)})
    family = [r for r in summary if r["contrast"] == "final rho0.5 - rho0" and r["metric"] == "bellman_rmse"]
    for row, adjusted in zip(family, holm([r["wilcoxon_p"] for r in family])):
        row["holm_p"] = adjusted
    with open(out / "critic_bias_summary.csv", "w", newline="", encoding="utf-8") as handle:
        keys = [k for k in summary[0] if k != "values"] + ["holm_p", "values"]
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in summary:
            writer.writerow({**{k: row.get(k, "") for k in keys}, "values": json.dumps(row["values"])})
    for row in family:
        print(f"{row['algorithm']:11s} {row['condition']:12s} {row['data']:8s} Bellman RMSE rho0.5-rho0 = {row['mean']:+8.1f} "
              f"[{row['t_ci_low']:+.1f},{row['t_ci_high']:+.1f}] p={row['wilcoxon_p']:.3f} holm={row['holm_p']:.3f}")


if __name__ == "__main__":
    main()
