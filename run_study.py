"""Execute isolated A -> sensor B -> A adaptation/retention branches."""
import argparse
import contextlib
import copy
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import gymnasium
import mujoco
import numpy as np
import torch

from learner import Learner, load_agent
from replay import load_episodes, sample_batch, save_episodes
from sensor_env import make_env

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def preserve_training_rng():
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
             torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    try:
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        if state[3] is not None:
            torch.cuda.set_rng_state_all(state[3])


@torch.inference_mode()
def rollout(model, algorithm, condition, config, environment_seed, deterministic=True, onset=0, max_steps=None):
    env = make_env(condition, config, onset=onset)
    observations, actions, rewards, terms = [], [], [], []
    true_readings, sensor_readings, ages = [], [], []
    if algorithm == "transformer":
        model.reset_context()
    try:
        obs, info = env.reset(seed=environment_seed)
        while True:
            observations.append(obs.copy())
            audit = info["sensor_audit"]
            true_readings.append(audit["true_observation"])
            sensor_readings.append(audit["observed_sensor"])
            ages.append(audit["observation_age"])
            if algorithm == "transformer":
                action = model.step_in_context(obs, adaptive=True, deterministic=deterministic, device="cpu")
            else:
                action = model.act(obs, deterministic=deterministic, device="cpu")
            obs, reward, terminated, truncated, info = env.step(action)
            if not np.isfinite(obs).all() or not np.isfinite(reward):
                raise FloatingPointError("Nonfinite environment transition")
            actions.append(action.copy())
            rewards.append(reward)
            terms.append(float(terminated))
            # A budget cap is a truncation: terminated stays 0, so the successor is bootstrapped.
            if terminated or truncated or (max_steps is not None and len(actions) >= max_steps):
                observations.append(obs.copy())
                break
    finally:
        env.close()
    episode = {"observations": np.asarray(observations, np.float32),
               "actions": np.asarray(actions, np.float32),
               "rewards": np.asarray(rewards, np.float32),
               "terminated": np.asarray(terms, np.float32)}
    audit = {"true_sensor": np.asarray(true_readings, np.float32),
             "observed_sensor": np.asarray(sensor_readings, np.float32),
             "observation_age": np.asarray(ages, np.int64),
             "step_rewards_float64": np.asarray(rewards, np.float64)}
    return episode, audit, float(np.sum(rewards, dtype=np.float64))


def prepare(output, config, smoke):
    output.mkdir(parents=True, exist_ok=False)
    (output / "source").mkdir()
    for path in HERE.iterdir():
        if path.suffix in {".py", ".json", ".md"}:
            shutil.copy2(path, output / "source" / path.name)
    sources = {p.name: sha256(p) for p in (output / "source").iterdir()}
    write_json(output / "config.json", config)
    (output / "inputs").mkdir()
    inputs = {}
    for algorithm in config["algorithms"]:
        for seed in config["seeds"]:
            original = (WORKSPACE / config.get("source_dir", "fault_benchmark_results")
                        / f"{algorithm}_seed_{seed}" / "best_validation_checkpoint.pt")
            dest = output / "inputs" / f"{algorithm}_{seed}.pt"
            shutil.copy2(original, dest)
            inputs[f"{algorithm}_{seed}"] = {"original": str(original), "copy": str(dest), "sha256": sha256(dest)}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WORKSPACE, text=True).strip()
    except subprocess.SubprocessError:
        commit = "unavailable"
    manifest = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source_files_sha256": sources, "inputs": inputs, "smoke": smoke,
                "git_head": commit, "source_capture": "Exact source snapshot, including untracked experiment files; git HEAD alone is insufficient",
                "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                "torch": torch.__version__, "gymnasium": gymnasium.__version__, "mujoco": mujoco.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    write_json(output / "manifest.json", manifest)
    return manifest


def evaluate(learner, config, condition, folder, budget, writer, checkpoint_hash, adaptation_seconds):
    with preserve_training_rng():
        model = learner.rollout_model()
        means = {}
        # A clean-continuation control has A == B; evaluate it once rather than writing duplicate rows.
        for measured in dict.fromkeys(["clean", condition]):
            returns = []
            for ep in range(config["evaluation_episodes"]):
                environment_seed = config["evaluation_seed_base"] + 100 * ep
                episode, audit, total = rollout(model, learner.algorithm, measured, config, environment_seed)
                returns.append(total)
                writer.writerow({"adaptation_steps": budget, "condition": measured,
                                 "episode": ep, "environment_seed": environment_seed,
                                 "return": total, "length": len(episode["actions"]),
                                 "checkpoint_sha256": checkpoint_hash,
                                 "adaptation_seconds": adaptation_seconds})
                np.savez_compressed(folder / f"trajectory_{budget}_{measured}_{ep}.npz", **episode, **audit)
            means[measured] = float(np.mean(returns))
        return means


def run_branch(output, manifest, config, algorithm, seed, condition, retained_fraction, old_episodes, device):
    name = f"{algorithm}_seed{seed}_{condition}_replay{int(retained_fraction * 100)}"
    folder = output / name
    folder.mkdir()
    source = manifest["inputs"][f"{algorithm}_{seed}"]
    # Same adaptation RNG and simulator schedule for paired retention arms.
    # Offsets 1 and 2 are unchanged from the pilot so its branches reproduce; 3 is the clean-continuation control.
    branch_seed = seed * 1000 + {"sensor_noise": 1, "sensor_delay": 2, "clean": 3}[condition]
    seed_all(branch_seed)
    learner = Learner(algorithm, source["copy"], config, device, branch_seed + 100)
    rng = np.random.default_rng(branch_seed + 200)
    new_episodes, checkpoints, curves = [], [], []
    cumulative_training_seconds, total_steps = 0.0, 0
    specification = {"algorithm": algorithm, "seed": seed, "condition": condition,
                     "retained_fraction": retained_fraction, "input_sha256": source["sha256"],
                     "context": learner.context, "adaptation_rng_seed": branch_seed}
    write_json(folder / "branch.json", specification)
    print(f"START {name}", flush=True)
    with (folder / "evaluations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["adaptation_steps", "condition", "episode", "environment_seed",
                                                   "return", "length", "checkpoint_sha256", "adaptation_seconds"])
        writer.writeheader()
        means = evaluate(learner, config, condition, folder, 0, writer, source["sha256"], 0.0)
        handle.flush()
        curves.append({"steps": 0, "training_seconds": 0.0, **means})
        print(f"  {name} step=0 A={means['clean']:.2f} B={means[condition]:.2f}", flush=True)
        # Collection is step-budgeted so environments with early termination (Ant-v5) keep exact
        # budgets: episodes are capped at the next evaluation point, and updates accrue at
        # updates_per_episode / episode_horizon per collected step. For HalfCheetah every episode
        # is a full 1,000 steps with 250 updates, identical to the pilot.
        updates_per_step = config["updates_per_episode"] / config["episode_horizon"]
        seed_stride = config.get("training_seed_stride", 100)
        update_credit, episode_index = 0.0, 0
        while total_steps < config["adaptation_steps"]:
            next_eval = min([s for s in config["evaluation_steps"] if s > total_steps] + [config["adaptation_steps"]])
            start = time.perf_counter()
            rollout_model = learner.rollout_model()
            environment_seed = config["training_seed_base"] + seed * seed_stride + episode_index
            episode_index += 1
            episode, audit, total_return = rollout(rollout_model, algorithm, condition, config, environment_seed,
                                                   deterministic=False,
                                                   max_steps=min(config["episode_horizon"], next_eval - total_steps))
            new_episodes.append(episode)
            total_steps += len(episode["actions"])
            update_credit += len(episode["actions"]) * updates_per_step
            n_updates = int(update_credit + 1e-9)
            update_credit -= n_updates
            losses = []
            for _ in range(n_updates):
                batch = sample_batch(new_episodes, old_episodes, config["batch_size"], learner.context,
                                     retained_fraction, rng, device)
                losses.append(learner.update(batch))
            if torch.cuda.is_available() and device.startswith("cuda"):
                torch.cuda.synchronize()
            cumulative_training_seconds += time.perf_counter() - start
            training_row = {"steps": total_steps, "episode_seed": environment_seed,
                            "episode_length": len(episode["actions"]),
                            "episode_return": total_return, "updates": learner.updates,
                            "training_seconds": cumulative_training_seconds,
                            **{key: float(np.mean([row[key] for row in losses])) for key in (losses[0] if losses else {})}}
            with (folder / "training.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(training_row, allow_nan=False) + "\n")
            print(f"  {name} collect={total_steps} updates={learner.updates} elapsed={cumulative_training_seconds:.1f}s", flush=True)
            if total_steps in config["evaluation_steps"]:
                checkpoint = folder / f"checkpoint_{total_steps}.pt"
                if total_steps == config["adaptation_steps"]:
                    learner.save(checkpoint)
                else:
                    torch.save({"agent": learner.model.state_dict(), "updates": learner.updates}, checkpoint)
                digest = sha256(checkpoint)
                checkpoints.append({"steps": total_steps, "file": checkpoint.name, "sha256": digest})
                means = evaluate(learner, config, condition, folder, total_steps, writer, digest, cumulative_training_seconds)
                handle.flush()
                curves.append({"steps": total_steps, "training_seconds": cumulative_training_seconds, **means})
                print(f"  {name} step={total_steps} A={means['clean']:.2f} B={means[condition]:.2f}", flush=True)
            write_json(output / "status.json", {"status": "RUNNING", "branch": name, "steps": total_steps,
                                               "branch_training_seconds": cumulative_training_seconds})
    if total_steps != config["adaptation_steps"]:
        raise RuntimeError("Unexpected episode termination changed the fixed budget")
    save_episodes(folder / "adaptation_replay.npz", new_episodes)
    original = torch.load(source["copy"], map_location="cpu", weights_only=True)
    now = learner.model.state_dict()
    max_change = max(float((now[key].detach().cpu() - value).abs().max()) for key, value in original.items())
    outside_support_change = None
    if algorithm == "transformer":
        outside_support_change = float((now['trunk.pos_emb'].detach().cpu()[:, 32:] - original['trunk.pos_emb'][:, 32:]).abs().max())
        if outside_support_change != 0:
            raise RuntimeError("Updated positional embeddings outside declared context support")
    if max_change == 0 or sha256(source["original"]) != source["sha256"]:
        raise RuntimeError("No learning update or original checkpoint changed")
    result = {**specification, "status": "COMPLETE", "environment_steps": total_steps,
              "updates": learner.updates, "curves": curves, "checkpoints": checkpoints,
              "max_parameter_change": max_change, "positional_change_outside_support": outside_support_change,
              "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(folder / "result.json", result)
    print(f"COMPLETE {name}", flush=True)
    del learner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json",
                        help="Protocol JSON; must live in this directory so the source snapshot captures it")
    parser.add_argument("--algorithms", nargs="+", help="Shard: subset of the protocol's algorithms")
    parser.add_argument("--seeds", nargs="+", type=int, help="Shard: subset of the protocol's seeds")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    protocol_path = args.protocol.resolve()
    if protocol_path.parent != HERE:
        raise ValueError("Protocol file must be inside the experiment directory")
    config = json.loads(protocol_path.read_text(encoding="utf-8"))
    config["protocol_file"] = protocol_path.name
    for key, subset in (("algorithms", args.algorithms), ("seeds", args.seeds)):
        if subset:
            if not set(subset) <= set(config[key]):
                raise ValueError(f"Shard {key} {subset} not in protocol {config[key]}")
            config[key] = [value for value in config[key] if value in subset]
            config[f"shard_{key}"] = True
    if args.smoke:
        config.update(seeds=[42], episode_horizon=32, adaptation_steps=64, evaluation_steps=[0, 32, 64],
                      evaluation_episodes=1, updates_per_episode=2, batch_size=8)
    started = time.perf_counter()
    manifest = prepare(args.output, config, args.smoke)
    all_results = []
    try:
        for algorithm in config["algorithms"]:
            for seed in config["seeds"]:
                seed_all(seed)
                source = manifest["inputs"][f"{algorithm}_{seed}"]["copy"]
                model = load_agent(algorithm, source, config["context"])
                old_episodes = []
                for index in range(config["old_replay_episodes"]):
                    old, _, _ = rollout(model, algorithm, "clean", config, config["old_replay_seed_base"] + seed * 100 + index)
                    old_episodes.append(old)
                save_episodes(args.output / "inputs" / f"old_replay_{algorithm}_{seed}.npz", old_episodes)
                for condition in config["conditions"]:
                    fractions = config.get("branch_fractions", {}).get(condition, config["retained_replay_fractions"])
                    for fraction in fractions:
                        result = run_branch(args.output, manifest, config, algorithm, seed, condition, fraction, old_episodes, args.device)
                        all_results.append(result)
        for source in manifest["inputs"].values():
            if sha256(source["original"]) != source["sha256"]:
                raise RuntimeError("Source checkpoint integrity check failed")
        write_json(args.output / "study_result.json", {"status": "COMPLETE", "branches": len(all_results),
                   "total_seconds": time.perf_counter() - started, "results": all_results})
        write_json(args.output / "status.json", {"status": "COMPLETE", "branches": len(all_results),
                                               "total_seconds": time.perf_counter() - started})
        print(f"STUDY COMPLETE: {len(all_results)} branches in {time.perf_counter() - started:.1f}s", flush=True)
    except Exception as error:
        write_json(args.output / "status.json", {"status": "FAILED", "completed_branches": len(all_results), "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
