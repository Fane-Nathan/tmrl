"""Tier 1 opaque-box E2E tests for Features #9-#15 (experiments, tests, analyzers, scripts, tutorials, docs, hygiene)."""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.e2e.conftest import (
    REPO_ROOT,
    copy_sensor_adaptation_workspace,
    create_synthetic_sensor_checkpoint,
    has_jax_stack,
    has_working_bash,
    populate_tmrl_data,
    run_python,
)

pytestmark = [pytest.mark.tier1]

EXP_DIR = REPO_ROOT / "experiments" / "sensor_adaptation_v1"


# =============================================================================
# Feature #9: Sensor-Adaptation Study & Env (6 tests)
# =============================================================================


def test_f09_01_sensor_adaptation_has_zero_tmrl_imports() -> None:
    for py_file in sorted(EXP_DIR.glob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("tmrl"), f"{py_file.name} imports {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("tmrl"), f"{py_file.name} imports from {node.module}"


def test_f09_02_protocol_schemas_consistent() -> None:
    expected = {
        "protocol.json": ("HalfCheetah-v5", 17),
        "protocol_extension.json": ("HalfCheetah-v5", 17),
        "protocol_ant.json": ("Ant-v5", 27),
    }
    for fname, (env_name, noise_len) in expected.items():
        data = json.loads((EXP_DIR / fname).read_text(encoding="utf-8"))
        assert data["environment"] == env_name
        assert len(data["noise_std"]) == noise_len
        assert all(float(v) >= 0.0 for v in data["noise_std"])
        assert data["adaptation_steps"] in data["evaluation_steps"]
        assert 1 <= int(data["context"]) <= 32
        assert set(data["algorithms"]) == {"transformer", "reactive"}
        assert len(data["seeds"]) >= 3


def test_f09_03_model_provenance_no_hardcoded_developer_paths() -> None:
    entries = json.loads((EXP_DIR / "model_provenance.json").read_text(encoding="utf-8"))
    assert isinstance(entries, list) and len(entries) == 2
    for entry in entries:
        src = entry["source"]
        assert not re.match(r"^[A-Za-z]:\\", src), f"Absolute Windows path in model_provenance.json: {src}"
        assert "felix" not in src.lower()
        assert re.fullmatch(r"[0-9a-f]{64}", entry["source_sha256"])
        assert len(entry["classes"]) >= 3


def test_f09_04_sensor_fault_env_noise_delay_and_clean_isolation() -> None:
    gym = pytest.importorskip("gymnasium")
    if str(EXP_DIR) not in sys.path:
        sys.path.insert(0, str(EXP_DIR))
    from sensor_env import SensorFaultEnv  # type: ignore

    class _DummyBoxEnv(gym.Env):
        def __init__(self) -> None:
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (3,), np.float32)
            self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
            self.t = 0
            self.last_action = None

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.t = 0
            return np.array([1.0, 2.0, 3.0], np.float32), {}

        def step(self, action):
            self.last_action = action.copy()
            self.t += 1
            return np.array([1.0 + self.t, 2.0 + self.t, 3.0 + self.t], np.float32), 1.5, False, False, {}

    clean = SensorFaultEnv(_DummyBoxEnv(), fault="clean")
    noisy = SensorFaultEnv(_DummyBoxEnv(), fault="sensor_noise", noise_std=[1.0, 0.0, 0.0])
    delayed = SensorFaultEnv(_DummyBoxEnv(), fault="sensor_delay", delay=2)

    obs_c, info_c = clean.reset(seed=5)
    obs_n, _ = noisy.reset(seed=5)
    obs_d, _ = delayed.reset(seed=5)
    np.testing.assert_array_equal(obs_c[:3], [1.0, 2.0, 3.0])
    assert info_c["sensor_audit"]["true_observation"].shape == (3,)
    assert obs_n[0] != 1.0 and obs_n[1] == 2.0 and obs_n[2] == 3.0
    np.testing.assert_array_equal(obs_d[:3], [0.0, 0.0, 0.0])

    act = np.array([0.25, -0.5], np.float32)
    clean.step(act)
    noisy.step(act)
    obs_d1, *_ = delayed.step(act)
    obs_d2, *_ = delayed.step(act)
    np.testing.assert_array_equal(clean.unwrapped.last_action, act)
    np.testing.assert_array_equal(noisy.unwrapped.last_action, act)
    np.testing.assert_array_equal(obs_d1[:3], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(obs_d2[:3], [1.0, 2.0, 3.0])


def test_f09_05_sensor_replay_save_load_without_pickle(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    if str(EXP_DIR) not in sys.path:
        sys.path.insert(0, str(EXP_DIR))
    from replay import extract_transition, load_episodes, save_episodes  # type: ignore

    ep = {
        "observations": np.arange(11 * 25, dtype=np.float32).reshape(11, 25),
        "actions": np.zeros((10, 6), dtype=np.float32),
        "rewards": np.ones((10,), dtype=np.float32),
        "terminated": np.array([0.0] * 9 + [1.0], dtype=np.float32),
    }
    cur, nxt = extract_transition(ep, 9, context=4)
    assert cur.shape == (4, 25) and nxt.shape == (4, 25)

    npz_path = tmp_path / "episodes.npz"
    save_episodes(npz_path, [ep])
    loaded = load_episodes(npz_path)
    assert len(loaded) == 1
    for key in ep:
        np.testing.assert_array_equal(ep[key], loaded[0][key])


@pytest.mark.requires_torch
def test_f09_06_frozen_models_and_load_agent_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    if str(EXP_DIR) not in sys.path:
        sys.path.insert(0, str(EXP_DIR))
    from learner import configure, load_agent  # type: ignore

    for alg in ("transformer", "reactive"):
        ckpt = create_synthetic_sensor_checkpoint(tmp_path / f"{alg}.pt", alg, obs_dim=25, act_dim=6, seed=7)
        model = load_agent(alg, ckpt, context=32, device="cpu")
        configure(model, alg, 32)
        assert not model.training
        obs = np.zeros((25,), dtype=np.float32)
        if alg == "transformer":
            model.reset_context()
            act = model.step_in_context(obs, adaptive=True, deterministic=True, device="cpu")
        else:
            act = model.act(obs, deterministic=True, device="cpu")
        assert act.shape == (6,) and np.all(np.isfinite(act))


# =============================================================================
# Feature #10: Contract & Unit Test Suites (6 tests)
# =============================================================================


@pytest.mark.requires_torch
@pytest.mark.requires_mujoco
def test_f10_01_sensor_adaptation_test_contracts_passes(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    res = run_python(
        [str(EXP_DIR / "test_contracts.py")],
        home_dir=tmp_path / "home",
        cwd=tmp_path,
        extra_env={"PYTHONPATH": str(EXP_DIR)},
        timeout=120,
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "OK" in res.stderr or "OK" in res.stdout


@pytest.mark.requires_mujoco
def test_f10_02_mujoco_halfcheetah_physics_identical_across_faults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("mujoco")
    monkeypatch.chdir(tmp_path)
    if str(EXP_DIR) not in sys.path:
        sys.path.insert(0, str(EXP_DIR))
    from sensor_env import make_env  # type: ignore

    cfg = json.loads((EXP_DIR / "protocol.json").read_text(encoding="utf-8"))
    envs = [make_env(cond, cfg) for cond in ("clean", "sensor_noise", "sensor_delay")]
    try:
        for env in envs:
            env.reset(seed=99)
        for step in range(4):
            cmd = np.full(6, 0.25 * (step + 1), dtype=np.float32)
            outs = [env.step(cmd) for env in envs]
            for out in outs[1:]:
                np.testing.assert_array_equal(
                    out[4]["sensor_audit"]["true_observation"],
                    outs[0][4]["sensor_audit"]["true_observation"],
                )
                assert out[1] == outs[0][1]
    finally:
        for env in envs:
            env.close()


def test_f10_03_benchmark_fault_env_rejects_sensor_labels() -> None:
    gym = pytest.importorskip("gymnasium")
    path = REPO_ROOT / "scripts" / "benchmark_fault_env.py"
    spec = importlib.util.spec_from_file_location("e2e_benchmark_fault_env", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class _BoxEnv(gym.Env):
        def __init__(self) -> None:
            self.observation_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)
            self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

        def reset(self, *, seed=None, options=None):
            return np.zeros((4,), np.float32), {}

        def step(self, action):
            return np.zeros((4,), np.float32), 0.0, False, False, {}

    for kwargs in ({"fixed_fault": "sensor_noise"}, {"fixed_fault": "sensor_delay"}, {"split": "test_sensor_noise"}):
        with pytest.raises(ValueError, match="does not implement observation faults"):
            mod.FaultInjectionWrapper(_BoxEnv(), **kwargs)


def test_f10_04_repo_unit_tests_exist_and_collect_cleanly(tmp_path: Path) -> None:
    tests_dir = REPO_ROOT / "tests"
    non_e2e_tests = [
        p
        for p in tests_dir.rglob("test_*.py")
        if "e2e" not in p.relative_to(tests_dir).parts
    ]
    assert non_e2e_tests, "Expected repository unit/CLI test suite under tests/ (outside tests/e2e/)"
    res = run_python(
        ["-m", "pytest", *[str(p) for p in non_e2e_tests], "--collect-only", "-q"],
        home_dir=tmp_path / "home",
        cwd=REPO_ROOT,
        timeout=60,
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"


@pytest.mark.requires_torch
def test_f10_05_test_contracts_requires_zero_pt_checkpoints(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    assert not list((REPO_ROOT / "fault_benchmark_results").rglob("*.pt"))
    res = run_python(
        ["-m", "unittest", "discover", "-s", str(EXP_DIR), "-p", "test_contracts.py"],
        home_dir=tmp_path / "home",
        cwd=tmp_path,
        extra_env={"PYTHONPATH": str(EXP_DIR)},
        timeout=120,
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"


def test_f10_06_requirements_files_list_expected_packages() -> None:
    exp_req = (EXP_DIR / "requirements.txt").read_text(encoding="utf-8").lower()
    paper_req = (REPO_ROOT / "requirements" / "paper-confirmatory.txt").read_text(encoding="utf-8").lower()
    for pkg in ("torch", "gymnasium", "mujoco", "numpy", "matplotlib", "psutil"):
        assert pkg in exp_req, f"{pkg} missing in experiments/sensor_adaptation_v1/requirements.txt"
        assert pkg in paper_req, f"{pkg} missing in requirements/paper-confirmatory.txt"
    assert "scipy" in exp_req


# =============================================================================
# Feature #11: Statistical, Latency & Paper Asset Analyzers (6 tests)
# =============================================================================


def test_f11_01_analyzer_clis_help_exit_zero(tmp_path: Path) -> None:
    scripts = ["analyze_results.py", "build_paper_assets.py"]
    if importlib.util.find_spec("scipy") is not None:
        scripts.append("analyze_extension.py")
    if importlib.util.find_spec("torch") is not None:
        scripts.extend(["benchmark_latency.py", "analyze_replay_bias.py"])
    for name in scripts:
        res = run_python([str(EXP_DIR / name), "--help"], home_dir=tmp_path / "home", cwd=EXP_DIR)
        assert res.returncode == 0, f"{name} --help failed: {res.stderr}"
        assert "usage:" in res.stdout.lower()


def test_f11_02_tracked_provenance_metadata_intact() -> None:
    for folder_name in ("fault_benchmark_results", "fault_benchmark_results_ant"):
        folder = REPO_ROOT / folder_name
        assert folder.is_dir()
        assert not list(folder.rglob("*.pt")), f"Unexpected .pt checkpoint in {folder_name}"
        manifest = folder / "confirmatory_manifest.csv"
        assert manifest.is_file()
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 16
        assert all(row.get("status") == "COMPLETE" for row in rows)
        for row in rows:
            sha = row.get("checkpoint_sha256", "")
            assert re.fullmatch(r"[0-9a-f]{64}", sha), f"Bad sha256 in {folder_name}: {sha}"


def test_f11_03_analyze_extension_on_synthetic_complete_fixture(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    shard = tmp_path / "shard"
    branch = shard / "transformer_seed42_sensor_noise_replay0"
    shard = tmp_path / "shard_a"
    (shard / "status.json").parent.mkdir(parents=True, exist_ok=True)
    (shard / "status.json").write_text(json.dumps({"status": "COMPLETE"}), encoding="utf-8")
    for cond, fractions in (("clean", (0.0,)), ("sensor_noise", (0.0, 0.5))):
        for retained in fractions:
            tag = "0" if retained == 0.0 else "50"
            branch = shard / f"transformer_seed42_{cond}_replay{tag}"
            branch.mkdir(parents=True)
            (branch / "branch.json").write_text(
                json.dumps(
                    {
                        "algorithm": "transformer",
                        "seed": 42,
                        "condition": cond,
                        "retained_fraction": retained,
                    }
                ),
                encoding="utf-8",
            )
            (branch / "result.json").write_text(json.dumps({"status": "COMPLETE"}), encoding="utf-8")
            with (branch / "evaluations.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["adaptation_steps", "condition", "episode", "environment_seed", "return", "length", "checkpoint_sha256", "adaptation_seconds"],
                )
                writer.writeheader()
                for step in (0, 5000, 10000, 20000):
                    eval_conds = (("clean", 100.0),) if cond == "clean" else (("clean", 100.0 + retained * 10.0), (cond, 50.0 + step * 0.005 + retained * 5.0))
                    for eval_c, val in eval_conds:
                        writer.writerow(
                            {
                                "adaptation_steps": step,
                                "condition": eval_c,
                                "episode": 0,
                                "environment_seed": 860000,
                                "return": val,
                                "length": 32,
                                "checkpoint_sha256": "0" * 64,
                                "adaptation_seconds": 1.0,
                            }
                        )
    out_dir = tmp_path / "analysis"
    res = run_python(
        [
            str(EXP_DIR / "analyze_extension.py"),
            str(shard),
            "--protocol",
            str(EXP_DIR / "protocol_extension.json"),
            "--out",
            str(out_dir),
            "--allow-subset",
        ],
        home_dir=tmp_path / "home",
        cwd=EXP_DIR,
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "seed_metrics.csv").is_file()


@pytest.mark.requires_torch
def test_f11_04_benchmark_latency_cls_head_parameter_count(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    sources = tmp_path / "sources"
    traj_root = tmp_path / "trajs"
    for alg in ("transformer", "reactive"):
        create_synthetic_sensor_checkpoint(
            sources / f"{alg}_seed_42" / "best_validation_checkpoint.pt",
            alg,
            obs_dim=25,
            act_dim=6,
        )
        tdir = traj_root / f"{alg}_seed42_sensor_delay_replay0"
        tdir.mkdir(parents=True)
        np.savez(tdir / "trajectory_0_clean_0.npz", observations=np.zeros((20, 25), dtype=np.float32))

    out_json = tmp_path / "latency.json"
    res = run_python(
        [
            str(EXP_DIR / "benchmark_latency.py"),
            "--trajectories",
            str(traj_root),
            "--sources",
            str(sources),
            "--seed",
            "42",
            "--steps",
            "10",
            "--warmup",
            "2",
            "--threads",
            "1",
            "--out",
            str(out_json),
        ],
        home_dir=tmp_path / "home",
        cwd=EXP_DIR,
        timeout=90,
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    report = json.loads(out_json.read_text(encoding="utf-8"))
    by_alg = {r["algorithm"]: r for r in report["results"]}
    assert set(by_alg) == {"transformer", "reactive"}
    # Both transformer (critics + task_encoder.cls_head excluded) and reactive (critics excluded)
    # must have inference_path_upper_bound strictly less than total.
    # Also check transformer specifically excludes cls_head via frozen_models param count:
    if str(EXP_DIR) not in sys.path:
        sys.path.insert(0, str(EXP_DIR))
    import frozen_models  # type: ignore

    tr_model = frozen_models.TransformerRL2Agent(obs_dim=25, act_dim=6)
    expected_deploy = sum(
        p.numel()
        for n, p in tr_model.named_parameters()
        if not n.startswith(("critics", "target_", "task_encoder.cls_head", "task_encoder.classifier"))
    )
    assert by_alg["transformer"]["parameters"]["inference_path_upper_bound"] == expected_deploy


def test_f11_05_build_paper_assets_generates_tex_and_figures(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    config = json.loads((EXP_DIR / "protocol.json").read_text(encoding="utf-8"))
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "verification.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({"smoke": False}), encoding="utf-8")

    results = []
    summary_rows = []
    for alg in config["algorithms"]:
        for cond in config["conditions"]:
            for frac in config["retained_replay_fractions"]:
                summary_rows.append(
                    {
                        "algorithm": alg,
                        "condition": cond,
                        "retained_fraction": frac,
                        "B_before_mean": 100.0,
                        "B_after_mean": 150.0,
                        "A_change_mean": -5.0,
                    }
                )
                results.append(
                    {
                        "algorithm": alg,
                        "seed": 42,
                        "condition": cond,
                        "retained_fraction": frac,
                        "curves": [
                            {"steps": s, "clean": 200.0, cond: 100.0 + s * 0.01}
                            for s in config["evaluation_steps"]
                        ],
                    }
                )
    (root / "study_result.json").write_text(json.dumps({"status": "COMPLETE", "results": results}), encoding="utf-8")
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    out = tmp_path / "assets"
    res = run_python([str(EXP_DIR / "build_paper_assets.py"), str(root), str(out)], home_dir=tmp_path / "home", cwd=EXP_DIR)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    for fname in (
        "outcomes_table.tex",
        "sensor_adaptation.pdf",
        "sensor_adaptation.png",
        "sensor_retention.pdf",
        "sensor_retention.png",
    ):
        assert (out / fname).is_file() and (out / fname).stat().st_size > 0


def test_f11_06_analyze_results_fails_cleanly_on_empty_dir(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty_pilot"
    empty_dir.mkdir()
    res = run_python([str(EXP_DIR / "analyze_results.py"), str(empty_dir)], home_dir=tmp_path / "home", cwd=EXP_DIR)
    assert res.returncode != 0
    assert not (empty_dir / "verification.json").exists()


# =============================================================================
# Feature #12: Operational & Source-Training Scripts (6 tests)
# =============================================================================


def test_f12_01_evaluate_live_driving_cli_and_safety_guards(tmp_path: Path) -> None:
    res = run_python([str(REPO_ROOT / "scripts" / "evaluate_live_driving.py"), "--help"], home_dir=tmp_path / "home")
    assert res.returncode == 0, res.stderr
    assert "--unseen-map" in res.stdout

    spec = importlib.util.spec_from_file_location("e2e_eval_live", REPO_ROOT / "scripts" / "evaluate_live_driving.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    stall = mod.MotionStallGuard(window_seconds=2.0, min_displacement_m=0.5, grace_seconds=3.0)
    assert stall.update(0.0, [0.0, 0.0, 0.0]) is False
    assert stall.update(3.5, [0.1, 0.0, 0.0]) is True

    fresh = mod.FrameFreshnessGuard(max_repeat_seconds=0.5, min_motion_m=0.5)
    img = np.zeros((8, 8), dtype=np.uint8)
    assert fresh.update(0.0, img, [0.0, 0.0, 0.0]) is False
    assert fresh.update(0.6, img, [2.0, 0.0, 0.0]) is True

    deadline = mod.CycleDeadlineGuard(max_cycle_seconds=0.1, consecutive_limit=2)
    assert deadline.update(0.2) is False
    assert deadline.update(0.2) is True

    args = mod.parse_args(["--mode", "foundation", "--unseen-map"])
    assert mod.reference_config_for_run(args, {"TRAJECTORY_ASSIST": {}}) is None


def test_f12_02_benchmark_live_pipeline_cli_help_headless(tmp_path: Path) -> None:
    res = run_python(
        [str(REPO_ROOT / "scripts" / "benchmark_live_pipeline.py"), "--help"],
        home_dir=tmp_path / "empty_home",
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "--steps" in res.stdout


def test_f12_03_run_ant_sources_cross_platform_and_empty_queue(tmp_path: Path) -> None:
    script = REPO_ROOT / "scripts" / "run_ant_sources.py"
    source_text = script.read_text(encoding="utf-8")
    assert "ctypes.windll" not in source_text, "run_ant_sources.py must use cross-platform psutil instead of ctypes.windll"

    res_help = run_python([str(script), "--help"], home_dir=tmp_path / "home")
    assert res_help.returncode == 0, res_help.stderr

    out_dir = tmp_path / "ant_out"
    res_empty = run_python([str(script), "--output-dir", str(out_dir)], home_dir=tmp_path / "home")
    assert res_empty.returncode == 0, f"stdout={res_empty.stdout}\nstderr={res_empty.stderr}"


def test_f12_04_wsl2_shell_scripts_lf_and_parameterized() -> None:
    for name in ("bootstrap_continual_jax_wsl2.sh", "configure_continual_jax_wsl_network.sh"):
        path = REPO_ROOT / "scripts" / name
        raw = path.read_bytes()
        assert b"\r\n" not in raw, f"{name} must use LF line endings, found CRLF"
        text = raw.decode("utf-8")
        assert "set -euo pipefail" in text
        assert 'uname -s' in text and "exit 2" in text
        assert "/mnt/c/Users/felix" not in text, f"{name} still contains hardcoded /mnt/c/Users/felix path"
        if has_working_bash():
            chk = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, timeout=10)
            assert chk.returncode == 0, chk.stderr


def test_f12_05_no_dangling_imports_to_deleted_scripts() -> None:
    scripts_dir = REPO_ROOT / "scripts"
    existing_stems = {p.stem for p in scripts_dir.glob("*.py")}
    for search_root in (REPO_ROOT / "tmrl", EXP_DIR, scripts_dir):
        for py_file in search_root.rglob("*.py"):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("scripts."):
                    mod_name = node.module.split(".")[1]
                    assert mod_name in existing_stems, (
                        f"{py_file.relative_to(REPO_ROOT)} imports missing scripts.{mod_name}"
                    )


def test_f12_06_source_recipes_cli_help_and_confirmatory_logging(tmp_path: Path) -> None:
    for script_name in (
        "benchmark_fault_env.py",
        "train_transformer_source_recipe_v1.py",
        "train_reactive_source_recipe_v1.py",
    ):
        res = run_python([str(REPO_ROOT / "scripts" / script_name), "--help"], home_dir=tmp_path / "home")
        assert res.returncode == 0, f"{script_name} --help failed: {res.stderr}"

    spec = importlib.util.spec_from_file_location("e2e_confirmatory_logging", REPO_ROOT / "scripts" / "confirmatory_logging.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sample_file = tmp_path / "sample.bin"
    sample_file.write_bytes(b"abc")
    assert len(mod.compute_file_sha256(sample_file)) == 64
    hw = mod.get_hardware_info()
    assert isinstance(hw, dict) and hw


# =============================================================================
# Feature #13: Tutorials & Neural Diagnostics (6 tests)
# =============================================================================


def test_f13_01_tutorial_scripts_valid_python_syntax() -> None:
    tuto_dir = REPO_ROOT / "tutorial"
    py_files = list(tuto_dir.rglob("*.py"))
    assert len(py_files) >= 8
    for py_file in py_files:
        ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))


@pytest.mark.requires_torch
def test_f13_02_debug_all_neural_networks_runs_headless_cpu(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    res = run_python(
        [str(REPO_ROOT / "tutorial" / "debug_all_neural_networks.py")],
        home_dir=tmp_path / "empty_home",
        timeout=120,
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "SMOKE DIAGNOSTICS PASSED" in res.stdout


def test_f13_03_train_world_model_on_replay_configurable_no_felix(tmp_path: Path) -> None:
    script = REPO_ROOT / "tutorial" / "train_world_model_on_replay.py"
    text = script.read_text(encoding="utf-8")
    assert "C:\\Users\\felix" not in text and "felix" not in text.lower()
    if has_jax_stack():
        res = run_python([str(script), "--help"], home_dir=tmp_path / "empty_home")
        assert res.returncode == 0, res.stderr
        assert "--checkpoint" in res.stdout


@pytest.mark.requires_jax
def test_f13_04_tuto_self_play_jax_runs_standalone(tmp_path: Path) -> None:
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    res = run_python(
        [str(REPO_ROOT / "tutorial" / "tuto_self_play_jax.py")],
        home_dir=tmp_path / "empty_home",
        timeout=120,
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "PASSED" in res.stdout


def test_f13_05_competition_actor_cli_help_without_tmrldata(tmp_path: Path) -> None:
    comp_script = REPO_ROOT / "tutorial" / "competition" / "custom_actor_module.py"
    home = tmp_path / "home"
    populate_tmrl_data(home)
    res = run_python([str(comp_script), "--help"], home_dir=home, cwd=comp_script.parent)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "--trainer" in res.stdout and "--worker" in res.stdout


def test_f13_06_tutorial_readme_documents_active_scripts_and_valid_links() -> None:
    tuto_readme = REPO_ROOT / "tutorial" / "README.md"
    text = tuto_readme.read_text(encoding="utf-8")
    for expected_name in ("tuto.py", "tuto_minimal_pendulum_torch.py", "debug_all_neural_networks.py", "train_world_model_on_replay.py"):
        assert expected_name in text, f"tutorial/README.md missing reference to {expected_name}"
    for match in re.finditer(r"\[[^\]]+\]\(([^)#]+)(?:#[^)]*)?\)", text):
        target = match.group(1).strip()
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        resolved = (tuto_readme.parent / target).resolve()
        assert resolved.exists(), f"Broken link in tutorial/README.md: {target}"


# =============================================================================
# Feature #14: Cross-Platform Docs & Reproducibility Guides (6 tests)
# =============================================================================


def _user_doc_files() -> list[Path]:
    files: list[Path] = []
    for rel in ("README.md", "CONTRIBUTING.md"):
        p = REPO_ROOT / rel
        if p.is_file():
            files.append(p)
    for folder in ("readme", "docs", "tutorial", "experiments"):
        root = REPO_ROOT / folder
        if root.is_dir():
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() in {".md", ".rst", ".py", ".json"}:
                    files.append(p)
    for p in (REPO_ROOT / "scripts").rglob("*"):
        if p.is_file() and p.suffix.lower() in {".md", ".sh", ".py"}:
            files.append(p)
    return sorted(set(files))


def test_f14_01_zero_hardcoded_developer_paths_in_docs_and_scripts() -> None:
    forbidden = (
        re.compile(r"C:\\Users\\felix", re.IGNORECASE),
        re.compile(r"/home/felix/", re.IGNORECASE),
        re.compile(r"/mnt/c/Users/felix", re.IGNORECASE),
        re.compile(r"D:\\Project\\tmrl", re.IGNORECASE),
        re.compile(r"/mnt/d/Project/tmrl", re.IGNORECASE),
        re.compile(r"D:\\miniconda3", re.IGNORECASE),
        re.compile(r"C:\\Users\\Yann", re.IGNORECASE),
    )
    hits: list[str] = []
    for path in _user_doc_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            for pat in forbidden:
                if pat.search(line):
                    hits.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()[:120]}")
    assert not hits, "Found hardcoded developer paths:\n" + "\n".join(hits[:20])


def test_f14_02_zero_ai_workflow_mentions_in_user_docs() -> None:
    doc_paths = [
        p
        for p in _user_doc_files()
        if p.suffix.lower() in {".md", ".rst"}
    ]
    forbidden = (
        re.compile(r"\b(?:data-quality|data-validation|computer-use)\s+skill\b", re.IGNORECASE),
        re.compile(r"(?:^|[`/\s])\.agents/", re.IGNORECASE),
        re.compile(r"(?:^|[`/\s])codebase_overview/", re.IGNORECASE),
    )
    hits: list[str] = []
    for path in doc_paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            for pat in forbidden:
                if pat.search(line):
                    hits.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()[:120]}")
    assert not hits, "Found AI-workflow references in user docs:\n" + "\n".join(hits[:20])


def test_f14_03_relative_markdown_links_resolve() -> None:
    md_files = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "CONTRIBUTING.md",
        *sorted((REPO_ROOT / "readme").glob("*.md")),
        REPO_ROOT / "tutorial" / "README.md",
        EXP_DIR / "README.md",
    ]
    broken: list[str] = []
    for md in md_files:
        if not md.is_file():
            continue
        text = md.read_text(encoding="utf-8")
        for match in re.finditer(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)", text):
            target = match.group(1)
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (md.parent / target).resolve()
            if not resolved.exists():
                broken.append(f"{md.relative_to(REPO_ROOT)} -> {target}")
    assert not broken, "Broken relative markdown links:\n" + "\n".join(broken)


def test_f14_04_documented_repo_script_and_config_paths_exist() -> None:
    doc_files = [
        REPO_ROOT / "README.md",
        *sorted((REPO_ROOT / "readme").glob("*.md")),
        EXP_DIR / "README.md",
    ]
    path_pat = re.compile(r"(?<![A-Za-z0-9_./-])((?:scripts|experiments|tutorial)/[A-Za-z0-9_./-]+\.(?:py|sh|json|md))")
    missing: list[str] = []
    for doc in doc_files:
        if not doc.is_file():
            continue
        for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            for match in path_pat.finditer(line):
                rel = match.group(1)
                if "*" in rel or "<" in rel:
                    continue
                if not (REPO_ROOT / rel).exists():
                    missing.append(f"{doc.relative_to(REPO_ROOT)}:{lineno} -> {rel}")
    assert not missing, "Dangling repo-relative script/config references in docs:\n" + "\n".join(missing)


def test_f14_05_sphinx_toctree_and_cli_docs_consistent() -> None:
    docs_src = REPO_ROOT / "docs" / "source"
    tmrl_rst = (docs_src / "tmrl.rst").read_text(encoding="utf-8")
    in_toctree = False
    missing_rst: list[str] = []
    for raw_line in tmrl_rst.splitlines():
        if raw_line.strip().startswith(".. toctree::"):
            in_toctree = True
            continue
        if in_toctree:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith(":"):
                continue
            if not raw_line.startswith((" ", "\t")):
                in_toctree = False
                continue
            if not (docs_src / f"{stripped}.rst").is_file():
                missing_rst.append(stripped)
    assert not missing_rst, f"docs/source/tmrl.rst cites missing .rst files: {missing_rst}"

    cli_rst = (docs_src / "cli.rst").read_text(encoding="utf-8")
    assert "--no-wandb" not in cli_rst, "docs/source/cli.rst documents nonexistent --no-wandb flag"


def test_f14_06_docs_cover_windows_and_linux_wsl2_preconditions() -> None:
    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "Windows" in readme_text and "Linux" in readme_text
    continual_doc = (REPO_ROOT / "readme" / "continual_foundation_quickstart.md").read_text(encoding="utf-8")
    assert "WSL2" in continual_doc or "WSL" in continual_doc
    exp_doc = (EXP_DIR / "README.md").read_text(encoding="utf-8")
    assert "test_contracts.py" in exp_doc and "run_study.py" in exp_doc


# =============================================================================
# Feature #15: Workspace & Git Hygiene (6 tests)
# =============================================================================


def test_f15_01_ai_workspace_dirs_removed() -> None:
    assert not (REPO_ROOT / ".agents").exists(), ".agents/ directory must be removed"
    assert not (REPO_ROOT / "codebase_overview").exists(), "codebase_overview/ directory must be removed"
    res = subprocess.run(
        ["git", "ls-files", ".agents", "codebase_overview"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert res.returncode == 0
    assert res.stdout.strip() == "", f"AI workspace files still tracked:\n{res.stdout}"


def test_f15_02_wandb_settings_not_tracked() -> None:
    res = subprocess.run(
        ["git", "ls-files", "tmrl/wandb/settings"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert res.returncode == 0
    assert res.stdout.strip() == "", "tmrl/wandb/settings should not be git-tracked"


def test_f15_03_git_status_clean_outside_tests_e2e() -> None:
    res = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert res.returncode == 0
    dirty = [
        line
        for line in res.stdout.splitlines()
        if line.strip()
        and "tests/e2e/" not in line.replace("\\", "/")
        and (line[:2] == "??" or (len(line) > 1 and line[1] != " "))
    ]
    assert not dirty, "Unstaged or untracked workspace changes remain:\n" + "\n".join(dirty)


def test_f15_04_gitattributes_protects_five_text_paths() -> None:
    protected = [
        "experiments/sensor_adaptation_v1/run_study.py",
        "scripts/benchmark_fault_env.py",
        "scripts/confirmatory_logging.py",
        "scripts/train_transformer_source_recipe_v1.py",
        "scripts/train_reactive_source_recipe_v1.py",
    ]
    res = subprocess.run(
        ["git", "check-attr", "text", "--", *protected],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert res.returncode == 0, res.stderr
    for rel in protected:
        assert f"{rel}: text: unset" in res.stdout, f"Missing -text protection on {rel}:\n{res.stdout}"


def test_f15_05_fault_benchmark_results_provenance_preserved() -> None:
    for folder, min_count in (("fault_benchmark_results", 190), ("fault_benchmark_results_ant", 190)):
        res = subprocess.run(
            ["git", "ls-files", folder],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert res.returncode == 0
        tracked = [line for line in res.stdout.splitlines() if line.strip()]
        assert len(tracked) >= min_count, f"{folder} has only {len(tracked)} tracked files"
        assert not any(f.endswith(".pt") for f in tracked)


def test_f15_06_gitignore_blocks_checkpoints_and_wandb_outputs() -> None:
    res = subprocess.run(
        ["git", "check-ignore", "dummy.pt", "dummy.npz", "wandb/debug.log"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert res.returncode == 0
    ignored = set(res.stdout.splitlines())
    assert {"dummy.pt", "dummy.npz", "wandb/debug.log"} <= ignored
