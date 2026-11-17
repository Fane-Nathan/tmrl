"""Tier 4 opaque-box E2E tests: 10 real-world external-researcher scenarios (T4-01..T4-10)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from tests.e2e.conftest import (
    REPO_ROOT,
    copy_sensor_adaptation_workspace,
    free_tcp_port,
    has_jax_stack,
    has_module,
    has_sphinx,
    has_working_bash,
    make_tmrl_config,
    make_valid_map_manifest,
    populate_tmrl_data,
    run_python,
)

pytestmark = [pytest.mark.tier4]

EXP_DIR = REPO_ROOT / "experiments" / "sensor_adaptation_v1"


def test_t4_01_fresh_shell_headless_package_setup(tmp_path: Path) -> None:
    """T4-01: Researcher installs/imports `tmrl` in a fresh shell with isolated HOME and no TrackMania/pywin32/network."""
    home = tmp_path / "fresh_home"
    r1 = run_python(
        ["-c", "import tmrl; print(tmrl.__file__)"],
        home_dir=home,
        block_network=True,
        netblock_dir=tmp_path / "netblock",
    )
    assert r1.returncode == 0, r1.stderr
    assert not (home / "TmrlData").exists()

    r2 = run_python(
        ["-m", "tmrl", "--help"],
        home_dir=home,
        block_network=True,
        netblock_dir=tmp_path / "netblock",
    )
    assert r2.returncode == 0, r2.stderr
    assert not (home / "TmrlData").exists()

    populate_tmrl_data(home)
    r3 = run_python(
        ["-m", "tmrl", "--install"],
        home_dir=home,
        block_network=True,
        netblock_dir=tmp_path / "netblock",
    )
    assert r3.returncode == 0, r3.stderr


@pytest.mark.requires_torch
@pytest.mark.requires_mujoco
def test_t4_02_cross_platform_unit_cli_and_contract_suite_gate(tmp_path: Path) -> None:
    """T4-02: Researcher runs repository unit/CLI tests + `test_contracts.py` on CPU without GPU or TrackMania."""
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")

    tests_dir = REPO_ROOT / "tests"
    non_e2e = [p for p in tests_dir.rglob("test_*.py") if "e2e" not in p.relative_to(tests_dir).parts]
    assert non_e2e, "Expected M3 unit/CLI tests under tests/"
    r_unit = run_python(["-m", "pytest", *[str(p) for p in non_e2e], "-q"], home_dir=tmp_path / "home", cwd=REPO_ROOT, timeout=120)
    assert r_unit.returncode == 0, f"Unit tests failed:\n{r_unit.stdout}\n{r_unit.stderr}"

    r_contracts = run_python(
        [str(EXP_DIR / "test_contracts.py")],
        home_dir=tmp_path / "home",
        cwd=tmp_path,
        extra_env={"PYTHONPATH": str(EXP_DIR)},
        timeout=120,
    )
    assert r_contracts.returncode == 0, f"test_contracts.py failed:\n{r_contracts.stdout}\n{r_contracts.stderr}"


@pytest.mark.requires_torch
@pytest.mark.requires_mujoco
def test_t4_03_standalone_mujoco_sensor_adaptation_study_pipeline(tmp_path: Path) -> None:
    """T4-03: Researcher reproduces source-smoke -> study-smoke -> analyze -> latency -> paper assets in isolated workspace."""
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    ws = copy_sensor_adaptation_workspace(tmp_path)
    exp = ws / "experiments" / "sensor_adaptation_v1"
    src_out = ws / "fault_benchmark_results"

    for script_name in ("train_transformer_source_recipe_v1.py", "train_reactive_source_recipe_v1.py"):
        r_src = run_python(
            [
                str(REPO_ROOT / "scripts" / script_name),
                "--env",
                "HalfCheetah-v5",
                "--smoke_test",
                "--seed",
                "42",
                "--num_threads",
                "1",
                "--output_dir",
                str(src_out),
            ],
            home_dir=tmp_path / "home",
            cwd=tmp_path,
            timeout=120,
        )
        assert r_src.returncode == 0, f"{script_name} failed:\n{r_src.stdout}\n{r_src.stderr}"

    pilot = tmp_path / "pilot_run"
    r_study = run_python(
        [str(exp / "run_study.py"), "--protocol", str(exp / "protocol.json"), "--output", str(pilot), "--smoke", "--device", "cpu"],
        home_dir=tmp_path / "home",
        cwd=exp,
        timeout=120,
    )
    assert r_study.returncode == 0, f"run_study failed:\n{r_study.stdout}\n{r_study.stderr}"

    r_ana = run_python([str(exp / "analyze_results.py"), str(pilot)], home_dir=tmp_path / "home", cwd=exp)
    assert r_ana.returncode == 0, f"analyze_results failed:\n{r_ana.stdout}\n{r_ana.stderr}"
    assert json.loads((pilot / "verification.json").read_text(encoding="utf-8"))["status"] == "PASS"

    latency_json = tmp_path / "latency.json"
    r_lat = run_python(
        [
            str(exp / "benchmark_latency.py"),
            "--trajectories",
            str(pilot),
            "--sources",
            str(src_out),
            "--seed",
            "42",
            "--steps",
            "8",
            "--warmup",
            "2",
            "--threads",
            "1",
            "--out",
            str(latency_json),
        ],
        home_dir=tmp_path / "home",
        cwd=exp,
        timeout=90,
    )
    assert r_lat.returncode == 0, f"benchmark_latency failed:\n{r_lat.stdout}\n{r_lat.stderr}"
    assert latency_json.is_file()

    manifest = json.loads((pilot / "manifest.json").read_text(encoding="utf-8"))
    manifest["smoke"] = False
    (pilot / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assets = tmp_path / "paper_assets"
    r_assets = run_python([str(exp / "build_paper_assets.py"), str(pilot), str(assets)], home_dir=tmp_path / "home", cwd=exp)
    assert r_assets.returncode == 0, f"build_paper_assets failed:\n{r_assets.stdout}\n{r_assets.stderr}"
    assert (assets / "outcomes_table.tex").is_file()


@pytest.mark.requires_torch
@pytest.mark.requires_mujoco
def test_t4_04_sharded_extension_study_and_ant_source_runner(tmp_path: Path) -> None:
    """T4-04: Researcher runs sharded extension study, extension/replay-bias analyzers, and checks Ant launcher."""
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    pytest.importorskip("scipy")
    ws = copy_sensor_adaptation_workspace(tmp_path)
    exp = ws / "experiments" / "sensor_adaptation_v1"

    r_src = run_python(
        [
            str(REPO_ROOT / "scripts" / "train_transformer_source_recipe_v1.py"),
            "--env",
            "HalfCheetah-v5",
            "--smoke_test",
            "--seed",
            "42",
            "--num_threads",
            "1",
            "--output_dir",
            str(ws / "fault_benchmark_results"),
        ],
        home_dir=tmp_path / "home",
        cwd=tmp_path,
        timeout=120,
    )
    assert r_src.returncode == 0, r_src.stderr

    shard = tmp_path / "shard_ext"
    r_ext_study = run_python(
        [
            str(exp / "run_study.py"),
            "--protocol",
            str(exp / "protocol_extension.json"),
            "--algorithms",
            "transformer",
            "--seeds",
            "42",
            "--smoke",
            "--device",
            "cpu",
            "--output",
            str(shard),
        ],
        home_dir=tmp_path / "home",
        cwd=exp,
        timeout=120,
    )
    assert r_ext_study.returncode == 0, r_ext_study.stderr

    ana_out = tmp_path / "ext_analysis"
    r_ext = run_python(
        [
            str(exp / "analyze_extension.py"),
            str(shard),
            "--protocol",
            str(exp / "protocol_extension.json"),
            "--allow-subset",
            "--out",
            str(ana_out),
        ],
        home_dir=tmp_path / "home",
        cwd=exp,
    )
    assert r_ext.returncode == 0, r_ext.stderr
    assert (ana_out / "summary.json").is_file()

    r_bias = run_python(
        [
            str(exp / "analyze_replay_bias.py"),
            str(shard),
            "--protocol",
            str(exp / "protocol_extension.json"),
            "--budget",
            "64",
            "--tail",
            "8",
            "--device",
            "cpu",
            "--out",
            str(ana_out),
        ],
        home_dir=tmp_path / "home",
        cwd=exp,
    )
    assert r_bias.returncode == 0, r_bias.stderr
    assert (ana_out / "critic_bias_summary.csv").is_file()

    ant_launcher = REPO_ROOT / "scripts" / "run_ant_sources.py"
    assert "ctypes.windll" not in ant_launcher.read_text(encoding="utf-8")
    assert run_python([str(ant_launcher), "--help"], home_dir=tmp_path / "home").returncode == 0


def test_t4_05_continual_jax_wsl2_config_preflight_freeze_and_shell(tmp_path: Path) -> None:
    """T4-05: Researcher prepares Continual JAX config, validates manifest, freezes baseline, and verifies WSL2 scripts."""
    home = tmp_path / "home"
    src_cfg = tmp_path / "base_config.json"
    out_cfg = home / "TmrlData" / "config" / "config.json"
    src_cfg.write_text(json.dumps(make_tmrl_config(grayscale=True, wandb_key="secret-123")), encoding="utf-8")

    r_prep = run_python(
        [
            "-m",
            "tmrl.tools.prepare_continual_jax_config",
            "--source",
            str(src_cfg),
            "--output",
            str(out_cfg),
            "--scrub-wandb-key",
            "--trainer-server-ip",
            "172.24.0.1",
        ],
        home_dir=home,
    )
    assert r_prep.returncode == 0, r_prep.stderr
    assert json.loads(out_cfg.read_text(encoding="utf-8"))["WANDB_KEY"] == ""

    manifest_file = tmp_path / "map_manifest.json"
    manifest_file.write_text(json.dumps(make_valid_map_manifest(frozen=True)), encoding="utf-8")
    r_man = run_python(["-m", "tmrl.tools.validate_continual_map_manifest", str(manifest_file), "--require-frozen"], home_dir=home)
    assert r_man.returncode == 0, r_man.stderr

    frozen_dir = tmp_path / "frozen"
    r_freeze = run_python(
        ["-m", "tmrl.tools.freeze_continual_baseline", "--source", str(out_cfg), "--output-dir", str(frozen_dir), "--repo-root", str(REPO_ROOT)],
        home_dir=home,
    )
    assert r_freeze.returncode == 0, r_freeze.stderr
    assert (frozen_dir / "baseline_manifest.json").is_file()

    for sh_name in ("bootstrap_continual_jax_wsl2.sh", "configure_continual_jax_wsl_network.sh"):
        sh_path = REPO_ROOT / "scripts" / sh_name
        raw = sh_path.read_bytes()
        assert b"\r\n" not in raw
        assert "/mnt/c/Users/felix" not in raw.decode("utf-8")
        if has_working_bash():
            assert subprocess.run(["bash", "-n", str(sh_path)], capture_output=True, text=True, timeout=10).returncode == 0

    if has_jax_stack():
        assert run_python(["-m", "tmrl.tools.continual_preflight", "--output", str(tmp_path / "preflight.json")], home_dir=home, timeout=90).returncode == 0
        assert run_python(["-m", "tmrl.tools.validate_continual_trainer_stack", "--output-directory", str(tmp_path / "stack")], home_dir=home, timeout=120).returncode == 0


@pytest.mark.requires_torch
def test_t4_06_tutorial_neural_diagnostics_and_replay_training(tmp_path: Path) -> None:
    """T4-06: Researcher runs Torch neural diagnostics and verifies configurable replay world-model tutorial."""
    pytest.importorskip("torch")
    r_diag = run_python([str(REPO_ROOT / "tutorial" / "debug_all_neural_networks.py")], home_dir=tmp_path / "empty_home", timeout=120)
    assert r_diag.returncode == 0, r_diag.stderr
    assert "SMOKE DIAGNOSTICS PASSED" in r_diag.stdout

    replay_script = REPO_ROOT / "tutorial" / "train_world_model_on_replay.py"
    assert "C:\\Users\\felix" not in replay_script.read_text(encoding="utf-8")
    if has_jax_stack():
        assert run_python([str(replay_script), "--help"], home_dir=tmp_path / "empty_home").returncode == 0
        assert run_python([str(REPO_ROOT / "tutorial" / "tuto_self_play_jax.py")], home_dir=tmp_path / "empty_home", timeout=120).returncode == 0


@pytest.mark.requires_torch
@pytest.mark.requires_tlspyo
def test_t4_07_distributed_rl_loopback_pendulum_sync(tmp_path: Path) -> None:
    """T4-07: Researcher runs minimal `Server` + `Trainer` + `RolloutWorker.run_synchronous` on `Pendulum-v1`."""
    pytest.importorskip("torch")
    pytest.importorskip("tlspyo")
    from tmrl.core.envs import GenericGymEnv
    from tmrl.core.networking import RolloutWorker, Server, Trainer
    from tmrl.core.torch.training_offline import TorchTrainingOffline
    from tmrl.core.util import partial
    from tmrl.custom.torch.custom_algorithms import SpinupSACAgent
    from tmrl.custom.torch.custom_memories import GenericTorchMemory
    from tmrl.custom.torch.custom_models import MLPActorCritic, SquashedGaussianMLPActor

    pub, srv_l, tr_l, wk_l = free_tcp_port(), free_tcp_port(), free_tcp_port(), free_tcp_port()
    env_cls = partial(GenericGymEnv, id="Pendulum-v1", gym_kwargs={"render_mode": None})
    dummy = env_cls()
    obs_space, act_space = dummy.observation_space, dummy.action_space
    dummy.close()

    tmod = tmp_path / "pendulum_trainer.tmod"
    tcpt = tmp_path / "pendulum_trainer.tcpt"
    server = Server(port=pub, password="pw", local_port=srv_l, security=None, keys_dir=None)
    errors: list[BaseException] = []
    try:
        training_cls = partial(
            TorchTrainingOffline,
            env_cls=(obs_space, act_space),
            memory_cls=partial(GenericTorchMemory, memory_size=64, batch_size=4),
            training_agent_cls=partial(SpinupSACAgent, model_cls=MLPActorCritic, device="cpu"),
            epochs=1,
            rounds=1,
            steps=4,
            update_model_interval=2,
            update_buffer_interval=1,
            max_training_steps_per_env_step=10.0,
            sleep_between_buffer_retrieval_attempts=0.05,
            start_training=4,
            device="cpu",
        )
        trainer = Trainer(
            training_cls=training_cls,
            server_ip="127.0.0.1",
            server_port=pub,
            local_com_port=tr_l,
            password="pw",
            model_path=str(tmod),
            checkpoint_path=str(tcpt),
            security=None,
        )
        worker = RolloutWorker(
            env_cls=env_cls,
            actor_module_cls=partial(SquashedGaussianMLPActor),
            sample_compressor=None,
            device="cpu",
            server_ip="127.0.0.1",
            server_port=pub,
            local_port=wk_l,
            password="pw",
            max_samples_per_episode=8,
            model_path=str(tmp_path / "pendulum_worker.tmod"),
            security=None,
        )

        def _worker() -> None:
            try:
                worker.run_synchronous(initial_steps=8, nb_steps=8, max_steps_per_update=4, end_episodes=False)
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        trainer.run()
        thread.join(timeout=15)
        assert not errors, f"Worker failed: {errors[0]!r}"
        assert tmod.is_file() and tcpt.is_file()
    finally:
        server.stop()


def test_t4_08_live_trackmania_tooling_and_competition_dry_cli(tmp_path: Path) -> None:
    """T4-08: Researcher verifies live-TM tooling, competition template CLI, and actor latency benchmark without TrackMania."""
    assert run_python([str(REPO_ROOT / "scripts" / "evaluate_live_driving.py"), "--help"], home_dir=tmp_path / "empty_home").returncode == 0
    assert run_python([str(REPO_ROOT / "scripts" / "benchmark_live_pipeline.py"), "--help"], home_dir=tmp_path / "empty_home").returncode == 0

    home = tmp_path / "home"
    populate_tmrl_data(home)
    comp = REPO_ROOT / "tutorial" / "competition" / "custom_actor_module.py"
    assert run_python([str(comp), "--help"], home_dir=home, cwd=comp.parent).returncode == 0

    if has_module("torch"):
        r_bench = run_python(
            ["-m", "tmrl.tools.benchmark_continual_actor", "--runtime", "torch", "--device", "cpu", "--iterations", "5", "--encoder-channels", "8,16,16,16"],
            home_dir=home,
            timeout=90,
        )
        assert r_bench.returncode == 0, r_bench.stderr


def test_t4_09_cross_platform_documentation_and_sphinx_parity_audit(tmp_path: Path) -> None:
    """T4-09: Independent reviewer audits docs for links, script refs, CLI parity, zero author paths, zero AI leaks, and Sphinx."""
    doc_files = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "CONTRIBUTING.md",
        *sorted((REPO_ROOT / "readme").glob("*.md")),
        REPO_ROOT / "tutorial" / "README.md",
        EXP_DIR / "README.md",
    ]
    bad_path_pat = re.compile(r"(?:C:\\Users\\felix|/home/felix/|/mnt/c/Users/felix|D:\\Project\\tmrl|/mnt/d/Project/tmrl|D:\\miniconda3|C:\\Users\\Yann)", re.IGNORECASE)
    ai_pat = re.compile(r"(?:data-quality skill|data-validation skill|computer-use skill|\.agents/|codebase_overview/)", re.IGNORECASE)
    rel_script_pat = re.compile(r"(?<![A-Za-z0-9_./-])((?:scripts|experiments|tutorial)/[A-Za-z0-9_./-]+\.(?:py|sh|json|md))")

    for doc in doc_files:
        text = doc.read_text(encoding="utf-8")
        assert not bad_path_pat.search(text), f"Hardcoded path in {doc.relative_to(REPO_ROOT)}"
        assert not ai_pat.search(text), f"AI-workflow mention in {doc.relative_to(REPO_ROOT)}"
        for match in re.finditer(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)", text):
            target = match.group(1)
            if not target.startswith(("http://", "https://", "mailto:")):
                assert (doc.parent / target).resolve().exists(), f"Broken link in {doc.relative_to(REPO_ROOT)}: {target}"
        for match in rel_script_pat.finditer(text):
            rel = match.group(1)
            if "*" not in rel and "<" not in rel:
                assert (REPO_ROOT / rel).exists(), f"Dangling path in {doc.relative_to(REPO_ROOT)}: {rel}"

    r_help = run_python(["-m", "tmrl", "--help"], home_dir=tmp_path / "empty_home")
    assert r_help.returncode == 0
    valid_flags = set(re.findall(r"--[a-z0-9-]+", r_help.stdout))
    cli_rst = (REPO_ROOT / "docs" / "source" / "cli.rst").read_text(encoding="utf-8")
    for match in re.finditer(r"python3?\s+-m\s+tmrl\b([^\n`]*)", cli_rst):
        for flag in re.findall(r"--[a-z0-9-]+", match.group(1)):
            assert flag in valid_flags, f"docs/source/cli.rst cites invalid flag {flag}"

    if has_sphinx():
        res = subprocess.run(
            ["sphinx-build", "-W", "-b", "html", str(REPO_ROOT / "docs" / "source"), str(tmp_path / "html")],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert res.returncode == 0, res.stderr


def test_t4_10_clean_fresh_clone_and_git_provenance_hygiene() -> None:
    """T4-10: Maintainer verifies repo hygiene prior to release (clean git status, no AI clutter, provenance preserved)."""
    assert not (REPO_ROOT / ".agents").exists()
    assert not (REPO_ROOT / "codebase_overview").exists()

    st = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=15)
    assert st.returncode == 0
    dirty = [
        line
        for line in st.stdout.splitlines()
        if line.strip()
        and "tests/e2e/" not in line.replace("\\", "/")
        and (line[:2] == "??" or (len(line) > 1 and line[1] != " "))
    ]
    assert not dirty, "Dirty git status:\n" + "\n".join(dirty)

    ls_wandb = subprocess.run(["git", "ls-files", "tmrl/wandb/settings"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=15)
    assert ls_wandb.stdout.strip() == ""

    assert (REPO_ROOT / "fault_benchmark_results" / "confirmatory_manifest.csv").is_file()
    assert (REPO_ROOT / "fault_benchmark_results_ant" / "confirmatory_manifest.csv").is_file()

    attrs = subprocess.run(
        [
            "git",
            "check-attr",
            "text",
            "--",
            "experiments/sensor_adaptation_v1/run_study.py",
            "scripts/benchmark_fault_env.py",
            "scripts/confirmatory_logging.py",
            "scripts/train_transformer_source_recipe_v1.py",
            "scripts/train_reactive_source_recipe_v1.py",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert attrs.returncode == 0 and attrs.stdout.count("text: unset") == 5
