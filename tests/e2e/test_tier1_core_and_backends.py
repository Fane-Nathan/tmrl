"""Tier 1 opaque-box E2E tests for Features #1-#8 (core framework, CLI, config, backends)."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import pickle
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.e2e.conftest import (
    REPO_ROOT,
    build_portable_jax_bundle,
    has_jax_stack,
    has_module,
    make_tmrl_config,
    make_valid_map_manifest,
    populate_tmrl_data,
    run_python,
)

pytestmark = [pytest.mark.tier1]


# =============================================================================
# Feature #1: Packaging & Lazy Initialization (6 tests)
# =============================================================================


def test_f01_01_pyproject_and_setup_valid_without_network(tmp_path: Path) -> None:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    assert pyproject_path.is_file()
    parsed = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    assert "build-system" in parsed

    home = tmp_path / "home"
    res = run_python(
        [str(REPO_ROOT / "setup.py"), "--name", "--version"],
        home_dir=home,
        cwd=REPO_ROOT,
        block_network=True,
        netblock_dir=tmp_path / "netblock",
    )
    assert res.returncode == 0, res.stderr
    assert "tmrl" in res.stdout.lower()
    assert not (home / "TmrlData").exists(), "Packaging metadata query must not create ~/TmrlData"


def test_f01_02_packaging_does_not_require_jax_unconditionally() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project_deps = pyproject.get("project", {}).get("dependencies", [])
    for dep in project_deps:
        pkg = dep.split("[")[0].split(">")[0].split("<")[0].split("=")[0].strip().lower()
        assert pkg not in {"jax", "jaxlib", "flax", "optax"}, f"Unexpected core dep in pyproject.toml: {dep}"

    setup_tree = ast.parse((REPO_ROOT / "setup.py").read_text(encoding="utf-8"))
    for node in ast.walk(setup_tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "install_req":
                    if isinstance(node.value, ast.List):
                        items = [
                            elt.value.lower()
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        ]
                        for jax_pkg in ("jax", "jaxlib", "flax", "optax"):
                            assert jax_pkg not in items, f"{jax_pkg} must not be in unconditional install_req"


def test_f01_03_import_tmrl_headless_no_tmrldata_no_network(tmp_path: Path) -> None:
    home = tmp_path / "empty_home"
    code = (
        "import io, sys, tmrl; "
        "assert isinstance(sys.stdout, io.TextIOBase) or hasattr(sys.stdout, 'buffer'); "
        "assert callable(tmrl.get_environment); "
        "print('IMPORT_OK')"
    )
    res = run_python(
        ["-c", code],
        home_dir=home,
        block_network=True,
        netblock_dir=tmp_path / "netblock",
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "IMPORT_OK" in res.stdout
    assert not (home / "TmrlData").exists(), "import tmrl must not create ~/TmrlData"


def test_f01_04_import_tmrl_without_pywin32_dxcam_vgamepad(tmp_path: Path) -> None:
    home = tmp_path / "empty_home"
    code = """
import sys
class _BlockWin:
    BLOCKED = {"win32gui", "win32ui", "win32con", "pywintypes", "dxcam", "vgamepad"}
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.BLOCKED:
            raise ModuleNotFoundError(f"Blocked optional Windows module: {fullname}")
        return None
sys.meta_path.insert(0, _BlockWin())
import tmrl
assert callable(tmrl.get_environment)
print("NO_WIN32_OK")
"""
    res = run_python(
        ["-c", code],
        home_dir=home,
        block_network=True,
        netblock_dir=tmp_path / "netblock",
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "NO_WIN32_OK" in res.stdout


def test_f01_05_init_tmrl_import_has_no_network_side_effect(tmp_path: Path) -> None:
    home = tmp_path / "empty_home"
    res = run_python(
        ["-c", "import tmrl.tools.init_package.init_tmrl as it; assert callable(it.init_tmrl_data); print('OK')"],
        home_dir=home,
        block_network=True,
        netblock_dir=tmp_path / "netblock",
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "OK" in res.stdout
    assert not (home / "TmrlData").exists()


def test_f01_06_init_pywin32_does_not_hijack_stdout_on_import(tmp_path: Path) -> None:
    home = tmp_path / "empty_home"
    code = """
import sys
orig_out, orig_err = sys.stdout, sys.stderr
import tmrl.tools.init_package.init_pywin32
assert sys.stdout is orig_out, "init_pywin32 mutated sys.stdout on import"
assert sys.stderr is orig_err, "init_pywin32 mutated sys.stderr on import"
print("STREAMS_OK")
"""
    res = run_python(["-c", code], home_dir=home)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "STREAMS_OK" in res.stdout


# =============================================================================
# Feature #2: CLI Entry Point (python -m tmrl) (6 tests)
# =============================================================================


def test_f02_01_cli_help_headless_without_tmrldata(tmp_path: Path) -> None:
    home = tmp_path / "empty_home"
    res = run_python(
        ["-m", "tmrl", "--help"],
        home_dir=home,
        block_network=True,
        netblock_dir=tmp_path / "netblock",
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    for flag in (
        "--install",
        "--server",
        "--trainer",
        "--worker",
        "--expert",
        "--test",
        "--benchmark",
        "--record-reward",
        "--use-keyboard",
        "--check-environment",
        "--wandb",
        "-d",
        "--config",
    ):
        assert flag in res.stdout, f"Missing CLI flag {flag} in --help output"
    assert not (home / "TmrlData").exists()


def test_f02_02_cli_help_without_pywin32_or_dxcam(tmp_path: Path) -> None:
    home = tmp_path / "empty_home"
    block_dir = tmp_path / "block_win"
    block_dir.mkdir()
    (block_dir / "sitecustomize.py").write_text(
        """import sys
class _Block:
    BLOCKED = {"win32gui", "win32ui", "win32con", "pywintypes", "dxcam", "vgamepad"}
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.BLOCKED:
            raise ModuleNotFoundError(f"Blocked: {fullname}")
        return None
sys.meta_path.insert(0, _Block())
""",
        encoding="utf-8",
    )
    res = run_python(
        ["-m", "tmrl", "--help"],
        home_dir=home,
        extra_env={"PYTHONPATH": str(block_dir)},
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "--server" in res.stdout


def test_f02_03_cli_no_arguments_exits_nonzero_cleanly(tmp_path: Path) -> None:
    home = tmp_path / "home"
    populate_tmrl_data(home)
    res = run_python(["-m", "tmrl"], home_dir=home)
    assert res.returncode != 0
    assert "Traceback" not in res.stderr or "ArgumentTypeError" in res.stderr or "usage:" in res.stderr.lower()


def test_f02_04_cli_unknown_flag_exits_2(tmp_path: Path) -> None:
    home = tmp_path / "empty_home"
    res = run_python(["-m", "tmrl", "--nonexistent-cli-flag"], home_dir=home)
    assert res.returncode == 2
    assert "unrecognized arguments" in res.stderr.lower()


def test_f02_05_cli_malformed_json_config_exits_2(tmp_path: Path) -> None:
    home = tmp_path / "home"
    populate_tmrl_data(home)
    res = run_python(["-m", "tmrl", "--server", "-d", "{bad_json"], home_dir=home)
    assert res.returncode == 2
    assert "invalid loads value" in res.stderr.lower() or "json" in res.stderr.lower()


def test_f02_06_cli_install_with_populated_tmrldata_offline(tmp_path: Path) -> None:
    home = tmp_path / "home"
    populate_tmrl_data(home)
    res = run_python(
        ["-m", "tmrl", "--install"],
        home_dir=home,
        block_network=True,
        netblock_dir=tmp_path / "netblock",
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert (home / "TmrlData" / "config" / "config.json").is_file()


# =============================================================================
# Feature #3: Configuration & Factory Wiring (6 tests)
# =============================================================================


def test_f03_01_config_constants_importable_without_tmrldata(tmp_path: Path) -> None:
    home = tmp_path / "empty_home"
    res = run_python(
        ["-c", "import tmrl.config.config_constants as cfg; print(cfg.__compatibility__)"],
        home_dir=home,
        block_network=True,
        netblock_dir=tmp_path / "netblock",
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "0.6.0" in res.stdout


def test_f03_02_config_constants_reads_temp_config_json(tmp_path: Path) -> None:
    home = tmp_path / "home"
    cfg_dict = make_tmrl_config(run_name="custom_e2e_cfg", img_width=80, img_height=48, img_hist_len=3)
    populate_tmrl_data(home, config_dict=cfg_dict)
    code = """
import tmrl.config.config_constants as cfg
assert cfg.RUN_NAME == "custom_e2e_cfg"
assert cfg.IMG_WIDTH == 80
assert cfg.IMG_HEIGHT == 48
assert cfg.IMG_HIST_LEN == 3
assert cfg.PASSWORD == "e2e-password"
print("CFG_OK")
"""
    res = run_python(["-c", code], home_dir=home)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "CFG_OK" in res.stdout


@pytest.mark.requires_torch
def test_f03_03_config_objects_sac_image_defaults_foundation_false(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("rtgym")
    home = tmp_path / "home"
    cfg_dict = make_tmrl_config(algorithm="SAC", interface="TM20FULL", grayscale=True, use_foundation_actor=None)
    assert "USE_FOUNDATION_ACTOR" not in cfg_dict["ALG"]
    populate_tmrl_data(home, config_dict=cfg_dict)
    code = """
import tmrl.config.config_objects as cfg_obj
policy_cls = getattr(cfg_obj.POLICY, "func", cfg_obj.POLICY)
assert policy_cls.__name__ == "SquashedGaussianVanillaCNNActor", f"Got {policy_cls}"
print("DEFAULT_NO_FOUNDATION_OK")
"""
    res = run_python(["-c", code], home_dir=home)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "DEFAULT_NO_FOUNDATION_OK" in res.stdout


@pytest.mark.requires_torch
def test_f03_04_config_objects_redqsac_lidar_wiring(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("rtgym")
    home = tmp_path / "home"
    cfg_dict = make_tmrl_config(algorithm="REDQSAC", interface="TM20LIDAR", grayscale=False)
    populate_tmrl_data(home, config_dict=cfg_dict)
    code = """
import tmrl.config.config_objects as cfg_obj
assert cfg_obj.ALG_NAME == "REDQSAC"
policy_cls = getattr(cfg_obj.POLICY, "func", cfg_obj.POLICY)
assert policy_cls.__name__ == "SquashedGaussianMLPActor"
assert callable(cfg_obj.ENV_CLS)
assert callable(cfg_obj.TRAINER)
print("LIDAR_REDQ_OK")
"""
    res = run_python(["-c", code], home_dir=home)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "LIDAR_REDQ_OK" in res.stdout


@pytest.mark.requires_torch
def test_f03_05_config_objects_torch_dreamer_wiring(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("rtgym")
    home = tmp_path / "home"
    cfg_dict = make_tmrl_config(
        algorithm="DREAMER",
        interface="TM20FULL",
        grayscale=True,
        use_foundation_actor=False,
        extra_alg={"DREAMER": {"FOUNDATION_ONLY": False, "FREEZE_FOUNDATION": False}},
    )
    populate_tmrl_data(home, config_dict=cfg_dict)
    code = """
import tmrl.config.config_objects as cfg_obj
assert cfg_obj.IS_TORCH_DREAMER is True
policy_cls = getattr(cfg_obj.POLICY, "func", cfg_obj.POLICY)
assert policy_cls.__name__ == "TorchDreamerActor"
print("DREAMER_WIRING_OK")
"""
    res = run_python(["-c", code], home_dir=home)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "DREAMER_WIRING_OK" in res.stdout


def test_f03_06_config_objects_rejects_unknown_algorithm(tmp_path: Path) -> None:
    home = tmp_path / "home"
    cfg_dict = make_tmrl_config(algorithm="NONEXISTENT_ALGO")
    populate_tmrl_data(home, config_dict=cfg_dict)
    res = run_python(["-c", "import tmrl.config.config_objects as o; _ = o.POLICY"], home_dir=home)
    assert res.returncode != 0
    assert "NONEXISTENT_ALGO" in res.stderr or "AssertionError" in res.stderr or "ValueError" in res.stderr


# =============================================================================
# Feature #4: Core Framework & Networking (6 tests)
# =============================================================================


def test_f04_01_core_modules_import_without_tmrldata(tmp_path: Path) -> None:
    home = tmp_path / "empty_home"
    code = """
import tmrl.core.actor
import tmrl.core.envs
import tmrl.core.memory
import tmrl.core.training
import tmrl.core.training_offline
import tmrl.core.networking
import tmrl.core.util
print("CORE_IMPORTS_OK")
"""
    res = run_python(["-c", code], home_dir=home, block_network=True, netblock_dir=tmp_path / "netblock")
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "CORE_IMPORTS_OK" in res.stdout


def test_f04_02_generic_gym_env_pendulum_reset_and_step() -> None:
    gym = pytest.importorskip("gymnasium")
    _ = gym
    from tmrl.core.envs import GenericGymEnv

    env = GenericGymEnv(id="Pendulum-v1", gym_kwargs={"render_mode": None})
    try:
        obs, info = env.reset(seed=7)
        assert env.observation_space.contains(obs)
        assert isinstance(info, dict)
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        assert env.observation_space.contains(next_obs)
        assert np.isfinite(float(reward))
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        assert isinstance(step_info, dict)
    finally:
        env.close()


@pytest.mark.requires_torch
def test_f04_03_generic_torch_memory_append_and_sample(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from tmrl.core.networking import Buffer
    from tmrl.custom.torch.custom_memories import GenericTorchMemory

    mem = GenericTorchMemory(
        device="cpu",
        memory_size=64,
        batch_size=4,
        dataset_path=str(tmp_path / "nonexistent_dataset"),
        crc_debug=False,
    )
    assert len(mem) == 0
    buf = Buffer(maxlen=32)
    for i in range(10):
        obs = np.full((3,), float(i), dtype=np.float32)
        act = np.array([0.1], dtype=np.float32)
        buf.append_sample((act, obs, -1.0, i == 9, False, {}))
    mem.append(buf)
    assert len(mem) >= 8
    last_obs, act, rew, new_obs, term, trunc = mem.sample()
    assert isinstance(last_obs, torch.Tensor) and last_obs.shape == (4, 3)
    assert act.shape == (4, 1)
    assert rew.shape == (4,)
    assert new_obs.shape == (4, 3)
    assert term.shape == (4,) and trunc.shape == (4,)


@pytest.mark.requires_torch
def test_f04_04_torch_training_offline_accepts_space_tuple() -> None:
    pytest.importorskip("torch")
    gym = pytest.importorskip("gymnasium")
    from tmrl.core.torch.training_offline import TorchTrainingOffline
    from tmrl.core.util import partial
    from tmrl.custom.torch.custom_algorithms import SpinupSACAgent
    from tmrl.custom.torch.custom_memories import GenericTorchMemory
    from tmrl.custom.torch.custom_models import MLPActorCritic

    obs_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    act_space = gym.spaces.Box(-2.0, 2.0, shape=(1,), dtype=np.float32)
    trainer = TorchTrainingOffline(
        env_cls=(obs_space, act_space),
        memory_cls=partial(GenericTorchMemory, memory_size=32, batch_size=4),
        training_agent_cls=partial(SpinupSACAgent, model_cls=MLPActorCritic, device="cpu"),
        epochs=1,
        rounds=1,
        steps=1,
        device="cpu",
    )
    assert trainer.device == "cpu"
    assert trainer.agent.get_actor() is not None


@pytest.mark.requires_torch
@pytest.mark.requires_tlspyo
def test_f04_05_rollout_worker_standalone_pendulum(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("tlspyo")
    from tmrl.core.envs import GenericGymEnv
    from tmrl.core.networking import RolloutWorker
    from tmrl.core.util import partial
    from tmrl.custom.torch.custom_models import SquashedGaussianMLPActor

    rw = RolloutWorker(
        env_cls=partial(GenericGymEnv, id="Pendulum-v1", gym_kwargs={"render_mode": None}),
        actor_module_cls=partial(SquashedGaussianMLPActor),
        sample_compressor=None,
        device="cpu",
        max_samples_per_episode=5,
        model_path=str(tmp_path / "worker.tmod"),
        standalone=True,
    )
    rw.run_episodes(max_samples_per_episode=5, nb_episodes=2)
    assert rw.buffer.stat_test_steps == 5
    assert np.isfinite(rw.buffer.stat_test_return)


@pytest.mark.requires_torch
def test_f04_06_core_training_offline_device_and_jax_jit_guard() -> None:
    pytest.importorskip("torch")
    from tmrl.core.torch.training_offline import TorchTrainingOffline

    assert TorchTrainingOffline.__name__ == "TorchTrainingOffline"
    if has_jax_stack():
        gym = pytest.importorskip("gymnasium")
        from tmrl.core.jax.training_offline import NNXTrainingOffline
        from tmrl.core.util import partial

        obs_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        act_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        with pytest.raises(ValueError, match="jit_sampling"):
            NNXTrainingOffline(
                env_cls=(obs_space, act_space),
                memory_cls=partial(lambda **kw: None),
                training_agent_cls=partial(lambda **kw: None),
                jit_sampling=True,
                jit_training=False,
            )


# =============================================================================
# Feature #5: Continual JAX & Diagnostic Tools (6 tests)
# =============================================================================


def test_f05_01_prepare_continual_jax_config_cli(tmp_path: Path) -> None:
    src = tmp_path / "source.json"
    out = tmp_path / "out.json"
    src.write_text(json.dumps(make_tmrl_config(grayscale=True, wandb_key="secret-token")), encoding="utf-8")

    res = run_python(
        [
            "-m",
            "tmrl.tools.prepare_continual_jax_config",
            "--source",
            str(src),
            "--output",
            str(out),
            "--run-name",
            "e2e_jax_run",
            "--scrub-wandb-key",
            "--trainer-server-ip",
            "192.168.50.10",
        ],
        home_dir=tmp_path / "home",
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    summary = json.loads(res.stdout)
    assert summary["algorithm"] == "CONTINUAL_DREAMER_JAX"
    assert summary["azr_enabled"] is False
    assert summary["experts_enabled"] is False
    assert summary["wandb_key_scrubbed"] is True
    assert summary["trainer_server_ip"] == "192.168.50.10"

    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["WANDB_KEY"] == ""
    assert written["LOCALHOST_TRAINER"] is False
    assert written["PUBLIC_IP_SERVER"] == "192.168.50.10"


def test_f05_02_validate_continual_map_manifest_valid(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(make_valid_map_manifest(frozen=True)), encoding="utf-8")
    res = run_python(
        ["-m", "tmrl.tools.validate_continual_map_manifest", str(manifest_path), "--require-frozen"],
        home_dir=tmp_path / "home",
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    payload = json.loads(res.stdout)
    assert payload["valid"] is True
    assert payload["errors"] == []
    assert len(payload["canonical_sha256"]) == 64


def test_f05_03_freeze_continual_baseline_sha256_manifest(tmp_path: Path) -> None:
    f1 = tmp_path / "ckpt.bin"
    f2 = tmp_path / "actor.bin"
    f1.write_bytes(b"checkpoint-bytes-v1")
    f2.write_bytes(b"actor-bytes-v1")
    out_dir = tmp_path / "frozen"

    res = run_python(
        [
            "-m",
            "tmrl.tools.freeze_continual_baseline",
            "--source",
            str(f1),
            "--source",
            str(f2),
            "--output-dir",
            str(out_dir),
            "--repo-root",
            str(REPO_ROOT),
        ],
        home_dir=tmp_path / "home",
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    manifest = json.loads((out_dir / "baseline_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert len(manifest["artifacts"]) == 2
    for item, original in zip(manifest["artifacts"], (f1, f2)):
        copied = out_dir / item["snapshot"]
        assert copied.is_file()
        assert item["sha256"] == hashlib.sha256(original.read_bytes()).hexdigest()
        assert hashlib.sha256(copied.read_bytes()).hexdigest() == item["sha256"]


@pytest.mark.requires_torch
def test_f05_04_benchmark_continual_actor_torch_cpu(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    out_json = tmp_path / "bench.json"
    res = run_python(
        [
            "-m",
            "tmrl.tools.benchmark_continual_actor",
            "--runtime",
            "torch",
            "--device",
            "cpu",
            "--iterations",
            "5",
            "--encoder-channels",
            "8,16,16,16",
            "--output",
            str(out_json),
        ],
        home_dir=tmp_path / "home",
        timeout=90,
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["runtime"] == "torch"
    assert data["passed"] is True
    assert data["iterations"] == 5
    for metric in ("mean", "p50", "p95", "p99", "max"):
        assert np.isfinite(data["latency_ms"][metric]) and data["latency_ms"][metric] > 0.0


def test_f05_05_tools_help_without_tmrldata(tmp_path: Path) -> None:
    home = tmp_path / "empty_home"
    modules = [
        "tmrl.tools.prepare_continual_jax_config",
        "tmrl.tools.validate_continual_map_manifest",
        "tmrl.tools.freeze_continual_baseline",
        "tmrl.tools.continual_preflight",
        "tmrl.tools.validate_continual_trainer_stack",
    ]
    if has_module("torch"):
        modules.append("tmrl.tools.benchmark_continual_actor")
    for mod in modules:
        res = run_python(["-m", mod, "--help"], home_dir=home)
        assert res.returncode == 0, f"{mod} --help failed: {res.stderr}"
        assert "usage:" in res.stdout.lower()


@pytest.mark.requires_jax
def test_f05_06_continual_preflight_cli(tmp_path: Path) -> None:
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    out_json = tmp_path / "preflight.json"
    res = run_python(
        ["-m", "tmrl.tools.continual_preflight", "--output", str(out_json)],
        home_dir=tmp_path / "home",
        timeout=90,
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["nnx_train_step"]["ok"] is True


# =============================================================================
# Feature #6: TrackMania Interfaces, Capture & Control (6 tests)
# =============================================================================


def test_f06_01_tm_modules_import_headless_without_pywin32(tmp_path: Path) -> None:
    home = tmp_path / "empty_home"
    code = """
import sys
class _BlockCapture:
    BLOCKED = {"win32gui", "win32ui", "win32con", "pywintypes", "dxcam", "vgamepad", "mss"}
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.BLOCKED:
            raise ModuleNotFoundError(f"Blocked: {fullname}")
        return None
sys.meta_path.insert(0, _BlockCapture())
import tmrl.custom.tm.tm_gym_interfaces
import tmrl.custom.tm.tm_preprocessors
import tmrl.custom.tm.utils.window
import tmrl.custom.tm.utils.control_mouse
import tmrl.custom.tm.utils.control_gamepad
import tmrl.custom.tm.utils.compute_reward
import tmrl.custom.tm.utils.trajectory_assist
import tmrl.custom.tm.utils.camera_preprocessing
print("TM_HEADLESS_IMPORTS_OK")
"""
    res = run_python(["-c", code], home_dir=home)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "TM_HEADLESS_IMPORTS_OK" in res.stdout


def test_f06_02_tm_interfaces_spaces_and_default_action(tmp_path: Path) -> None:
    home = tmp_path / "home"
    populate_tmrl_data(home)
    code = """
import numpy as np
from tmrl.custom.tm.tm_gym_interfaces import (
    TM2020Interface,
    TM2020InterfaceLidar,
    TM2020InterfaceLidarProgress,
)
for cls, kwargs in (
    (TM2020Interface, dict(img_hist_len=4, gamepad=False, grayscale=True, resize_to=(64, 64))),
    (TM2020InterfaceLidar, dict(img_hist_len=4, gamepad=False)),
    (TM2020InterfaceLidarProgress, dict(img_hist_len=4, gamepad=False)),
):
    iface = cls(**kwargs)
    obs_space = iface.get_observation_space()
    act_space = iface.get_action_space()
    default_act = iface.get_default_action()
    assert len(obs_space) >= 2
    assert act_space.shape == (3,)
    assert default_act.shape == (3,)
    assert act_space.contains(default_act)
print("SPACES_OK")
"""
    res = run_python(["-c", code], home_dir=home)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "SPACES_OK" in res.stdout


def test_f06_03_reward_function_trajectory_and_universal_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tmrl.custom.tm.utils.compute_reward import RewardFunction

    traj = np.column_stack([np.arange(0, 30, dtype=np.float32), np.zeros(30, dtype=np.float32), np.zeros(30, dtype=np.float32)])
    rew_path = tmp_path / "reward.pkl"
    with rew_path.open("wb") as handle:
        pickle.dump(traj, handle)

    monkeypatch.setenv("TMRL_UNIVERSAL_REWARD", "0")
    rf = RewardFunction(str(rew_path), nb_obs_forward=5, nb_obs_backward=5, nb_zero_rew_before_failure=3, min_nb_steps_before_failure=2)
    r1, term1 = rf.compute_reward(np.array([4.0, 0.0, 0.0], dtype=np.float32))
    assert r1 > 0.0 and term1 is False
    # Stall past min_nb_steps_before_failure + nb_zero_rew_before_failure
    terminated_seen = False
    for _ in range(10):
        _, term = rf.compute_reward(np.array([4.0, 0.0, 0.0], dtype=np.float32))
        terminated_seen = terminated_seen or term
    assert terminated_seen
    rf.reset()
    assert rf.cur_idx == 0 and rf.step_counter == 0 and rf.failure_counter == 0

    monkeypatch.setenv("TMRL_UNIVERSAL_REWARD", "1")
    rf_univ = RewardFunction(str(rew_path))
    r_init, t_init = rf_univ.compute_reward(np.array([10.0, 0.0, 0.0], dtype=np.float32), speed=50.0)
    assert t_init is False and r_init == 0.0
    r_fwd, t_fwd = rf_univ.compute_reward(np.array([15.0, 0.0, 0.0], dtype=np.float32), speed=50.0)
    assert r_fwd > 0.0 and t_fwd is False


def test_f06_04_preprocessors_and_camera_preprocessing() -> None:
    from tmrl.custom.tm.tm_preprocessors import (
        obs_preprocessor_tm_act_in_obs,
        obs_preprocessor_tm_lidar_act_in_obs,
        obs_preprocessor_tm_lidar_progress_act_in_obs,
    )
    from tmrl.custom.tm.utils.camera_preprocessing import preprocess_camera_frame

    img_obs = (
        np.array([100.0], np.float32),
        np.array([3.0], np.float32),
        np.array([5000.0], np.float32),
        np.full((4, 64, 64), 128, dtype=np.uint8),
        np.zeros((3,), np.float32),
    )
    out_img = obs_preprocessor_tm_act_in_obs(img_obs)
    assert np.isclose(out_img[0][0], 0.1)
    assert np.isclose(out_img[1][0], 0.3)
    assert np.isclose(out_img[2][0], 0.5)
    assert out_img[3].dtype == np.float32 and np.isclose(out_img[3][0, 0, 0], 0.5)

    lidar_obs = (np.array([50.0], np.float32), np.ones((4, 19), np.float32), np.zeros((3,), np.float32))
    out_lidar = obs_preprocessor_tm_lidar_act_in_obs(lidar_obs)
    assert out_lidar[1].shape == (76,)

    prog_obs = (np.array([50.0], np.float32), np.array([0.2], np.float32), np.ones((4, 19), np.float32))
    out_prog = obs_preprocessor_tm_lidar_progress_act_in_obs(prog_obs)
    assert out_prog[2].shape == (76,)

    bgra = np.full((120, 160, 4), 200, dtype=np.uint8)
    gray = preprocess_camera_frame(bgra, resize_to=(64, 48), grayscale=True)
    rgb = preprocess_camera_frame(bgra[:, :, :3], resize_to=(64, 48), grayscale=False)
    assert gray.shape == (48, 64) and gray.dtype == np.uint8
    assert rgb.shape == (48, 64, 3) and rgb.dtype == np.uint8


def test_f06_05_signed_trigger_controls_and_control_mouse_time_import(tmp_path: Path) -> None:
    from tmrl.custom.tm.utils.control_gamepad import signed_trigger_controls

    assert signed_trigger_controls([-1.0, 0.0, 0.25]) == [0.0, 0.5, 0.25]
    with pytest.raises(ValueError):
        signed_trigger_controls([1.5, 0.0, 0.0])

    # Stub pyautogui in a subprocess so control_mouse runs safely without touching user mouse
    code = """
import sys, types
stub = types.ModuleType("pyautogui")
stub.moveTo = lambda *a, **k: None
stub.click = lambda *a, **k: None
stub.mouseUp = lambda *a, **k: None
stub.mouseDown = lambda *a, **k: None
stub.press = lambda *a, **k: None
stub.keyDown = lambda *a, **k: None
stub.keyUp = lambda *a, **k: None
sys.modules["pyautogui"] = stub
from tmrl.custom.tm.utils import control_mouse
control_mouse.mouse_close_finish_pop_up_tm20(small_window=True)
print("MOUSE_OK")
"""
    res = run_python(["-c", code], home_dir=tmp_path / "home")
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "MOUSE_OK" in res.stdout


def test_f06_06_demonstration_trajectory_assist_replay_and_pursuit(tmp_path: Path) -> None:
    from tmrl.custom.tm.utils.trajectory_assist import DemonstrationTrajectoryAssist

    t = np.linspace(0.0, 50.0, 40, dtype=np.float32)
    positions = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    actions = np.tile(np.array([[0.8, -1.0, 0.1]], dtype=np.float32), (40, 1))
    speeds = np.full((40,), 15.0, dtype=np.float32)
    traj_path = tmp_path / "demo.npz"
    np.savez(traj_path, positions=positions, actions=actions, speeds_mps=speeds)

    for mode in ("replay", "pursuit"):
        assist = DemonstrationTrajectoryAssist(
            {
                "ENABLED": True,
                "TRAJECTORY_PATH": str(traj_path),
                "STEERING_MODE": mode,
                "CORRECTION_ENABLED": True,
            }
        )
        telemetry = [10.0, 0.0, 5.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 3000.0]
        act, info = assist.action(telemetry, np.zeros(3, dtype=np.float32))
        assert act.shape == (3,)
        assert np.all(np.isfinite(act))
        assert np.all(act >= -1.0) and np.all(act <= 1.0)
        assert "deviation_m" in info and "reference_index" in info


# =============================================================================
# Feature #7: PyTorch Algorithms, Models & Extensions (6 tests)
# =============================================================================


@pytest.mark.requires_torch
def test_f07_01_torch_modules_import_without_tmrldata(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    code = """
import tmrl.custom.torch.custom_models
import tmrl.custom.torch.custom_algorithms
import tmrl.custom.torch.dreamer
import tmrl.custom.torch.foundation_actor
import tmrl.custom.torch.jax_dreamer_actor
import tmrl.custom.torch.world_model
import tmrl.custom.torch.car_brain
print("TORCH_IMPORTS_OK")
"""
    res = run_python(["-c", code], home_dir=tmp_path / "empty_home")
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "TORCH_IMPORTS_OK" in res.stdout


@pytest.mark.requires_torch
def test_f07_02_torch_mlp_and_cnn_actors_act_and_save_load(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    gym = pytest.importorskip("gymnasium")
    from tmrl.custom.torch.custom_models import (
        MLPActorCritic,
        REDQMLPActorCritic,
        SquashedGaussianMLPActor,
        SquashedGaussianVanillaCNNActor,
        VanillaCNNActorCritic,
    )

    vec_obs = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
    vec_act = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    mlp_ac = MLPActorCritic(vec_obs, vec_act)
    redq_ac = REDQMLPActorCritic(vec_obs, vec_act, n=3)
    assert len(redq_ac.qs) == 3
    sample_vec = np.zeros((4,), dtype=np.float32)
    a1 = mlp_ac.actor.act_(sample_vec, test=True)
    assert vec_act.contains(a1)

    ckpt_mlp = tmp_path / "mlp.tmod"
    mlp_ac.actor.save(ckpt_mlp)
    reloaded = SquashedGaussianMLPActor(vec_obs, vec_act)
    reloaded.load(ckpt_mlp, device="cpu")
    np.testing.assert_allclose(a1, reloaded.act_(sample_vec, test=True), atol=1e-6)

    img_obs = gym.spaces.Tuple(
        (
            gym.spaces.Box(0.0, 1000.0, (1,), np.float32),
            gym.spaces.Box(0.0, 6.0, (1,), np.float32),
            gym.spaces.Box(0.0, 10000.0, (1,), np.float32),
            gym.spaces.Box(0.0, 1.0, (4, 64, 64), np.float32),
            gym.spaces.Box(-1.0, 1.0, (3,), np.float32),
            gym.spaces.Box(-1.0, 1.0, (3,), np.float32),
        )
    )
    img_act = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    cnn_ac = VanillaCNNActorCritic(img_obs, img_act)
    obs_inst = tuple(np.zeros(s.shape, dtype=s.dtype) for s in img_obs)
    a_cnn = cnn_ac.actor.act_(obs_inst, test=True)
    assert img_act.contains(a_cnn)

    ckpt_cnn = tmp_path / "cnn.tmod"
    cnn_ac.actor.save(ckpt_cnn)
    reloaded_cnn = SquashedGaussianVanillaCNNActor(img_obs, img_act)
    reloaded_cnn.load(ckpt_cnn, device="cpu")
    np.testing.assert_allclose(a_cnn, reloaded_cnn.act_(obs_inst, test=True), atol=1e-6)


@pytest.mark.requires_torch
def test_f07_03_torch_dreamer_and_jax_mirror_actor_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    gym = pytest.importorskip("gymnasium")
    from tmrl.custom.torch.dreamer import TorchDreamerActor
    from tmrl.custom.torch.jax_dreamer_actor import TorchJAXDreamerActor

    obs_space = gym.spaces.Tuple(
        (
            gym.spaces.Box(0.0, 1.0, (1,), np.float32),
            gym.spaces.Box(0.0, 1.0, (1,), np.float32),
            gym.spaces.Box(0.0, 1.0, (1,), np.float32),
            gym.spaces.Box(0.0, 1.0, (4, 64, 64), np.float32),
            gym.spaces.Box(-1.0, 1.0, (3,), np.float32),
            gym.spaces.Box(-1.0, 1.0, (3,), np.float32),
        )
    )
    act_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)
    obs = tuple(np.zeros(s.shape, dtype=s.dtype) for s in obs_space)

    dreamer_actor = TorchDreamerActor(
        obs_space,
        act_space,
        latent_dim=32,
        hidden_dim=64,
        policy_hidden_dim=64,
        img_channels=4,
        img_height=64,
        img_width=64,
        use_foundation_encoder=False,
        foundation_only=False,
    )
    act_d = dreamer_actor.act_(obs, test=True)
    assert act_space.contains(act_d)
    dreamer_ckpt = tmp_path / "dreamer.tmod"
    dreamer_actor.save(dreamer_ckpt)
    reloaded_d = TorchDreamerActor(
        obs_space,
        act_space,
        latent_dim=32,
        hidden_dim=64,
        policy_hidden_dim=64,
        img_channels=4,
        img_height=64,
        img_width=64,
        use_foundation_encoder=False,
        foundation_only=False,
    )
    reloaded_d.load(dreamer_ckpt, device="cpu")
    reloaded_d.reset()
    np.testing.assert_allclose(act_d, reloaded_d.act_(obs, test=True), atol=1e-6)

    jax_mirror = TorchJAXDreamerActor(
        obs_space,
        act_space,
        latent_dim=32,
        hidden_dim=64,
        policy_hidden_dim=64,
        encoder_channels=(8, 16, 16, 16),
        img_channels=4,
        img_height=64,
        img_width=64,
    )
    act_m = jax_mirror.act_(obs, test=True)
    assert act_space.contains(act_m)
    bundle_path = tmp_path / "mirror.tmod"
    bundle_path.write_bytes(pickle.dumps(build_portable_jax_bundle(jax_mirror)))
    reloaded = TorchJAXDreamerActor(
        obs_space,
        act_space,
        latent_dim=32,
        hidden_dim=64,
        policy_hidden_dim=64,
        encoder_channels=(8, 16, 16, 16),
        img_channels=4,
        img_height=64,
        img_width=64,
    )
    reloaded.load(bundle_path, device="cpu")
    assert reloaded.last_load_succeeded is True
    np.testing.assert_allclose(act_m, reloaded.act_(obs, test=True), atol=1e-6)


@pytest.mark.requires_torch
def test_f07_04_spinup_and_redq_optimizer_betas_and_train_step() -> None:
    torch = pytest.importorskip("torch")
    gym = pytest.importorskip("gymnasium")
    from tmrl.custom.torch.custom_algorithms import REDQSACAgent, SpinupSACAgent
    from tmrl.custom.torch.custom_models import MLPActorCritic, REDQMLPActorCritic

    obs_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)
    act_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
    batch = (
        torch.zeros((6, 4), dtype=torch.float32),
        torch.zeros((6, 2), dtype=torch.float32),
        torch.ones((6,), dtype=torch.float32),
        torch.zeros((6, 4), dtype=torch.float32),
        torch.zeros((6,), dtype=torch.float32),
        torch.zeros((6,), dtype=torch.float32),
    )

    for agent_cls, model_cls, extra in (
        (SpinupSACAgent, MLPActorCritic, {}),
        (REDQSACAgent, REDQMLPActorCritic, {"m": 2, "q_updates_per_policy_update": 1}),
    ):
        agent = agent_cls(
            observation_space=obs_space,
            action_space=act_space,
            device="cpu",
            model_cls=model_cls,
            optimizer_actor="adamw",
            optimizer_critic="adam",
            betas_actor=(0.8, 0.99),
            betas_critic=(0.85, 0.995),
            l2_actor=1e-4,
            **extra,
        )
        assert isinstance(agent.pi_optimizer, torch.optim.AdamW)
        assert isinstance(agent.q_optimizer, torch.optim.Adam)
        assert agent.pi_optimizer.param_groups[0]["betas"] == (0.8, 0.99)
        assert agent.q_optimizer.param_groups[0]["betas"] == (0.85, 0.995)
        metrics = agent.train(batch)
        assert isinstance(metrics, dict) and metrics
        assert all(np.isfinite(float(v)) for v in metrics.values())


@pytest.mark.requires_torch
def test_f07_05_dream_sac_and_torch_dreamer_train_steps() -> None:
    torch = pytest.importorskip("torch")
    gym = pytest.importorskip("gymnasium")
    from tmrl.custom.torch.custom_algorithms import DreamSACAgent
    from tmrl.custom.torch.custom_models import VanillaCNNActorCritic
    from tmrl.custom.torch.dreamer import TorchDreamerAgent

    obs_space = gym.spaces.Tuple(
        (
            gym.spaces.Box(0.0, 100.0, (1,), np.float32),
            gym.spaces.Box(0.0, 6.0, (1,), np.float32),
            gym.spaces.Box(0.0, 10000.0, (1,), np.float32),
            gym.spaces.Box(0, 255, (4, 64, 64), np.uint8),
            gym.spaces.Box(-1.0, 1.0, (3,), np.float32),
            gym.spaces.Box(-1.0, 1.0, (3,), np.float32),
        )
    )
    act_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)

    dream_sac = DreamSACAgent(
        observation_space=obs_space,
        action_space=act_space,
        model_cls=VanillaCNNActorCritic,
        device="cpu",
    )
    b = 4
    obs_b = (
        torch.zeros((b, 1)),
        torch.ones((b, 1)),
        torch.ones((b, 1)) * 1000.0,
        torch.zeros((b, 4, 64, 64), dtype=torch.uint8),
        torch.zeros((b, 3)),
        torch.zeros((b, 3)),
    )
    metrics = dream_sac.train((obs_b, torch.zeros((b, 3)), torch.ones((b, 1)), obs_b, torch.zeros((b, 1)), None))
    assert all(np.isfinite(float(v)) for v in metrics.values())

    dreamer = TorchDreamerAgent(
        observation_space=obs_space,
        action_space=act_space,
        device="cpu",
        latent_dim=32,
        hidden_dim=64,
        policy_hidden_dim=64,
        use_foundation_encoder=False,
        foundation_only=False,
        world_model_warmup_steps=0,
        imagination_horizon=2,
        imagination_batch_size=4,
        burn_in=1,
    )
    t_len = 4
    seq_obs = (
        torch.zeros((2, t_len, 1)),
        torch.ones((2, t_len, 1)),
        torch.ones((2, t_len, 1)),
        torch.zeros((2, t_len, 4, 64, 64), dtype=torch.float32),
        torch.zeros((2, t_len, 3)),
        torch.zeros((2, t_len, 3)),
    )
    is_first = torch.zeros((2, t_len))
    is_first[:, 0] = 1.0
    seq_batch = (
        seq_obs,
        torch.zeros((2, t_len, 3)),
        torch.ones((2, t_len)),
        seq_obs,
        torch.zeros((2, t_len)),
        torch.zeros((2, t_len)),
        is_first,
    )
    d_metrics = dreamer.train(seq_batch)
    assert isinstance(d_metrics, dict) and d_metrics
    assert all(np.isfinite(float(v)) for v in d_metrics.values())


@pytest.mark.requires_torch
def test_f07_06_torch_extensions_use_logging_not_print_plus() -> None:
    torch_dir = REPO_ROOT / "tmrl" / "custom" / "torch"
    for fname in ("car_brain.py", "foundation_actor.py", "foundation_encoder.py"):
        tree = ast.parse((torch_dir / fname).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    assert not node.args[0].value.startswith(("[+]", "[-]", "[!]")), (
                        f"{fname} still uses bare print({node.args[0].value!r}) instead of logging"
                    )


# =============================================================================
# Feature #8: JAX/NNX Algorithms & Continual Dreamer (6 tests)
# =============================================================================


def test_f08_01_non_jax_workflows_succeed_when_jax_absent(tmp_path: Path) -> None:
    block_dir = tmp_path / "block_jax"
    block_dir.mkdir()
    (block_dir / "sitecustomize.py").write_text(
        """import sys
class _BlockJAX:
    BLOCKED = {"jax", "jaxlib", "flax", "optax"}
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.BLOCKED:
            raise ModuleNotFoundError(f"Blocked optional JAX module: {fullname}")
        return None
sys.meta_path.insert(0, _BlockJAX())
""",
        encoding="utf-8",
    )
    res_help = run_python(
        ["-m", "tmrl", "--help"],
        home_dir=tmp_path / "home",
        extra_env={"PYTHONPATH": str(block_dir)},
    )
    assert res_help.returncode == 0, res_help.stderr

    if has_module("torch"):
        res_torch = run_python(
            ["-c", "import tmrl.custom.torch.custom_algorithms, tmrl.custom.torch.jax_dreamer_actor; print('OK')"],
            home_dir=tmp_path / "home",
            extra_env={"PYTHONPATH": str(block_dir)},
        )
        assert res_torch.returncode == 0, res_torch.stderr


@pytest.mark.requires_jax
def test_f08_02_jax_custom_modules_import_without_tmrldata(tmp_path: Path) -> None:
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    code = """
import tmrl.custom.jax.custom_models
import tmrl.custom.jax.custom_algorithms
import tmrl.custom.jax.continual_dreamer
import tmrl.custom.jax.world_model
print("JAX_IMPORTS_OK")
"""
    res = run_python(["-c", code], home_dir=tmp_path / "empty_home")
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "JAX_IMPORTS_OK" in res.stdout


@pytest.mark.requires_jax
def test_f08_03_jax_nnx_models_explicit_dims_act() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    gym = pytest.importorskip("gymnasium")
    from tmrl.custom.jax.custom_models import NNXREDQMLPActorCritic, NNXREDQVanillaCNNActorCritic

    vec_obs = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)
    vec_act = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
    mlp_ac = NNXREDQMLPActorCritic(vec_obs, vec_act)
    act = mlp_ac.actor.act_(np.zeros((4,), dtype=np.float32), test=True)
    assert vec_act.contains(act)

    img_obs = gym.spaces.Tuple(
        (
            gym.spaces.Box(0.0, 100.0, (1,), np.float32),
            gym.spaces.Box(0.0, 6.0, (1,), np.float32),
            gym.spaces.Box(0.0, 10000.0, (1,), np.float32),
            gym.spaces.Box(0.0, 1.0, (4, 64, 64), np.float32),
            gym.spaces.Box(-1.0, 1.0, (3,), np.float32),
            gym.spaces.Box(-1.0, 1.0, (3,), np.float32),
        )
    )
    img_act = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)
    cnn_ac = NNXREDQVanillaCNNActorCritic(img_obs, img_act, img_hist_len=4, img_height=64, img_width=64)
    obs_inst = tuple(np.zeros(s.shape, dtype=s.dtype) for s in img_obs)
    act_cnn = cnn_ac.actor.act_(obs_inst, test=True)
    assert img_act.contains(act_cnn)


@pytest.mark.requires_jax
def test_f08_04_jax_nnxsac_agent_train_step() -> None:
    pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    gym = pytest.importorskip("gymnasium")
    from tmrl.custom.jax.custom_algorithms import NNXSACAgent
    from tmrl.custom.jax.custom_models import NNXREDQMLPActorCritic

    obs_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)
    act_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
    agent = NNXSACAgent(obs_space, act_space, model_cls=NNXREDQMLPActorCritic)
    batch = (
        jnp.zeros((4, 4), dtype=jnp.float32),
        jnp.zeros((4, 2), dtype=jnp.float32),
        jnp.ones((4,), dtype=jnp.float32),
        jnp.zeros((4, 4), dtype=jnp.float32),
        jnp.zeros((4,), dtype=jnp.float32),
        jnp.zeros((4,), dtype=jnp.float32),
    )
    metrics = agent.train(batch)
    assert isinstance(metrics, dict) and metrics
    assert all(np.isfinite(float(v)) for v in metrics.values())


@pytest.mark.requires_jax
@pytest.mark.requires_torch
def test_f08_05_jax_dreamer_actor_portable_bundle_torch_parity(tmp_path: Path) -> None:
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    pytest.importorskip("torch")
    gym = pytest.importorskip("gymnasium")
    from tmrl.custom.jax.continual_dreamer import JAXDreamerActor
    from tmrl.custom.torch.jax_dreamer_actor import TorchJAXDreamerActor

    obs_space = gym.spaces.Tuple(
        (
            gym.spaces.Box(0.0, 1.0, (1,), np.float32),
            gym.spaces.Box(0.0, 1.0, (1,), np.float32),
            gym.spaces.Box(0.0, 1.0, (1,), np.float32),
            gym.spaces.Box(0.0, 1.0, (4, 64, 64), np.float32),
            gym.spaces.Box(-1.0, 1.0, (3,), np.float32),
            gym.spaces.Box(-1.0, 1.0, (3,), np.float32),
        )
    )
    act_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)
    kwargs = dict(latent_dim=32, hidden_dim=64, policy_hidden_dim=64, encoder_channels=(8, 16, 16, 16), img_channels=4, img_height=64, img_width=64)
    jax_actor = JAXDreamerActor(obs_space, act_space, **kwargs)
    bundle = tmp_path / "portable.tmod"
    jax_actor.save(bundle)

    torch_actor = TorchJAXDreamerActor(obs_space, act_space, **kwargs)
    torch_actor.load(bundle, device="cpu")
    obs = tuple(np.full(s.shape, 0.25, dtype=s.dtype) for s in obs_space)
    np.testing.assert_allclose(jax_actor.act_(obs, test=True), torch_actor.act_(obs, test=True), atol=1e-4)


@pytest.mark.requires_jax
def test_f08_06_continual_jax_checkpoint_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    gym = pytest.importorskip("gymnasium")
    from tmrl.core.jax.training_offline import NNXTrainingOffline
    from tmrl.core.util import partial
    from tmrl.custom.jax.continual_checkpoints import (
        dump_continual_jax_run_instance,
        load_continual_jax_run_instance,
        validate_continual_jax_run_instance,
    )
    from tmrl.custom.jax.continual_dreamer import JAXDreamerAgent
    from tmrl.custom.jax.continual_replay import ArrayJaxMemoryTMFullSequence

    obs_space = gym.spaces.Tuple(
        (
            gym.spaces.Box(0.0, 1.0, (1,), np.float32),
            gym.spaces.Box(0.0, 1.0, (1,), np.float32),
            gym.spaces.Box(0.0, 1.0, (1,), np.float32),
            gym.spaces.Box(0.0, 1.0, (4, 64, 64), np.float32),
            gym.spaces.Box(-1.0, 1.0, (3,), np.float32),
            gym.spaces.Box(-1.0, 1.0, (3,), np.float32),
        )
    )
    act_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)
    trainer_cls = partial(
        NNXTrainingOffline,
        env_cls=(obs_space, act_space),
        memory_cls=partial(ArrayJaxMemoryTMFullSequence, memory_size=16, batch_size=2, sequence_length=4, imgs_obs=4, act_buf_len=2),
        training_agent_cls=partial(JAXDreamerAgent, latent_dim=32, hidden_dim=64, policy_hidden_dim=64, encoder_channels=(8, 16, 16, 16)),
        jit_sampling=False,
        jit_training=False,
    )
    trainer = trainer_cls()
    ckpt_path = tmp_path / "jax_trainer.tcpt"
    dump_continual_jax_run_instance(trainer, ckpt_path)
    restored = load_continual_jax_run_instance(ckpt_path)
    validate_continual_jax_run_instance(restored, trainer_cls)
