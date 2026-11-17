"""Tier 2 opaque-box E2E tests: boundary, negative, corner, and local integration cases across Features #1-#15."""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from tests.e2e.conftest import (
    REPO_ROOT,
    build_portable_jax_bundle,
    copy_sensor_adaptation_workspace,
    create_synthetic_sensor_checkpoint,
    free_tcp_port,
    has_jax_stack,
    has_module,
    has_sphinx,
    make_tmrl_config,
    make_valid_map_manifest,
    populate_tmrl_data,
    run_python,
)

pytestmark = [pytest.mark.tier2]

EXP_DIR = REPO_ROOT / "experiments" / "sensor_adaptation_v1"


# =============================================================================
# Feature #1 Boundaries: Packaging & Lazy Initialization (5 tests)
# =============================================================================


def test_t2_f01_01_import_tmrl_idempotent_and_get_environment(tmp_path: Path) -> None:
    home = tmp_path / "empty_home"
    code = """
import importlib, tmrl
mod2 = importlib.reload(tmrl)
env_cls = mod2.get_environment()
assert type(env_cls) is type(tmrl.get_environment())
print("IDEMPOTENT_OK")
"""
    res = run_python(["-c", code], home_dir=home, block_network=True, netblock_dir=tmp_path / "netblock")
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "IDEMPOTENT_OK" in res.stdout
    assert not (home / "TmrlData").exists()


def test_t2_f01_02_manifest_in_targets_exist_and_exclude_wandb() -> None:
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "readme/pypi.md" in manifest
    assert (REPO_ROOT / "readme" / "pypi.md").is_file()
    assert "tmrl/wandb" in manifest


def test_t2_f01_03_import_tmrl_survives_corrupt_tmrldata_config(tmp_path: Path) -> None:
    home = tmp_path / "home"
    cfg_dir = home / "TmrlData" / "config"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.json").write_text("{corrupt json", encoding="utf-8")
    res = run_python(
        ["-c", "import tmrl; print('LAZY_CORRUPT_OK')"],
        home_dir=home,
        block_network=True,
        netblock_dir=tmp_path / "netblock",
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "LAZY_CORRUPT_OK" in res.stdout


def test_t2_f01_04_import_tmrl_does_not_create_missing_parent_home(tmp_path: Path) -> None:
    nonexistent_home = tmp_path / "does_not_exist" / "nested_user"
    env = os.environ.copy()
    env["HOME"] = str(nonexistent_home)
    env["USERPROFILE"] = str(nonexistent_home)
    env["PYTHONPATH"] = str(REPO_ROOT)
    res = subprocess.run(
        [sys.executable, "-c", "import tmrl; print('OK')"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res.returncode == 0, res.stderr
    assert not (nonexistent_home / "TmrlData").exists()


def test_t2_f01_05_init_tmrl_data_offline_raises_connection_error(tmp_path: Path) -> None:
    home = tmp_path / "home"
    code = """
from tmrl.tools.init_package.init_tmrl import init_tmrl_data
try:
    init_tmrl_data()
except ConnectionError:
    print("CONN_ERROR_OK")
else:
    raise AssertionError("Expected ConnectionError when network is blocked")
"""
    res = run_python(
        ["-c", code],
        home_dir=home,
        block_network=True,
        netblock_dir=tmp_path / "netblock",
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "CONN_ERROR_OK" in res.stdout
    assert not (home / "TmrlData" / "config" / "config.json").exists()


# =============================================================================
# Feature #2 Boundaries: CLI Entry Point (5 tests)
# =============================================================================


def test_t2_f02_01_cli_install_with_empty_config_modifier_offline(tmp_path: Path) -> None:
    home = tmp_path / "home"
    populate_tmrl_data(home)
    res = run_python(
        ["-m", "tmrl", "--install", "-d", "{}"],
        home_dir=home,
        block_network=True,
        netblock_dir=tmp_path / "netblock",
    )
    assert res.returncode == 0, res.stderr


def test_t2_f02_02_cli_config_modifier_rejects_non_dict_json(tmp_path: Path) -> None:
    home = tmp_path / "home"
    populate_tmrl_data(home)
    res = run_python(["-m", "tmrl", "--test", "-d", "[1, 2, 3]"], home_dir=home)
    assert res.returncode != 0


def test_t2_f02_03_cli_short_help_flag_headless(tmp_path: Path) -> None:
    home = tmp_path / "empty_home"
    res = run_python(["-m", "tmrl", "-h"], home_dir=home, block_network=True, netblock_dir=tmp_path / "netblock")
    assert res.returncode == 0, res.stderr
    assert "--config" in res.stdout


def test_t2_f02_04_cli_worker_without_tmrldata_fails_without_network(tmp_path: Path) -> None:
    home = tmp_path / "empty_home"
    res = run_python(
        ["-m", "tmrl", "--worker"],
        home_dir=home,
        block_network=True,
        netblock_dir=tmp_path / "netblock",
    )
    assert res.returncode != 0
    assert not (home / "TmrlData" / "config" / "config.json").exists()


def test_t2_f02_05_cli_record_reward_flags_accepted_by_parser(tmp_path: Path) -> None:
    home = tmp_path / "empty_home"
    res = run_python(["-m", "tmrl", "--help"], home_dir=home)
    assert res.returncode == 0
    assert "--record-reward" in res.stdout and "--use-keyboard" in res.stdout


# =============================================================================
# Feature #3 Boundaries: Configuration & Factory Wiring (5 tests)
# =============================================================================


def test_t2_f03_01_outdated_config_version_rejected(tmp_path: Path) -> None:
    home = tmp_path / "home"
    populate_tmrl_data(home, config_dict=make_tmrl_config(version="0.5.0"))
    res = run_python(["-c", "import tmrl.config.config_objects"], home_dir=home)
    assert res.returncode != 0
    assert "0.5.0" in res.stderr or "0.6.0" in res.stderr or "version" in res.stderr.lower()


def test_t2_f03_02_missing_version_key_rejected(tmp_path: Path) -> None:
    home = tmp_path / "home"
    cfg_dict = make_tmrl_config()
    cfg_dict.pop("__VERSION__")
    populate_tmrl_data(home, config_dict=cfg_dict)
    res = run_python(["-c", "import tmrl.config.config_objects"], home_dir=home)
    assert res.returncode != 0
    assert "__VERSION__" in res.stderr or "outdated" in res.stderr.lower()


@pytest.mark.requires_torch
def test_t2_f03_03_dreamer_rejects_lidar_interface(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    home = tmp_path / "home"
    cfg_dict = make_tmrl_config(algorithm="DREAMER", interface="TM20LIDAR", grayscale=True)
    populate_tmrl_data(home, config_dict=cfg_dict)
    res = run_python(["-c", "import tmrl.config.config_objects as o; _ = o.POLICY"], home_dir=home)
    assert res.returncode != 0
    assert "image" in res.stderr.lower() or "lidar" in res.stderr.lower() or "AssertionError" in res.stderr


@pytest.mark.requires_torch
def test_t2_f03_04_dreamer_rejects_rgb_non_grayscale(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    home = tmp_path / "home"
    cfg_dict = make_tmrl_config(algorithm="DREAMER", interface="TM20FULL", grayscale=False)
    populate_tmrl_data(home, config_dict=cfg_dict)
    res = run_python(["-c", "import tmrl.config.config_objects as o; _ = o.POLICY"], home_dir=home)
    assert res.returncode != 0
    assert "grayscale" in res.stderr.lower() or "AssertionError" in res.stderr


@pytest.mark.requires_jax
def test_t2_f03_05_continual_jax_rejects_azr_and_experts(tmp_path: Path) -> None:
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    home = tmp_path / "home"
    cfg_dict = make_tmrl_config(
        algorithm="CONTINUAL_DREAMER_JAX",
        interface="TM20FULL",
        grayscale=True,
        extra_alg={"AZR_IMAGINATION": {"ENABLED": True}},
    )
    populate_tmrl_data(home, config_dict=cfg_dict)
    res = run_python(["-c", "import tmrl.config.config_objects as o; _ = o.POLICY"], home_dir=home)
    assert res.returncode != 0
    assert "AZR" in res.stderr


# =============================================================================
# Feature #4 Boundaries: Core Framework & Networking (5 tests)
# =============================================================================


def test_t2_f04_01_buffer_clips_to_maxlen_and_accumulates() -> None:
    from tmrl.core.networking import Buffer

    b1 = Buffer(maxlen=4)
    for i in range(6):
        b1.append_sample((i, i, float(i), False, False, {}))
    assert len(b1) == 4
    assert b1.memory[0][0] == 2

    b2 = Buffer(maxlen=4)
    b2.stat_train_return = 42.0
    b2.stat_test_steps = 17
    b2.append_sample((99, 99, 1.0, True, False, {}))
    b1 += b2
    assert len(b1) == 4
    assert b1.stat_train_return == 42.0
    assert b1.stat_test_steps == 17


@pytest.mark.requires_torch
@pytest.mark.requires_tlspyo
def test_t2_f04_02_rollout_worker_standalone_forbids_run_and_sync(tmp_path: Path) -> None:
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
        model_path=str(tmp_path / "w.tmod"),
        standalone=True,
    )
    with pytest.raises(RuntimeError, match="standalone"):
        rw.run(nb_episodes=1)
    with pytest.raises(RuntimeError, match="standalone"):
        rw.run_synchronous(initial_steps=1, nb_steps=1)


@pytest.mark.requires_torch
@pytest.mark.requires_tlspyo
def test_t2_f04_03_rollout_worker_run_synchronous_rejects_zero_initial_steps(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("tlspyo")
    from tmrl.core.envs import GenericGymEnv
    from tmrl.core.networking import RolloutWorker, Server
    from tmrl.core.util import partial
    from tmrl.custom.torch.custom_models import SquashedGaussianMLPActor

    pub_port = free_tcp_port()
    srv_local = free_tcp_port()
    w_local = free_tcp_port()
    server = Server(port=pub_port, password="p", local_port=srv_local, security=None, keys_dir=None)
    try:
        rw = RolloutWorker(
            env_cls=partial(GenericGymEnv, id="Pendulum-v1", gym_kwargs={"render_mode": None}),
            actor_module_cls=partial(SquashedGaussianMLPActor),
            sample_compressor=None,
            device="cpu",
            server_ip="127.0.0.1",
            server_port=pub_port,
            local_port=w_local,
            password="p",
            max_samples_per_episode=5,
            model_path=str(tmp_path / "w.tmod"),
            security=None,
            standalone=False,
        )
        with pytest.raises(RuntimeError, match="initial_steps must be >= 1"):
            rw.run_synchronous(initial_steps=0, nb_steps=2)
    finally:
        server.stop()


@pytest.mark.requires_torch
@pytest.mark.requires_tlspyo
def test_t2_f04_04_loopback_server_trainer_worker_run_and_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("tlspyo")
    from tmrl.core.envs import GenericGymEnv
    from tmrl.core.networking import RolloutWorker, Server, Trainer
    from tmrl.core.torch.training_offline import TorchTrainingOffline
    from tmrl.core.util import partial
    from tmrl.custom.torch.custom_algorithms import SpinupSACAgent
    from tmrl.custom.torch.custom_memories import GenericTorchMemory
    from tmrl.custom.torch.custom_models import MLPActorCritic, SquashedGaussianMLPActor

    pub_port = free_tcp_port()
    srv_local = free_tcp_port()
    tr_local = free_tcp_port()
    w_local = free_tcp_port()
    env_cls = partial(GenericGymEnv, id="Pendulum-v1", gym_kwargs={"render_mode": None})
    dummy = env_cls()
    obs_space, act_space = dummy.observation_space, dummy.action_space
    dummy.close()

    tmod_t = tmp_path / "trainer.tmod"
    tcpt_t = tmp_path / "trainer.tcpt"
    tmod_w = tmp_path / "worker.tmod"

    server = Server(port=pub_port, password="p", local_port=srv_local, security=None, keys_dir=None)
    try:
        training_cls = partial(
            TorchTrainingOffline,
            env_cls=(obs_space, act_space),
            memory_cls=partial(GenericTorchMemory, memory_size=64, batch_size=4),
            training_agent_cls=partial(SpinupSACAgent, model_cls=MLPActorCritic, device="cpu"),
            epochs=1,
            rounds=1,
            steps=2,
            update_model_interval=1,
            update_buffer_interval=1,
            max_training_steps_per_env_step=10.0,
            sleep_between_buffer_retrieval_attempts=0.05,
            start_training=4,
            device="cpu",
        )
        trainer = Trainer(
            training_cls=training_cls,
            server_ip="127.0.0.1",
            server_port=pub_port,
            local_com_port=tr_local,
            password="p",
            model_path=str(tmod_t),
            checkpoint_path=str(tcpt_t),
            security=None,
        )
        worker = RolloutWorker(
            env_cls=env_cls,
            actor_module_cls=partial(SquashedGaussianMLPActor),
            sample_compressor=None,
            device="cpu",
            server_ip="127.0.0.1",
            server_port=pub_port,
            local_port=w_local,
            password="p",
            max_samples_per_episode=8,
            model_path=str(tmod_w),
            security=None,
        )
        w_thread = threading.Thread(
            target=lambda: worker.run(nb_episodes=2, verbose=False),
            daemon=True,
        )
        w_thread.start()
        trainer.run()
        w_thread.join(timeout=15)
        assert tmod_t.is_file()
        assert tcpt_t.is_file()
    finally:
        server.stop()


@pytest.mark.requires_torch
@pytest.mark.requires_tlspyo
def test_t2_f04_05_loopback_run_synchronous_initial_steps_completes(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("tlspyo")
    from tmrl.core.envs import GenericGymEnv
    from tmrl.core.networking import RolloutWorker, Server, Trainer
    from tmrl.core.torch.training_offline import TorchTrainingOffline
    from tmrl.core.util import partial
    from tmrl.custom.torch.custom_algorithms import SpinupSACAgent
    from tmrl.custom.torch.custom_memories import GenericTorchMemory
    from tmrl.custom.torch.custom_models import MLPActorCritic, SquashedGaussianMLPActor

    pub_port = free_tcp_port()
    srv_local = free_tcp_port()
    tr_local = free_tcp_port()
    w_local = free_tcp_port()
    env_cls = partial(GenericGymEnv, id="Pendulum-v1", gym_kwargs={"render_mode": None})
    dummy = env_cls()
    obs_space, act_space = dummy.observation_space, dummy.action_space
    dummy.close()

    server = Server(port=pub_port, password="p", local_port=srv_local, security=None, keys_dir=None)
    worker_errors: list[BaseException] = []
    try:
        training_cls = partial(
            TorchTrainingOffline,
            env_cls=(obs_space, act_space),
            memory_cls=partial(GenericTorchMemory, memory_size=64, batch_size=4),
            training_agent_cls=partial(SpinupSACAgent, model_cls=MLPActorCritic, device="cpu"),
            epochs=1,
            rounds=1,
            steps=3,
            update_model_interval=1,
            update_buffer_interval=1,
            max_training_steps_per_env_step=10.0,
            sleep_between_buffer_retrieval_attempts=0.05,
            start_training=4,
            device="cpu",
        )
        trainer = Trainer(
            training_cls=training_cls,
            server_ip="127.0.0.1",
            server_port=pub_port,
            local_com_port=tr_local,
            password="p",
            model_path=str(tmp_path / "tr.tmod"),
            checkpoint_path=str(tmp_path / "tr.tcpt"),
            security=None,
        )
        worker = RolloutWorker(
            env_cls=env_cls,
            actor_module_cls=partial(SquashedGaussianMLPActor),
            sample_compressor=None,
            device="cpu",
            server_ip="127.0.0.1",
            server_port=pub_port,
            local_port=w_local,
            password="p",
            max_samples_per_episode=6,
            model_path=str(tmp_path / "wk.tmod"),
            security=None,
        )

        def _run_sync() -> None:
            try:
                worker.run_synchronous(
                    initial_steps=6,
                    nb_steps=6,
                    max_steps_per_update=6,
                    end_episodes=False,
                )
            except BaseException as exc:  # pragma: no cover
                worker_errors.append(exc)

        w_thread = threading.Thread(target=_run_sync, daemon=True)
        w_thread.start()
        trainer.run()
        w_thread.join(timeout=15)
        assert not worker_errors, f"RolloutWorker.run_synchronous failed: {worker_errors[0]!r}"
        assert (tmp_path / "tr.tmod").is_file()
    finally:
        server.stop()


# =============================================================================
# Feature #5 Boundaries: Continual JAX & Diagnostic Tools (5 tests)
# =============================================================================


def test_t2_f05_01_prepare_continual_jax_config_refuses_overwrite_and_bad_inputs(tmp_path: Path) -> None:
    good_src = tmp_path / "good.json"
    out = tmp_path / "out.json"
    good_src.write_text(json.dumps(make_tmrl_config(grayscale=True)), encoding="utf-8")
    out.write_text("{}", encoding="utf-8")

    # Refuses overwrite without --overwrite
    res_no_ow = run_python(
        ["-m", "tmrl.tools.prepare_continual_jax_config", "--source", str(good_src), "--output", str(out)],
        home_dir=tmp_path / "home",
    )
    assert res_no_ow.returncode != 0

    # Succeeds with --overwrite
    res_ow = run_python(
        ["-m", "tmrl.tools.prepare_continual_jax_config", "--source", str(good_src), "--output", str(out), "--overwrite"],
        home_dir=tmp_path / "home",
    )
    assert res_ow.returncode == 0, res_ow.stderr

    # Rejects LIDAR and non-grayscale
    for bad_cfg in (
        make_tmrl_config(interface="TM2020InterfaceLidar", grayscale=True),
        make_tmrl_config(interface="TM2020Interface", grayscale=False),
    ):
        bad_src = tmp_path / "bad.json"
        bad_src.write_text(json.dumps(bad_cfg), encoding="utf-8")
        res_bad = run_python(
            ["-m", "tmrl.tools.prepare_continual_jax_config", "--source", str(bad_src), "--output", str(tmp_path / "bad_out.json")],
            home_dir=tmp_path / "home",
        )
        assert res_bad.returncode != 0


def test_t2_f05_02_validate_continual_map_manifest_negative_cases(tmp_path: Path) -> None:
    # 1. unfrozen with --require-frozen
    unfrozen = make_valid_map_manifest(frozen=False)
    p1 = tmp_path / "unfrozen.json"
    p1.write_text(json.dumps(unfrozen), encoding="utf-8")
    r1 = run_python(["-m", "tmrl.tools.validate_continual_map_manifest", str(p1), "--require-frozen"], home_dir=tmp_path / "home")
    assert r1.returncode == 2 and json.loads(r1.stdout)["valid"] is False

    # 2. missing split + duplicate map_id + invalid sha256
    bad = make_valid_map_manifest(frozen=True)
    bad["maps"][1]["map_id"] = bad["maps"][0]["map_id"]
    bad["maps"][1]["split"] = "foundation_train"
    bad["maps"][2]["track_sha256"] = "not-a-sha256"
    p2 = tmp_path / "bad.json"
    p2.write_text(json.dumps(bad), encoding="utf-8")
    r2 = run_python(["-m", "tmrl.tools.validate_continual_map_manifest", str(p2)], home_dir=tmp_path / "home")
    assert r2.returncode == 2
    errs = " ".join(json.loads(r2.stdout)["errors"])
    assert "duplicate" in errs and "missing map splits" in errs and "track_sha256" in errs


def test_t2_f05_03_freeze_continual_baseline_negative_cases(tmp_path: Path) -> None:
    src = tmp_path / "ckpt.bin"
    src.write_bytes(b"data")
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "existing.txt").write_text("x", encoding="utf-8")

    r_nonempty = run_python(
        ["-m", "tmrl.tools.freeze_continual_baseline", "--source", str(src), "--output-dir", str(nonempty)],
        home_dir=tmp_path / "home",
    )
    assert r_nonempty.returncode == 2

    r_missing = run_python(
        ["-m", "tmrl.tools.freeze_continual_baseline", "--source", str(tmp_path / "missing.bin"), "--output-dir", str(tmp_path / "o2")],
        home_dir=tmp_path / "home",
    )
    assert r_missing.returncode == 2


@pytest.mark.requires_torch
def test_t2_f05_04_benchmark_continual_actor_bad_channels_and_budget_fail(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    r_bad = run_python(
        ["-m", "tmrl.tools.benchmark_continual_actor", "--encoder-channels", "8,16"],
        home_dir=tmp_path / "home",
    )
    assert r_bad.returncode == 2

    out = tmp_path / "tight.json"
    r_tight = run_python(
        [
            "-m",
            "tmrl.tools.benchmark_continual_actor",
            "--runtime",
            "torch",
            "--device",
            "cpu",
            "--iterations",
            "3",
            "--encoder-channels",
            "8,16,16,16",
            "--max-p95-ms",
            "0.000001",
            "--output",
            str(out),
        ],
        home_dir=tmp_path / "home",
        timeout=90,
    )
    assert r_tight.returncode == 2
    assert json.loads(out.read_text(encoding="utf-8"))["passed"] is False


@pytest.mark.requires_jax
def test_t2_f05_05_validate_continual_trainer_stack_rejects_sac_config(tmp_path: Path) -> None:
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    home = tmp_path / "home"
    populate_tmrl_data(home, config_dict=make_tmrl_config(algorithm="SAC"))
    res = run_python(
        ["-m", "tmrl.tools.validate_continual_trainer_stack", "--output-directory", str(tmp_path / "out")],
        home_dir=home,
    )
    assert res.returncode != 0
    assert "CONTINUAL_DREAMER_JAX" in res.stderr


# =============================================================================
# Feature #6 Boundaries: TrackMania Interfaces, Capture & Control (5 tests)
# =============================================================================


def test_t2_f06_01_tm2020_interface_rejects_signed_triggers_without_gamepad(tmp_path: Path) -> None:
    home = tmp_path / "home"
    populate_tmrl_data(home)
    code = """
from tmrl.custom.tm.tm_gym_interfaces import TM2020Interface
try:
    TM2020Interface(gamepad=False, signed_analog_triggers=True)
except ValueError:
    print("VALUE_ERROR_OK")
else:
    raise AssertionError("Expected ValueError")
"""
    res = run_python(["-c", code], home_dir=home)
    assert res.returncode == 0, res.stderr
    assert "VALUE_ERROR_OK" in res.stdout


def test_t2_f06_02_reward_function_stray_distance_zero_reward_and_stall(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tmrl.custom.tm.utils.compute_reward import RewardFunction

    traj = np.column_stack([np.arange(0, 20, dtype=np.float32), np.zeros(20, dtype=np.float32), np.zeros(20, dtype=np.float32)])
    rew_path = tmp_path / "reward.pkl"
    with rew_path.open("wb") as handle:
        pickle.dump(traj, handle)
    monkeypatch.setenv("TMRL_UNIVERSAL_REWARD", "0")
    rf = RewardFunction(str(rew_path), nb_obs_forward=5, nb_zero_rew_before_failure=2, min_nb_steps_before_failure=1, max_dist_from_traj=5.0)
    r_far, term_far = rf.compute_reward(np.array([10.0, 0.0, 100.0], dtype=np.float32))
    assert r_far == 0.0 and term_far is False


def test_t2_f06_03_preprocess_camera_frame_invalid_shapes_raise() -> None:
    from tmrl.custom.tm.utils.camera_preprocessing import preprocess_camera_frame

    for bad in (np.zeros((0, 0, 3), np.uint8), np.zeros((16, 16), np.uint8), np.zeros((16, 16, 1), np.uint8)):
        with pytest.raises(ValueError):
            preprocess_camera_frame(bad)


def test_t2_f06_04_signed_trigger_controls_rejects_bad_inputs() -> None:
    from tmrl.custom.tm.utils.control_gamepad import signed_trigger_controls

    for bad in ([0.0, 0.0], [float("nan"), 0.0, 0.0], [0.0, float("inf"), 0.0], [0.0, 0.0, -1.5]):
        with pytest.raises(ValueError):
            signed_trigger_controls(bad)


def test_t2_f06_05_trajectory_assist_negative_and_deviation_stop(tmp_path: Path) -> None:
    from tmrl.custom.tm.utils.trajectory_assist import DemonstrationTrajectoryAssist

    with pytest.raises(ValueError):
        DemonstrationTrajectoryAssist({"STEERING_MODE": "invalid"})
    with pytest.raises(ValueError):
        DemonstrationTrajectoryAssist({"TRACKING_MIN_SPEED": 30.0, "TRACKING_MAX_SPEED": 10.0})
    with pytest.raises(FileNotFoundError):
        DemonstrationTrajectoryAssist({"ENABLED": True, "TRAJECTORY_PATH": str(tmp_path / "missing.npz")})

    # Deviation exceeding MAX_DEVIATION_METERS triggers emergency_stop
    t = np.linspace(0.0, 30.0, 20, dtype=np.float32)
    traj_path = tmp_path / "demo.npz"
    np.savez(
        traj_path,
        positions=np.column_stack([t, np.zeros_like(t), np.zeros_like(t)]),
        actions=np.zeros((20, 3), dtype=np.float32),
        speeds_mps=np.full((20,), 10.0, dtype=np.float32),
    )
    assist = DemonstrationTrajectoryAssist({"ENABLED": True, "TRAJECTORY_PATH": str(traj_path), "MAX_DEVIATION_METERS": 2.0})
    _, info = assist.action([5.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1000.0], np.zeros(3, np.float32))
    assert info["emergency_stop"] is True


# =============================================================================
# Feature #7 Boundaries: PyTorch Algorithms, Models & Extensions (5 tests)
# =============================================================================


@pytest.mark.requires_torch
def test_t2_f07_01_vanilla_cnn_custom_image_dimensions_without_tmrldata(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    code = """
import gymnasium as gym, numpy as np
from tmrl.custom.torch.custom_models import VanillaCNNActorCritic
obs_space = gym.spaces.Tuple((
    gym.spaces.Box(0.0, 100.0, (1,), np.float32),
    gym.spaces.Box(0.0, 6.0, (1,), np.float32),
    gym.spaces.Box(0.0, 10000.0, (1,), np.float32),
    gym.spaces.Box(0.0, 1.0, (2, 64, 80), np.float32),
    gym.spaces.Box(-1.0, 1.0, (3,), np.float32),
    gym.spaces.Box(-1.0, 1.0, (3,), np.float32),
))
act_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)
ac = VanillaCNNActorCritic(obs_space, act_space)
obs = tuple(np.zeros(s.shape, dtype=s.dtype) for s in obs_space)
act = ac.actor.act_(obs, test=True)
assert act.shape == (3,)
print("CUSTOM_DIMS_OK")
"""
    res = run_python(["-c", code], home_dir=tmp_path / "empty_home")
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "CUSTOM_DIMS_OK" in res.stdout


@pytest.mark.requires_torch
def test_t2_f07_02_spinup_sac_sgd_and_fixed_entropy() -> None:
    torch = pytest.importorskip("torch")
    gym = pytest.importorskip("gymnasium")
    from tmrl.custom.torch.custom_algorithms import SpinupSACAgent
    from tmrl.custom.torch.custom_models import MLPActorCritic

    obs_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)
    act_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
    agent = SpinupSACAgent(
        observation_space=obs_space,
        action_space=act_space,
        device="cpu",
        model_cls=MLPActorCritic,
        optimizer_actor="sgd",
        optimizer_critic="sgd",
        learn_entropy_coef=False,
        alpha=0.15,
    )
    assert isinstance(agent.pi_optimizer, torch.optim.SGD)
    batch = (
        torch.zeros((4, 3)),
        torch.zeros((4, 1)),
        torch.ones((4,)),
        torch.zeros((4, 3)),
        torch.zeros((4,)),
        torch.zeros((4,)),
    )
    metrics = agent.train(batch)
    assert np.isclose(float(agent.alpha_t), 0.15)
    assert all(np.isfinite(float(v)) for v in metrics.values())


@pytest.mark.requires_torch
def test_t2_f07_03_torch_jax_dreamer_actor_rejects_tampered_bundle(tmp_path: Path) -> None:
    import pickle as pkl

    pytest.importorskip("torch")
    gym = pytest.importorskip("gymnasium")
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
    actor = TorchJAXDreamerActor(
        obs_space, act_space, latent_dim=32, hidden_dim=64, policy_hidden_dim=64, encoder_channels=(8, 16, 16, 16), img_channels=4, img_height=64, img_width=64
    )
    bundle = tmp_path / "portable.tmod"
    payload = build_portable_jax_bundle(actor)
    bundle.write_bytes(pkl.dumps(payload))
    actor.load(bundle, device="cpu")
    assert actor.last_load_succeeded is True

    tampered = tmp_path / "tampered.tmod"
    payload["parameter_checksum"] = "0" * 64
    tampered.write_bytes(pkl.dumps(payload))
    actor.load(tampered, device="cpu")
    assert actor.last_load_succeeded is False
    assert actor.last_load_error and "checksum" in actor.last_load_error.lower()


@pytest.mark.requires_torch
def test_t2_f07_04_torch_dreamer_actor_rejects_invalid_foundation_flags() -> None:
    pytest.importorskip("torch")
    gym = pytest.importorskip("gymnasium")
    from tmrl.custom.torch.dreamer import TorchDreamerActor

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
    with pytest.raises(ValueError, match="foundation_only"):
        TorchDreamerActor(obs_space, act_space, img_channels=4, img_height=64, img_width=64, use_foundation_encoder=False, foundation_only=True)
    with pytest.raises(ValueError, match="foundation_steer_threshold"):
        TorchDreamerActor(obs_space, act_space, img_channels=4, img_height=64, img_width=64, use_foundation_encoder=False, foundation_only=False, foundation_steer_threshold=1.5)


@pytest.mark.requires_torch
def test_t2_f07_05_world_model_symlog_symexp_extreme_range() -> None:
    torch = pytest.importorskip("torch")
    from tmrl.custom.torch.world_model import TorchLatentRSSM, symexp, symlog

    vals = torch.tensor([-1e4, -100.0, -1.0, 0.0, 1.0, 100.0, 1e4], dtype=torch.float32)
    rec = symexp(symlog(vals))
    torch.testing.assert_close(vals, rec, atol=1e-2, rtol=1e-4)

    rssm = TorchLatentRSSM(latent_dim=16, action_dim=3, hidden_dim=32)
    h = torch.zeros((2, 32))
    _, _, prior_std = rssm.compute_prior(h)
    _, _, post_std = rssm.compute_posterior(h, torch.zeros((2, 16)))
    assert (prior_std > 0).all() and (post_std > 0).all()


# =============================================================================
# Feature #8 Boundaries: JAX/NNX Algorithms & Continual Dreamer (5 tests)
# =============================================================================


@pytest.mark.requires_jax
def test_t2_f08_01_jax_world_model_imports_without_tmrldata(tmp_path: Path) -> None:
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    res = run_python(["-c", "import tmrl.custom.jax.world_model; print('OK')"], home_dir=tmp_path / "empty_home")
    assert res.returncode == 0, res.stderr


@pytest.mark.requires_jax
def test_t2_f08_02_jax_dreamer_actor_deterministic_action_bounds() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    gym = pytest.importorskip("gymnasium")
    from tmrl.custom.jax.continual_dreamer import JAXDreamerActor

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
    actor = JAXDreamerActor(obs_space, act_space, latent_dim=32, hidden_dim=64, policy_hidden_dim=64, encoder_channels=(8, 16, 16, 16), img_channels=4, img_height=64, img_width=64)
    obs = tuple(np.zeros(s.shape, dtype=s.dtype) for s in obs_space)
    a1 = actor.act_(obs, test=True)
    actor.reset()
    a2 = actor.act_(obs, test=True)
    np.testing.assert_allclose(a1, a2, atol=1e-6)
    assert act_space.contains(a1)


@pytest.mark.requires_jax
def test_t2_f08_03_jax_dreamer_agent_train_step() -> None:
    pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    gym = pytest.importorskip("gymnasium")
    from tmrl.custom.jax.continual_dreamer import JAXDreamerAgent

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
    agent = JAXDreamerAgent(obs_space, act_space, latent_dim=32, hidden_dim=64, policy_hidden_dim=64, encoder_channels=(8, 16, 16, 16), world_model_warmup_steps=0, imagination_horizon=2, imagination_batch_size=4, burn_in=1)
    seq_obs = (
        jnp.zeros((2, 4, 1)),
        jnp.ones((2, 4, 1)),
        jnp.ones((2, 4, 1)),
        jnp.zeros((2, 4, 4, 64, 64)),
        jnp.zeros((2, 4, 3)),
        jnp.zeros((2, 4, 3)),
    )
    batch = (seq_obs, jnp.zeros((2, 4, 3)), jnp.ones((2, 4)), seq_obs, jnp.zeros((2, 4)), jnp.zeros((2, 4)))
    metrics = agent.train(batch)
    assert all(np.isfinite(float(v)) for v in metrics.values())


@pytest.mark.requires_jax
def test_t2_f08_04_array_jax_memory_sequence_append_and_sample(tmp_path: Path) -> None:
    pytest.importorskip("jax")
    from tmrl.core.networking import Buffer
    from tmrl.custom.jax.continual_replay import ArrayJaxMemoryTMFullSequence

    mem = ArrayJaxMemoryTMFullSequence(device="cpu", memory_size=32, batch_size=2, sequence_length=3, imgs_obs=4, act_buf_len=2, dataset_path=str(tmp_path / "none"))
    buf = Buffer(maxlen=32)
    for t in range(8):
        obs = (np.array([1.0], np.float32), np.array([1.0], np.float32), np.array([1.0], np.float32), np.zeros((4, 64, 64), np.float32), np.zeros((3,), np.float32), np.zeros((3,), np.float32))
        buf.append_sample((np.zeros((3,), np.float32), obs, 1.0, t == 7, False, {}))
    mem.append(buf)
    sample = mem.sample()
    assert len(sample) == 6


@pytest.mark.requires_jax
def test_t2_f08_05_continual_checkpoints_rejects_corrupted_file(tmp_path: Path) -> None:
    pytest.importorskip("jax")
    from tmrl.custom.jax.continual_checkpoints import load_continual_jax_run_instance

    bad_path = tmp_path / "bad.tcpt"
    bad_path.write_bytes(b"not a valid checkpoint")
    with pytest.raises(Exception):
        load_continual_jax_run_instance(bad_path)


# =============================================================================
# Feature #9 Boundaries: Sensor-Adaptation Study & Env (5 tests)
# =============================================================================


def test_t2_f09_01_sensor_fault_env_negative_and_onset_boundaries() -> None:
    gym = pytest.importorskip("gymnasium")
    if str(EXP_DIR) not in sys.path:
        sys.path.insert(0, str(EXP_DIR))
    from sensor_env import SensorFaultEnv  # type: ignore

    class _Counter(gym.Env):
        def __init__(self) -> None:
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (2,), np.float32)
            self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
            self.t = 0

        def reset(self, *, seed=None, options=None):
            self.t = 0
            return np.array([10.0, 20.0], np.float32), {}

        def step(self, action):
            self.t += 1
            return np.array([10.0 + self.t, 20.0 + self.t], np.float32), 1.0, False, False, {}

    for bad_kw in ({"fault": "action_noise"}, {"fault": "sensor_noise", "noise_std": 0.0}, {"fault": "sensor_delay", "delay": -1}, {"onset": -1}):
        with pytest.raises(ValueError):
            SensorFaultEnv(_Counter(), **bad_kw)

    env = SensorFaultEnv(_Counter(), fault="sensor_delay", delay=2, onset=3)
    np.testing.assert_array_equal(env.reset(seed=1)[0][:2], [10.0, 20.0])
    env.step(np.array([0.0], np.float32))
    obs2, *_ = env.step(np.array([0.0], np.float32))
    np.testing.assert_array_equal(obs2[:2], [12.0, 22.0])
    obs3, _, _, _, info3 = env.step(np.array([0.0], np.float32))
    np.testing.assert_array_equal(obs3[:2], [11.0, 21.0])
    assert info3["sensor_audit"]["active"] is True


@pytest.mark.requires_torch
def test_t2_f09_02_learner_rejects_invalid_context_and_missing_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    if str(EXP_DIR) not in sys.path:
        sys.path.insert(0, str(EXP_DIR))
    from frozen_models import TransformerRL2Agent  # type: ignore
    from learner import configure, load_agent  # type: ignore

    model = TransformerRL2Agent()
    for bad_ctx in (0, -1, 33):
        with pytest.raises(ValueError):
            configure(model, "transformer", bad_ctx)
    with pytest.raises(FileNotFoundError):
        load_agent("transformer", tmp_path / "missing.pt")


@pytest.mark.requires_torch
def test_t2_f09_03_replay_sample_batch_empty_pool_raises() -> None:
    pytest.importorskip("torch")
    if str(EXP_DIR) not in sys.path:
        sys.path.insert(0, str(EXP_DIR))
    from replay import sample_batch  # type: ignore

    ep = {
        "observations": np.zeros((5, 25), np.float32),
        "actions": np.zeros((4, 6), np.float32),
        "rewards": np.zeros((4,), np.float32),
        "terminated": np.zeros((4,), np.float32),
    }
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="empty"):
        sample_batch([], [], 4, 32, 0.0, rng, "cpu")
    with pytest.raises(ValueError, match="empty"):
        sample_batch([ep], [], 4, 32, 0.5, rng, "cpu")


@pytest.mark.requires_torch
@pytest.mark.requires_mujoco
def test_t2_f09_04_run_study_guardrails_missing_ckpt_external_protocol_existing_out(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    ws = copy_sensor_adaptation_workspace(tmp_path)
    exp = ws / "experiments" / "sensor_adaptation_v1"

    # 1. Missing .pt checkpoint -> FileNotFoundError
    r_missing = run_python(
        [str(exp / "run_study.py"), "--protocol", str(exp / "protocol.json"), "--output", str(tmp_path / "o1"), "--smoke", "--device", "cpu"],
        home_dir=tmp_path / "home",
        cwd=exp,
    )
    assert r_missing.returncode != 0 and "FileNotFoundError" in r_missing.stderr

    # 2. Protocol outside experiment dir -> ValueError
    ext_proto = tmp_path / "external_protocol.json"
    ext_proto.write_text((exp / "protocol.json").read_text(encoding="utf-8"), encoding="utf-8")
    r_proto = run_python(
        [str(exp / "run_study.py"), "--protocol", str(ext_proto), "--output", str(tmp_path / "o2"), "--smoke", "--device", "cpu"],
        home_dir=tmp_path / "home",
        cwd=exp,
    )
    assert r_proto.returncode != 0 and "ValueError" in r_proto.stderr

    # 3. Unknown seed -> ValueError
    r_seed = run_python(
        [str(exp / "run_study.py"), "--protocol", str(exp / "protocol.json"), "--seeds", "999", "--output", str(tmp_path / "o3"), "--device", "cpu"],
        home_dir=tmp_path / "home",
        cwd=exp,
    )
    assert r_seed.returncode != 0 and "ValueError" in r_seed.stderr

    # 4. Pre-existing output directory -> FileExistsError
    existing = tmp_path / "existing_out"
    existing.mkdir()
    r_exists = run_python(
        [str(exp / "run_study.py"), "--protocol", str(exp / "protocol.json"), "--output", str(existing), "--smoke", "--device", "cpu"],
        home_dir=tmp_path / "home",
        cwd=exp,
    )
    assert r_exists.returncode != 0 and "FileExistsError" in r_exists.stderr


@pytest.mark.requires_torch
@pytest.mark.requires_mujoco
def test_t2_f09_05_run_study_smoke_halfcheetah_and_ant_synthetic_checkpoints(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    ws = copy_sensor_adaptation_workspace(tmp_path)
    exp = ws / "experiments" / "sensor_adaptation_v1"

    for alg in ("transformer", "reactive"):
        create_synthetic_sensor_checkpoint(
            ws / "fault_benchmark_results" / f"{alg}_seed_42" / "best_validation_checkpoint.pt",
            alg,
            obs_dim=25,
            act_dim=6,
        )
    create_synthetic_sensor_checkpoint(
        ws / "fault_benchmark_results_ant" / "transformer_seed_42" / "best_validation_checkpoint.pt",
        "transformer",
        obs_dim=37,
        act_dim=8,
    )

    hc_out = tmp_path / "hc_smoke"
    r_hc = run_python(
        [str(exp / "run_study.py"), "--protocol", str(exp / "protocol.json"), "--algorithms", "transformer", "--seeds", "42", "--output", str(hc_out), "--smoke", "--device", "cpu"],
        home_dir=tmp_path / "home",
        cwd=exp,
        timeout=120,
    )
    assert r_hc.returncode == 0, f"stdout={r_hc.stdout}\nstderr={r_hc.stderr}"
    assert json.loads((hc_out / "status.json").read_text(encoding="utf-8"))["status"] == "COMPLETE"
    study = json.loads((hc_out / "study_result.json").read_text(encoding="utf-8"))
    assert study["branches"] == 4
    for branch_res in study["results"]:
        assert branch_res["max_parameter_change"] > 0.0
        if branch_res["algorithm"] == "transformer":
            assert branch_res["positional_change_outside_support"] == 0.0

    ant_out = tmp_path / "ant_smoke"
    r_ant = run_python(
        [str(exp / "run_study.py"), "--protocol", str(exp / "protocol_ant.json"), "--algorithms", "transformer", "--seeds", "42", "--output", str(ant_out), "--smoke", "--device", "cpu"],
        home_dir=tmp_path / "home",
        cwd=exp,
        timeout=120,
    )
    assert r_ant.returncode == 0, f"stdout={r_ant.stdout}\nstderr={r_ant.stderr}"
    assert json.loads((ant_out / "study_result.json").read_text(encoding="utf-8"))["branches"] == 5


# =============================================================================
# Feature #10 Boundaries: Contract & Unit Test Suites (5 tests)
# =============================================================================


def test_t2_f10_01_sensor_noise_rng_repeatability_across_resets() -> None:
    gym = pytest.importorskip("gymnasium")
    if str(EXP_DIR) not in sys.path:
        sys.path.insert(0, str(EXP_DIR))
    from sensor_env import SensorFaultEnv  # type: ignore

    class _ConstEnv(gym.Env):
        def __init__(self) -> None:
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (2,), np.float32)
            self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)

        def reset(self, *, seed=None, options=None):
            return np.array([1.0, 2.0], np.float32), {}

        def step(self, action):
            return np.array([1.0, 2.0], np.float32), 0.0, False, False, {}

    env = SensorFaultEnv(_ConstEnv(), "sensor_noise", noise_std=0.5)
    o1, _ = env.reset(seed=123)
    o2, _ = env.reset(seed=123)
    o3, _ = env.reset(seed=124)
    np.testing.assert_array_equal(o1, o2)
    assert not np.array_equal(o1[:2], o3[:2])


@pytest.mark.requires_torch
def test_t2_f10_02_replay_terminal_step_sampling_and_lengths() -> None:
    pytest.importorskip("torch")
    if str(EXP_DIR) not in sys.path:
        sys.path.insert(0, str(EXP_DIR))
    from replay import sample_batch  # type: ignore

    ep = {
        "observations": np.arange(4 * 25, dtype=np.float32).reshape(4, 25),
        "actions": np.zeros((3, 6), np.float32),
        "rewards": np.ones((3,), np.float32),
        "terminated": np.array([0.0, 0.0, 1.0], np.float32),
    }
    batch = sample_batch([ep], [ep], 100, 32, 0.5, np.random.default_rng(7), "cpu")
    assert batch["origins"].count("old") == 50
    assert batch["origins"].count("new") == 50
    assert float(batch["terminated"].sum()) > 0.0
    assert (batch["next_lengths"] <= 4).all()


@pytest.mark.requires_torch
def test_t2_f10_03_transformer_right_padding_invariance() -> None:
    torch = pytest.importorskip("torch")
    if str(EXP_DIR) not in sys.path:
        sys.path.insert(0, str(EXP_DIR))
    from frozen_models import TransformerRL2Agent  # type: ignore
    from learner import configure, features  # type: ignore

    torch.manual_seed(11)
    model = configure(TransformerRL2Agent(), "transformer", 32)
    short = torch.randn(2, 5, 25)
    padded = torch.cat([short, torch.randn(2, 27, 25)], dim=1)
    with torch.no_grad():
        f_short = features(model, "transformer", short, torch.tensor([5, 5]))
        f_pad = features(model, "transformer", padded, torch.tensor([5, 5]))
    torch.testing.assert_close(f_short, f_pad, atol=2e-6, rtol=2e-6)


@pytest.mark.requires_torch
def test_t2_f10_04_learner_update_preserves_unused_pos_emb_and_source(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if str(EXP_DIR) not in sys.path:
        sys.path.insert(0, str(EXP_DIR))
    from frozen_models import TransformerRL2Agent  # type: ignore
    from learner import Learner  # type: ignore
    from replay import sample_batch  # type: ignore

    cfg = json.loads((EXP_DIR / "protocol.json").read_text(encoding="utf-8"))
    model = TransformerRL2Agent()
    initial = model.trunk.pos_emb.detach().clone()
    ckpt = tmp_path / "source.pt"
    torch.save(model.state_dict(), ckpt)

    learner = Learner("transformer", ckpt, cfg, "cpu", 5)
    ep = {
        "observations": np.ones((36, 25), np.float32),
        "actions": np.zeros((35, 6), np.float32),
        "rewards": np.ones((35,), np.float32),
        "terminated": np.array([0.0] * 34 + [1.0], np.float32),
    }
    batch = sample_batch([ep], [ep], 8, 32, 0.5, np.random.default_rng(1), "cpu")
    metrics = learner.update(batch)
    assert np.isfinite(list(metrics.values())).all()
    torch.testing.assert_close(initial[:, 32:], learner.model.trunk.pos_emb[:, 32:], rtol=0, atol=0)
    reloaded = torch.load(ckpt, weights_only=True)
    torch.testing.assert_close(initial, reloaded["trunk.pos_emb"], rtol=0, atol=0)


def test_t2_f10_05_repo_unit_tests_execute_and_pass_cpu(tmp_path: Path) -> None:
    tests_dir = REPO_ROOT / "tests"
    non_e2e = [p for p in tests_dir.rglob("test_*.py") if "e2e" not in p.relative_to(tests_dir).parts]
    assert non_e2e, "Expected M3 repository unit/CLI tests under tests/"
    res = run_python(
        ["-m", "pytest", *[str(p) for p in non_e2e], "-v"],
        home_dir=tmp_path / "home",
        cwd=REPO_ROOT,
        timeout=120,
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"


# =============================================================================
# Feature #11 Boundaries: Analyzers & Paper Asset Builders (5 tests)
# =============================================================================


@pytest.mark.requires_torch
@pytest.mark.requires_mujoco
def test_t2_f11_01_analyze_results_and_build_paper_assets_roundtrip(
    tmp_path: Path, smoke_pilot_template: tuple[Path, Path]
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    ws_src, pilot_src = smoke_pilot_template
    exp = ws_src / "experiments" / "sensor_adaptation_v1"
    pilot = tmp_path / "pilot"
    shutil.copytree(pilot_src, pilot)

    r_ana = run_python([str(exp / "analyze_results.py"), str(pilot)], home_dir=tmp_path / "home", cwd=exp)
    assert r_ana.returncode == 0, f"stdout={r_ana.stdout}\nstderr={r_ana.stderr}"
    assert json.loads((pilot / "verification.json").read_text(encoding="utf-8"))["status"] == "PASS"
    assert (pilot / "REPORT.md").is_file()

    # Smoke=True must fail build_paper_assets.py
    r_smoke_fail = run_python(
        [str(exp / "build_paper_assets.py"), str(pilot), str(tmp_path / "assets_fail")],
        home_dir=tmp_path / "home",
        cwd=exp,
    )
    assert r_smoke_fail.returncode != 0

    # Patch manifest["smoke"] = False to test full paper asset generation
    manifest_data = json.loads((pilot / "manifest.json").read_text(encoding="utf-8"))
    manifest_data["smoke"] = False
    (pilot / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")
    assets_ok = tmp_path / "assets_ok"
    r_assets = run_python(
        [str(exp / "build_paper_assets.py"), str(pilot), str(assets_ok)],
        home_dir=tmp_path / "home",
        cwd=exp,
    )
    assert r_assets.returncode == 0, r_assets.stderr
    assert (assets_ok / "outcomes_table.tex").is_file()
    assert (assets_ok / "sensor_adaptation.pdf").is_file()


def test_t2_f11_02_analyze_extension_rejects_incomplete_or_missing_seeds(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    shard = tmp_path / "shard"
    shard.mkdir()
    (shard / "status.json").write_text(json.dumps({"status": "RUNNING"}), encoding="utf-8")
    res = run_python(
        [str(EXP_DIR / "analyze_extension.py"), str(shard), "--protocol", str(EXP_DIR / "protocol.json")],
        home_dir=tmp_path / "home",
        cwd=EXP_DIR,
    )
    assert res.returncode != 0


@pytest.mark.requires_torch
@pytest.mark.requires_mujoco
def test_t2_f11_03_analyze_replay_bias_cpu_and_missing_log_alpha_error(
    tmp_path: Path, smoke_pilot_template: tuple[Path, Path]
) -> None:
    import torch

    pytest.importorskip("scipy")
    pytest.importorskip("mujoco")
    ws_src, pilot_src = smoke_pilot_template
    exp = ws_src / "experiments" / "sensor_adaptation_v1"
    pilot = tmp_path / "pilot_bias"
    shutil.copytree(pilot_src, pilot)

    out_bias = tmp_path / "bias_out"
    r_bias = run_python(
        [str(exp / "analyze_replay_bias.py"), str(pilot), "--protocol", str(exp / "protocol.json"), "--budget", "64", "--tail", "8", "--device", "cpu", "--out", str(out_bias)],
        home_dir=tmp_path / "home",
        cwd=exp,
    )
    assert r_bias.returncode == 0, f"stdout={r_bias.stdout}\nstderr={r_bias.stderr}"
    assert (out_bias / "critic_bias_rows.csv").is_file()
    assert (out_bias / "critic_bias_summary.csv").is_file()

    # Strip log_alpha from final checkpoint and confirm failure
    ckpt64 = pilot / "transformer_seed42_sensor_noise_replay0" / "checkpoint_64.pt"
    payload = torch.load(ckpt64, map_location="cpu", weights_only=True)
    payload.pop("log_alpha", None)
    torch.save(payload, ckpt64)
    r_fail = run_python(
        [str(exp / "analyze_replay_bias.py"), str(pilot), "--protocol", str(exp / "protocol.json"), "--budget", "64", "--tail", "8", "--device", "cpu"],
        home_dir=tmp_path / "home",
        cwd=exp,
    )
    assert r_fail.returncode != 0


@pytest.mark.requires_torch
def test_t2_f11_04_benchmark_latency_cpu_name_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    if str(EXP_DIR) not in sys.path:
        sys.path.insert(0, str(EXP_DIR))
    import benchmark_latency  # type: ignore

    def _no_powershell(*args, **kwargs):
        raise FileNotFoundError("powershell not installed")

    monkeypatch.setattr(benchmark_latency.subprocess, "check_output", _no_powershell)
    name = benchmark_latency.cpu_name()
    assert isinstance(name, str)


def test_t2_f11_05_camera_ready_assets_reconciled_and_consistent(tmp_path: Path) -> None:
    cam_script = REPO_ROOT / "scripts" / "build_camera_ready_assets.py"
    if cam_script.is_file():
        res = run_python([str(cam_script), "--help"], home_dir=tmp_path / "home")
        assert res.returncode == 0, res.stderr
        assert "--env" in res.stdout
    else:
        # If M3 consolidated/retired build_camera_ready_assets.py, docs must not cite it as active
        exp_readme = (EXP_DIR / "README.md").read_text(encoding="utf-8")
        assert "build_camera_ready_assets.py" not in exp_readme


# =============================================================================
# Feature #12 Boundaries: Operational & Source-Training Scripts (5 tests)
# =============================================================================


@pytest.mark.parametrize(
    "bad_args",
    [
        ["--unseen-map", "--mode", "tracking"],
        ["--mode", "tracking", "--foundation-weights", "foo.pt"],
        ["--mode", "foundation", "--signed-analog-triggers"],
        ["--steps", "0"],
        ["--stall-seconds", "-1"],
    ],
)
def test_t2_f12_01_evaluate_live_driving_invalid_arg_combos(tmp_path: Path, bad_args: list[str]) -> None:
    res = run_python([str(REPO_ROOT / "scripts" / "evaluate_live_driving.py"), *bad_args], home_dir=tmp_path / "home")
    assert res.returncode == 2


def test_t2_f12_02_evaluate_live_driving_replay_parser_and_guard_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = importlib.util.spec_from_file_location("e2e_eval_live_t2", REPO_ROOT / "scripts" / "evaluate_live_driving.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    replay_dir = tmp_path / "Documents" / "Trackmania2020" / "Replays" / "My Replays"
    replay_dir.mkdir(parents=True)
    replay_file = replay_dir / "TrackB_(00_21_345).Replay.Gbx"
    replay_file.write_bytes(b"replay-payload")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    parsed = mod.extract_latest_replay_info()
    assert parsed is not None
    assert parsed["official_race_time_ms"] == 21345
    assert len(parsed["replay_sha256"]) == 64

    with pytest.raises(ValueError):
        mod.MotionStallGuard().update(1.0, [float("nan"), 0.0, 0.0])
    with pytest.raises(ValueError):
        mod.FrameFreshnessGuard().update(1.0, np.array([], dtype=np.uint8), [0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        mod.CycleDeadlineGuard(0.1).update(-0.5)


def test_t2_f12_03_run_ant_sources_rejects_unknown_algorithm_job(tmp_path: Path) -> None:
    res = run_python(
        [str(REPO_ROOT / "scripts" / "run_ant_sources.py"), "--jobs", "unknown_alg:42", "--output-dir", str(tmp_path / "ant")],
        home_dir=tmp_path / "home",
    )
    assert res.returncode != 0
    assert "Unknown algorithm" in res.stderr or "ValueError" in res.stderr


@pytest.mark.requires_mujoco
def test_t2_f12_04_benchmark_fault_env_smoke_and_all_splits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("mujoco")
    monkeypatch.chdir(tmp_path)
    script = REPO_ROOT / "scripts" / "benchmark_fault_env.py"
    res = run_python([str(script), "--smoke_test"], home_dir=tmp_path / "home", cwd=tmp_path, timeout=90)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"

    spec = importlib.util.spec_from_file_location("e2e_bfe_splits", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for split in ("train", "val", "test_latency", "test_dead", "test_sign_flip"):
        env = mod.make_fault_env("HalfCheetah-v5", split=split, seed=42)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (25,)
        env.close()


@pytest.mark.requires_torch
@pytest.mark.requires_mujoco
def test_t2_f12_05_source_recipes_smoke_test_and_load_agent(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    if str(EXP_DIR) not in sys.path:
        sys.path.insert(0, str(EXP_DIR))
    from learner import load_agent  # type: ignore

    for script_name, env_id, alg, out_sub in (
        ("train_transformer_source_recipe_v1.py", "HalfCheetah-v5", "transformer", "tr_src"),
        ("train_reactive_source_recipe_v1.py", "Ant-v5", "reactive", "re_src"),
    ):
        out_dir = tmp_path / out_sub
        res = run_python(
            [
                str(REPO_ROOT / "scripts" / script_name),
                "--env",
                env_id,
                "--smoke_test",
                "--seed",
                "42",
                "--num_threads",
                "1",
                "--output_dir",
                str(out_dir),
            ],
            home_dir=tmp_path / "home",
            cwd=tmp_path,
            timeout=120,
        )
        assert res.returncode == 0, f"{script_name} failed:\nstdout={res.stdout}\nstderr={res.stderr}"
        seed_dir = out_dir / f"{alg}_seed_42"
        ckpt = seed_dir / "best_validation_checkpoint.pt"
        assert ckpt.is_file()
        assert (out_dir / "confirmatory_manifest.csv").is_file()
        assert (seed_dir / "checkpoint_metadata.json").is_file()
        assert (seed_dir / "final_test_summary.json").is_file()
        model = load_agent(alg, ckpt, context=32, device="cpu")
        assert not model.training


# =============================================================================
# Feature #13 Boundaries: Tutorials & Neural Diagnostics (5 tests)
# =============================================================================


def test_t2_f13_01_train_world_model_on_replay_missing_checkpoint_clean(tmp_path: Path) -> None:
    if not has_jax_stack():
        pytest.skip("JAX stack not installed; testing missing checkpoint CLI behavior requires jax")
    missing = tmp_path / "nonexistent.tcpt"
    res = run_python(
        [str(REPO_ROOT / "tutorial" / "train_world_model_on_replay.py"), "--checkpoint", str(missing)],
        home_dir=tmp_path / "empty_home",
    )
    assert "not found" in (res.stdout + res.stderr).lower()


@pytest.mark.requires_jax
def test_t2_f13_02_train_world_model_on_replay_synthetic_tcpt(tmp_path: Path) -> None:
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    from tmrl.core.util import dump

    class _DummyMemory:
        def __len__(self) -> int:
            return 2

        def sample(self):
            obs = (
                np.zeros((2, 1), np.float32),
                np.ones((2, 1), np.float32),
                np.ones((2, 1), np.float32),
                np.zeros((2, 4, 96, 96), np.uint8),
                np.zeros((2, 3), np.float32),
                np.zeros((2, 3), np.float32),
            )
            act = np.zeros((2, 3), np.float32)
            rew = np.ones((2,), np.float32)
            term = np.zeros((2,), np.float32)
            trunc = np.zeros((2,), np.float32)
            return obs, act, rew, obs, term, trunc

    ckpt = tmp_path / "sample.tcpt"
    dump({"memory": _DummyMemory()}, ckpt)
    res = run_python(
        [str(REPO_ROOT / "tutorial" / "train_world_model_on_replay.py"), "--checkpoint", str(ckpt)],
        home_dir=tmp_path / "empty_home",
        timeout=120,
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"


def test_t2_f13_03_dummy_rc_drone_interface_reset_and_step(tmp_path: Path) -> None:
    pytest.importorskip("rtgym")
    code = """
import sys
sys.path.insert(0, r"%s")
from tuto_envs.dummy_rc_drone_interface import DummyRCDroneInterface
import gymnasium as gym
import rtgym
cfg = rtgym.DEFAULT_CONFIG_DICT.copy()
cfg["interface"] = DummyRCDroneInterface
env = gym.make("real-time-gym-ts-v1", config=cfg)
obs, info = env.reset()
for _ in range(3):
    obs, rew, term, trunc, info = env.step(env.action_space.sample())
env.close()
print("DRONE_OK")
""" % str(REPO_ROOT / "tutorial")
    res = run_python(["-c", code], home_dir=tmp_path / "empty_home")
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "DRONE_OK" in res.stdout


@pytest.mark.requires_torch
def test_t2_f13_04_tuto_minimal_pendulum_torch_imports_without_tmrldata(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    code = """
import sys
sys.path.insert(0, r"%s")
import tuto_minimal_pendulum_torch
assert callable(tuto_minimal_pendulum_torch.run_worker)
print("IMPORT_TUTO_OK")
""" % str(REPO_ROOT / "tutorial")
    res = run_python(["-c", code], home_dir=tmp_path / "empty_home")
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "IMPORT_TUTO_OK" in res.stdout


@pytest.mark.requires_torch
def test_t2_f13_05_competition_custom_actor_module_acts(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    home = tmp_path / "home"
    populate_tmrl_data(home, config_dict=make_tmrl_config(algorithm="SAC", interface="TM20FULL", grayscale=True))
    code = """
import sys, numpy as np, gymnasium as gym
sys.path.insert(0, r"%s")
from custom_actor_module import MyActorModule
obs_space = gym.spaces.Tuple((
    gym.spaces.Box(0.0, 100.0, (1,), np.float32),
    gym.spaces.Box(0.0, 6.0, (1,), np.float32),
    gym.spaces.Box(0.0, 10000.0, (1,), np.float32),
    gym.spaces.Box(0.0, 1.0, (4, 64, 64), np.float32),
    gym.spaces.Box(-1.0, 1.0, (3,), np.float32),
    gym.spaces.Box(-1.0, 1.0, (3,), np.float32),
))
act_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)
actor = MyActorModule(obs_space, act_space)
obs = tuple(np.zeros(s.shape, dtype=s.dtype) for s in obs_space)
act = actor.act_(obs, test=True)
assert act_space.contains(act)
print("COMP_ACTOR_OK")
""" % str(REPO_ROOT / "tutorial" / "competition")
    res = run_python(["-c", code], home_dir=home)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "COMP_ACTOR_OK" in res.stdout


# =============================================================================
# Feature #14 Boundaries: Cross-Platform Docs & Reproducibility Guides (5 tests)
# =============================================================================


def test_t2_f14_01_documented_tmrl_cli_flags_match_live_help(tmp_path: Path) -> None:
    help_res = run_python(["-m", "tmrl", "--help"], home_dir=tmp_path / "empty_home")
    assert help_res.returncode == 0, help_res.stderr
    help_flags = set(re.findall(r"--[a-z0-9-]+", help_res.stdout))

    doc_paths = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "readme" / "get_started.md",
        REPO_ROOT / "readme" / "reference_guide.md",
        REPO_ROOT / "docs" / "source" / "cli.rst",
    ]
    mismatches: list[str] = []
    cmd_pat = re.compile(r"python3?\s+-m\s+tmrl\b([^\n`]*)")
    for doc in doc_paths:
        if not doc.is_file():
            continue
        for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            for match in cmd_pat.finditer(line):
                for flag in re.findall(r"--[a-z0-9-]+", match.group(1)):
                    if flag not in help_flags:
                        mismatches.append(f"{doc.relative_to(REPO_ROOT)}:{lineno}: {flag}")
    assert not mismatches, "Documented `python -m tmrl` flags absent from --help:\n" + "\n".join(mismatches)


def test_t2_f14_02_documented_sensor_adaptation_flags_match_help(tmp_path: Path) -> None:
    readme_text = (EXP_DIR / "README.md").read_text(encoding="utf-8")
    for script_name in ("run_study.py", "analyze_extension.py", "analyze_replay_bias.py", "benchmark_latency.py"):
        res = run_python([str(EXP_DIR / script_name), "--help"], home_dir=tmp_path / "home", cwd=EXP_DIR)
        assert res.returncode == 0, res.stderr
        help_flags = set(re.findall(r"--[a-z0-9-]+", res.stdout))
        for match in re.finditer(rf"python(?:\s+-u)?\s+{re.escape(script_name)}\b([^\n`]*)", readme_text):
            for flag in re.findall(r"--[a-z0-9-]+", match.group(1)):
                assert flag in help_flags, f"experiments/sensor_adaptation_v1/README.md documents unknown {flag} for {script_name}"


def test_t2_f14_03_documented_continual_tool_flags_match_help(tmp_path: Path) -> None:
    doc = (REPO_ROOT / "readme" / "continual_foundation_quickstart.md").read_text(encoding="utf-8")
    pat = re.compile(r"-m\s+(tmrl\.tools\.[a-zA-Z0-9_]+)\b([^\n`\\]*)")
    checked = 0
    for match in pat.finditer(doc):
        mod, tail = match.group(1), match.group(2)
        if mod == "tmrl.tools.benchmark_continual_actor" and not has_module("torch"):
            continue
        res = run_python(["-m", mod, "--help"], home_dir=tmp_path / "empty_home")
        assert res.returncode == 0, f"{mod} --help failed: {res.stderr}"
        help_flags = set(re.findall(r"--[a-z0-9-]+", res.stdout))
        for flag in re.findall(r"--[a-z0-9-]+", tail):
            assert flag in help_flags, f"{doc} documents unknown flag {flag} for {mod}"
        checked += 1
    assert checked >= 2


@pytest.mark.requires_sphinx
def test_t2_f14_04_sphinx_html_build_zero_warnings(tmp_path: Path) -> None:
    if not has_sphinx():
        pytest.skip("sphinx-build is not installed in test environment")
    out_html = tmp_path / "sphinx_html"
    res = subprocess.run(
        ["sphinx-build", "-W", "-b", "html", str(REPO_ROOT / "docs" / "source"), str(out_html)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"


def test_t2_f14_05_docs_contain_zero_legacy_moved_module_paths() -> None:
    legacy = (
        "tmrl/tuto/",
        "tmrl/actor.py",
        "tmrl/networking.py",
        "tmrl/training_offline.py",
        "tmrl/custom/custom_memories.py",
        "tmrl/custom/custom_preprocessors.py",
    )
    hits: list[str] = []
    for doc in [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "readme").glob("*.md")), *sorted((REPO_ROOT / "docs" / "source").glob("*.rst"))]:
        text = doc.read_text(encoding="utf-8")
        for needle in legacy:
            if needle in text:
                hits.append(f"{doc.relative_to(REPO_ROOT)} contains legacy path {needle!r}")
    assert not hits, "\n".join(hits)


# =============================================================================
# Feature #15 Boundaries: Workspace & Git Hygiene (5 tests)
# =============================================================================


def test_t2_f15_01_no_binary_checkpoints_or_archives_tracked_in_git() -> None:
    res = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert res.returncode == 0
    bad_exts = (".pt", ".pth", ".tmod", ".tcpt", ".npz", ".zip", ".onnx")
    tracked_bad = [line for line in res.stdout.splitlines() if line.lower().endswith(bad_exts)]
    assert not tracked_bad, f"Tracked binary/checkpoint artifacts found: {tracked_bad}"


def test_t2_f15_02_git_diff_has_no_unstaged_changes() -> None:
    res = subprocess.run(
        ["git", "diff", "--name-only"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert res.returncode == 0
    changed = [line for line in res.stdout.splitlines() if line.strip() and "tests/e2e/" not in line]
    assert not changed, f"Unstaged git changes remain:\n" + "\n".join(changed)


def test_t2_f15_03_gitattributes_protected_files_valid_syntax_and_consistent_eol() -> None:
    protected = [
        REPO_ROOT / "experiments" / "sensor_adaptation_v1" / "run_study.py",
        REPO_ROOT / "scripts" / "benchmark_fault_env.py",
        REPO_ROOT / "scripts" / "confirmatory_logging.py",
        REPO_ROOT / "scripts" / "train_transformer_source_recipe_v1.py",
        REPO_ROOT / "scripts" / "train_reactive_source_recipe_v1.py",
    ]
    for path in protected:
        raw = path.read_bytes()
        ast.parse(raw.decode("utf-8"), filename=str(path))
        has_crlf = b"\r\n" in raw
        has_lf_only = b"\n" in raw.replace(b"\r\n", b"")
        assert not (has_crlf and has_lf_only), f"{path.name} has mixed CRLF and LF line endings"


def test_t2_f15_04_fault_benchmark_results_seed_dirs_complete() -> None:
    for folder_name in ("fault_benchmark_results", "fault_benchmark_results_ant"):
        root = REPO_ROOT / folder_name
        for alg in ("transformer", "reactive"):
            for seed in range(42, 50):
                sdir = root / f"{alg}_seed_{seed}"
                assert sdir.is_dir(), f"Missing {sdir}"
                assert (sdir / "checkpoint_metadata.json").is_file()
                assert (sdir / "final_test_summary.json").is_file()


def test_t2_f15_05_tracked_configs_have_zero_developer_paths() -> None:
    targets = [
        REPO_ROOT / ".readthedocs.yaml",
        *sorted((REPO_ROOT / "experiments" / "sensor_adaptation_v1").glob("*.json")),
    ]
    for path in targets:
        text = path.read_text(encoding="utf-8")
        for bad in ("D:\\Project\\tmrl", "C:\\Users\\felix", "D:\\miniconda3", "C:\\Users\\Yann"):
            assert bad not in text, f"{path.relative_to(REPO_ROOT)} contains hardcoded path {bad}"
