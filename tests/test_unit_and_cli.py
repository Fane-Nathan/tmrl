"""Repository unit and CLI smoke tests (runnable on CPU without GPU or live TrackMania)."""

import importlib.util
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(args: list[str], tmp_path: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    home = tmp_path / "isolated_home"
    home.mkdir(parents=True, exist_ok=True)
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_import_tmrl_and_cli_help_headless(tmp_path: Path) -> None:
    res = _run_cli(["-m", "tmrl", "--help"], tmp_path)
    assert res.returncode == 0, res.stderr
    for flag in ("--server", "--trainer", "--worker", "--test", "--benchmark", "--install"):
        assert flag in res.stdout


def test_core_util_wrappers_and_buffer(tmp_path: Path) -> None:
    from tmrl.core.memory import check_samples_crc
    from tmrl.core.networking import Buffer
    from tmrl.core.util import (
        dump,
        load,
        load_json,
        partial,
        partial_from_args,
        partial_to_dict,
        prod,
        save_json,
    )
    from tmrl.core.wrappers import AffineObservationWrapper, Float64ToFloat32

    assert prod([2, 3, 4]) == 24

    def _add(a: int, b: int = 1) -> int:
        return a + b

    p = partial(_add, a=5, b=3)
    d = partial_to_dict(p)
    assert d["a"] == 5 and d["b"] == 3
    p2 = partial_from_args(_add, {"a": "5", "b": "3"})
    assert p2() == 8

    json_file = tmp_path / "sample.json"
    save_json({"k": 42}, json_file)
    assert load_json(json_file) == {"k": 42}

    bin_file = tmp_path / "obj.pkl"
    dump({"arr": [1, 2, 3]}, bin_file)
    assert load(bin_file) == {"arr": [1, 2, 3]}

    buf = Buffer(maxlen=4)
    buf.append_sample((np.zeros(2, np.float32), np.ones(3, np.float32), 1.0, False, False, {}))
    assert len(buf) == 1
    buf.clear()
    assert len(buf) == 0

    check_samples_crc(
        np.zeros(2), np.ones(3), np.zeros(2), 1.0, False, False,
        np.zeros(2), np.ones(3), np.zeros(2), 1.0, False, False,
        10, 2,
    )

    class _Float64Env(gym.Env):
        def __init__(self) -> None:
            self.observation_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float64)
            self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)

        def reset(self, *, seed=None, options=None):
            return np.array([0.5, -0.5], dtype=np.float64), {}

        def step(self, action):
            return np.array([0.25, 0.75], dtype=np.float64), 1.0, False, False, {}

    wrapped = AffineObservationWrapper(Float64ToFloat32(_Float64Env()), shift=1.0, scale=2.0)
    obs, _ = wrapped.reset()
    assert obs.dtype == np.float32
    np.testing.assert_allclose(obs, [3.0, 1.0])


def test_continual_tools_cli_roundtrip(tmp_path: Path) -> None:
    src_cfg = tmp_path / "config.json"
    out_cfg = tmp_path / "config_jax.json"
    src_cfg.write_text(
        json.dumps(
            {
                "RUN_NAME": "unit_test",
                "ENV": {"RTGYM_INTERFACE": "TM20FULL", "IMG_GRAYSCALE": True, "IMG_HIST_LEN": 4, "IMG_WIDTH": 64, "IMG_HEIGHT": 64, "RTGYM_CONFIG": {"act_buf_len": 2}},
                "ALG": {"ALGORITHM": "DREAMER"},
                "WANDB_KEY": "secret",
            }
        ),
        encoding="utf-8",
    )
    res_cfg = _run_cli(
        [
            "-m",
            "tmrl.tools.prepare_continual_jax_config",
            "--source",
            str(src_cfg),
            "--output",
            str(out_cfg),
            "--scrub-wandb-key",
        ],
        tmp_path,
    )
    assert res_cfg.returncode == 0, res_cfg.stderr
    written = json.loads(out_cfg.read_text(encoding="utf-8"))
    assert written["ALG"]["ALGORITHM"] == "CONTINUAL_DREAMER_JAX"
    assert written["WANDB_KEY"] == ""

    frozen_dir = tmp_path / "frozen"
    res_frz = _run_cli(
        ["-m", "tmrl.tools.freeze_continual_baseline", "--source", str(out_cfg), "--output-dir", str(frozen_dir)],
        tmp_path,
    )
    assert res_frz.returncode == 0, res_frz.stderr
    assert (frozen_dir / "baseline_manifest.json").is_file()


def test_tm_utils_controls_preprocessing_and_reward(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tmrl.custom.tm.utils.camera_preprocessing import preprocess_camera_frame
    from tmrl.custom.tm.utils.compute_reward import RewardFunction
    from tmrl.custom.tm.utils.control_gamepad import signed_trigger_controls

    assert signed_trigger_controls([-1.0, 1.0, -0.5]) == [0.0, 1.0, -0.5]
    frame = np.full((64, 64, 4), 120, dtype=np.uint8)
    out = preprocess_camera_frame(frame, resize_to=(32, 32), grayscale=True)
    assert out.shape == (32, 32) and out.dtype == np.uint8

    traj = np.column_stack([np.arange(0, 15, dtype=np.float32), np.zeros(15, np.float32), np.zeros(15, np.float32)])
    rew_file = tmp_path / "reward.pkl"
    rew_file.write_bytes(pickle.dumps(traj))
    monkeypatch.setenv("TMRL_UNIVERSAL_REWARD", "0")
    rf = RewardFunction(str(rew_file), nb_obs_forward=5)
    r, term = rf.compute_reward(np.array([3.0, 0.0, 0.0], np.float32))
    assert r > 0.0 and term is False


def test_torch_models_and_sac_optimizer_betas() -> None:
    from tmrl.custom.torch.custom_algorithms import REDQSACAgent, SpinupSACAgent
    from tmrl.custom.torch.custom_models import MLPActorCritic, REDQMLPActorCritic, VanillaCNNActorCritic

    obs_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)
    act_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
    sac = SpinupSACAgent(
        obs_space,
        act_space,
        device="cpu",
        model_cls=MLPActorCritic,
        optimizer_actor="adamw",
        optimizer_critic="adam",
        betas_actor=(0.8, 0.99),
        betas_critic=(0.85, 0.995),
    )
    assert isinstance(sac.pi_optimizer, torch.optim.AdamW)
    assert sac.pi_optimizer.param_groups[0]["betas"] == (0.8, 0.99)
    assert sac.q_optimizer.param_groups[0]["betas"] == (0.85, 0.995)

    redq = REDQSACAgent(
        obs_space,
        act_space,
        device="cpu",
        model_cls=REDQMLPActorCritic,
        optimizer_actor="adam",
        optimizer_critic="adamw",
        betas_actor=(0.7, 0.95),
        betas_critic=(0.75, 0.96),
    )
    assert redq.pi_optimizer.param_groups[0]["betas"] == (0.7, 0.95)
    assert redq.q_optimizer.param_groups[0]["betas"] == (0.75, 0.96)

    img_obs = gym.spaces.Tuple(
        (
            gym.spaces.Box(0.0, 100.0, (1,), np.float32),
            gym.spaces.Box(0.0, 6.0, (1,), np.float32),
            gym.spaces.Box(0.0, 10000.0, (1,), np.float32),
            gym.spaces.Box(0.0, 1.0, (2, 64, 80), np.float32),
            gym.spaces.Box(-1.0, 1.0, (3,), np.float32),
            gym.spaces.Box(-1.0, 1.0, (3,), np.float32),
        )
    )
    img_act = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)
    cnn_ac = VanillaCNNActorCritic(img_obs, img_act)
    sample = tuple(np.zeros(s.shape, dtype=s.dtype) for s in img_obs)
    act = cnn_ac.actor.act_(sample, test=True)
    assert act.shape == (3,)


def test_scripts_fault_env_logging_ant_runner_and_guards(tmp_path: Path) -> None:
    def _load_script(name: str):
        path = REPO_ROOT / "scripts" / name
        spec = importlib.util.spec_from_file_location(f"unit_{path.stem}", path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    bfe = _load_script("benchmark_fault_env.py")

    class _DummyEnv(gym.Env):
        def __init__(self) -> None:
            self.observation_space = gym.spaces.Box(-2.0, 2.0, (3,), np.float32)
            self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

        def reset(self, *, seed=None, options=None):
            return np.zeros((3,), np.float32), {}

        def step(self, action):
            return np.ones((3,), np.float32), 1.0, False, False, {}

    with pytest.raises(ValueError, match="does not implement observation faults"):
        bfe.FaultInjectionWrapper(_DummyEnv(), fixed_fault="sensor_noise")

    wrapped = bfe.FaultInjectionWrapper(_DummyEnv(), split="train", seed=42)
    obs, info = wrapped.reset(seed=42)
    assert obs.shape == (7,) and info["fault_type"] in ("action_scale", "action_noise")

    clog = _load_script("confirmatory_logging.py")
    f = tmp_path / "x.txt"
    f.write_text("hello", encoding="utf-8")
    assert len(clog.compute_file_sha256(f)) == 64

    ant = _load_script("run_ant_sources.py")
    assert ant.free_gb() > 0.0

    eval_live = _load_script("evaluate_live_driving.py")
    assert eval_live.CycleDeadlineGuard(0.05, consecutive_limit=2).update(0.1) is False
