"""Shared fixtures, capability probes, and opaque-box isolation helpers for tests/e2e/."""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_configure(config: pytest.Config) -> None:
    markers = [
        "tier1: Fast offline opaque-box contract tests",
        "tier2: Boundary, negative, and local component integration tests",
        "tier3: Pairwise cross-feature integration tests",
        "tier4: End-to-end external-researcher scenario tests",
        "requires_torch: Requires PyTorch",
        "requires_mujoco: Requires Gymnasium + MuJoCo",
        "requires_jax: Requires JAX + Flax + Optax",
        "requires_tlspyo: Requires tlspyo networking",
        "requires_sphinx: Requires sphinx-build",
        "requires_bash: Requires a working bash shell",
        "external_runtime: Requires external runtime (live TrackMania, CUDA GPU, or WSL2)",
    ]
    for marker in markers:
        config.addinivalue_line("markers", marker)


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def has_jax_stack() -> bool:
    return all(has_module(mod) for mod in ("jax", "flax", "optax"))


def has_working_bash() -> bool:
    bash_path = shutil.which("bash")
    if not bash_path:
        return False
    try:
        res = subprocess.run(
            [bash_path, "-c", "exit 0"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return res.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def has_sphinx() -> bool:
    return shutil.which("sphinx-build") is not None or has_module("sphinx")


def free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_network_blocker(netblock_dir: Path) -> Path:
    netblock_dir.mkdir(parents=True, exist_ok=True)
    sitecustomize = netblock_dir / "sitecustomize.py"
    sitecustomize.write_text(
        """import socket
import urllib.request

def _blocked(*args, **kwargs):
    raise ConnectionError("Network calls are blocked in offline E2E tests")

socket.create_connection = _blocked
socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
urllib.request.urlopen = _blocked
urllib.request.urlretrieve = _blocked
""",
        encoding="utf-8",
    )
    return netblock_dir


def build_isolated_env(
    home_dir: Path,
    extra_env: dict[str, str] | None = None,
    *,
    include_repo: bool = True,
    block_network: bool = False,
    netblock_dir: Path | None = None,
) -> dict[str, str]:
    home_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HOME"] = str(home_dir)
    env["USERPROFILE"] = str(home_dir)
    env["MPLBACKEND"] = "Agg"
    env["WANDB_MODE"] = "disabled"
    env["WANDB_SILENT"] = "true"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    py_paths: list[str] = []
    if block_network:
        blocker = _write_network_blocker(netblock_dir or (home_dir / "_netblock"))
        py_paths.append(str(blocker))
    if include_repo:
        py_paths.append(str(REPO_ROOT))
    if extra_env and "PYTHONPATH" in extra_env:
        py_paths.append(extra_env["PYTHONPATH"])
    elif env.get("PYTHONPATH"):
        py_paths.append(env["PYTHONPATH"])

    if py_paths:
        env["PYTHONPATH"] = os.pathsep.join(py_paths)
    if extra_env:
        for key, value in extra_env.items():
            if key != "PYTHONPATH":
                env[key] = value
    return env


def run_python(
    args: list[str],
    *,
    home_dir: Path,
    cwd: Path = REPO_ROOT,
    timeout: int = 60,
    extra_env: dict[str, str] | None = None,
    block_network: bool = False,
    netblock_dir: Path | None = None,
    include_repo: bool = True,
) -> subprocess.CompletedProcess[str]:
    env = build_isolated_env(
        home_dir,
        extra_env=extra_env,
        include_repo=include_repo,
        block_network=block_network,
        netblock_dir=netblock_dir,
    )
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def make_tmrl_config(
    *,
    algorithm: str = "SAC",
    interface: str = "TM20FULL",
    grayscale: bool = True,
    img_width: int = 64,
    img_height: int = 64,
    img_hist_len: int = 4,
    act_buf_len: int = 2,
    use_foundation_actor: bool | None = None,
    version: str = "0.7.1",
    run_name: str = "e2e_test_run",
    wandb_key: str = "dummy-key-for-test",
    extra_alg: dict[str, Any] | None = None,
    extra_env: dict[str, Any] | None = None,
) -> dict[str, Any]:
    alg_cfg: dict[str, Any] = {
        "ALGORITHM": algorithm,
        "REDQ_N": 4,
        "REDQ_M": 2,
        "REDQ_Q_UPDATES_PER_POLICY_UPDATE": 1,
        "DROPOUT_CRITIC": 0.0,
        "LAYER_NORM_CRITIC": False,
        "LAYER_NORM_ACTOR": False,
        "GAMMA": 0.99,
        "POLYAK": 0.995,
        "ALPHA": 0.2,
        "LEARN_ENTROPY_COEF": True,
        "TARGET_ENTROPY": None,
        "LR_ACTOR": 1e-4,
        "LR_CRITIC": 1e-4,
        "LR_ENTROPY": 1e-4,
        "OPTIMIZER_ACTOR": "adam",
        "OPTIMIZER_CRITIC": "adam",
    }
    if use_foundation_actor is not None:
        alg_cfg["USE_FOUNDATION_ACTOR"] = use_foundation_actor
    if extra_alg:
        alg_cfg.update(extra_alg)

    env_cfg: dict[str, Any] = {
        "RTGYM_INTERFACE": interface,
        "SLEEP_TIME_AT_RESET": 0.1,
        "IMG_HIST_LEN": img_hist_len,
        "IMG_GRAYSCALE": grayscale,
        "IMG_WIDTH": img_width,
        "IMG_HEIGHT": img_height,
        "WINDOW_WIDTH": 256,
        "WINDOW_HEIGHT": 128,
        "REWARD_CONFIG": {
            "END_OF_TRACK": 100.0,
            "CONSTANT_PENALTY": -0.1,
            "CHECK_FORWARD": 50,
            "CHECK_BACKWARD": 10,
            "FAILURE_COUNTDOWN": 10,
            "MIN_STEPS": 70,
            "MAX_STRAY": 60.0,
        },
        "RTGYM_CONFIG": {
            "time_step_duration": 0.05,
            "start_obs_capture": 0.05,
            "time_step_timeout_factor": 1.0,
            "ep_max_length": 1000,
            "act_buf_len": act_buf_len,
            "reset_act_buf": False,
            "benchmark": False,
            "benchmark_polyak": 0.2,
            "wait_on_done": False,
        },
    }
    if extra_env:
        env_cfg.update(extra_env)

    return {
        "__VERSION__": version,
        "RUN_NAME": run_name,
        "BUFFERS_MAXLEN": 100,
        "RW_MAX_SAMPLES_PER_EPISODE": 50,
        "CUDA_TRAINING": False,
        "CUDA_INFERENCE": False,
        "VIRTUAL_GAMEPAD": False,
        "LOCALHOST_WORKER": True,
        "LOCALHOST_TRAINER": True,
        "PUBLIC_IP_SERVER": "127.0.0.1",
        "SAVE_MODEL_EVERY": 0,
        "MEMORY_SIZE": 256,
        "BATCH_SIZE": 8,
        "MAX_EPOCHS": 1,
        "ROUNDS_PER_EPOCH": 1,
        "TRAINING_STEPS_PER_ROUND": 2,
        "UPDATE_MODEL_INTERVAL": 1,
        "UPDATE_BUFFER_INTERVAL": 1,
        "MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP": 1.0,
        "ENVIRONMENT_STEPS_BEFORE_TRAINING": 4,
        "WANDB_PROJECT": "tmrl-e2e",
        "WANDB_ENTITY": "tmrl-e2e",
        "WANDB_KEY": wandb_key,
        "PORT": 55555,
        "LOCAL_PORT_SERVER": 55556,
        "LOCAL_PORT_TRAINER": 55557,
        "LOCAL_PORT_WORKER": 55558,
        "PASSWORD": "e2e-password",
        "TLS": False,
        "TLS_CREDENTIALS_DIRECTORY": "",
        "TLS_HOSTNAME": "localhost",
        "BUFFER_SIZE": 4194304,
        "HEADER_SIZE": 10,
        "ENV": env_cfg,
        "ALG": alg_cfg,
    }


def populate_tmrl_data(
    home_dir: Path,
    config_dict: dict[str, Any] | None = None,
    *,
    with_reward: bool = True,
) -> Path:
    tmrl_data = home_dir / "TmrlData"
    for sub in ("checkpoints", "dataset", "reward", "weights", "config"):
        (tmrl_data / sub).mkdir(parents=True, exist_ok=True)
    cfg = config_dict if config_dict is not None else make_tmrl_config()
    (tmrl_data / "config" / "config.json").write_text(
        json.dumps(cfg, indent=2),
        encoding="utf-8",
    )
    if with_reward:
        traj = np.column_stack(
            [
                np.linspace(0.0, 50.0, 60, dtype=np.float32),
                np.zeros(60, dtype=np.float32),
                np.zeros(60, dtype=np.float32),
            ]
        )
        with (tmrl_data / "reward" / "reward.pkl").open("wb") as handle:
            pickle.dump(traj, handle)
    return tmrl_data


def make_valid_map_manifest(frozen: bool = True) -> dict[str, Any]:
    splits = (
        ("map_train_01", "foundation_train"),
        ("map_val_01", "foundation_validation"),
        ("map_stream_01", "continual_stream"),
        ("map_zeroshot_01", "held_out_zero_shot"),
    )
    maps = []
    for idx, (map_id, split) in enumerate(splits):
        maps.append(
            {
                "map_id": map_id,
                "split": split,
                "track_sha256": f"{idx + 1:064x}",
                "reward_sha256": f"{idx + 11:064x}",
                "ghost_sha256": None if idx % 2 else f"{idx + 21:064x}",
            }
        )
    return {
        "schema_version": 1,
        "frozen": frozen,
        "maps": maps,
    }


def copy_sensor_adaptation_workspace(tmp_path: Path) -> Path:
    """Create an isolated workspace copy of experiments/sensor_adaptation_v1/ in tmp_path."""
    workspace = tmp_path / "sensor_workspace"
    src_dir = REPO_ROOT / "experiments" / "sensor_adaptation_v1"
    dst_dir = workspace / "experiments" / "sensor_adaptation_v1"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        if item.is_file() and item.suffix in {".py", ".json", ".md", ".txt"}:
            shutil.copy2(item, dst_dir / item.name)
    # Also create empty source dirs so callers can populate synthetic checkpoints
    (workspace / "fault_benchmark_results").mkdir(parents=True, exist_ok=True)
    (workspace / "fault_benchmark_results_ant").mkdir(parents=True, exist_ok=True)
    return workspace


def create_synthetic_sensor_checkpoint(
    dest_path: Path,
    algorithm: str,
    *,
    obs_dim: int = 25,
    act_dim: int = 6,
    seed: int = 42,
) -> Path:
    """Create a valid synthetic .pt checkpoint matching frozen_models.py."""
    import torch

    exp_dir = str(REPO_ROOT / "experiments" / "sensor_adaptation_v1")
    if exp_dir not in sys.path:
        sys.path.insert(0, exp_dir)
    import frozen_models  # type: ignore

    torch.manual_seed(seed)
    if algorithm == "transformer":
        model = frozen_models.TransformerRL2Agent(obs_dim=obs_dim, act_dim=act_dim)
    elif algorithm == "reactive":
        model = frozen_models.ReactiveREDQAgent(obs_dim=obs_dim, act_dim=act_dim)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), dest_path)
    return dest_path


def build_portable_jax_bundle(actor: Any) -> dict[str, Any]:
    """Export a valid v4 portable bundle dict from a TorchJAXDreamerActor for offline load tests."""
    from tmrl.custom.torch.jax_dreamer_actor import (
        ACTOR_BUNDLE_MAGIC,
        ACTOR_BUNDLE_VERSION,
        _parameter_checksum,
    )

    params: dict[str, np.ndarray] = {}
    for name in ("conv1", "conv2", "conv3", "conv4"):
        mod = getattr(actor.encoder, name)
        params[f"world_model/encoder/{name}/kernel"] = np.transpose(
            mod.weight.detach().cpu().numpy(), (2, 3, 1, 0)
        ).copy()
        params[f"world_model/encoder/{name}/bias"] = mod.bias.detach().cpu().numpy().copy()
    for mod, prefix in (
        (actor.encoder.telemetry_fc, "world_model/encoder/telemetry_fc"),
        (actor.encoder.fc_fuse, "world_model/encoder/fc_fuse"),
        (actor.rssm.fc_h, "world_model/rssm/fc_h"),
        (actor.rssm.fc_prior_mean, "world_model/rssm/fc_prior_mean"),
        (actor.rssm.fc_prior_std, "world_model/rssm/fc_prior_std"),
        (actor.rssm.fc_post_mean, "world_model/rssm/fc_post_mean"),
        (actor.rssm.fc_post_std, "world_model/rssm/fc_post_std"),
        (actor.policy.hidden_1, "policy/hidden_1"),
        (actor.policy.hidden_2, "policy/hidden_2"),
        (actor.policy.mu_head, "policy/mu_head"),
        (actor.policy.log_std_head, "policy/log_std_head"),
    ):
        params[f"{prefix}/kernel"] = mod.weight.detach().cpu().numpy().T.copy()
        params[f"{prefix}/bias"] = mod.bias.detach().cpu().numpy().copy()
    params["policy/norm/scale"] = actor.policy.norm.weight.detach().cpu().numpy().copy()
    params["policy/norm/bias"] = actor.policy.norm.bias.detach().cpu().numpy().copy()
    return {
        "magic": ACTOR_BUNDLE_MAGIC,
        "version": ACTOR_BUNDLE_VERSION,
        "architecture": actor.architecture_manifest(),
        "parameters": params,
        "parameter_checksum": _parameter_checksum(params),
    }


@pytest.fixture
def isolated_home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    return home


@pytest.fixture(scope="session")
def smoke_pilot_template(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Generate one genuine synthetic-checkpoint `--smoke` pilot study once per session.

    Downstream analyzer/asset tests copy this tree into their own `tmp_path` so each
    test stays isolated and fast without repeating identical MuJoCo `--smoke` runs.
    """
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    base = tmp_path_factory.mktemp("smoke_pilot_template")
    ws = copy_sensor_adaptation_workspace(base)
    exp = ws / "experiments" / "sensor_adaptation_v1"
    for alg in ("transformer", "reactive"):
        create_synthetic_sensor_checkpoint(
            ws / "fault_benchmark_results" / f"{alg}_seed_42" / "best_validation_checkpoint.pt",
            alg,
            obs_dim=25,
            act_dim=6,
        )
    pilot = base / "pilot_smoke"
    res = run_python(
        [
            str(exp / "run_study.py"),
            "--protocol",
            str(exp / "protocol.json"),
            "--output",
            str(pilot),
            "--smoke",
            "--device",
            "cpu",
        ],
        home_dir=base / "home",
        cwd=exp,
        timeout=180,
    )
    assert res.returncode == 0, f"Session smoke pilot failed:\nstdout={res.stdout}\nstderr={res.stderr}"
    return ws, pilot
