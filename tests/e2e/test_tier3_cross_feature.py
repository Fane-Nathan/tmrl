"""Tier 3 opaque-box E2E tests: 18 pairwise cross-feature combinations (T3-01..T3-18)."""

from __future__ import annotations

import ast
import importlib.util
import json
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
    copy_sensor_adaptation_workspace,
    create_synthetic_sensor_checkpoint,
    free_tcp_port,
    has_jax_stack,
    has_module,
    make_tmrl_config,
    populate_tmrl_data,
    run_python,
)

pytestmark = [pytest.mark.tier3]

EXP_DIR = REPO_ROOT / "experiments" / "sensor_adaptation_v1"


def test_t3_01_lazy_init_and_cli_help_without_tmrldata_or_pywin32(tmp_path: Path) -> None:
    """#1 x #2: Headless `import tmrl` + `python -m tmrl --help` without TmrlData, pywin32, or network."""
    home = tmp_path / "empty_home"
    block_dir = tmp_path / "block_win"
    block_dir.mkdir()
    (block_dir / "sitecustomize.py").write_text(
        """import socket, sys, urllib.request
def _deny(*a, **k):
    raise ConnectionError("Blocked")
socket.create_connection = _deny
socket.socket.connect = _deny
urllib.request.urlopen = _deny
class _BlockWin:
    BLOCKED = {"win32gui", "win32ui", "win32con", "pywintypes", "dxcam", "vgamepad"}
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.BLOCKED:
            raise ModuleNotFoundError(fullname)
        return None
sys.meta_path.insert(0, _BlockWin())
""",
        encoding="utf-8",
    )
    r_imp = run_python(["-c", "import tmrl; assert callable(tmrl.get_environment)"], home_dir=home, extra_env={"PYTHONPATH": str(block_dir)})
    assert r_imp.returncode == 0, r_imp.stderr
    r_cli = run_python(["-m", "tmrl", "--help"], home_dir=home, extra_env={"PYTHONPATH": str(block_dir)})
    assert r_cli.returncode == 0, r_cli.stderr
    assert "--worker" in r_cli.stdout and not (home / "TmrlData").exists()


@pytest.mark.requires_torch
def test_t3_02_install_and_config_objects_foundation_default_false(tmp_path: Path) -> None:
    """#1 x #3: `--install` with temp TmrlData + `config_objects` defaults `USE_FOUNDATION_ACTOR=False`."""
    pytest.importorskip("torch")
    pytest.importorskip("rtgym")
    home = tmp_path / "home"
    populate_tmrl_data(home, config_dict=make_tmrl_config(algorithm="SAC", interface="TM20FULL", grayscale=True))
    r_inst = run_python(["-m", "tmrl", "--install"], home_dir=home, block_network=True, netblock_dir=tmp_path / "netblock")
    assert r_inst.returncode == 0, r_inst.stderr

    code = """
import tmrl.config.config_objects as cfg_obj
policy_cls = getattr(cfg_obj.POLICY, "func", cfg_obj.POLICY)
assert policy_cls.__name__ == "SquashedGaussianVanillaCNNActor", policy_cls
"""
    r_cfg = run_python(["-c", code], home_dir=home)
    assert r_cfg.returncode == 0, r_cfg.stderr


@pytest.mark.requires_torch
def test_t3_03_cli_config_json_override_and_invalid_json_error(tmp_path: Path) -> None:
    """#2 x #3: `python -m tmrl` `-d/--config` JSON override parsing & error behavior."""
    pytest.importorskip("torch")
    home = tmp_path / "home"
    populate_tmrl_data(home)
    r_ok = run_python(["-m", "tmrl", "--install", "-d", '{"ep_max_length": 42}'], home_dir=home)
    assert r_ok.returncode == 0, r_ok.stderr
    r_bad = run_python(["-m", "tmrl", "--worker", "-d", "{bad-json"], home_dir=home)
    assert r_bad.returncode == 2


@pytest.mark.requires_torch
def test_t3_04_torch_models_decoupled_and_optimizer_validation(tmp_path: Path) -> None:
    """#3 x #7: `tmrl.custom.torch.custom_models` decoupled from `cfg` + `adam`/`adamw` validation."""
    pytest.importorskip("torch")
    code = """
import gymnasium as gym, numpy as np, torch
from tmrl.custom.torch.custom_models import MLPActorCritic, VanillaCNNActorCritic
from tmrl.custom.torch.custom_algorithms import SpinupSACAgent

obs_vec = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)
act_vec = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
agent = SpinupSACAgent(
    obs_vec, act_vec, device="cpu", model_cls=MLPActorCritic,
    optimizer_actor="adamw", optimizer_critic="adam",
    betas_actor=(0.75, 0.98), betas_critic=(0.85, 0.99),
)
assert isinstance(agent.pi_optimizer, torch.optim.AdamW)
assert agent.pi_optimizer.param_groups[0]["betas"] == (0.75, 0.98)
assert agent.q_optimizer.param_groups[0]["betas"] == (0.85, 0.99)
"""
    res = run_python(["-c", code], home_dir=tmp_path / "empty_home")
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"


@pytest.mark.requires_jax
def test_t3_05_jax_models_decoupled_and_nnxsac_step(tmp_path: Path) -> None:
    """#3 x #8: `tmrl.custom.jax.custom_models` decoupled from `cfg` + `NNXSACAgent.train`."""
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    code = """
import gymnasium as gym, jax.numpy as jnp, numpy as np
from tmrl.custom.jax.custom_models import NNXREDQMLPActorCritic
from tmrl.custom.jax.custom_algorithms import NNXSACAgent

obs = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)
act = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
agent = NNXSACAgent(obs, act, model_cls=NNXREDQMLPActorCritic)
batch = (jnp.zeros((4, 4)), jnp.zeros((4, 2)), jnp.ones((4,)), jnp.zeros((4, 4)), jnp.zeros((4,)), jnp.zeros((4,)))
m = agent.train(batch)
assert all(np.isfinite(float(v)) for v in m.values())
"""
    res = run_python(["-c", code], home_dir=tmp_path / "empty_home")
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"


def test_t3_06_tm_interfaces_headless_and_generic_gym_env_contract(tmp_path: Path) -> None:
    """#4 x #6: Headless `tmrl.custom.tm` imports (no pywin32 / cycle_map_macro) + `GenericGymEnv`."""
    code = """
import sys
class _BlockWin:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"win32gui", "win32ui", "win32con", "dxcam", "vgamepad"}:
            raise ModuleNotFoundError(fullname)
        return None
sys.meta_path.insert(0, _BlockWin())
import tmrl.custom.tm.tm_gym_interfaces
import tmrl.custom.tm.utils.window
from tmrl.core.envs import GenericGymEnv
env = GenericGymEnv(id="Pendulum-v1", gym_kwargs={"render_mode": None})
obs, _ = env.reset(seed=0)
assert env.observation_space.contains(obs)
env.close()
"""
    res = run_python(["-c", code], home_dir=tmp_path / "empty_home")
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"


@pytest.mark.requires_torch
@pytest.mark.requires_tlspyo
def test_t3_07_networking_server_trainer_worker_run_synchronous_pendulum(tmp_path: Path) -> None:
    """#4 x #13: End-to-end `Server` + `Trainer` + `RolloutWorker.run_synchronous` on `Pendulum-v1`."""
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

    server = Server(port=pub, password="pw", local_port=srv_l, security=None, keys_dir=None)
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
            server_port=pub,
            local_com_port=tr_l,
            password="pw",
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
            server_port=pub,
            local_port=wk_l,
            password="pw",
            max_samples_per_episode=6,
            model_path=str(tmp_path / "wk.tmod"),
            security=None,
        )

        def _worker_fn() -> None:
            try:
                worker.run_synchronous(initial_steps=8, nb_steps=6, max_steps_per_update=6, end_episodes=False)
            except BaseException as exc:  # pragma: no cover
                worker_errors.append(exc)

        t = threading.Thread(target=_worker_fn, daemon=True)
        t.start()
        trainer.run()
        t.join(timeout=15)
        assert not worker_errors, f"run_synchronous failed: {worker_errors[0]!r}"
        assert (tmp_path / "tr.tmod").is_file()
    finally:
        server.stop()


@pytest.mark.requires_jax
def test_t3_08_prepare_continual_jax_config_consumed_by_config_objects(tmp_path: Path) -> None:
    """#5 x #3: `prepare_continual_jax_config` output is accepted cleanly by `config_objects`."""
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    home = tmp_path / "home"
    populate_tmrl_data(home, config_dict=make_tmrl_config(grayscale=True, wandb_key="secret"))
    cfg_file = home / "TmrlData" / "config" / "config.json"
    res = run_python(
        ["-m", "tmrl.tools.prepare_continual_jax_config", "--source", str(cfg_file), "--output", str(cfg_file), "--scrub-wandb-key", "--overwrite"],
        home_dir=home,
    )
    assert res.returncode == 0, res.stderr
    r_obj = run_python(["-c", "import tmrl.config.config_objects as o; assert o.IS_JAX_DREAMER"], home_dir=home)
    assert r_obj.returncode == 0, r_obj.stderr


@pytest.mark.requires_jax
@pytest.mark.requires_torch
def test_t3_09_validate_continual_trainer_stack_end_to_end(tmp_path: Path) -> None:
    """#5 x #8: `validate_continual_trainer_stack` exercises `JAXDreamerAgent` + `continual_checkpoints`."""
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    pytest.importorskip("torch")
    home = tmp_path / "home"
    populate_tmrl_data(home, config_dict=make_tmrl_config(grayscale=True))
    cfg_file = home / "TmrlData" / "config" / "config.json"
    assert run_python(["-m", "tmrl.tools.prepare_continual_jax_config", "--source", str(cfg_file), "--output", str(cfg_file), "--overwrite"], home_dir=home).returncode == 0
    stack_out = tmp_path / "stack"
    res = run_python(["-m", "tmrl.tools.validate_continual_trainer_stack", "--output-directory", str(stack_out)], home_dir=home, timeout=120)
    assert res.returncode == 0, res.stderr
    assert (stack_out / "trainer_stack.json").is_file()


@pytest.mark.requires_torch
def test_t3_10_benchmark_continual_actor_torch_jax_mirror(tmp_path: Path) -> None:
    """#5 x #7: `benchmark_continual_actor --runtime torch` benchmarks `TorchJAXDreamerActor`."""
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
    assert res.returncode == 0, res.stderr
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["passed"] is True and payload["runtime"] == "torch"


def test_t3_11_live_driving_and_pipeline_scripts_cli_and_tm_imports(tmp_path: Path) -> None:
    """#6 x #12: `evaluate_live_driving.py` and `benchmark_live_pipeline.py` `--help` & imports headless."""
    for script in ("evaluate_live_driving.py", "benchmark_live_pipeline.py"):
        res = run_python([str(REPO_ROOT / "scripts" / script), "--help"], home_dir=tmp_path / "empty_home")
        assert res.returncode == 0, f"{script} --help failed: {res.stderr}"


@pytest.mark.requires_torch
@pytest.mark.requires_mujoco
def test_t3_12_test_contracts_imports_benchmark_fault_env_and_passes(tmp_path: Path) -> None:
    """#9 x #10: `experiments/sensor_adaptation_v1/test_contracts.py` validates `benchmark_fault_env.py` boundary."""
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


@pytest.mark.requires_torch
@pytest.mark.requires_mujoco
def test_t3_13_run_study_smoke_consumed_by_analyzers_and_latency(
    tmp_path: Path, smoke_pilot_template: tuple[Path, Path]
) -> None:
    """#9 x #11: `run_study.py --smoke` outputs consumed cleanly by `analyze_results.py`, `analyze_replay_bias.py`, `benchmark_latency.py`."""
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    pytest.importorskip("scipy")
    ws, pilot_src = smoke_pilot_template
    exp = ws / "experiments" / "sensor_adaptation_v1"
    pilot = tmp_path / "pilot"
    shutil.copytree(pilot_src, pilot)
    assert run_python([str(exp / "analyze_results.py"), str(pilot)], home_dir=tmp_path / "home", cwd=exp).returncode == 0
    assert run_python(
        [str(exp / "analyze_replay_bias.py"), str(pilot), "--protocol", str(exp / "protocol.json"), "--budget", "64", "--tail", "8", "--device", "cpu"],
        home_dir=tmp_path / "home",
        cwd=exp,
    ).returncode == 0
    lat_out = tmp_path / "latency.json"
    r_lat = run_python(
        [
            str(exp / "benchmark_latency.py"),
            "--trajectories",
            str(pilot),
            "--sources",
            str(ws / "fault_benchmark_results"),
            "--seed",
            "42",
            "--steps",
            "8",
            "--warmup",
            "2",
            "--threads",
            "1",
            "--out",
            str(lat_out),
        ],
        home_dir=tmp_path / "home",
        cwd=exp,
        timeout=90,
    )
    assert r_lat.returncode == 0, r_lat.stderr
    lat = json.loads(lat_out.read_text(encoding="utf-8"))
    tr = next(r for r in lat["results"] if r["algorithm"] == "transformer")
    assert tr["parameters"]["inference_path_upper_bound"] < tr["parameters"]["total"]


@pytest.mark.requires_torch
@pytest.mark.requires_mujoco
def test_t3_14_source_recipes_smoke_and_gitattributes_text_unset(tmp_path: Path) -> None:
    """#10 x #12: Source recipes `--smoke_test` + `-text` gitattributes protection."""
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    assert run_python([str(REPO_ROOT / "scripts" / "benchmark_fault_env.py"), "--smoke_test"], home_dir=tmp_path / "home", cwd=tmp_path, timeout=90).returncode == 0
    assert run_python(
        [str(REPO_ROOT / "scripts" / "train_transformer_source_recipe_v1.py"), "--env", "HalfCheetah-v5", "--smoke_test", "--seed", "42", "--num_threads", "1", "--output_dir", str(tmp_path / "src_tr")],
        home_dir=tmp_path / "home",
        cwd=tmp_path,
        timeout=120,
    ).returncode == 0
    attr = subprocess.run(
        ["git", "check-attr", "text", "--", "scripts/benchmark_fault_env.py", "scripts/train_transformer_source_recipe_v1.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert attr.returncode == 0 and "text: unset" in attr.stdout


@pytest.mark.requires_torch
@pytest.mark.requires_mujoco
def test_t3_15_paper_assets_and_provenance_outputs_consistent(
    tmp_path: Path, smoke_pilot_template: tuple[Path, Path]
) -> None:
    """#11 x #12: `build_paper_assets.py` (and reconciled camera-ready assets if present) consume verified outputs."""
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    ws, pilot_src = smoke_pilot_template
    exp = ws / "experiments" / "sensor_adaptation_v1"
    pilot = tmp_path / "pilot"
    shutil.copytree(pilot_src, pilot)
    assert run_python([str(exp / "analyze_results.py"), str(pilot)], home_dir=tmp_path / "home", cwd=exp).returncode == 0
    manifest_data = json.loads((pilot / "manifest.json").read_text(encoding="utf-8"))
    manifest_data["smoke"] = False
    (pilot / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")
    assets = tmp_path / "assets"
    assert run_python([str(exp / "build_paper_assets.py"), str(pilot), str(assets)], home_dir=tmp_path / "home", cwd=exp).returncode == 0
    assert (assets / "outcomes_table.tex").is_file()


def test_t3_16_model_provenance_paths_resolve_and_results_intact() -> None:
    """#9 x #15: `model_provenance.json` repo-relative paths resolve & `fault_benchmark_results*` intact."""
    entries = json.loads((EXP_DIR / "model_provenance.json").read_text(encoding="utf-8"))
    for entry in entries:
        src_path = REPO_ROOT / entry["source"]
        assert src_path.is_file(), f"Provenance source path missing: {entry['source']}"
    assert (REPO_ROOT / "fault_benchmark_results" / "confirmatory_manifest.csv").is_file()
    assert (REPO_ROOT / "fault_benchmark_results_ant" / "confirmatory_manifest.csv").is_file()


def test_t3_17_documented_tmrl_cli_flags_parity_with_live_help(tmp_path: Path) -> None:
    """#2 x #14: Every `python -m tmrl` flag in docs exists in `python -m tmrl --help`."""
    res = run_python(["-m", "tmrl", "--help"], home_dir=tmp_path / "empty_home")
    assert res.returncode == 0, res.stderr
    valid_flags = set(re.findall(r"--[a-z0-9-]+", res.stdout))
    for doc in (REPO_ROOT / "README.md", REPO_ROOT / "readme" / "get_started.md", REPO_ROOT / "readme" / "reference_guide.md", REPO_ROOT / "docs" / "source" / "cli.rst"):
        if not doc.is_file():
            continue
        for match in re.finditer(r"python3?\s+-m\s+tmrl\b([^\n`]*)", doc.read_text(encoding="utf-8")):
            for flag in re.findall(r"--[a-z0-9-]+", match.group(1)):
                assert flag in valid_flags, f"{doc.relative_to(REPO_ROOT)} cites unknown flag {flag}"


def test_t3_18_tutorial_and_doc_module_links_resolve_on_disk() -> None:
    """#13 x #14: `tutorial/README.md`, `readme/tuto_library.md`, `readme/competition.md`, `docs/source/index.rst` resolve."""
    for doc in (REPO_ROOT / "tutorial" / "README.md", REPO_ROOT / "readme" / "competition.md", REPO_ROOT / "readme" / "tuto_library.md", REPO_ROOT / "docs" / "source" / "index.rst"):
        text = doc.read_text(encoding="utf-8")
        assert "tmrl/tuto/" not in text, f"{doc.relative_to(REPO_ROOT)} still references moved tmrl/tuto/"
    tuto_readme = (REPO_ROOT / "tutorial" / "README.md").read_text(encoding="utf-8")
    for fname in ("tuto.py", "debug_all_neural_networks.py", "train_world_model_on_replay.py"):
        assert fname in tuto_readme and (REPO_ROOT / "tutorial" / fname).is_file()
