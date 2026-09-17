import logging

import rtgym
import torch

import tmrl.config.config_constants as cfg
from tmrl.envs import GenericGymEnv
from tmrl.networking import Trainer, RolloutWorker
from tmrl.training_offline import TorchTrainingOffline
from tmrl.custom.custom_memories import get_local_buffer_sample_tm20_imgs
from tmrl.custom.tm.tm_gym_interfaces import TM2020Interface
from tmrl.custom.tm.tm_preprocessors import obs_preprocessor_tm_act_in_obs
from tmrl.util import partial
from tmrl.world_model.actor import WorldModelActor
from tmrl.world_model.agent import WorldModelAgent
from tmrl.world_model.config import WorldModelConfig
from tmrl.world_model.memory import WorldModelMemory


if cfg.PRAGMA_LIDAR:
    raise RuntimeError("world-model v1 currently targets the original TMRL image environment, not LIDAR.")
if not cfg.GRAYSCALE:
    raise RuntimeError("world-model v1 currently expects grayscale images.")

RAW_WM_CONFIG = cfg.TMRL_CONFIG.get("WORLD_MODEL", {})
WM_CONFIG = WorldModelConfig.from_mapping(RAW_WM_CONFIG)
RUN_NAME = RAW_WM_CONFIG.get("RUN_NAME", "WORLD_MODEL_V1")


def _requested_device(explicit_device, use_cuda):
    return str(explicit_device) if explicit_device else ("cuda:0" if use_cuda else "cpu")


def _validate_device(device, role):
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{role} is configured for {device}, but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build for this Python environment, then verify with: "
                "python -c \"import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())\""
            )
        index = torch.device(device).index
        index = 0 if index is None else index
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"{role} requested {device}, but only {torch.cuda.device_count()} CUDA device(s) are visible."
            )
    return device


TRAINING_DEVICE = _requested_device(
    RAW_WM_CONFIG.get("TRAINING_DEVICE"),
    cfg.CUDA_TRAINING,
)
INFERENCE_DEVICE = _requested_device(
    RAW_WM_CONFIG.get("INFERENCE_DEVICE"),
    cfg.CUDA_INFERENCE,
)


def _device_description(device):
    if not device.startswith("cuda"):
        return "cpu"
    index = torch.device(device).index
    index = 0 if index is None else index
    props = torch.cuda.get_device_properties(index)
    return f"cuda:{index} ({props.name}, {props.total_memory / (1024 ** 3):.1f} GiB)"


MODEL_PATH_WORKER = str(cfg.WEIGHTS_FOLDER / f"{RUN_NAME}.tmod")
MODEL_PATH_TRAINER = str(cfg.WEIGHTS_FOLDER / f"{RUN_NAME}_t.tmod")
CHECKPOINT_PATH = str(cfg.CHECKPOINTS_FOLDER / f"{RUN_NAME}_t.tcpt")

INT = partial(
    TM2020Interface,
    img_hist_len=cfg.IMG_HIST_LEN,
    gamepad=cfg.PRAGMA_GAMEPAD,
    grayscale=cfg.GRAYSCALE,
    resize_to=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT),
)
CONFIG_DICT = rtgym.DEFAULT_CONFIG_DICT.copy()
for key, value in cfg.ENV_CONFIG["RTGYM_CONFIG"].items():
    CONFIG_DICT[key] = value
CONFIG_DICT["interface"] = INT
ENV_CLS = partial(GenericGymEnv, id=cfg.RTGYM_VERSION, gym_kwargs={"config": CONFIG_DICT})

POLICY = partial(
    WorldModelActor,
    wm_config=WM_CONFIG,
    img_hist_len=cfg.IMG_HIST_LEN,
    action_history_len=cfg.ACT_BUF_LEN,
)
SAMPLE_COMPRESSOR = get_local_buffer_sample_tm20_imgs
OBS_PREPROCESSOR = obs_preprocessor_tm_act_in_obs

MEMORY = partial(
    WorldModelMemory,
    memory_size=cfg.TMRL_CONFIG["MEMORY_SIZE"],
    batch_size=WM_CONFIG.batch_size,
    dataset_path=cfg.DATASET_PATH,
    imgs_obs=cfg.IMG_HIST_LEN,
    act_buf_len=cfg.ACT_BUF_LEN,
    crc_debug=cfg.CRC_DEBUG,
    horizon=WM_CONFIG.horizon,
)

AGENT = partial(
    WorldModelAgent,
    wm_config=WM_CONFIG,
    img_hist_len=cfg.IMG_HIST_LEN,
    action_history_len=cfg.ACT_BUF_LEN,
)

TRAINER = partial(
    TorchTrainingOffline,
    env_cls=ENV_CLS,
    memory_cls=MEMORY,
    training_agent_cls=AGENT,
    epochs=cfg.TMRL_CONFIG["MAX_EPOCHS"],
    rounds=cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"],
    steps=cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"],
    update_model_interval=RAW_WM_CONFIG.get("UPDATE_MODEL_INTERVAL", cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"]),
    update_buffer_interval=RAW_WM_CONFIG.get("UPDATE_BUFFER_INTERVAL", cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"]),
    max_training_steps_per_env_step=RAW_WM_CONFIG.get("UPDATES_PER_ENV_STEP", 1.0),
    start_training=RAW_WM_CONFIG.get("START_TRAINING", 5000),
    device=TRAINING_DEVICE,
)


def make_worker(standalone=False):
    device = _validate_device(INFERENCE_DEVICE, "World-model inference")
    logging.info("World-model inference device: %s", _device_description(device))
    return RolloutWorker(
        env_cls=ENV_CLS,
        actor_module_cls=POLICY,
        sample_compressor=SAMPLE_COMPRESSOR,
        device=device,
        server_ip=cfg.SERVER_IP_FOR_WORKER,
        max_samples_per_episode=cfg.RW_MAX_SAMPLES_PER_EPISODE,
        model_path=MODEL_PATH_WORKER,
        obs_preprocessor=OBS_PREPROCESSOR,
        crc_debug=cfg.CRC_DEBUG,
        standalone=standalone,
    )


def make_trainer():
    device = _validate_device(TRAINING_DEVICE, "World-model training")
    if device.startswith("cuda"):
        # Fixed-size image batches benefit from cuDNN autotuning. Keep numerical
        # precision in float32 for v1; mixed precision can be benchmarked later.
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    logging.info("World-model training device: %s", _device_description(device))
    return Trainer(
        training_cls=TRAINER,
        server_ip=cfg.SERVER_IP_FOR_TRAINER,
        model_path=MODEL_PATH_TRAINER,
        checkpoint_path=CHECKPOINT_PATH,
    )
