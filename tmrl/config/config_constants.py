"""Configuration constants loaded from ``config.json`` with safe defaults."""

from copy import deepcopy
import json
import logging
import os
from pathlib import Path
import platform

from packaging import version

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

__compatibility__ = "0.6.0"
__err_msg = (
    "Perform a clean installation:\n"
    "(1) Uninstall TMRL,\n"
    "(2) Delete the TmrlData folder,\n"
    "(3) Reinstall TMRL."
)

SYSTEM = platform.system()
RTGYM_VERSION = "real-time-gym-v1" if SYSTEM == "Windows" else "real-time-gym-ts-v1"

_DEFAULT_TMRL_CONFIG = {
    "RUN_NAME": "SAC_4_imgs_pretrained",
    "RESET_TRAINING": False,
    "BUFFERS_MAXLEN": 500000,
    "RW_MAX_SAMPLES_PER_EPISODE": 1000,
    "CUDA_TRAINING": True,
    "CUDA_INFERENCE": False,
    "VIRTUAL_GAMEPAD": True,
    "LOCALHOST_WORKER": True,
    "LOCALHOST_TRAINER": True,
    "PUBLIC_IP_SERVER": "0.0.0.0",
    "PASSWORD": "YourRandomPasswordHere",
    "TLS": False,
    "TLS_HOSTNAME": "default",
    "TLS_CREDENTIALS_DIRECTORY": "",
    "WANDB_PROJECT": "tmrl",
    "WANDB_ENTITY": "tmrl",
    "WANDB_KEY": "",
    "PORT": 55555,
    "LOCAL_PORT_SERVER": 55556,
    "LOCAL_PORT_TRAINER": 55557,
    "LOCAL_PORT_WORKER": 55558,
    "BUFFER_SIZE": 536870912,
    "HEADER_SIZE": 12,
    "MAX_EPOCHS": 10000,
    "ROUNDS_PER_EPOCH": 100,
    "TRAINING_STEPS_PER_ROUND": 200,
    "MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP": 4.0,
    "ENVIRONMENT_STEPS_BEFORE_TRAINING": 1000,
    "UPDATE_MODEL_INTERVAL": 200,
    "UPDATE_BUFFER_INTERVAL": 200,
    "SAVE_MODEL_EVERY": 0,
    "MEMORY_SIZE": 1000000,
    "BATCH_SIZE": 256,
    "DEBUG_MODE": False,
    "ALG": {
        "ALGORITHM": "SAC",
        "USE_FOUNDATION_ACTOR": False,
        "LEARN_ENTROPY_COEF": False,
        "LR_ACTOR": 0.00001,
        "LR_CRITIC": 0.00005,
        "LR_ENTROPY": 0.0003,
        "GAMMA": 0.995,
        "POLYAK": 0.995,
        "TARGET_ENTROPY": -0.5,
        "ALPHA": 0.01,
        "REDQ_N": 10,
        "REDQ_M": 2,
        "REDQ_Q_UPDATES_PER_POLICY_UPDATE": 20,
        "OPTIMIZER_ACTOR": "adam",
        "OPTIMIZER_CRITIC": "adam",
        "BETAS_ACTOR": [0.997, 0.997],
        "BETAS_CRITIC": [0.997, 0.997],
        "L2_ACTOR": 0.0,
        "L2_CRITIC": 0.0,
    },
    "ENV": {
        "RTGYM_INTERFACE": "TM20FULL",
        "WINDOW_WIDTH": 256,
        "WINDOW_HEIGHT": 128,
        "IMG_WIDTH": 64,
        "IMG_HEIGHT": 64,
        "IMG_GRAYSCALE": True,
        "SLEEP_TIME_AT_RESET": 1.5,
        "IMG_HIST_LEN": 4,
        "CAPTURE_BACKEND": "gdi",
        "CAPTURE_FPS": 60,
        "CAPTURE_KEEP_ON_TOP": False,
        "LINUX_X_OFFSET": 64,
        "LINUX_Y_OFFSET": 70,
        "IMG_SCALE_CHECK_ENV": 1.0,
        "RTGYM_CONFIG": {
            "time_step_duration": 0.05,
            "start_obs_capture": 0.04,
            "time_step_timeout_factor": 1.0,
            "act_buf_len": 2,
            "benchmark": False,
            "wait_on_done": True,
            "ep_max_length": 1000,
            "interface_kwargs": {"save_replays": False},
        },
        "REWARD_CONFIG": {
            "END_OF_TRACK": 100.0,
            "CONSTANT_PENALTY": 0.0,
            "CHECK_FORWARD": 500,
            "CHECK_BACKWARD": 10,
            "FAILURE_COUNTDOWN": 10,
            "MIN_STEPS": 70,
            "MAX_STRAY": 100.0,
        },
    },
    "__VERSION__": "0.6.0",
}


def _default_tmrl_config() -> dict:
    """Return a fresh copy of the canonical fallback TMRL configuration."""
    return deepcopy(_DEFAULT_TMRL_CONFIG)


def get_tmrl_folder() -> Path:
    """Resolve the TMRL data folder (honors ``TMRL_DATA_DIR`` when set)."""
    override = os.environ.get("TMRL_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / "TmrlData"


def get_config_file(tmrl_folder: Path | None = None) -> Path:
    """Resolve ``config.json`` path (honors ``TMRL_CONFIG_PATH`` when set)."""
    override = os.environ.get("TMRL_CONFIG_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    base = Path(tmrl_folder).expanduser() if tmrl_folder is not None else get_tmrl_folder()
    return base / "config" / "config.json"


def _validate_config_version(tmrl_config: dict) -> str:
    if "__VERSION__" not in tmrl_config:
        raise RuntimeError("config.json is outdated. " + __err_msg)
    config_version = str(tmrl_config["__VERSION__"])
    if version.parse(config_version) < version.parse(__compatibility__):
        raise RuntimeError(
            f"config.json version ({config_version}) must be >= {__compatibility__}. "
            + __err_msg
        )
    return config_version


def load_config(config_file: Path | str | None = None) -> tuple[dict, bool]:
    """Load ``config.json`` if present, or return in-memory defaults without creating files."""
    explicit_path = config_file is not None
    target = Path(config_file).expanduser() if explicit_path else get_config_file()

    if target.is_file():
        with open(target, encoding="utf-8") as fid:
            loaded = json.load(fid)
        _validate_config_version(loaded)
        return loaded, True

    if explicit_path:
        raise FileNotFoundError(f"TMRL config file not found: {target}")

    logging.debug(
        "TMRL config file not found at %s; using built-in default configuration.",
        target,
    )
    return _default_tmrl_config(), False


def _apply_config(
    tmrl_config: dict,
    tmrl_folder: Path,
    config_file: Path,
    loaded: bool,
) -> dict:
    """Bind module-level configuration constants from ``tmrl_config``."""
    global TMRL_FOLDER, CHECKPOINTS_FOLDER, DATASET_FOLDER, REWARD_FOLDER
    global WEIGHTS_FOLDER, CONFIG_FOLDER, CONFIG_FILE, TMRL_CONFIG, CONFIG_LOADED
    global CONFIG_VERSION, RUN_NAME, BUFFERS_MAXLEN, RW_MAX_SAMPLES_PER_EPISODE
    global PRAGMA_RNN, CUDA_TRAINING, CUDA_INFERENCE, PRAGMA_GAMEPAD
    global LOCALHOST_WORKER, LOCALHOST_TRAINER, PUBLIC_IP_SERVER
    global SERVER_IP_FOR_WORKER, SERVER_IP_FOR_TRAINER
    global ENV_CONFIG, RTGYM_INTERFACE, PRAGMA_LIDAR, PRAGMA_PROGRESS
    global LIDAR_BLACK_THRESHOLD, REWARD_CONFIG, SLEEP_TIME_AT_RESET
    global IMG_HIST_LEN, ACT_BUF_LEN, WINDOW_WIDTH, WINDOW_HEIGHT
    global GRAYSCALE, IMG_WIDTH, IMG_HEIGHT
    global CAPTURE_BACKEND, CAPTURE_FPS, CAPTURE_KEEP_ON_TOP
    global LINUX_X_OFFSET, LINUX_Y_OFFSET, IMG_SCALE_CHECK_ENV
    global CRC_DEBUG, CRC_DEBUG_SAMPLES, PROFILE_TRAINER, SYNCHRONIZE_CUDA, DEBUG_MODE
    global PATH_DATA, MODEL_HISTORY, MODEL_PATH_WORKER, MODEL_PATH_SAVE_HISTORY
    global MODEL_PATH_TRAINER, CHECKPOINT_PATH, DATASET_PATH, REWARD_PATH
    global WANDB_RUN_ID, WANDB_PROJECT, WANDB_ENTITY, WANDB_KEY
    global PRINT_BYTESIZES, PORT, LOCAL_PORT_SERVER, LOCAL_PORT_TRAINER
    global LOCAL_PORT_WORKER, PASSWORD, SECURITY, CREDENTIALS_DIRECTORY
    global HOSTNAME, BUFFER_SIZE, HEADER_SIZE

    TMRL_FOLDER = tmrl_folder
    CHECKPOINTS_FOLDER = TMRL_FOLDER / "checkpoints"
    DATASET_FOLDER = TMRL_FOLDER / "dataset"
    REWARD_FOLDER = TMRL_FOLDER / "reward"
    WEIGHTS_FOLDER = TMRL_FOLDER / "weights"
    CONFIG_FOLDER = TMRL_FOLDER / "config"
    CONFIG_FILE = config_file

    TMRL_CONFIG = tmrl_config
    CONFIG_LOADED = loaded
    CONFIG_VERSION = _validate_config_version(TMRL_CONFIG)

    # General
    RUN_NAME = TMRL_CONFIG["RUN_NAME"]
    BUFFERS_MAXLEN = TMRL_CONFIG["BUFFERS_MAXLEN"]
    RW_MAX_SAMPLES_PER_EPISODE = TMRL_CONFIG["RW_MAX_SAMPLES_PER_EPISODE"]

    PRAGMA_RNN = False

    CUDA_TRAINING = TMRL_CONFIG["CUDA_TRAINING"]
    CUDA_INFERENCE = TMRL_CONFIG["CUDA_INFERENCE"]

    PRAGMA_GAMEPAD = TMRL_CONFIG["VIRTUAL_GAMEPAD"]

    LOCALHOST_WORKER = TMRL_CONFIG["LOCALHOST_WORKER"]
    LOCALHOST_TRAINER = TMRL_CONFIG["LOCALHOST_TRAINER"]
    PUBLIC_IP_SERVER = TMRL_CONFIG["PUBLIC_IP_SERVER"]

    SERVER_IP_FOR_WORKER = PUBLIC_IP_SERVER if not LOCALHOST_WORKER else "127.0.0.1"
    SERVER_IP_FOR_TRAINER = PUBLIC_IP_SERVER if not LOCALHOST_TRAINER else "127.0.0.1"

    # Environment
    ENV_CONFIG = TMRL_CONFIG["ENV"]
    RTGYM_INTERFACE = ENV_CONFIG["RTGYM_INTERFACE"]
    PRAGMA_LIDAR = RTGYM_INTERFACE.endswith("LIDAR")
    PRAGMA_PROGRESS = RTGYM_INTERFACE.endswith("LIDARPROGRESS")
    if PRAGMA_PROGRESS:
        PRAGMA_LIDAR = True
    LIDAR_BLACK_THRESHOLD = [55, 55, 55]
    REWARD_CONFIG = ENV_CONFIG["REWARD_CONFIG"]
    SLEEP_TIME_AT_RESET = ENV_CONFIG["SLEEP_TIME_AT_RESET"]
    IMG_HIST_LEN = ENV_CONFIG["IMG_HIST_LEN"]
    ACT_BUF_LEN = ENV_CONFIG["RTGYM_CONFIG"]["act_buf_len"]
    WINDOW_WIDTH = ENV_CONFIG["WINDOW_WIDTH"]
    WINDOW_HEIGHT = ENV_CONFIG["WINDOW_HEIGHT"]
    GRAYSCALE = ENV_CONFIG["IMG_GRAYSCALE"] if "IMG_GRAYSCALE" in ENV_CONFIG else False
    IMG_WIDTH = ENV_CONFIG["IMG_WIDTH"] if "IMG_WIDTH" in ENV_CONFIG else 64
    IMG_HEIGHT = ENV_CONFIG["IMG_HEIGHT"] if "IMG_HEIGHT" in ENV_CONFIG else 64
    CAPTURE_BACKEND = ENV_CONFIG.get("CAPTURE_BACKEND", "gdi")
    CAPTURE_FPS = int(ENV_CONFIG.get("CAPTURE_FPS", 60))
    CAPTURE_KEEP_ON_TOP = bool(ENV_CONFIG.get("CAPTURE_KEEP_ON_TOP", False))
    LINUX_X_OFFSET = ENV_CONFIG["LINUX_X_OFFSET"] if "LINUX_X_OFFSET" in ENV_CONFIG else 64
    LINUX_Y_OFFSET = ENV_CONFIG["LINUX_Y_OFFSET"] if "LINUX_Y_OFFSET" in ENV_CONFIG else 70
    IMG_SCALE_CHECK_ENV = (
        ENV_CONFIG["IMG_SCALE_CHECK_ENV"] if "IMG_SCALE_CHECK_ENV" in ENV_CONFIG else 1.0
    )

    # Debugging and benchmarking
    CRC_DEBUG = False
    CRC_DEBUG_SAMPLES = 100
    PROFILE_TRAINER = False
    SYNCHRONIZE_CUDA = False
    DEBUG_MODE = TMRL_CONFIG["DEBUG_MODE"] if "DEBUG_MODE" in TMRL_CONFIG else False

    # File system
    PATH_DATA = TMRL_FOLDER
    logging.debug(f" PATH_DATA:{PATH_DATA}")

    MODEL_HISTORY = TMRL_CONFIG["SAVE_MODEL_EVERY"]

    MODEL_PATH_WORKER = str(WEIGHTS_FOLDER / (RUN_NAME + ".tmod"))
    MODEL_PATH_SAVE_HISTORY = str(WEIGHTS_FOLDER / (RUN_NAME + "_"))
    MODEL_PATH_TRAINER = str(WEIGHTS_FOLDER / (RUN_NAME + "_t.tmod"))
    CHECKPOINT_PATH = str(CHECKPOINTS_FOLDER / (RUN_NAME + "_t.tcpt"))
    DATASET_PATH = str(DATASET_FOLDER)
    REWARD_PATH = str(REWARD_FOLDER / "reward.pkl")

    # WandB
    WANDB_RUN_ID = RUN_NAME
    WANDB_PROJECT = TMRL_CONFIG["WANDB_PROJECT"]
    WANDB_ENTITY = TMRL_CONFIG["WANDB_ENTITY"]
    WANDB_KEY = TMRL_CONFIG["WANDB_KEY"]

    if WANDB_KEY:
        os.environ["WANDB_API_KEY"] = WANDB_KEY

    # Networking
    PRINT_BYTESIZES = True

    PORT = TMRL_CONFIG["PORT"]
    LOCAL_PORT_SERVER = TMRL_CONFIG["LOCAL_PORT_SERVER"]
    LOCAL_PORT_TRAINER = TMRL_CONFIG["LOCAL_PORT_TRAINER"]
    LOCAL_PORT_WORKER = TMRL_CONFIG["LOCAL_PORT_WORKER"]
    PASSWORD = TMRL_CONFIG["PASSWORD"]
    SECURITY = "TLS" if TMRL_CONFIG["TLS"] else None
    CREDENTIALS_DIRECTORY = (
        TMRL_CONFIG["TLS_CREDENTIALS_DIRECTORY"]
        if TMRL_CONFIG["TLS_CREDENTIALS_DIRECTORY"] != ""
        else None
    )
    HOSTNAME = TMRL_CONFIG["TLS_HOSTNAME"]

    BUFFER_SIZE = TMRL_CONFIG["BUFFER_SIZE"]
    HEADER_SIZE = TMRL_CONFIG["HEADER_SIZE"]

    return TMRL_CONFIG


def reload_config(config_file: Path | str | None = None) -> dict:
    """Reload configuration from disk (or defaults) and refresh module globals."""
    tmrl_folder = get_tmrl_folder()
    resolved_file = (
        Path(config_file).expanduser()
        if config_file is not None
        else get_config_file(tmrl_folder)
    )
    tmrl_config, loaded = load_config(config_file=config_file)
    return _apply_config(tmrl_config, tmrl_folder, resolved_file, loaded)


def ensure_tmrl_data(force: bool = False) -> Path:
    """Explicitly initialize ``TmrlData`` on demand and reload constants."""
    from tmrl.tools.init_package.init_tmrl import ensure_tmrl_data as _ensure_tmrl_data

    tmrl_folder = _ensure_tmrl_data(tmrl_folder=get_tmrl_folder(), force=force)
    reload_config()
    return tmrl_folder


reload_config()
