# third-party imports
# from tmrl.custom.custom_checkpoints import load_run_instance_images_dataset, dump_run_instance_images_dataset
# third-party imports

import rtgym

# local imports
import tmrl.config.config_constants as cfg
from tmrl.training_offline import TorchTrainingOffline
from tmrl.custom.tm.tm_gym_interfaces import TM2020Interface, TM2020InterfaceLidar, TM2020InterfaceLidarProgress
from tmrl.custom.custom_memories import MemoryTMFull, MemoryTMLidar, MemoryTMLidarProgress, get_local_buffer_sample_lidar, get_local_buffer_sample_lidar_progress, get_local_buffer_sample_tm20_imgs
from tmrl.custom.tm.tm_preprocessors import obs_preprocessor_tm_act_in_obs, obs_preprocessor_tm_lidar_act_in_obs, obs_preprocessor_tm_lidar_progress_act_in_obs
from tmrl.envs import GenericGymEnv
from tmrl.custom.custom_models import SquashedGaussianMLPActor, MLPActorCritic, REDQMLPActorCritic, RNNActorCritic, SquashedGaussianRNNActor, SquashedGaussianVanillaCNNActor, VanillaCNNActorCritic, SquashedGaussianVanillaColorCNNActor, VanillaColorCNNActorCritic
from tmrl.custom.custom_algorithms import SpinupSacAgent as SAC_Agent
from tmrl.custom.custom_algorithms import REDQSACAgent as REDQ_Agent
from tmrl.custom.custom_checkpoints import update_run_instance
from tmrl.util import partial


ALG_CONFIG = cfg.TMRL_CONFIG["ALG"]
ALG_NAME = ALG_CONFIG["ALGORITHM"]
assert ALG_NAME in ["SAC", "REDQSAC"], f"If you wish to implement {ALG_NAME}, do not use 'ALG' in config.json for that."


# MODEL, GYM ENVIRONMENT, REPLAY MEMORY AND TRAINING: ===========

if cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_RNN:
        assert ALG_NAME == "SAC", f"{ALG_NAME} is not implemented here."
        TRAIN_MODEL = RNNActorCritic
        POLICY = SquashedGaussianRNNActor
    else:
        TRAIN_MODEL = MLPActorCritic if ALG_NAME == "SAC" else REDQMLPActorCritic
        POLICY = SquashedGaussianMLPActor
else:
    assert not cfg.PRAGMA_RNN, "RNNs not supported yet"
    assert ALG_NAME == "SAC", f"{ALG_NAME} is not implemented here."
    TRAIN_MODEL = VanillaCNNActorCritic if cfg.GRAYSCALE else VanillaColorCNNActorCritic
    POLICY = SquashedGaussianVanillaCNNActor if cfg.GRAYSCALE else SquashedGaussianVanillaColorCNNActor

if cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_PROGRESS:
        INT = partial(TM2020InterfaceLidarProgress, img_hist_len=cfg.IMG_HIST_LEN, gamepad=cfg.PRAGMA_GAMEPAD)
    else:
        INT = partial(TM2020InterfaceLidar, img_hist_len=cfg.IMG_HIST_LEN, gamepad=cfg.PRAGMA_GAMEPAD)
else:
    INT = partial(TM2020Interface,
                  img_hist_len=cfg.IMG_HIST_LEN,
                  gamepad=cfg.PRAGMA_GAMEPAD,
                  grayscale=cfg.GRAYSCALE,
                  resize_to=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT))

CONFIG_DICT = rtgym.DEFAULT_CONFIG_DICT.copy()
CONFIG_DICT["interface"] = INT
CONFIG_DICT_MODIFIERS = cfg.ENV_CONFIG["RTGYM_CONFIG"]
for k, v in CONFIG_DICT_MODIFIERS.items():
    CONFIG_DICT[k] = v

# to compress a sample before sending it over the local network/Internet:
if cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_PROGRESS:
        SAMPLE_COMPRESSOR = get_local_buffer_sample_lidar_progress
    else:
        SAMPLE_COMPRESSOR = get_local_buffer_sample_lidar
else:
    SAMPLE_COMPRESSOR = get_local_buffer_sample_tm20_imgs

# to preprocess observations that come out of the gymnasium environment:
if cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_PROGRESS:
        OBS_PREPROCESSOR = obs_preprocessor_tm_lidar_progress_act_in_obs
    else:
        OBS_PREPROCESSOR = obs_preprocessor_tm_lidar_act_in_obs
else:
    OBS_PREPROCESSOR = obs_preprocessor_tm_act_in_obs
# to augment data that comes out of the replay buffer:
SAMPLE_PREPROCESSOR = None

assert not cfg.PRAGMA_RNN, "RNNs not supported yet"

if cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_RNN:
        assert False, "not implemented"
    else:
        if cfg.PRAGMA_PROGRESS:
            MEM = MemoryTMLidarProgress
        else:
            MEM = MemoryTMLidar
else:
    MEM = MemoryTMFull

MEMORY = partial(MEM,
                 memory_size=cfg.TMRL_CONFIG["MEMORY_SIZE"],
                 batch_size=cfg.TMRL_CONFIG["BATCH_SIZE"],
                 sample_preprocessor=SAMPLE_PREPROCESSOR,
                 dataset_path=cfg.DATASET_PATH,
                 imgs_obs=cfg.IMG_HIST_LEN,
                 act_buf_len=cfg.ACT_BUF_LEN,
                 crc_debug=cfg.CRC_DEBUG)

# ALGORITHM: ===================================================

if ALG_NAME == "SAC":
    AGENT = partial(
        SAC_Agent,
        device='cuda' if cfg.CUDA_TRAINING else 'cpu',
        model_cls=TRAIN_MODEL,
        lr_actor=ALG_CONFIG["LR_ACTOR"],
        lr_critic=ALG_CONFIG["LR_CRITIC"],
        lr_entropy=ALG_CONFIG["LR_ENTROPY"],
        gamma=ALG_CONFIG["GAMMA"],
        polyak=ALG_CONFIG["POLYAK"],
        learn_entropy_coef=ALG_CONFIG["LEARN_ENTROPY_COEF"],  # False for SAC v2 with no temperature autotuning
        target_entropy=ALG_CONFIG["TARGET_ENTROPY"],  # None for automatic
        alpha=ALG_CONFIG["ALPHA"],  # inverse of reward scale
        optimizer_actor=ALG_CONFIG["OPTIMIZER_ACTOR"],
        optimizer_critic=ALG_CONFIG["OPTIMIZER_CRITIC"],
        betas_actor=ALG_CONFIG["BETAS_ACTOR"] if "BETAS_ACTOR" in ALG_CONFIG else None,
        betas_critic=ALG_CONFIG["BETAS_CRITIC"] if "BETAS_CRITIC" in ALG_CONFIG else None,
        l2_actor=ALG_CONFIG["L2_ACTOR"] if "L2_ACTOR" in ALG_CONFIG else None,
        l2_critic=ALG_CONFIG["L2_CRITIC"] if "L2_CRITIC" in ALG_CONFIG else None
    )
else:
    AGENT = partial(
        REDQ_Agent,
        device='cuda' if cfg.CUDA_TRAINING else 'cpu',
        model_cls=TRAIN_MODEL,
        lr_actor=ALG_CONFIG["LR_ACTOR"],
        lr_critic=ALG_CONFIG["LR_CRITIC"],
        lr_entropy=ALG_CONFIG["LR_ENTROPY"],
        gamma=ALG_CONFIG["GAMMA"],
        polyak=ALG_CONFIG["POLYAK"],
        learn_entropy_coef=ALG_CONFIG["LEARN_ENTROPY_COEF"],  # False for SAC v2 with no temperature autotuning
        target_entropy=ALG_CONFIG["TARGET_ENTROPY"],  # None for automatic
        alpha=ALG_CONFIG["ALPHA"],  # inverse of reward scale
        n=ALG_CONFIG["REDQ_N"],  # number of Q networks
        m=ALG_CONFIG["REDQ_M"],  # number of Q targets
        q_updates_per_policy_update=ALG_CONFIG["REDQ_Q_UPDATES_PER_POLICY_UPDATE"]
    )

# TRAINER: =====================================================


def sac_v2_entropy_scheduler(agent, epoch):
    start_ent = -0.0
    end_ent = -7.0
    end_epoch = 200
    if epoch <= end_epoch:
        agent.target_entropy = start_ent + (end_ent - start_ent) * epoch / end_epoch


ENV_CLS = partial(GenericGymEnv, id=cfg.RTGYM_VERSION, gym_kwargs={"config": CONFIG_DICT})

if cfg.PRAGMA_LIDAR:  # lidar
    TRAINER = partial(
        TorchTrainingOffline,
        env_cls=ENV_CLS,
        memory_cls=MEMORY,
        epochs=cfg.TMRL_CONFIG["MAX_EPOCHS"],
        rounds=cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"],
        steps=cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"],
        update_model_interval=cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"],
        update_buffer_interval=cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"],
        max_training_steps_per_env_step=cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"],
        profiling=cfg.PROFILE_TRAINER,
        training_agent_cls=AGENT,
        agent_scheduler=None,  # sac_v2_entropy_scheduler
        start_training=cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"])  # set this > 0 to start from an existing policy (fills the buffer up to this number of samples before starting training)
else:  # images
    TRAINER = partial(
        TorchTrainingOffline,
        env_cls=ENV_CLS,
        memory_cls=MEMORY,
        epochs=cfg.TMRL_CONFIG["MAX_EPOCHS"],
        rounds=cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"],
        steps=cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"],
        update_model_interval=cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"],
        update_buffer_interval=cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"],
        max_training_steps_per_env_step=cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"],
        profiling=cfg.PROFILE_TRAINER,
        training_agent_cls=AGENT,
        agent_scheduler=None,  # sac_v2_entropy_scheduler
        start_training=cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"])

# CHECKPOINTS: ===================================================

DUMP_RUN_INSTANCE_FN = None if cfg.PRAGMA_LIDAR else None  # dump_run_instance_images_dataset
LOAD_RUN_INSTANCE_FN = None if cfg.PRAGMA_LIDAR else None  # load_run_instance_images_dataset
UPDATER_FN = update_run_instance if ALG_NAME in ["SAC", "REDQSAC"] else None


# RL² OVERRIDE (Transformer-trunk REDQ-SAC) ====================================
#
# When cfg.PRAGMA_RL2 is True, replace the model / agent / memory / interface
# selections above with the RL² stack. Existing config paths are unaffected when
# the flag is False (default).
#
# Required config_constants keys (all optional, defaults given):
#   PRAGMA_RL2: bool                  — toggle the RL² stack
#   RL2_BURN_IN: int = 20
#   RL2_TRAIN_WINDOW: int = 64
#   RL2_TRANSFORMER_D_MODEL: int = 64
#   RL2_TRANSFORMER_LAYERS: int = 2
#   RL2_TRANSFORMER_HEADS: int = 2
#   RL2_TRANSFORMER_FFN: int = 128
#   RL2_TRANSFORMER_MAX_LEN: int = 128

if getattr(cfg, "PRAGMA_RL2", False):
    from tmrl.custom.custom_models_transformer import TransformerREDQActorCritic
    from tmrl.custom.custom_algorithms import RecurrentREDQSACAgent
    from tmrl.custom.custom_memories_sequence import (
        SequenceMemory,
        get_local_buffer_sample_lidar_progress_rl2,
        get_local_buffer_sample_tm20_imgs_rl2,
    )
    from tmrl.custom.tm.tm_gym_interfaces import (
        TM2020InterfaceLidarRL2,
        TM2020InterfaceRL2,
    )

    # Model: Transformer-trunk REDQ actor-critic, configured via cfg.
    TRAIN_MODEL = partial(
        TransformerREDQActorCritic,
        n=ALG_CONFIG.get("REDQ_N", 10),
        d_model=cfg.RL2_TRANSFORMER_D_MODEL,
        n_layers=cfg.RL2_TRANSFORMER_LAYERS,
        n_heads=cfg.RL2_TRANSFORMER_HEADS,
        ffn_dim=cfg.RL2_TRANSFORMER_FFN,
        max_len=cfg.RL2_TRANSFORMER_MAX_LEN,
    )

    # Interface: lidar-progress for initial smoke-test, image for production.
    if cfg.PRAGMA_LIDAR:
        assert cfg.PRAGMA_PROGRESS, \
            "PRAGMA_RL2 requires the lidar-progress interface (PRAGMA_PROGRESS=True) " \
            "or the image interface (PRAGMA_LIDAR=False)."
        INT = partial(
            TM2020InterfaceLidarRL2,
            img_hist_len=cfg.IMG_HIST_LEN,
            gamepad=cfg.PRAGMA_GAMEPAD,
        )
        SAMPLE_COMPRESSOR = get_local_buffer_sample_lidar_progress_rl2
        OBS_PREPROCESSOR = obs_preprocessor_tm_lidar_progress_act_in_obs
    else:
        INT = partial(
            TM2020InterfaceRL2,
            img_hist_len=cfg.IMG_HIST_LEN,
            gamepad=cfg.PRAGMA_GAMEPAD,
            grayscale=cfg.GRAYSCALE,
            resize_to=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT),
        )
        SAMPLE_COMPRESSOR = get_local_buffer_sample_tm20_imgs_rl2
        OBS_PREPROCESSOR = obs_preprocessor_tm_act_in_obs

    # Re-build CONFIG_DICT so rtgym uses the RL² interface.
    CONFIG_DICT = rtgym.DEFAULT_CONFIG_DICT.copy()
    CONFIG_DICT["interface"] = INT
    for k, v in CONFIG_DICT_MODIFIERS.items():
        CONFIG_DICT[k] = v

    # Memory: SequenceMemory (no imgs_obs / act_buf_len args).
    MEM = SequenceMemory
    MEMORY = partial(
        SequenceMemory,
        burn_in=cfg.RL2_BURN_IN,
        train_window=cfg.RL2_TRAIN_WINDOW,
        memory_size=cfg.TMRL_CONFIG["MEMORY_SIZE"],
        batch_size=cfg.TMRL_CONFIG["BATCH_SIZE"],
        sample_preprocessor=None,
        dataset_path=cfg.DATASET_PATH,
        crc_debug=cfg.CRC_DEBUG,
    )

    # Agent: RecurrentREDQSACAgent.
    AGENT = partial(
        RecurrentREDQSACAgent,
        device='cuda' if cfg.CUDA_TRAINING else 'cpu',
        model_cls=TRAIN_MODEL,
        lr_actor=ALG_CONFIG["LR_ACTOR"],
        lr_critic=ALG_CONFIG["LR_CRITIC"],
        lr_entropy=ALG_CONFIG["LR_ENTROPY"],
        gamma=ALG_CONFIG["GAMMA"],
        polyak=ALG_CONFIG["POLYAK"],
        learn_entropy_coef=ALG_CONFIG["LEARN_ENTROPY_COEF"],
        target_entropy=ALG_CONFIG["TARGET_ENTROPY"],
        alpha=ALG_CONFIG["ALPHA"],
        n=ALG_CONFIG.get("REDQ_N", 10),
        m=ALG_CONFIG.get("REDQ_M", 2),
        q_updates_per_policy_update=ALG_CONFIG.get("REDQ_Q_UPDATES_PER_POLICY_UPDATE", 20),
        burn_in=cfg.RL2_BURN_IN,
        train_window=cfg.RL2_TRAIN_WINDOW,
    )

    # Rebuild ENV_CLS and TRAINER so they capture the new CONFIG_DICT / MEMORY / AGENT.
    ENV_CLS = partial(GenericGymEnv, id=cfg.RTGYM_VERSION, gym_kwargs={"config": CONFIG_DICT})
    TRAINER = partial(
        TorchTrainingOffline,
        env_cls=ENV_CLS,
        memory_cls=MEMORY,
        epochs=cfg.TMRL_CONFIG["MAX_EPOCHS"],
        rounds=cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"],
        steps=cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"],
        update_model_interval=cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"],
        update_buffer_interval=cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"],
        max_training_steps_per_env_step=cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"],
        profiling=cfg.PROFILE_TRAINER,
        training_agent_cls=AGENT,
        agent_scheduler=None,
        start_training=cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"],
    )
