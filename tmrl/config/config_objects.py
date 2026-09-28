"""
This file sets up the example TMRL pipeline according to the content of config.json.
"""

# Note for developers of the TMRL library:
# config_objects.py is the bottom of the import graph, it defines the behavior of the CLI.
# No module other than top __main__.py and __init__.py should import it within the library.
# For this reason, the Trainer class has no default for the training_cls argument.
# (In previous TMRL versions, training_cls was using TRAINER as default)

import rtgym

# local imports

# core
import tmrl.config.config_constants as cfg
from tmrl.core.envs import GenericGymEnv
from tmrl.core.util import partial

# core (torch dependent)
from tmrl.core.torch.training_offline import TorchTrainingOffline

# custom (trackmania dependent)
from tmrl.custom.tm.tm_gym_interfaces import TM2020Interface, TM2020InterfaceLidar, TM2020InterfaceLidarProgress
from tmrl.custom.tm.tm_preprocessors import obs_preprocessor_tm_act_in_obs, obs_preprocessor_tm_lidar_act_in_obs, obs_preprocessor_tm_lidar_progress_act_in_obs

# custom (torch dependent)
from tmrl.custom.torch.custom_memories import ArrayTorchMemoryTMFull, ArrayTorchMemoryTMFullSequence, MemoryTMLidar, MemoryTMLidarProgress, get_local_buffer_sample_lidar, get_local_buffer_sample_lidar_progress, get_local_buffer_sample_tm20_imgs
from tmrl.custom.torch.custom_models import SquashedGaussianMLPActor, MLPActorCritic, REDQMLPActorCritic, RNNActorCritic, SquashedGaussianRNNActor, SquashedGaussianVanillaCNNActor, VanillaCNNActorCritic, SquashedGaussianVanillaColorCNNActor, VanillaColorCNNActorCritic, REDQVanillaCNNActorCritic
from tmrl.custom.torch.custom_algorithms import DreamSACAgent, SpinupSACAgent
from tmrl.custom.torch.custom_algorithms import REDQSACAgent as REDQ_Agent
from tmrl.custom.torch.custom_checkpoints import update_run_instance, dump_dreamer_run_instance, load_dreamer_run_instance
from tmrl.custom.torch.dreamer import TorchDreamerActor, TorchDreamerAgent
from tmrl.custom.torch.poincare_memory import replay_sampler_kwargs_from_config
from tmrl.custom.torch.foundation_actor import SquashedGaussianFoundationActor, FoundationCNNActorCritic, REDQFoundationCNNActorCritic


ALG_CONFIG = cfg.TMRL_CONFIG["ALG"]
ALG_NAME = ALG_CONFIG["ALGORITHM"]
DREAMER_CONFIG = ALG_CONFIG.get("DREAMER", {})
CONTINUAL_DREAMER_CONFIG = ALG_CONFIG.get("CONTINUAL_DREAMER", {})
FOUNDATION_CONFIG = CONTINUAL_DREAMER_CONFIG.get("FOUNDATION", {})
IS_TORCH_DREAMER = ALG_NAME == "DREAMER"
IS_JAX_DREAMER = ALG_NAME == "CONTINUAL_DREAMER_JAX"
IS_DREAMER = IS_TORCH_DREAMER or IS_JAX_DREAMER
ACTIVE_DREAMER_CONFIG = (
    CONTINUAL_DREAMER_CONFIG if IS_JAX_DREAMER else DREAMER_CONFIG
)
AZR_CONFIG = ALG_CONFIG.get("AZR_IMAGINATION", {})
if isinstance(AZR_CONFIG, bool):
    AZR_CONFIG = {"ENABLED": AZR_CONFIG}
USE_AZR_IMAGINATION = bool(AZR_CONFIG.get("ENABLED", False))
USE_EXPERIMENTAL_WORLD_MODEL = bool(
    ALG_NAME == "SAC"
    and (ALG_CONFIG.get("EXPERIMENTAL_WORLD_MODEL", False) or USE_AZR_IMAGINATION)
)
SAC_Agent = DreamSACAgent if USE_EXPERIMENTAL_WORLD_MODEL else SpinupSACAgent
REDQ_N = ALG_CONFIG["REDQ_N"] if "REDQ_N" in ALG_CONFIG else 10
DROPOUT_CRITIC = ALG_CONFIG["DROPOUT_CRITIC"] if "DROPOUT_CRITIC" in ALG_CONFIG else 0.0
LAYER_NORM_CRITIC = ALG_CONFIG["LAYER_NORM_CRITIC"] if "LAYER_NORM_CRITIC" in ALG_CONFIG else False
LAYER_NORM_ACTOR = ALG_CONFIG["LAYER_NORM_ACTOR"] if "LAYER_NORM_ACTOR" in ALG_CONFIG else False
assert ALG_NAME in ["SAC", "REDQSAC", "DREAMER", "CONTINUAL_DREAMER_JAX"], f"Invalid config.json: TMRL has no example pipeline for {ALG_NAME}. config.json defines the default kwargs internal to the TMRL framework: you should avoid tempering with this file when using the TMRL Python library. To implement custom TMRL pipelines, please read the TMRL tutorial on GitHub."
if IS_JAX_DREAMER:
    import numpy as np
    from gymnasium import spaces

    from tmrl.core.jax.training_offline import NNXTrainingOffline
    from tmrl.custom.jax.continual_checkpoints import (
        dump_continual_jax_run_instance,
        load_continual_jax_run_instance,
        validate_continual_jax_run_instance,
    )
    from tmrl.custom.jax.continual_dreamer import JAXDreamerAgent
    from tmrl.custom.jax.continual_replay import ArrayJaxMemoryTMFullSequence
    from tmrl.custom.torch.jax_dreamer_actor import TorchJAXDreamerActor

    assert not USE_AZR_IMAGINATION, (
        "M1 CONTINUAL_DREAMER_JAX requires AZR_IMAGINATION.ENABLED=false. "
        "AZR is introduced only after continual retention gates pass."
    )
    assert not bool(CONTINUAL_DREAMER_CONFIG.get("EXPERTS", {}).get("ENABLED", False)), (
        "Dynamic experts are disabled until the deployed JAX foundation passes M1."
    )
if USE_EXPERIMENTAL_WORLD_MODEL:
    assert ALG_NAME == "SAC", "AZR imagination currently requires SAC."
    assert not cfg.PRAGMA_LIDAR, "AZR imagination currently requires image observations."
if IS_DREAMER:
    assert not cfg.PRAGMA_LIDAR, "DREAMER requires image observations."
    assert cfg.GRAYSCALE, "The compact DREAMER controller currently requires grayscale images."
    assert not cfg.PRAGMA_RNN, "DREAMER owns its recurrent state; PRAGMA_RNN must stay disabled."


DREAM_AGENT_KWARGS = {}
if USE_EXPERIMENTAL_WORLD_MODEL:
    DREAM_AGENT_KWARGS = dict(
        enable_azr_imagination=USE_AZR_IMAGINATION,
        horizon=int(AZR_CONFIG.get("HORIZON", 3)),
        lr_world_model=float(AZR_CONFIG.get("LR_WORLD_MODEL", 3e-4)),
        lr_adversary=float(AZR_CONFIG.get("LR_PROPOSER", 1e-4)),
        free_nats=float(AZR_CONFIG.get("FREE_NATS", 1.0)),
        world_model_warmup_steps=int(AZR_CONFIG.get("WARMUP_STEPS", 1000)),
        solver_attempts=int(AZR_CONFIG.get("SOLVER_ATTEMPTS", 8)),
        tasks_per_proposal=int(AZR_CONFIG.get("TASKS_PER_PROPOSAL", 8)),
        task_batch_size=int(AZR_CONFIG.get("TASK_BATCH_SIZE", 8)),
        task_buffer_capacity=int(AZR_CONFIG.get("TASK_BUFFER_SIZE", 1024)),
        task_max_age=int(AZR_CONFIG.get("TASK_MAX_AGE", 2000)),
        task_proposal_interval=int(AZR_CONFIG.get("PROPOSAL_INTERVAL", 4)),
        imagination_interval=int(AZR_CONFIG.get("IMAGINATION_INTERVAL", 2)),
        min_task_continuation=float(AZR_CONFIG.get("MIN_CONTINUATION", 0.5)),
        max_latent_perturbation=float(AZR_CONFIG.get("MAX_PERTURBATION", 0.25)),
        target_quantile=float(AZR_CONFIG.get("TARGET_QUANTILE", 0.5)),
        imagination_actor_scale=float(AZR_CONFIG.get("ACTOR_LOSS_SCALE", 0.05)),
        imagination_reward_clip=float(AZR_CONFIG.get("REWARD_CLIP", 10.0)),
        imagination_grad_clip=float(AZR_CONFIG.get("GRAD_CLIP", 10.0)),
        azr_seed=int(AZR_CONFIG.get("SEED", 0)),
    )


# MODEL, GYM ENVIRONMENT, REPLAY MEMORY AND TRAINING: ===========

# model:

if IS_TORCH_DREAMER:
    TRAIN_MODEL = None
    use_foundation = bool(ALG_CONFIG.get("USE_FOUNDATION_ACTOR", True))
    weights_path = ALG_CONFIG.get("FOUNDATION_WEIGHTS_PATH", "weights/car_brain_1m_curriculum/car_brain_multimodal.pt")
    POLICY = partial(
        TorchDreamerActor,
        latent_dim=int(DREAMER_CONFIG.get("LATENT_DIM", 128)),
        hidden_dim=int(DREAMER_CONFIG.get("HIDDEN_DIM", 256)),
        policy_hidden_dim=int(DREAMER_CONFIG.get("POLICY_HIDDEN_DIM", 256)),
        img_channels=cfg.IMG_HIST_LEN,
        img_height=cfg.IMG_HEIGHT,
        img_width=cfg.IMG_WIDTH,
        use_foundation_encoder=use_foundation,
        foundation_weights_path=weights_path,
        freeze_foundation=bool(DREAMER_CONFIG.get("FREEZE_FOUNDATION", True)),
        foundation_only=bool(DREAMER_CONFIG.get("FOUNDATION_ONLY", True)),
        foundation_discrete_actions=bool(
            DREAMER_CONFIG.get("FOUNDATION_DISCRETE_ACTIONS", False)
        ),
        foundation_steer_threshold=float(
            DREAMER_CONFIG.get("FOUNDATION_STEER_THRESHOLD", 0.05)
        ),
        residual_scale=float(DREAMER_CONFIG.get("RESIDUAL_SCALE", 0.25)),
        reload_foundation_on_actor_load=bool(
            DREAMER_CONFIG.get("RELOAD_FOUNDATION_ON_ACTOR_LOAD", True)
        ),
    )
elif IS_JAX_DREAMER:
    TRAIN_MODEL = None
    POLICY = partial(
        TorchJAXDreamerActor,
        latent_dim=int(FOUNDATION_CONFIG.get("LATENT_DIM", 128)),
        hidden_dim=int(FOUNDATION_CONFIG.get("HIDDEN_DIM", 256)),
        policy_hidden_dim=int(
            FOUNDATION_CONFIG.get("POLICY_HIDDEN_DIM", 256)
        ),
        img_channels=cfg.IMG_HIST_LEN,
        img_height=cfg.IMG_HEIGHT,
        img_width=cfg.IMG_WIDTH,
        encoder_channels=tuple(
            FOUNDATION_CONFIG.get("ENCODER_CHANNELS", [16, 32, 64, 64])
        ),
        realtime_cpu_tuning=bool(
            FOUNDATION_CONFIG.get("WORKER_REALTIME_CPU_TUNING", True)
        ),
        realtime_cpu_threads=int(
            FOUNDATION_CONFIG.get("WORKER_CPU_THREADS", 8)
        ),
        realtime_cpu_affinity_count=int(
            FOUNDATION_CONFIG.get("WORKER_CPU_AFFINITY_COUNT", 8)
        ),
        realtime_high_priority=bool(
            FOUNDATION_CONFIG.get("WORKER_HIGH_PRIORITY", True)
        ),
    )
elif cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_RNN:
        assert ALG_NAME == "SAC", f"{ALG_NAME} is not implemented here."
        TRAIN_MODEL = RNNActorCritic
        POLICY = SquashedGaussianRNNActor
    else:
        assert ALG_NAME in ["SAC", "REDQSAC"], f"{ALG_NAME} is not implemented here."
        if ALG_NAME == "SAC":
            TRAIN_MODEL = partial(MLPActorCritic, critic_dropout=DROPOUT_CRITIC, critic_layer_norm=LAYER_NORM_CRITIC, actor_layer_norm=LAYER_NORM_ACTOR)
        else:
            TRAIN_MODEL = partial(REDQMLPActorCritic, n=REDQ_N, critic_dropout=DROPOUT_CRITIC, critic_layer_norm=LAYER_NORM_CRITIC, actor_layer_norm=LAYER_NORM_ACTOR)
        POLICY = partial(SquashedGaussianMLPActor, layer_norm=LAYER_NORM_ACTOR)
else:
    assert not cfg.PRAGMA_RNN, "RNNs not supported yet"
    assert ALG_NAME in ["SAC", "REDQSAC"], f"{ALG_NAME} is not implemented here."
    use_foundation = bool(ALG_CONFIG.get("USE_FOUNDATION_ACTOR", True))
    if use_foundation:
        if ALG_NAME == "SAC":
            TRAIN_MODEL = partial(FoundationCNNActorCritic, critic_dropout=DROPOUT_CRITIC, critic_layer_norm=LAYER_NORM_CRITIC, actor_layer_norm=LAYER_NORM_ACTOR)
            POLICY = SquashedGaussianFoundationActor
        else:
            TRAIN_MODEL = partial(REDQFoundationCNNActorCritic, n=REDQ_N, critic_dropout=DROPOUT_CRITIC, critic_layer_norm=LAYER_NORM_CRITIC, actor_layer_norm=LAYER_NORM_ACTOR)
            POLICY = SquashedGaussianFoundationActor
    elif ALG_NAME == "SAC":
        TRAIN_MODEL = partial(VanillaCNNActorCritic, critic_dropout=DROPOUT_CRITIC, critic_layer_norm=LAYER_NORM_CRITIC, actor_layer_norm=LAYER_NORM_ACTOR) if cfg.GRAYSCALE else VanillaColorCNNActorCritic
        POLICY = partial(SquashedGaussianVanillaCNNActor, layer_norm=LAYER_NORM_ACTOR) if cfg.GRAYSCALE else SquashedGaussianVanillaColorCNNActor
    else:
        assert cfg.GRAYSCALE, f"{ALG_NAME} is not implemented here."
        TRAIN_MODEL = partial(REDQVanillaCNNActorCritic, n=REDQ_N, critic_dropout=DROPOUT_CRITIC, critic_layer_norm=LAYER_NORM_CRITIC, actor_layer_norm=LAYER_NORM_ACTOR)
        POLICY = partial(SquashedGaussianVanillaCNNActor, layer_norm=LAYER_NORM_ACTOR)

# rtgym interface:

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

if IS_TORCH_DREAMER:
    MEM = ArrayTorchMemoryTMFullSequence
elif IS_JAX_DREAMER:
    MEM = ArrayJaxMemoryTMFullSequence
elif cfg.PRAGMA_LIDAR:
    if cfg.PRAGMA_RNN:
        assert False, "not implemented"
    else:
        if cfg.PRAGMA_PROGRESS:
            MEM = MemoryTMLidarProgress
        else:
            MEM = MemoryTMLidar
else:
    MEM = ArrayTorchMemoryTMFull

MEMORY_KWARGS = dict(
    memory_size=cfg.TMRL_CONFIG["MEMORY_SIZE"],
    batch_size=(
        int(ACTIVE_DREAMER_CONFIG.get("BATCH_SIZE", 8))
        if IS_DREAMER
        else cfg.TMRL_CONFIG["BATCH_SIZE"]
    ),
    sample_preprocessor=SAMPLE_PREPROCESSOR,
    dataset_path=cfg.DATASET_PATH,
    imgs_obs=cfg.IMG_HIST_LEN,
    act_buf_len=cfg.ACT_BUF_LEN,
    crc_debug=cfg.CRC_DEBUG,
)
if IS_DREAMER:
    MEMORY_KWARGS["sequence_length"] = int(
        ACTIVE_DREAMER_CONFIG.get("SEQUENCE_LENGTH", 16)
    )
    MEMORY_KWARGS.update(
        replay_sampler_kwargs_from_config(ACTIVE_DREAMER_CONFIG)
    )
MEMORY = partial(MEM, **MEMORY_KWARGS)


# ALGORITHM: ===================================================


assert ALG_NAME in ["SAC", "REDQSAC", "DREAMER", "CONTINUAL_DREAMER_JAX"], f"{ALG_NAME} is not implemented here."

if IS_TORCH_DREAMER:
    use_foundation = bool(ALG_CONFIG.get("USE_FOUNDATION_ACTOR", True))
    weights_path = ALG_CONFIG.get("FOUNDATION_WEIGHTS_PATH", "weights/car_brain_1m_curriculum/car_brain_multimodal.pt")
    AGENT = partial(
        TorchDreamerAgent,
        device='cuda' if cfg.CUDA_TRAINING else 'cpu',
        latent_dim=int(DREAMER_CONFIG.get("LATENT_DIM", 128)),
        hidden_dim=int(DREAMER_CONFIG.get("HIDDEN_DIM", 256)),
        policy_hidden_dim=int(DREAMER_CONFIG.get("POLICY_HIDDEN_DIM", 256)),
        reconstruction_size=int(DREAMER_CONFIG.get("RECONSTRUCTION_SIZE", 24)),
        lr_world_model=float(DREAMER_CONFIG.get("LR_WORLD_MODEL", 3e-4)),
        lr_actor=float(DREAMER_CONFIG.get("LR_ACTOR", 8e-5)),
        lr_critic=float(DREAMER_CONFIG.get("LR_CRITIC", 8e-5)),
        lr_foundation=float(DREAMER_CONFIG.get("LR_FOUNDATION", 0.0)),
        use_foundation_encoder=use_foundation,
        foundation_weights_path=weights_path,
        freeze_foundation=bool(DREAMER_CONFIG.get("FREEZE_FOUNDATION", True)),
        foundation_only=bool(DREAMER_CONFIG.get("FOUNDATION_ONLY", True)),
        foundation_discrete_actions=bool(
            DREAMER_CONFIG.get("FOUNDATION_DISCRETE_ACTIONS", False)
        ),
        foundation_steer_threshold=float(
            DREAMER_CONFIG.get("FOUNDATION_STEER_THRESHOLD", 0.05)
        ),
        residual_scale=float(DREAMER_CONFIG.get("RESIDUAL_SCALE", 0.25)),
        reload_foundation_on_actor_load=bool(
            DREAMER_CONFIG.get("RELOAD_FOUNDATION_ON_ACTOR_LOAD", True)
        ),
        gamma=float(ALG_CONFIG.get("GAMMA", 0.99)),
        lambda_=float(DREAMER_CONFIG.get("LAMBDA", 0.95)),
        free_nats=float(DREAMER_CONFIG.get("FREE_NATS", 1.0)),
        imagination_horizon=int(DREAMER_CONFIG.get("HORIZON", 8)),
        imagination_batch_size=int(
            DREAMER_CONFIG.get("IMAGINATION_BATCH_SIZE", 32)
        ),
        burn_in=int(DREAMER_CONFIG.get("BURN_IN", 5)),
        world_model_warmup_steps=int(
            DREAMER_CONFIG.get("WARMUP_STEPS", 2000)
        ),
        entropy_scale=float(DREAMER_CONFIG.get("ENTROPY_SCALE", 3e-4)),
        target_polyak=float(DREAMER_CONFIG.get("TARGET_POLYAK", 0.99)),
        grad_clip=float(DREAMER_CONFIG.get("GRAD_CLIP", 100.0)),
        enable_azr_imagination=USE_AZR_IMAGINATION,
        lr_adversary=float(AZR_CONFIG.get("LR_PROPOSER", 1e-4)),
        solver_attempts=int(AZR_CONFIG.get("SOLVER_ATTEMPTS", 4)),
        tasks_per_proposal=int(AZR_CONFIG.get("TASKS_PER_PROPOSAL", 4)),
        task_batch_size=int(AZR_CONFIG.get("TASK_BATCH_SIZE", 8)),
        task_buffer_capacity=int(AZR_CONFIG.get("TASK_BUFFER_SIZE", 1024)),
        task_max_age=int(AZR_CONFIG.get("TASK_MAX_AGE", 2000)),
        task_proposal_interval=int(AZR_CONFIG.get("PROPOSAL_INTERVAL", 8)),
        min_task_continuation=float(
            AZR_CONFIG.get("MIN_CONTINUATION", 0.5)
        ),
        max_latent_perturbation=float(AZR_CONFIG.get("MAX_PERTURBATION", 0.25)),
        target_quantile=float(AZR_CONFIG.get("TARGET_QUANTILE", 0.5)),
        azr_seed=int(AZR_CONFIG.get("SEED", 0)),
    )
elif IS_JAX_DREAMER:
    AGENT = partial(
        JAXDreamerAgent,
        latent_dim=int(FOUNDATION_CONFIG.get("LATENT_DIM", 128)),
        hidden_dim=int(FOUNDATION_CONFIG.get("HIDDEN_DIM", 256)),
        policy_hidden_dim=int(
            FOUNDATION_CONFIG.get("POLICY_HIDDEN_DIM", 256)
        ),
        encoder_channels=tuple(
            FOUNDATION_CONFIG.get("ENCODER_CHANNELS", [16, 32, 64, 64])
        ),
        lr_world_model=float(
            CONTINUAL_DREAMER_CONFIG.get("LR_WORLD_MODEL", 1e-4)
        ),
        lr_actor=float(CONTINUAL_DREAMER_CONFIG.get("LR_ACTOR", 3e-5)),
        lr_critic=float(CONTINUAL_DREAMER_CONFIG.get("LR_CRITIC", 3e-5)),
        gamma=float(ALG_CONFIG.get("GAMMA", 0.99)),
        lambda_=float(CONTINUAL_DREAMER_CONFIG.get("LAMBDA", 0.95)),
        free_nats=float(CONTINUAL_DREAMER_CONFIG.get("FREE_NATS", 1.0)),
        imagination_horizon=int(
            CONTINUAL_DREAMER_CONFIG.get("HORIZON", 8)
        ),
        imagination_batch_size=int(
            CONTINUAL_DREAMER_CONFIG.get("IMAGINATION_BATCH_SIZE", 32)
        ),
        burn_in=int(CONTINUAL_DREAMER_CONFIG.get("BURN_IN", 5)),
        world_model_warmup_steps=int(
            CONTINUAL_DREAMER_CONFIG.get("WARMUP_STEPS", 2000)
        ),
        entropy_scale=float(
            CONTINUAL_DREAMER_CONFIG.get("ENTROPY_SCALE", 3e-4)
        ),
        target_polyak=float(
            CONTINUAL_DREAMER_CONFIG.get("TARGET_POLYAK", 0.99)
        ),
        grad_clip=float(CONTINUAL_DREAMER_CONFIG.get("GRAD_CLIP", 10.0)),
        seed=int(CONTINUAL_DREAMER_CONFIG.get("SEED", 0)),
    )
elif ALG_NAME == "SAC":
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
        l2_critic=ALG_CONFIG["L2_CRITIC"] if "L2_CRITIC" in ALG_CONFIG else None,
        **DREAM_AGENT_KWARGS,
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
        optimizer_actor=ALG_CONFIG["OPTIMIZER_ACTOR"],
        optimizer_critic=ALG_CONFIG["OPTIMIZER_CRITIC"],
        betas_actor=ALG_CONFIG["BETAS_ACTOR"] if "BETAS_ACTOR" in ALG_CONFIG else None,
        betas_critic=ALG_CONFIG["BETAS_CRITIC"] if "BETAS_CRITIC" in ALG_CONFIG else None,
        l2_actor=ALG_CONFIG["L2_ACTOR"] if "L2_ACTOR" in ALG_CONFIG else None,
        l2_critic=ALG_CONFIG["L2_CRITIC"] if "L2_CRITIC" in ALG_CONFIG else None,
        m=ALG_CONFIG["REDQ_M"],  # number of Q targets
        q_updates_per_policy_update=ALG_CONFIG["REDQ_Q_UPDATES_PER_POLICY_UPDATE"]
    )


# TRAINER: =====================================================


def sac_v2_entropy_scheduler(agent, epoch):
    start_ent = -0.0
    end_ent = -7.0
    end_epoch = 200
    if epoch <= end_epoch:
        agent.entopy_target = start_ent + (end_ent - start_ent) * epoch / end_epoch


if IS_JAX_DREAMER:
    # The WSL2 trainer must not initialize the native-Windows TrackMania
    # interface merely to discover tensor shapes.
    JAX_OBSERVATION_SPACE = spaces.Tuple(
        (
            spaces.Box(0.0, 1000.0, shape=(1,), dtype=np.float32),
            spaces.Box(0.0, 6.0, shape=(1,), dtype=np.float32),
            spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float32),
            spaces.Box(
                0.0,
                1.0,
                shape=(cfg.IMG_HIST_LEN, cfg.IMG_HEIGHT, cfg.IMG_WIDTH),
                dtype=np.float32,
            ),
            *(
                spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
                for _ in range(cfg.ACT_BUF_LEN)
            ),
        )
    )
    JAX_ACTION_SPACE = spaces.Box(
        -1.0, 1.0, shape=(3,), dtype=np.float32
    )
    ENV_CLS = (JAX_OBSERVATION_SPACE, JAX_ACTION_SPACE)
else:
    ENV_CLS = partial(GenericGymEnv, id=cfg.RTGYM_VERSION, gym_kwargs={"config": CONFIG_DICT})

if IS_JAX_DREAMER:
    TRAINER = partial(
        NNXTrainingOffline,
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
        device=None,
        jit_sampling=False,
        jit_training=False,
    )
elif cfg.PRAGMA_LIDAR:  # lidar
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


DUMP_RUN_INSTANCE_FN = (
    dump_continual_jax_run_instance
    if IS_JAX_DREAMER
    else dump_dreamer_run_instance
    if IS_TORCH_DREAMER
    else None
)
LOAD_RUN_INSTANCE_FN = (
    load_continual_jax_run_instance
    if IS_JAX_DREAMER
    else load_dreamer_run_instance
    if IS_TORCH_DREAMER
    else None
)
UPDATER_FN = (
    validate_continual_jax_run_instance
    if IS_JAX_DREAMER
    else update_run_instance
    if ALG_NAME in ["SAC", "REDQSAC", "DREAMER"]
    else None
)
