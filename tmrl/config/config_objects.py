"""Modular, lazy-loaded pipeline factory wiring for TMRL."""

from dataclasses import dataclass
from typing import Any

import tmrl.config.config_constants as cfg


def _partial(*args, **kwargs):
    from tmrl.core.util import partial

    return partial(*args, **kwargs)


_SUPPORTED_ALGORITHMS = ("SAC", "REDQSAC", "DREAMER", "CONTINUAL_DREAMER_JAX")


@dataclass(frozen=True)
class AlgorithmSettings:
    """Parsed algorithm configuration and derived flags."""

    alg_config: dict[str, Any]
    alg_name: str
    dreamer_config: dict[str, Any]
    continual_dreamer_config: dict[str, Any]
    foundation_config: dict[str, Any]
    is_torch_dreamer: bool
    is_jax_dreamer: bool
    is_dreamer: bool
    active_dreamer_config: dict[str, Any]
    azr_config: dict[str, Any]
    use_azr_imagination: bool
    use_experimental_world_model: bool
    use_foundation_actor: bool
    redq_n: int
    dropout_critic: float
    layer_norm_critic: bool
    layer_norm_actor: bool
    dream_agent_kwargs: dict[str, Any]


def _build_dream_agent_kwargs(
    use_experimental_world_model: bool,
    use_azr_imagination: bool,
    azr_config: dict[str, Any],
) -> dict[str, Any]:
    if not use_experimental_world_model:
        return {}
    return dict(
        enable_azr_imagination=use_azr_imagination,
        horizon=int(azr_config.get("HORIZON", 3)),
        lr_world_model=float(azr_config.get("LR_WORLD_MODEL", 3e-4)),
        lr_adversary=float(azr_config.get("LR_PROPOSER", 1e-4)),
        free_nats=float(azr_config.get("FREE_NATS", 1.0)),
        world_model_warmup_steps=int(azr_config.get("WARMUP_STEPS", 1000)),
        solver_attempts=int(azr_config.get("SOLVER_ATTEMPTS", 8)),
        tasks_per_proposal=int(azr_config.get("TASKS_PER_PROPOSAL", 8)),
        task_batch_size=int(azr_config.get("TASK_BATCH_SIZE", 8)),
        task_buffer_capacity=int(azr_config.get("TASK_BUFFER_SIZE", 1024)),
        task_max_age=int(azr_config.get("TASK_MAX_AGE", 2000)),
        task_proposal_interval=int(azr_config.get("PROPOSAL_INTERVAL", 4)),
        imagination_interval=int(azr_config.get("IMAGINATION_INTERVAL", 2)),
        min_task_continuation=float(azr_config.get("MIN_CONTINUATION", 0.5)),
        max_latent_perturbation=float(azr_config.get("MAX_PERTURBATION", 0.25)),
        target_quantile=float(azr_config.get("TARGET_QUANTILE", 0.5)),
        imagination_actor_scale=float(azr_config.get("ACTOR_LOSS_SCALE", 0.05)),
        imagination_reward_clip=float(azr_config.get("REWARD_CLIP", 10.0)),
        imagination_grad_clip=float(azr_config.get("GRAD_CLIP", 10.0)),
        azr_seed=int(azr_config.get("SEED", 0)),
    )


def build_algorithm_settings(tmrl_config: dict[str, Any] | None = None) -> AlgorithmSettings:
    """Parse algorithm settings from ``tmrl_config`` (defaults to ``cfg.TMRL_CONFIG``)."""
    source = cfg.TMRL_CONFIG if tmrl_config is None else tmrl_config
    alg_config = source["ALG"]
    alg_name = alg_config["ALGORITHM"]
    dreamer_config = alg_config.get("DREAMER", {})
    continual_dreamer_config = alg_config.get("CONTINUAL_DREAMER", {})
    foundation_config = continual_dreamer_config.get("FOUNDATION", {})
    is_torch_dreamer = alg_name == "DREAMER"
    is_jax_dreamer = alg_name == "CONTINUAL_DREAMER_JAX"
    is_dreamer = is_torch_dreamer or is_jax_dreamer
    active_dreamer_config = (
        continual_dreamer_config if is_jax_dreamer else dreamer_config
    )
    azr_config = alg_config.get("AZR_IMAGINATION", {})
    if isinstance(azr_config, bool):
        azr_config = {"ENABLED": azr_config}
    use_azr_imagination = bool(azr_config.get("ENABLED", False))
    use_experimental_world_model = bool(
        alg_name == "SAC"
        and (alg_config.get("EXPERIMENTAL_WORLD_MODEL", False) or use_azr_imagination)
    )
    use_foundation_actor = bool(alg_config.get("USE_FOUNDATION_ACTOR", False))
    redq_n = alg_config["REDQ_N"] if "REDQ_N" in alg_config else 10
    dropout_critic = alg_config["DROPOUT_CRITIC"] if "DROPOUT_CRITIC" in alg_config else 0.0
    layer_norm_critic = (
        alg_config["LAYER_NORM_CRITIC"] if "LAYER_NORM_CRITIC" in alg_config else False
    )
    layer_norm_actor = (
        alg_config["LAYER_NORM_ACTOR"] if "LAYER_NORM_ACTOR" in alg_config else False
    )
    dream_agent_kwargs = _build_dream_agent_kwargs(
        use_experimental_world_model=use_experimental_world_model,
        use_azr_imagination=use_azr_imagination,
        azr_config=azr_config,
    )
    return AlgorithmSettings(
        alg_config=alg_config,
        alg_name=alg_name,
        dreamer_config=dreamer_config,
        continual_dreamer_config=continual_dreamer_config,
        foundation_config=foundation_config,
        is_torch_dreamer=is_torch_dreamer,
        is_jax_dreamer=is_jax_dreamer,
        is_dreamer=is_dreamer,
        active_dreamer_config=active_dreamer_config,
        azr_config=azr_config,
        use_azr_imagination=use_azr_imagination,
        use_experimental_world_model=use_experimental_world_model,
        use_foundation_actor=use_foundation_actor,
        redq_n=redq_n,
        dropout_critic=dropout_critic,
        layer_norm_critic=layer_norm_critic,
        layer_norm_actor=layer_norm_actor,
        dream_agent_kwargs=dream_agent_kwargs,
    )


def validate_algorithm_settings(settings: AlgorithmSettings) -> None:
    """Validate configuration invariants without importing ML/CV frameworks."""
    assert settings.alg_name in _SUPPORTED_ALGORITHMS, (
        f"Invalid config.json: TMRL has no example pipeline for {settings.alg_name}. "
        "config.json defines the default kwargs internal to the TMRL framework: "
        "you should avoid tempering with this file when using the TMRL Python library. "
        "To implement custom TMRL pipelines, please read the TMRL tutorial on GitHub."
    )
    if settings.is_jax_dreamer:
        assert not settings.use_azr_imagination, (
            "M1 CONTINUAL_DREAMER_JAX requires AZR_IMAGINATION.ENABLED=false. "
            "AZR is introduced only after continual retention gates pass."
        )
        assert not bool(
            settings.continual_dreamer_config.get("EXPERTS", {}).get("ENABLED", False)
        ), "Dynamic experts are disabled until the deployed JAX foundation passes M1."
    if settings.use_experimental_world_model:
        assert settings.alg_name == "SAC", "AZR imagination currently requires SAC."
        assert not cfg.PRAGMA_LIDAR, "AZR imagination currently requires image observations."
    if settings.is_dreamer:
        assert not cfg.PRAGMA_LIDAR, "DREAMER requires image observations."
        assert cfg.GRAYSCALE, "The compact DREAMER controller currently requires grayscale images."
        assert not cfg.PRAGMA_RNN, (
            "DREAMER owns its recurrent state; PRAGMA_RNN must stay disabled."
        )


def build_rtgym_interface():
    """Return a partial constructor for the configured TrackMania rtgym interface."""
    from tmrl.custom.tm.tm_gym_interfaces import (
        TM2020Interface,
        TM2020InterfaceLidar,
        TM2020InterfaceLidarProgress,
    )

    if cfg.PRAGMA_LIDAR:
        if cfg.PRAGMA_PROGRESS:
            return _partial(
                TM2020InterfaceLidarProgress,
                img_hist_len=cfg.IMG_HIST_LEN,
                gamepad=cfg.PRAGMA_GAMEPAD,
            )
        return _partial(
            TM2020InterfaceLidar,
            img_hist_len=cfg.IMG_HIST_LEN,
            gamepad=cfg.PRAGMA_GAMEPAD,
        )
    return _partial(
        TM2020Interface,
        img_hist_len=cfg.IMG_HIST_LEN,
        gamepad=cfg.PRAGMA_GAMEPAD,
        grayscale=cfg.GRAYSCALE,
        resize_to=(cfg.IMG_WIDTH, cfg.IMG_HEIGHT),
    )


def build_config_dict(interface=None) -> dict[str, Any]:
    """Build the rtgym configuration dictionary."""
    import rtgym

    config_dict = rtgym.DEFAULT_CONFIG_DICT.copy()
    config_dict["interface"] = build_rtgym_interface() if interface is None else interface
    for key, value in cfg.ENV_CONFIG["RTGYM_CONFIG"].items():
        config_dict[key] = value
    return config_dict


def build_sample_compressor():
    """Return the sample compressor function matching the configured observation mode."""
    from tmrl.custom.torch.custom_memories import (
        get_local_buffer_sample_lidar,
        get_local_buffer_sample_lidar_progress,
        get_local_buffer_sample_tm20_imgs,
    )

    if cfg.PRAGMA_LIDAR:
        if cfg.PRAGMA_PROGRESS:
            return get_local_buffer_sample_lidar_progress
        return get_local_buffer_sample_lidar
    return get_local_buffer_sample_tm20_imgs


def build_obs_preprocessor():
    """Return the observation preprocessor matching the configured observation mode."""
    from tmrl.custom.tm.tm_preprocessors import (
        obs_preprocessor_tm_act_in_obs,
        obs_preprocessor_tm_lidar_act_in_obs,
        obs_preprocessor_tm_lidar_progress_act_in_obs,
    )

    if cfg.PRAGMA_LIDAR:
        if cfg.PRAGMA_PROGRESS:
            return obs_preprocessor_tm_lidar_progress_act_in_obs
        return obs_preprocessor_tm_lidar_act_in_obs
    return obs_preprocessor_tm_act_in_obs


def _build_torch_dreamer_policy(settings: AlgorithmSettings):
    from tmrl.custom.torch.dreamer import TorchDreamerActor

    dreamer_cfg = settings.dreamer_config
    use_foundation = settings.use_foundation_actor
    weights_path = settings.alg_config.get(
        "FOUNDATION_WEIGHTS_PATH",
        "weights/car_brain_1m_curriculum/car_brain_multimodal.pt",
    )
    foundation_only = bool(dreamer_cfg.get("FOUNDATION_ONLY", use_foundation))
    policy = _partial(
        TorchDreamerActor,
        latent_dim=int(dreamer_cfg.get("LATENT_DIM", 128)),
        hidden_dim=int(dreamer_cfg.get("HIDDEN_DIM", 256)),
        policy_hidden_dim=int(dreamer_cfg.get("POLICY_HIDDEN_DIM", 256)),
        img_channels=cfg.IMG_HIST_LEN,
        img_height=cfg.IMG_HEIGHT,
        img_width=cfg.IMG_WIDTH,
        use_foundation_encoder=use_foundation,
        foundation_weights_path=weights_path,
        freeze_foundation=bool(dreamer_cfg.get("FREEZE_FOUNDATION", True)),
        foundation_only=foundation_only,
        foundation_discrete_actions=bool(
            dreamer_cfg.get("FOUNDATION_DISCRETE_ACTIONS", False)
        ),
        foundation_steer_threshold=float(
            dreamer_cfg.get("FOUNDATION_STEER_THRESHOLD", 0.05)
        ),
        residual_scale=float(dreamer_cfg.get("RESIDUAL_SCALE", 0.25)),
        reload_foundation_on_actor_load=bool(
            dreamer_cfg.get("RELOAD_FOUNDATION_ON_ACTOR_LOAD", True)
        ),
    )
    return None, policy


def _build_jax_dreamer_policy(settings: AlgorithmSettings):
    from tmrl.custom.torch.jax_dreamer_actor import TorchJAXDreamerActor

    foundation_cfg = settings.foundation_config
    policy = _partial(
        TorchJAXDreamerActor,
        latent_dim=int(foundation_cfg.get("LATENT_DIM", 128)),
        hidden_dim=int(foundation_cfg.get("HIDDEN_DIM", 256)),
        policy_hidden_dim=int(foundation_cfg.get("POLICY_HIDDEN_DIM", 256)),
        img_channels=cfg.IMG_HIST_LEN,
        img_height=cfg.IMG_HEIGHT,
        img_width=cfg.IMG_WIDTH,
        encoder_channels=tuple(
            foundation_cfg.get("ENCODER_CHANNELS", [16, 32, 64, 64])
        ),
        realtime_cpu_tuning=bool(
            foundation_cfg.get("WORKER_REALTIME_CPU_TUNING", True)
        ),
        realtime_cpu_threads=int(foundation_cfg.get("WORKER_CPU_THREADS", 8)),
        realtime_cpu_affinity_count=int(
            foundation_cfg.get("WORKER_CPU_AFFINITY_COUNT", 8)
        ),
        realtime_high_priority=bool(
            foundation_cfg.get("WORKER_HIGH_PRIORITY", True)
        ),
    )
    return None, policy


def _build_lidar_model_and_policy(settings: AlgorithmSettings):
    from tmrl.custom.torch.custom_models import (
        MLPActorCritic,
        REDQMLPActorCritic,
        RNNActorCritic,
        SquashedGaussianMLPActor,
        SquashedGaussianRNNActor,
    )

    if cfg.PRAGMA_RNN:
        assert settings.alg_name == "SAC", f"{settings.alg_name} is not implemented here."
        return RNNActorCritic, SquashedGaussianRNNActor

    assert settings.alg_name in ["SAC", "REDQSAC"], (
        f"{settings.alg_name} is not implemented here."
    )
    if settings.alg_name == "SAC":
        train_model = _partial(
            MLPActorCritic,
            critic_dropout=settings.dropout_critic,
            critic_layer_norm=settings.layer_norm_critic,
            actor_layer_norm=settings.layer_norm_actor,
        )
    else:
        train_model = _partial(
            REDQMLPActorCritic,
            n=settings.redq_n,
            critic_dropout=settings.dropout_critic,
            critic_layer_norm=settings.layer_norm_critic,
            actor_layer_norm=settings.layer_norm_actor,
        )
    policy = _partial(SquashedGaussianMLPActor, layer_norm=settings.layer_norm_actor)
    return train_model, policy


def _build_vision_model_and_policy(settings: AlgorithmSettings):
    assert not cfg.PRAGMA_RNN, "RNNs not supported yet"
    assert settings.alg_name in ["SAC", "REDQSAC"], (
        f"{settings.alg_name} is not implemented here."
    )
    if settings.use_foundation_actor:
        from tmrl.custom.torch.foundation_actor import (
            FoundationCNNActorCritic,
            REDQFoundationCNNActorCritic,
            SquashedGaussianFoundationActor,
        )

        if settings.alg_name == "SAC":
            train_model = _partial(
                FoundationCNNActorCritic,
                critic_dropout=settings.dropout_critic,
                critic_layer_norm=settings.layer_norm_critic,
                actor_layer_norm=settings.layer_norm_actor,
            )
        else:
            train_model = _partial(
                REDQFoundationCNNActorCritic,
                n=settings.redq_n,
                critic_dropout=settings.dropout_critic,
                critic_layer_norm=settings.layer_norm_critic,
                actor_layer_norm=settings.layer_norm_actor,
            )
        return train_model, SquashedGaussianFoundationActor

    from tmrl.custom.torch.custom_models import (
        REDQVanillaCNNActorCritic,
        SquashedGaussianVanillaCNNActor,
        SquashedGaussianVanillaColorCNNActor,
        VanillaCNNActorCritic,
        VanillaColorCNNActorCritic,
    )

    if settings.alg_name == "SAC":
        train_model = (
            _partial(
                VanillaCNNActorCritic,
                critic_dropout=settings.dropout_critic,
                critic_layer_norm=settings.layer_norm_critic,
                actor_layer_norm=settings.layer_norm_actor,
            )
            if cfg.GRAYSCALE
            else VanillaColorCNNActorCritic
        )
        policy = (
            _partial(SquashedGaussianVanillaCNNActor, layer_norm=settings.layer_norm_actor)
            if cfg.GRAYSCALE
            else SquashedGaussianVanillaColorCNNActor
        )
        return train_model, policy

    assert cfg.GRAYSCALE, f"{settings.alg_name} is not implemented here."
    train_model = _partial(
        REDQVanillaCNNActorCritic,
        n=settings.redq_n,
        critic_dropout=settings.dropout_critic,
        critic_layer_norm=settings.layer_norm_critic,
        actor_layer_norm=settings.layer_norm_actor,
    )
    policy = _partial(SquashedGaussianVanillaCNNActor, layer_norm=settings.layer_norm_actor)
    return train_model, policy


def build_model_and_policy(settings: AlgorithmSettings | None = None):
    """Build ``(TRAIN_MODEL, POLICY)`` for the active algorithm and observation mode."""
    active = _SETTINGS if settings is None else settings
    if active.is_torch_dreamer:
        return _build_torch_dreamer_policy(active)
    if active.is_jax_dreamer:
        return _build_jax_dreamer_policy(active)
    if cfg.PRAGMA_LIDAR:
        return _build_lidar_model_and_policy(active)
    return _build_vision_model_and_policy(active)


def build_memory(settings: AlgorithmSettings | None = None):
    """Build ``(MEM, MEMORY_KWARGS, MEMORY)`` for the active algorithm."""
    active = _SETTINGS if settings is None else settings
    assert not cfg.PRAGMA_RNN, "RNNs not supported yet"

    if active.is_torch_dreamer:
        from tmrl.custom.torch.custom_memories import ArrayTorchMemoryTMFullSequence

        mem_cls = ArrayTorchMemoryTMFullSequence
    elif active.is_jax_dreamer:
        from tmrl.custom.jax.continual_replay import ArrayJaxMemoryTMFullSequence

        mem_cls = ArrayJaxMemoryTMFullSequence
    elif cfg.PRAGMA_LIDAR:
        from tmrl.custom.torch.custom_memories import MemoryTMLidar, MemoryTMLidarProgress

        mem_cls = MemoryTMLidarProgress if cfg.PRAGMA_PROGRESS else MemoryTMLidar
    else:
        from tmrl.custom.torch.custom_memories import ArrayTorchMemoryTMFull

        mem_cls = ArrayTorchMemoryTMFull

    memory_kwargs = dict(
        memory_size=cfg.TMRL_CONFIG["MEMORY_SIZE"],
        batch_size=(
            int(active.active_dreamer_config.get("BATCH_SIZE", 8))
            if active.is_dreamer
            else cfg.TMRL_CONFIG["BATCH_SIZE"]
        ),
        sample_preprocessor=SAMPLE_PREPROCESSOR,
        dataset_path=cfg.DATASET_PATH,
        imgs_obs=cfg.IMG_HIST_LEN,
        act_buf_len=cfg.ACT_BUF_LEN,
        crc_debug=cfg.CRC_DEBUG,
    )
    if active.is_dreamer:
        from tmrl.custom.torch.poincare_memory import replay_sampler_kwargs_from_config

        memory_kwargs["sequence_length"] = int(
            active.active_dreamer_config.get("SEQUENCE_LENGTH", 16)
        )
        memory_kwargs.update(
            replay_sampler_kwargs_from_config(active.active_dreamer_config)
        )
    memory_partial = _partial(mem_cls, **memory_kwargs)
    return mem_cls, memory_kwargs, memory_partial


def _build_torch_dreamer_agent(settings: AlgorithmSettings):
    from tmrl.custom.torch.dreamer import TorchDreamerAgent

    dreamer_cfg = settings.dreamer_config
    azr_cfg = settings.azr_config
    use_foundation = settings.use_foundation_actor
    weights_path = settings.alg_config.get(
        "FOUNDATION_WEIGHTS_PATH",
        "weights/car_brain_1m_curriculum/car_brain_multimodal.pt",
    )
    foundation_only = bool(dreamer_cfg.get("FOUNDATION_ONLY", use_foundation))
    return _partial(
        TorchDreamerAgent,
        device="cuda" if cfg.CUDA_TRAINING else "cpu",
        latent_dim=int(dreamer_cfg.get("LATENT_DIM", 128)),
        hidden_dim=int(dreamer_cfg.get("HIDDEN_DIM", 256)),
        policy_hidden_dim=int(dreamer_cfg.get("POLICY_HIDDEN_DIM", 256)),
        reconstruction_size=int(dreamer_cfg.get("RECONSTRUCTION_SIZE", 24)),
        lr_world_model=float(dreamer_cfg.get("LR_WORLD_MODEL", 3e-4)),
        lr_actor=float(dreamer_cfg.get("LR_ACTOR", 8e-5)),
        lr_critic=float(dreamer_cfg.get("LR_CRITIC", 8e-5)),
        lr_foundation=float(dreamer_cfg.get("LR_FOUNDATION", 0.0)),
        use_foundation_encoder=use_foundation,
        foundation_weights_path=weights_path,
        freeze_foundation=bool(dreamer_cfg.get("FREEZE_FOUNDATION", True)),
        foundation_only=foundation_only,
        foundation_discrete_actions=bool(
            dreamer_cfg.get("FOUNDATION_DISCRETE_ACTIONS", False)
        ),
        foundation_steer_threshold=float(
            dreamer_cfg.get("FOUNDATION_STEER_THRESHOLD", 0.05)
        ),
        residual_scale=float(dreamer_cfg.get("RESIDUAL_SCALE", 0.25)),
        reload_foundation_on_actor_load=bool(
            dreamer_cfg.get("RELOAD_FOUNDATION_ON_ACTOR_LOAD", True)
        ),
        gamma=float(settings.alg_config.get("GAMMA", 0.99)),
        lambda_=float(dreamer_cfg.get("LAMBDA", 0.95)),
        free_nats=float(dreamer_cfg.get("FREE_NATS", 1.0)),
        imagination_horizon=int(dreamer_cfg.get("HORIZON", 8)),
        imagination_batch_size=int(dreamer_cfg.get("IMAGINATION_BATCH_SIZE", 32)),
        burn_in=int(dreamer_cfg.get("BURN_IN", 5)),
        world_model_warmup_steps=int(dreamer_cfg.get("WARMUP_STEPS", 2000)),
        entropy_scale=float(dreamer_cfg.get("ENTROPY_SCALE", 3e-4)),
        target_polyak=float(dreamer_cfg.get("TARGET_POLYAK", 0.99)),
        grad_clip=float(dreamer_cfg.get("GRAD_CLIP", 100.0)),
        enable_azr_imagination=settings.use_azr_imagination,
        lr_adversary=float(azr_cfg.get("LR_PROPOSER", 1e-4)),
        solver_attempts=int(azr_cfg.get("SOLVER_ATTEMPTS", 4)),
        tasks_per_proposal=int(azr_cfg.get("TASKS_PER_PROPOSAL", 4)),
        task_batch_size=int(azr_cfg.get("TASK_BATCH_SIZE", 8)),
        task_buffer_capacity=int(azr_cfg.get("TASK_BUFFER_SIZE", 1024)),
        task_max_age=int(azr_cfg.get("TASK_MAX_AGE", 2000)),
        task_proposal_interval=int(azr_cfg.get("PROPOSAL_INTERVAL", 8)),
        min_task_continuation=float(azr_cfg.get("MIN_CONTINUATION", 0.5)),
        max_latent_perturbation=float(azr_cfg.get("MAX_PERTURBATION", 0.25)),
        target_quantile=float(azr_cfg.get("TARGET_QUANTILE", 0.5)),
        azr_seed=int(azr_cfg.get("SEED", 0)),
    )


def _build_jax_dreamer_agent(settings: AlgorithmSettings):
    from tmrl.custom.jax.continual_dreamer import JAXDreamerAgent

    foundation_cfg = settings.foundation_config
    continual_cfg = settings.continual_dreamer_config
    return _partial(
        JAXDreamerAgent,
        latent_dim=int(foundation_cfg.get("LATENT_DIM", 128)),
        hidden_dim=int(foundation_cfg.get("HIDDEN_DIM", 256)),
        policy_hidden_dim=int(foundation_cfg.get("POLICY_HIDDEN_DIM", 256)),
        encoder_channels=tuple(
            foundation_cfg.get("ENCODER_CHANNELS", [16, 32, 64, 64])
        ),
        lr_world_model=float(continual_cfg.get("LR_WORLD_MODEL", 1e-4)),
        lr_actor=float(continual_cfg.get("LR_ACTOR", 3e-5)),
        lr_critic=float(continual_cfg.get("LR_CRITIC", 3e-5)),
        gamma=float(settings.alg_config.get("GAMMA", 0.99)),
        lambda_=float(continual_cfg.get("LAMBDA", 0.95)),
        free_nats=float(continual_cfg.get("FREE_NATS", 1.0)),
        imagination_horizon=int(continual_cfg.get("HORIZON", 8)),
        imagination_batch_size=int(continual_cfg.get("IMAGINATION_BATCH_SIZE", 32)),
        burn_in=int(continual_cfg.get("BURN_IN", 5)),
        world_model_warmup_steps=int(continual_cfg.get("WARMUP_STEPS", 2000)),
        entropy_scale=float(continual_cfg.get("ENTROPY_SCALE", 3e-4)),
        target_polyak=float(continual_cfg.get("TARGET_POLYAK", 0.99)),
        grad_clip=float(continual_cfg.get("GRAD_CLIP", 10.0)),
        seed=int(continual_cfg.get("SEED", 0)),
    )


def _get_sac_agent_cls(settings: AlgorithmSettings):
    from tmrl.custom.torch.custom_algorithms import DreamSACAgent, SpinupSACAgent

    return DreamSACAgent if settings.use_experimental_world_model else SpinupSACAgent


def _build_sac_agent(settings: AlgorithmSettings, train_model):
    alg_cfg = settings.alg_config
    sac_cls = _get_sac_agent_cls(settings)
    return _partial(
        sac_cls,
        device="cuda" if cfg.CUDA_TRAINING else "cpu",
        model_cls=train_model,
        lr_actor=alg_cfg["LR_ACTOR"],
        lr_critic=alg_cfg["LR_CRITIC"],
        lr_entropy=alg_cfg["LR_ENTROPY"],
        gamma=alg_cfg["GAMMA"],
        polyak=alg_cfg["POLYAK"],
        learn_entropy_coef=alg_cfg["LEARN_ENTROPY_COEF"],
        target_entropy=alg_cfg["TARGET_ENTROPY"],
        alpha=alg_cfg["ALPHA"],
        optimizer_actor=alg_cfg["OPTIMIZER_ACTOR"],
        optimizer_critic=alg_cfg["OPTIMIZER_CRITIC"],
        betas_actor=alg_cfg["BETAS_ACTOR"] if "BETAS_ACTOR" in alg_cfg else None,
        betas_critic=alg_cfg["BETAS_CRITIC"] if "BETAS_CRITIC" in alg_cfg else None,
        l2_actor=alg_cfg["L2_ACTOR"] if "L2_ACTOR" in alg_cfg else None,
        l2_critic=alg_cfg["L2_CRITIC"] if "L2_CRITIC" in alg_cfg else None,
        **settings.dream_agent_kwargs,
    )


def _build_redq_agent(settings: AlgorithmSettings, train_model):
    from tmrl.custom.torch.custom_algorithms import REDQSACAgent as REDQ_Agent

    alg_cfg = settings.alg_config
    return _partial(
        REDQ_Agent,
        device="cuda" if cfg.CUDA_TRAINING else "cpu",
        model_cls=train_model,
        lr_actor=alg_cfg["LR_ACTOR"],
        lr_critic=alg_cfg["LR_CRITIC"],
        lr_entropy=alg_cfg["LR_ENTROPY"],
        gamma=alg_cfg["GAMMA"],
        polyak=alg_cfg["POLYAK"],
        learn_entropy_coef=alg_cfg["LEARN_ENTROPY_COEF"],
        target_entropy=alg_cfg["TARGET_ENTROPY"],
        alpha=alg_cfg["ALPHA"],
        optimizer_actor=alg_cfg["OPTIMIZER_ACTOR"],
        optimizer_critic=alg_cfg["OPTIMIZER_CRITIC"],
        betas_actor=alg_cfg["BETAS_ACTOR"] if "BETAS_ACTOR" in alg_cfg else None,
        betas_critic=alg_cfg["BETAS_CRITIC"] if "BETAS_CRITIC" in alg_cfg else None,
        l2_actor=alg_cfg["L2_ACTOR"] if "L2_ACTOR" in alg_cfg else None,
        l2_critic=alg_cfg["L2_CRITIC"] if "L2_CRITIC" in alg_cfg else None,
        m=alg_cfg["REDQ_M"],
        q_updates_per_policy_update=alg_cfg["REDQ_Q_UPDATES_PER_POLICY_UPDATE"],
    )


def build_agent(settings: AlgorithmSettings | None = None, train_model=None):
    """Build the training agent partial constructor for the configured algorithm."""
    active = _SETTINGS if settings is None else settings
    assert active.alg_name in _SUPPORTED_ALGORITHMS, (
        f"{active.alg_name} is not implemented here."
    )
    if active.is_torch_dreamer:
        return _build_torch_dreamer_agent(active)
    if active.is_jax_dreamer:
        return _build_jax_dreamer_agent(active)
    resolved_model = _get_lazy("TRAIN_MODEL") if train_model is None else train_model
    if active.alg_name == "SAC":
        return _build_sac_agent(active, resolved_model)
    return _build_redq_agent(active, resolved_model)


def sac_v2_entropy_scheduler(agent, epoch):
    """Linearly anneal target entropy across the first 200 epochs."""
    start_ent = -0.0
    end_ent = -7.0
    end_epoch = 200
    if epoch <= end_epoch:
        agent.entopy_target = start_ent + (end_ent - start_ent) * epoch / end_epoch


def build_jax_spaces():
    """Build ``(JAX_OBSERVATION_SPACE, JAX_ACTION_SPACE)`` without opening TrackMania."""
    from gymnasium import spaces
    import numpy as np

    observation_space = spaces.Tuple(
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
    action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
    return observation_space, action_space


def build_env_cls(
    settings: AlgorithmSettings | None = None,
    config_dict: dict[str, Any] | None = None,
):
    """Build ``ENV_CLS`` (spaces tuple for JAX Dreamer, ``GenericGymEnv`` partial otherwise)."""
    active = _SETTINGS if settings is None else settings
    if active.is_jax_dreamer:
        return (
            _get_lazy("JAX_OBSERVATION_SPACE"),
            _get_lazy("JAX_ACTION_SPACE"),
        )

    from tmrl.core.envs import GenericGymEnv

    resolved_config = _get_lazy("CONFIG_DICT") if config_dict is None else config_dict
    return _partial(
        GenericGymEnv,
        id=cfg.RTGYM_VERSION,
        gym_kwargs={"config": resolved_config},
    )


def build_trainer(
    settings: AlgorithmSettings | None = None,
    env_cls=None,
    memory=None,
    agent=None,
):
    """Build ``TRAINER`` partial for either JAX NNX or PyTorch offline training."""
    active = _SETTINGS if settings is None else settings
    resolved_env_cls = _get_lazy("ENV_CLS") if env_cls is None else env_cls
    resolved_memory = _get_lazy("MEMORY") if memory is None else memory
    resolved_agent = _get_lazy("AGENT") if agent is None else agent

    common_kwargs = dict(
        env_cls=resolved_env_cls,
        memory_cls=resolved_memory,
        epochs=cfg.TMRL_CONFIG["MAX_EPOCHS"],
        rounds=cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"],
        steps=cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"],
        update_model_interval=cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"],
        update_buffer_interval=cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"],
        max_training_steps_per_env_step=cfg.TMRL_CONFIG[
            "MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"
        ],
        profiling=cfg.PROFILE_TRAINER,
        training_agent_cls=resolved_agent,
        agent_scheduler=None,
        start_training=cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"],
    )

    if active.is_jax_dreamer:
        from tmrl.core.jax.training_offline import NNXTrainingOffline

        return _partial(
            NNXTrainingOffline,
            **common_kwargs,
            device=None,
            jit_sampling=False,
            jit_training=False,
        )

    from tmrl.core.torch.training_offline import TorchTrainingOffline

    return _partial(TorchTrainingOffline, **common_kwargs)


def build_checkpoint_handlers(settings: AlgorithmSettings | None = None):
    """Build ``(DUMP_RUN_INSTANCE_FN, LOAD_RUN_INSTANCE_FN, UPDATER_FN)``."""
    active = _SETTINGS if settings is None else settings
    if active.is_jax_dreamer:
        from tmrl.custom.jax.continual_checkpoints import (
            dump_continual_jax_run_instance,
            load_continual_jax_run_instance,
            validate_continual_jax_run_instance,
        )

        return (
            dump_continual_jax_run_instance,
            load_continual_jax_run_instance,
            validate_continual_jax_run_instance,
        )

    if active.is_torch_dreamer:
        from tmrl.custom.torch.custom_checkpoints import (
            dump_dreamer_run_instance,
            load_dreamer_run_instance,
            update_run_instance,
        )

        return (
            dump_dreamer_run_instance,
            load_dreamer_run_instance,
            update_run_instance,
        )

    if active.alg_name in ["SAC", "REDQSAC"]:
        from tmrl.custom.torch.custom_checkpoints import update_run_instance

        return None, None, update_run_instance

    return None, None, None


SAMPLE_PREPROCESSOR = None

_LAZY_ATTRS = frozenset(
    {
        "INT",
        "CONFIG_DICT_MODIFIERS",
        "CONFIG_DICT",
        "SAMPLE_COMPRESSOR",
        "OBS_PREPROCESSOR",
        "TRAIN_MODEL",
        "POLICY",
        "MEM",
        "MEMORY_KWARGS",
        "MEMORY",
        "SAC_Agent",
        "AGENT",
        "JAX_OBSERVATION_SPACE",
        "JAX_ACTION_SPACE",
        "ENV_CLS",
        "TRAINER",
        "DUMP_RUN_INSTANCE_FN",
        "LOAD_RUN_INSTANCE_FN",
        "UPDATER_FN",
    }
)


def reload() -> AlgorithmSettings:
    """Refresh parsed settings from ``cfg.TMRL_CONFIG`` and clear cached lazy exports."""
    global _SETTINGS
    global ALG_CONFIG, ALG_NAME, DREAMER_CONFIG, CONTINUAL_DREAMER_CONFIG
    global FOUNDATION_CONFIG, IS_TORCH_DREAMER, IS_JAX_DREAMER, IS_DREAMER
    global ACTIVE_DREAMER_CONFIG, AZR_CONFIG, USE_AZR_IMAGINATION
    global USE_EXPERIMENTAL_WORLD_MODEL, USE_FOUNDATION_ACTOR
    global REDQ_N, DROPOUT_CRITIC, LAYER_NORM_CRITIC, LAYER_NORM_ACTOR
    global DREAM_AGENT_KWARGS

    for attr in _LAZY_ATTRS:
        globals().pop(attr, None)

    settings = build_algorithm_settings()
    validate_algorithm_settings(settings)

    _SETTINGS = settings
    ALG_CONFIG = settings.alg_config
    ALG_NAME = settings.alg_name
    DREAMER_CONFIG = settings.dreamer_config
    CONTINUAL_DREAMER_CONFIG = settings.continual_dreamer_config
    FOUNDATION_CONFIG = settings.foundation_config
    IS_TORCH_DREAMER = settings.is_torch_dreamer
    IS_JAX_DREAMER = settings.is_jax_dreamer
    IS_DREAMER = settings.is_dreamer
    ACTIVE_DREAMER_CONFIG = settings.active_dreamer_config
    AZR_CONFIG = settings.azr_config
    USE_AZR_IMAGINATION = settings.use_azr_imagination
    USE_EXPERIMENTAL_WORLD_MODEL = settings.use_experimental_world_model
    USE_FOUNDATION_ACTOR = settings.use_foundation_actor
    REDQ_N = settings.redq_n
    DROPOUT_CRITIC = settings.dropout_critic
    LAYER_NORM_CRITIC = settings.layer_norm_critic
    LAYER_NORM_ACTOR = settings.layer_norm_actor
    DREAM_AGENT_KWARGS = settings.dream_agent_kwargs
    return settings


def _get_lazy(name: str):
    if name in globals():
        return globals()[name]
    return __getattr__(name)


def __getattr__(name: str):
    if name == "INT":
        value = build_rtgym_interface()
        globals()["INT"] = value
        return value

    if name == "CONFIG_DICT_MODIFIERS":
        value = cfg.ENV_CONFIG["RTGYM_CONFIG"]
        globals()["CONFIG_DICT_MODIFIERS"] = value
        return value

    if name == "CONFIG_DICT":
        value = build_config_dict(interface=_get_lazy("INT"))
        globals()["CONFIG_DICT"] = value
        return value

    if name == "SAMPLE_COMPRESSOR":
        value = build_sample_compressor()
        globals()["SAMPLE_COMPRESSOR"] = value
        return value

    if name == "OBS_PREPROCESSOR":
        value = build_obs_preprocessor()
        globals()["OBS_PREPROCESSOR"] = value
        return value

    if name in ("TRAIN_MODEL", "POLICY"):
        train_model, policy = build_model_and_policy(_SETTINGS)
        globals()["TRAIN_MODEL"] = train_model
        globals()["POLICY"] = policy
        return globals()[name]

    if name in ("MEM", "MEMORY_KWARGS", "MEMORY"):
        mem_cls, memory_kwargs, memory_partial = build_memory(_SETTINGS)
        globals()["MEM"] = mem_cls
        globals()["MEMORY_KWARGS"] = memory_kwargs
        globals()["MEMORY"] = memory_partial
        return globals()[name]

    if name == "SAC_Agent":
        value = _get_sac_agent_cls(_SETTINGS)
        globals()["SAC_Agent"] = value
        return value

    if name == "AGENT":
        value = build_agent(_SETTINGS)
        globals()["AGENT"] = value
        return value

    if name in ("JAX_OBSERVATION_SPACE", "JAX_ACTION_SPACE"):
        if not _SETTINGS.is_jax_dreamer:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        obs_space, act_space = build_jax_spaces()
        globals()["JAX_OBSERVATION_SPACE"] = obs_space
        globals()["JAX_ACTION_SPACE"] = act_space
        return globals()[name]

    if name == "ENV_CLS":
        value = build_env_cls(_SETTINGS)
        globals()["ENV_CLS"] = value
        return value

    if name == "TRAINER":
        value = build_trainer(_SETTINGS)
        globals()["TRAINER"] = value
        return value

    if name in ("DUMP_RUN_INSTANCE_FN", "LOAD_RUN_INSTANCE_FN", "UPDATER_FN"):
        dump_fn, load_fn, updater_fn = build_checkpoint_handlers(_SETTINGS)
        globals()["DUMP_RUN_INSTANCE_FN"] = dump_fn
        globals()["LOAD_RUN_INSTANCE_FN"] = load_fn
        globals()["UPDATER_FN"] = updater_fn
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    available = set(globals()) | set(_LAZY_ATTRS)
    if not _SETTINGS.is_jax_dreamer:
        available.discard("JAX_OBSERVATION_SPACE")
        available.discard("JAX_ACTION_SPACE")
    return sorted(available)


reload()
