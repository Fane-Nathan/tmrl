import os
import tarfile
from pathlib import Path
import itertools

from torch.optim import Adam
import numpy as np
import torch

from tmrl.config import config_constants as cfg
from tmrl.core.util import dump, load
import logging


def load_run_instance_images_dataset(checkpoint_path):
    """
    function used to load trainers from checkpoint path
    Args:
        checkpoint_path: the path where instances of run_cls are checkpointed
    Returns:
        An instance of run_cls loaded from checkpoint_path
    """
    chk_path = Path(checkpoint_path)
    parent_path = chk_path.parent.absolute()
    tar_path = str(parent_path / 'dataset.tar')
    dataset_path = str(cfg.DATASET_PATH)
    logging.debug(f" load: tar_path :{tar_path}")
    logging.debug(f" load: dataset_path :{dataset_path}")
    with tarfile.open(tar_path, 'r') as t:
        t.extractall(dataset_path)
    return load(checkpoint_path)


def dump_run_instance_images_dataset(run_instance, checkpoint_path):
    """
    function used to dump trainers to checkpoint path
    Args:
        run_instance: the instance of run_cls to checkpoint
        checkpoint_path: the path where instances of run_cls are checkpointed
    """
    chk_path = Path(checkpoint_path)
    parent_path = chk_path.parent.absolute()
    tar_path = str(parent_path / 'dataset.tar')
    dataset_path = str(cfg.DATASET_PATH)
    logging.debug(f" dump: tar_path :{tar_path}")
    logging.debug(f" dump: dataset_path :{dataset_path}")
    with tarfile.open(tar_path, 'w') as tar_handle:
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                tar_handle.add(os.path.join(root, file), arcname=file)
    dump(run_instance, checkpoint_path)


def update_memory(run_instance):
    steps = cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"]
    memory_size = cfg.TMRL_CONFIG["MEMORY_SIZE"]
    batch_size = cfg.TMRL_CONFIG["BATCH_SIZE"]
    if run_instance.steps != steps \
            or run_instance.memory.batch_size != batch_size \
            or run_instance.memory.memory_size != memory_size:
        run_instance.steps = steps
        # run_instance.memory.nb_steps = steps
        run_instance.memory.batch_size = batch_size
        run_instance.memory.memory_size = memory_size
        logging.info(f"Memory updated with steps:{steps}, batch size:{batch_size}, memory size:{memory_size}.")

    # Trim loaded memory if it exceeds the target memory size to free system RAM
    if hasattr(run_instance, "memory") and hasattr(run_instance.memory, "data") and len(run_instance.memory.data) > 0:
        current_len = len(run_instance.memory)
        to_trim = current_len - int(memory_size)
        if to_trim > 0:
            for i in range(len(run_instance.memory.data)):
                arr = run_instance.memory.data[i]
                if isinstance(arr, torch.Tensor):
                    run_instance.memory.data[i] = arr[to_trim:].clone()
                elif isinstance(arr, np.ndarray):
                    run_instance.memory.data[i] = arr[to_trim:].copy()
                else:
                    run_instance.memory.data[i] = arr[to_trim:]
            logging.info(f"Trimmed loaded replay buffer from {current_len} down to {len(run_instance.memory)} samples to prevent RAM exhaustion.")
    return run_instance


def dump_dreamer_run_instance(run_instance, checkpoint_path):
    """
    Safely dumps the TorchDreamer trainer checkpoint.
    Guarantees replay memory is trimmed to memory_size before serialization
    to eliminate MemoryError during torch storage pickling.
    """
    memory_size = int(cfg.TMRL_CONFIG.get("MEMORY_SIZE", 50000))
    if hasattr(run_instance, "memory") and hasattr(run_instance.memory, "data") and len(run_instance.memory.data) > 0:
        current_len = len(run_instance.memory)
        to_trim = current_len - memory_size
        if to_trim > 0:
            for i in range(len(run_instance.memory.data)):
                arr = run_instance.memory.data[i]
                if isinstance(arr, torch.Tensor):
                    run_instance.memory.data[i] = arr[to_trim:].clone()
                elif isinstance(arr, np.ndarray):
                    run_instance.memory.data[i] = arr[to_trim:].copy()
                else:
                    run_instance.memory.data[i] = arr[to_trim:]
            logging.info(f"Trimmed replay buffer from {current_len} to {len(run_instance.memory)} samples before checkpointing.")

    try:
        dump(run_instance, checkpoint_path)
    except MemoryError:
        logging.warning("MemoryError during standard dump! Trimming replay buffer to 25,000 samples and retrying...")
        if hasattr(run_instance, "memory") and hasattr(run_instance.memory, "data") and len(run_instance.memory.data) > 0:
            to_trim = max(0, len(run_instance.memory) - 25000)
            for i in range(len(run_instance.memory.data)):
                arr = run_instance.memory.data[i]
                if isinstance(arr, torch.Tensor):
                    run_instance.memory.data[i] = arr[to_trim:].clone()
                elif isinstance(arr, np.ndarray):
                    run_instance.memory.data[i] = arr[to_trim:].copy()
                else:
                    run_instance.memory.data[i] = arr[to_trim:]
        dump(run_instance, checkpoint_path)


def load_dreamer_run_instance(checkpoint_path):
    return load(checkpoint_path)


def update_run_instance(run_instance, training_cls):
    """
    Updates the checkpoint after loading with compatible values from config.json

    Args:
        run_instance: the instance of the checkpoint to update
        training_cls: partially instantiated class of a new checkpoint (to replace run_instance if needed)

    Returns:
        run_instance: the updated checkpoint
    """
    # check whether we should start a new experiment entirely and keep only the memory:
    if "RESET_TRAINING" in cfg.TMRL_CONFIG and cfg.TMRL_CONFIG["RESET_TRAINING"]:
        new_run_instance = training_cls()
        new_run_instance.memory = run_instance.memory
        new_run_instance = update_memory(new_run_instance)
        new_run_instance.total_samples = len(new_run_instance.memory)
        return new_run_instance

    # update training Agent:
    ALG_CONFIG = cfg.TMRL_CONFIG["ALG"]
    ALG_NAME = ALG_CONFIG["ALGORITHM"]
    assert ALG_NAME in ["SAC", "REDQSAC", "DREAMER"], f"{ALG_NAME} is not supported by this checkpoint updater."

    if ALG_NAME in ["SAC", "REDQSAC"]:
        lr_actor = ALG_CONFIG["LR_ACTOR"]
        lr_critic = ALG_CONFIG["LR_CRITIC"]
        lr_entropy = ALG_CONFIG["LR_ENTROPY"]
        gamma = ALG_CONFIG["GAMMA"]
        polyak = ALG_CONFIG["POLYAK"]
        learn_entropy_coef = ALG_CONFIG["LEARN_ENTROPY_COEF"]
        target_entropy = ALG_CONFIG["TARGET_ENTROPY"]
        alpha = ALG_CONFIG["ALPHA"]

        if ALG_NAME == "SAC":
            if run_instance.agent.lr_actor != lr_actor:
                old = run_instance.agent.lr_actor
                run_instance.agent.lr_actor = lr_actor
                run_instance.agent.pi_optimizer = Adam(run_instance.agent.model.actor.parameters(), lr=lr_actor)
                logging.info(f"Actor optimizer reinitialized with new lr: {lr_actor} (old lr: {old}).")

            if run_instance.agent.lr_critic != lr_critic:
                old = run_instance.agent.lr_critic
                run_instance.agent.lr_critic = lr_critic
                run_instance.agent.q_optimizer = Adam(itertools.chain(run_instance.agent.model.q1.parameters(), run_instance.agent.model.q2.parameters()), lr=lr_critic)
                logging.info(f"Critic optimizer reinitialized with new lr: {lr_critic} (old lr: {old}).")

        if run_instance.agent.learn_entropy_coef != learn_entropy_coef:
            logging.warning(f"Cannot switch entropy learning.")

        if run_instance.agent.lr_entropy != lr_entropy or run_instance.agent.alpha != alpha:
            run_instance.agent.lr_entropy = lr_entropy
            run_instance.agent.alpha = alpha
            device = run_instance.device or ("cuda" if torch.cuda.is_available() else "cpu")
            if run_instance.agent.learn_entropy_coef:
                run_instance.agent.log_alpha = torch.log(torch.ones(1) * run_instance.agent.alpha).to(device).requires_grad_(True)
                run_instance.agent.alpha_optimizer = Adam([run_instance.agent.log_alpha], lr=lr_entropy)
                logging.info(f"Entropy optimizer reinitialized.")
            else:
                run_instance.agent.alpha_t = torch.tensor(float(run_instance.agent.alpha)).to(device)
                logging.info(f"Alpha changed to {alpha}.")
        
        if run_instance.agent.gamma != gamma:
            old = run_instance.agent.gamma
            run_instance.agent.gamma = gamma
            logging.info(f"Gamma coefficient changed to {gamma} (old: {old}).")

        if run_instance.agent.polyak != polyak:
            old = run_instance.agent.polyak
            run_instance.agent.polyak = polyak
            logging.info(f"Polyak coefficient changed to {polyak} (old: {old}).")

        if target_entropy is None:  # automatic entropy coefficient
            action_space = run_instance.agent.action_space
            run_instance.agent.target_entropy = -np.prod(action_space.shape)  # .astype(np.float32)
        else:
            run_instance.agent.target_entropy = float(target_entropy)
        logging.info(f"Target entropy: {run_instance.agent.target_entropy}.")

        if ALG_NAME == "REDQSAC":
            m = ALG_CONFIG["REDQ_M"]
            q_updates_per_policy_update = ALG_CONFIG["REDQ_Q_UPDATES_PER_POLICY_UPDATE"]

            if run_instance.agent.q_updates_per_policy_update != q_updates_per_policy_update:
                old = run_instance.agent.q_updates_per_policy_update
                run_instance.agent.q_updates_per_policy_update = q_updates_per_policy_update
                logging.info(f"Q update ratio switched to {q_updates_per_policy_update} (old: {old}).")

            if run_instance.agent.m != m:
                old = run_instance.agent.m
                run_instance.agent.m = m
                logging.info(f"M switched to {m} (old: {old}).")

    elif ALG_NAME == "DREAMER":
        dreamer_cfg = ALG_CONFIG.get("DREAMER", {})
        warmup_steps = int(dreamer_cfg.get("WARMUP_STEPS", 100))
        if hasattr(run_instance.agent, "world_model_warmup_steps") and run_instance.agent.world_model_warmup_steps != warmup_steps:
            old_warmup = run_instance.agent.world_model_warmup_steps
            run_instance.agent.world_model_warmup_steps = warmup_steps
            logging.info(f"Dreamer warmup steps updated to {warmup_steps} (old: {old_warmup}).")

        # Migrate legacy checkpoints away from the failure mode where Dreamer's
        # world-model optimizer rewrote the demonstrated visual driving policy.
        # The worker also remains foundation-only until that gate is explicitly
        # disabled after closed-loop evaluation.
        agent = run_instance.agent
        if hasattr(agent, "configure_foundation_safety"):
            agent.foundation_weights_path = ALG_CONFIG.get(
                "FOUNDATION_WEIGHTS_PATH",
                "weights/car_brain_1m_curriculum/car_brain_multimodal.pt",
            )
            agent.lr_foundation = float(dreamer_cfg.get("LR_FOUNDATION", 0.0))
            agent.reload_foundation_on_actor_load = bool(
                dreamer_cfg.get("RELOAD_FOUNDATION_ON_ACTOR_LOAD", True)
            )
            freeze_foundation = bool(
                dreamer_cfg.get("FREEZE_FOUNDATION", True)
            )
            foundation_only = bool(dreamer_cfg.get("FOUNDATION_ONLY", True))
            foundation_discrete_actions = bool(
                dreamer_cfg.get("FOUNDATION_DISCRETE_ACTIONS", False)
            )
            foundation_steer_threshold = float(
                dreamer_cfg.get("FOUNDATION_STEER_THRESHOLD", 0.05)
            )
            residual_scale = float(dreamer_cfg.get("RESIDUAL_SCALE", 0.25))
            agent.configure_foundation_safety(
                freeze_foundation=freeze_foundation,
                foundation_only=foundation_only,
                foundation_discrete_actions=foundation_discrete_actions,
                foundation_steer_threshold=foundation_steer_threshold,
                residual_scale=residual_scale,
                reload_weights=True,
            )
            logging.info(
                "Dreamer foundation safety applied: frozen=%s, "
                "foundation_only=%s, discrete_actions=%s, "
                "steer_threshold=%s, residual_scale=%s, source=%s.",
                freeze_foundation,
                foundation_only,
                foundation_discrete_actions,
                foundation_steer_threshold,
                residual_scale,
                agent.foundation_weights_path,
            )

    epochs = cfg.TMRL_CONFIG["MAX_EPOCHS"]
    rounds = cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"]
    update_model_interval = cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"]
    update_buffer_interval = cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"]
    max_training_steps_per_env_step = cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"]
    profiling = cfg.PROFILE_TRAINER
    start_training = cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"]

    if run_instance.epochs != epochs:
        old = run_instance.epochs
        run_instance.epochs = epochs
        logging.info(f"Max epochs changed to {epochs} (old: {old}).")

    if run_instance.rounds != rounds:
        old = run_instance.rounds
        run_instance.rounds = rounds
        logging.info(f"Rounds per epoch changed to {rounds} (old: {old}).")

    if run_instance.update_model_interval != update_model_interval:
        old = run_instance.update_model_interval
        run_instance.update_model_interval = update_model_interval
        logging.info(f"Model update interval changed to {update_model_interval} (old: {old}).")

    if run_instance.update_buffer_interval != update_buffer_interval:
        old = run_instance.update_buffer_interval
        run_instance.update_buffer_interval = update_buffer_interval
        logging.info(f"Buffer update interval changed to {update_buffer_interval} (old: {old}).")

    if run_instance.max_training_steps_per_env_step != max_training_steps_per_env_step:
        old = run_instance.max_training_steps_per_env_step
        run_instance.max_training_steps_per_env_step = max_training_steps_per_env_step
        logging.info(f"Max train/env step ratio changed to {max_training_steps_per_env_step} (old: {old}).")

    if run_instance.profiling != profiling:
        old = run_instance.profiling
        run_instance.profiling = profiling
        logging.info(f"Profiling witched to {profiling} (old: {old}).")

    if run_instance.start_training != start_training:
        old = run_instance.start_training
        run_instance.start_training = start_training
        logging.info(f"Number of environment steps before training changed to {start_training} (old: {old}).")

    run_instance = update_memory(run_instance)

    return run_instance
