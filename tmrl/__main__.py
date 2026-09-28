import json
import logging
import sys
import time
from argparse import ArgumentParser, ArgumentTypeError

from tmrl.tools.init_package.init_tmrl import TMRL_FOLDER as _TMRL_FOLDER
import tmrl.config.config_constants as cfg

# Desktop Duplication must be created before importing the torch-dependent
# configuration graph on hybrid-GPU Windows systems.
_CAPTURE_COMMANDS = {
    "--worker",
    "--expert",
    "--test",
    "--benchmark",
    "--check-environment",
    "--record-reward",
}
if any(argument in _CAPTURE_COMMANDS for argument in sys.argv[1:]):
    from tmrl.custom.tm.utils.window import preinitialize_dxcam_capture

    preinitialize_dxcam_capture()

import tmrl.config.config_objects as cfg_obj

from tmrl.core.envs import GenericGymEnv
from tmrl.core.networking import Server, Trainer, RolloutWorker
from tmrl.core.util import partial

from tmrl.tools.check_environment import check_env_tm20lidar, check_env_tm20full
from tmrl.tools.record import record_reward_dist


def main(args):
    if args.server:
        serv = Server()
        while True:
            time.sleep(1.0)
    elif args.worker or args.test or args.benchmark or args.expert:
        config = cfg_obj.CONFIG_DICT
        config_modifiers = args.config
        for k, v in config_modifiers.items():
            config[k] = v
        worker_device = (
            str(cfg_obj.FOUNDATION_CONFIG.get("WORKER_DEVICE", "cpu"))
            if cfg_obj.IS_JAX_DREAMER
            else "cuda"
            if cfg.CUDA_INFERENCE
            else "cpu"
        )
        rw = RolloutWorker(env_cls=partial(GenericGymEnv, id=cfg.RTGYM_VERSION, gym_kwargs={"config": config}),
                           actor_module_cls=cfg_obj.POLICY,
                           sample_compressor=cfg_obj.SAMPLE_COMPRESSOR,
                           device=worker_device,
                           server_ip=cfg.SERVER_IP_FOR_WORKER,
                           max_samples_per_episode=cfg.RW_MAX_SAMPLES_PER_EPISODE,
                           model_path=cfg.MODEL_PATH_WORKER,
                           obs_preprocessor=cfg_obj.OBS_PREPROCESSOR,
                           crc_debug=cfg.CRC_DEBUG,
                           standalone=args.test)
        if args.test:
            rw.run_episodes(10000)
        elif args.worker:
            rw.run()
        elif args.expert:
            rw.run(expert=True)
        elif args.benchmark:
            rw.run_env_benchmark(nb_steps=1000, test=False)
        else:
            rw.run_episodes(10000)
    elif args.trainer:
        trainer = Trainer(training_cls=cfg_obj.TRAINER,
                          server_ip=cfg.SERVER_IP_FOR_TRAINER,
                          model_path=cfg.MODEL_PATH_TRAINER,
                          checkpoint_path=cfg.CHECKPOINT_PATH,
                          dump_run_instance_fn=cfg_obj.DUMP_RUN_INSTANCE_FN,
                          load_run_instance_fn=cfg_obj.LOAD_RUN_INSTANCE_FN,
                          updater_fn=cfg_obj.UPDATER_FN)
        logging.info(f"--- NOW RUNNING {cfg_obj.ALG_NAME} on TrackMania ---")
        if cfg_obj.IS_JAX_DREAMER:
            dreamer = cfg_obj.CONTINUAL_DREAMER_CONFIG
            logging.info(
                "M1 JAX Dreamer foundation is enabled: replay sequences=%s, "
                "batch=%s, warm-up=%s, imagination horizon=%s. Dynamic experts, "
                "protected replay, consolidation, and AZR remain disabled.",
                dreamer.get("SEQUENCE_LENGTH", 16),
                dreamer.get("BATCH_SIZE", 8),
                dreamer.get("WARMUP_STEPS", 2000),
                dreamer.get("HORIZON", 8),
            )
        elif cfg_obj.ALG_NAME == "DREAMER":
            dreamer = cfg_obj.DREAMER_CONFIG
            logging.info(
                "Recurrent Dreamer control is enabled: replay sequences=%s, "
                "batch=%s, warm-up=%s, imagination horizon=%s. The broadcast "
                "actor contains the encoder, RSSM filter, and latent policy.",
                dreamer.get("SEQUENCE_LENGTH", 16),
                dreamer.get("BATCH_SIZE", 8),
                dreamer.get("WARMUP_STEPS", 2000),
                dreamer.get("HORIZON", 8),
            )
            if cfg_obj.USE_AZR_IMAGINATION:
                logging.info(
                    "AZR latent tasks are connected as prioritized Dreamer "
                    "imagination start states (attempts=%s, proposal interval=%s).",
                    cfg_obj.AZR_CONFIG.get("SOLVER_ATTEMPTS", 4),
                    cfg_obj.AZR_CONFIG.get("PROPOSAL_INTERVAL", 8),
                )
        elif cfg_obj.USE_AZR_IMAGINATION:
            azr = cfg_obj.AZR_CONFIG
            logging.info(
                "AZR imagination is enabled: real-replay warm-up=%s, horizon=%s, "
                "solver attempts=%s. Imagined updates will target the broadcast actor.",
                azr.get("WARMUP_STEPS", 1000),
                azr.get("HORIZON", 3),
                azr.get("SOLVER_ATTEMPTS", 8),
            )
        if args.wandb:
            trainer.run_with_wandb(entity=cfg.WANDB_ENTITY,
                                   project=cfg.WANDB_PROJECT,
                                   run_id=cfg.WANDB_RUN_ID)
        else:
            trainer.run()
    elif args.record_reward:
        record_reward_dist(path_reward=cfg.REWARD_PATH, use_keyboard=args.use_keyboard)
    elif args.check_env:
        if cfg.PRAGMA_LIDAR:
            check_env_tm20lidar()
        else:
            check_env_tm20full()
    elif args.install:
        logging.info(f"TMRL folder: {cfg.TMRL_FOLDER}")
    else:
        raise ArgumentTypeError('Enter a valid argument')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--install', action='store_true', help='checks TMRL installation')
    parser.add_argument('--server', action='store_true', help='launches the server')
    parser.add_argument('--trainer', action='store_true', help='launches the trainer')
    parser.add_argument('--worker', action='store_true', help='launches a rollout worker')
    parser.add_argument('--expert', action='store_true', help='launches an expert rollout worker (no model update)')
    parser.add_argument('--test', action='store_true', help='runs inference without training')
    parser.add_argument('--benchmark', action='store_true', help='runs a benchmark of the environment')
    parser.add_argument('--record-reward', dest='record_reward', action='store_true', help='utility to record a reward function in TM20')
    parser.add_argument('--use-keyboard', dest='use_keyboard', action='store_true', help='modifier for --record-reward')
    parser.add_argument('--check-environment', dest='check_env', action='store_true', help='utility to check the environment')
    parser.add_argument('--wandb', dest='wandb', action='store_true', help='(use with --trainer) if you want to log results on Weights and Biases, use this option')
    parser.add_argument('-d', '--config', type=json.loads, default={}, help='dictionary containing configuration options (modifiers) for the rtgym environment')
    arguments = parser.parse_args()

    main(arguments)
