import argparse
import logging
import time

import tmrl.config.config_constants as cfg
from tmrl.networking import Server
from tmrl.world_model.config_objects import RUN_NAME, make_trainer, make_worker


def main():
    parser = argparse.ArgumentParser(description="TMRL latent world-model baseline")
    parser.add_argument("mode", choices=("server", "trainer", "worker", "test"))
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    if args.mode == "server":
        # Keep a strong reference to the Server for the lifetime of this process.
        # Without it, the object may be garbage-collected immediately, which
        # tears down the tlspyo relay and causes trainer/worker connection refusals.
        server = Server()
        while True:
            time.sleep(1.0)
    elif args.mode == "trainer":
        trainer = make_trainer()
        logging.info("--- NOW RUNNING WORLD_MODEL_V1 on TrackMania ---")
        if args.wandb:
            trainer.run_with_wandb(
                entity=cfg.WANDB_ENTITY,
                project=cfg.WANDB_PROJECT,
                run_id=RUN_NAME,
            )
        else:
            trainer.run()
    else:
        worker = make_worker(standalone=args.mode == "test")
        if args.mode == "test":
            worker.run_episodes(10000)
        else:
            worker.run()


if __name__ == "__main__":
    main()
