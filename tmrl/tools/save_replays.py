"""Run standalone rollout episodes with TrackMania replay saving enabled."""

from __future__ import annotations

import argparse
import logging
import math
import platform
import sys
from typing import Sequence


def _maybe_preinitialize_dxcam() -> None:
    """Initialize DXcam on Windows before torch/config_objects are imported."""
    if platform.system() != "Windows":
        return
    try:
        from tmrl.custom.tm.utils.window import preinitialize_dxcam_capture
    except ImportError as exc:
        raise RuntimeError(
            f"Saving replays on Windows requires optional dependencies (missing: {exc.name}). "
            'Install them with `pip install "tmrl[torch,trackmania]"`.'
        ) from exc
    preinitialize_dxcam_capture()


def save_replays(nb_replays: int | float = math.inf) -> None:
    """Run standalone episodes with ``interface_kwargs['save_replays'] = True``."""
    if nb_replays <= 0:
        raise ValueError("nb_replays must be >= 1")

    _maybe_preinitialize_dxcam()

    try:
        import tmrl.config.config_constants as cfg
        import tmrl.config.config_objects as cfg_obj
        from tmrl.core.envs import GenericGymEnv
        from tmrl.core.networking import RolloutWorker
        from tmrl.core.util import partial

        config = dict(cfg_obj.CONFIG_DICT)
        interface_kwargs = dict(config.get("interface_kwargs") or {})
        interface_kwargs["save_replays"] = True
        config["interface_kwargs"] = interface_kwargs

        worker_device = (
            str(cfg_obj.FOUNDATION_CONFIG.get("WORKER_DEVICE", "cpu"))
            if cfg_obj.IS_JAX_DREAMER
            else "cuda"
            if cfg.CUDA_INFERENCE
            else "cpu"
        )
        policy_cls = cfg_obj.POLICY
        sample_compressor = cfg_obj.SAMPLE_COMPRESSOR
        obs_preprocessor = cfg_obj.OBS_PREPROCESSOR
    except ImportError as exc:
        raise RuntimeError(
            f"Saving replays requires optional runtime dependencies (missing: {exc.name}). "
            'Install them with `pip install "tmrl[torch,trackmania]"`.'
        ) from exc

    rw = RolloutWorker(
        env_cls=partial(GenericGymEnv, id=cfg.RTGYM_VERSION, gym_kwargs={"config": config}),
        actor_module_cls=policy_cls,
        sample_compressor=sample_compressor,
        device=worker_device,
        server_ip=cfg.SERVER_IP_FOR_WORKER,
        model_path=cfg.MODEL_PATH_WORKER,
        obs_preprocessor=obs_preprocessor,
        crc_debug=cfg.CRC_DEBUG,
        standalone=True,
    )
    try:
        rw.run_episodes(10000, nb_episodes=nb_replays)
    finally:
        rw.stop()


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for replay recording."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nb_replays",
        type=int,
        default=math.inf,
        help="Number of replays to record (default: unlimited).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and record TrackMania replays."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.nb_replays < 1:
        parser.error("--nb_replays must be >= 1")
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    try:
        save_replays(args.nb_replays)
    except KeyboardInterrupt:
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Replay recording failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
