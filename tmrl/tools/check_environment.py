"""Live TrackMania 2020 environment and screen-capture verification tool."""

from __future__ import annotations

import argparse
import logging
import platform
import sys
from typing import Any, Sequence

import tmrl.config.config_constants as cfg


def _require_trackmania_runtime() -> tuple[Any, ...]:
    """Lazily import TrackMania runtime dependencies with actionable errors."""
    try:
        import cv2
        import gymnasium
        from rtgym.envs.real_time_env import DEFAULT_CONFIG_DICT

        from tmrl.custom.tm.tm_gym_interfaces import (
            TM2020Interface,
            TM2020InterfaceLidar,
        )
        from tmrl.custom.tm.utils.tools import Lidar
        from tmrl.custom.tm.utils.window import WindowInterface
    except ImportError as exc:
        platform_hint = (
            "pywin32 and dxcam on Windows"
            if platform.system() == "Windows"
            else "mss on Linux"
        )
        raise RuntimeError(
            f"TrackMania environment check requires optional dependencies (missing: {exc.name}). "
            f'Install them with `pip install "tmrl[trackmania]"` (including {platform_hint}).'
        ) from exc
    return (
        gymnasium,
        cv2,
        DEFAULT_CONFIG_DICT,
        TM2020Interface,
        TM2020InterfaceLidar,
        WindowInterface,
        Lidar,
    )


def _maybe_preinitialize_dxcam() -> None:
    """Initialize DXcam on Windows before live TrackMania capture starts."""
    if platform.system() != "Windows":
        return
    try:
        from tmrl.custom.tm.utils.window import preinitialize_dxcam_capture
    except ImportError as exc:
        raise RuntimeError(
            f"TrackMania Windows capture requires optional dependencies (missing: {exc.name}). "
            'Install them with `pip install "tmrl[trackmania]"` (including pywin32 and dxcam).'
        ) from exc
    preinitialize_dxcam_capture()


def check_env_tm20lidar() -> None:
    """Run an interactive LiDAR TrackMania environment check loop."""
    (
        gymnasium,
        cv2,
        default_config_dict,
        _,
        tm2020_interface_lidar,
        window_interface_cls,
        lidar_cls,
    ) = _require_trackmania_runtime()

    window_interface = window_interface_cls("Trackmania")
    if cfg.SYSTEM != "Windows":
        window_interface.move_and_resize()  # needed on Linux
    lidar = lidar_cls(window_interface.screenshot())
    env_config = default_config_dict.copy()
    env_config["interface"] = tm2020_interface_lidar
    env_config["wait_on_done"] = True
    env_config["interface_kwargs"] = {"img_hist_len": 1, "gamepad": False}

    env = None
    try:
        env = gymnasium.make(cfg.RTGYM_VERSION, config=env_config)
        env.reset()
        while True:
            _, r, d, t, _ = env.step(None)
            logging.info(f"r:{r}, d:{d}, t:{t}")
            if d or t:
                env.reset()
            img = window_interface.screenshot()[:, :, :3]
            lidar.lidar_20(img, True)
    finally:
        if env is not None:
            env.close()
        cv2.destroyAllWindows()


def show_imgs(imgs: Any, scale: float | None = None) -> None:
    """Display a batch of grayscale or color observation frames via OpenCV."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            f"Displaying environment frames requires OpenCV (missing: {exc.name}). "
            'Install it with `pip install "tmrl[trackmania]"`.'
        ) from exc

    resolved_scale = cfg.IMG_SCALE_CHECK_ENV if scale is None else scale
    imshape = imgs.shape
    if len(imshape) == 3:  # grayscale
        nb, h, w = imshape
        concat = imgs.reshape((nb * h, w))
    elif len(imshape) == 4:  # color
        nb, h, w, c = imshape
        concat = imgs.reshape((nb * h, w, c))
    else:
        raise ValueError(f"Expected a 3D or 4D image batch, got shape {imshape}.")

    width = max(1, int(concat.shape[1] * resolved_scale))
    height = max(1, int(concat.shape[0] * resolved_scale))
    cv2.imshow(
        "Environment",
        cv2.resize(concat, (width, height), interpolation=cv2.INTER_NEAREST),
    )
    cv2.waitKey(1)


def check_env_tm20full() -> None:
    """Run an interactive image-based TrackMania environment check loop."""
    (
        gymnasium,
        cv2,
        default_config_dict,
        tm2020_interface,
        _,
        _,
        _,
    ) = _require_trackmania_runtime()

    env_config = default_config_dict.copy()
    env_config["interface"] = tm2020_interface
    env_config["wait_on_done"] = True
    env_config["interface_kwargs"] = {
        "gamepad": False,
        "grayscale": cfg.GRAYSCALE,
        "resize_to": (cfg.IMG_WIDTH, cfg.IMG_HEIGHT),
    }

    env = None
    try:
        env = gymnasium.make(cfg.RTGYM_VERSION, config=env_config)
        o, _ = env.reset()
        show_imgs(o[3])
        logging.info(
            f"o:[{o[0].item():05.01f}, {o[1].item():03.01f}, {o[2].item():07.01f}, imgs({len(o[3])})]"
        )
        while True:
            o, r, d, t, _ = env.step(None)
            show_imgs(o[3])
            logging.info(
                f"r:{r:.2f}, d:{d}, t:{t}, o:[{o[0].item():05.01f}, {o[1].item():03.01f}, {o[2].item():07.01f}, imgs({len(o[3])})]"
            )
            if d or t:
                o, _ = env.reset()
                show_imgs(o[3])
                logging.info(
                    f"o:[{o[0].item():05.01f}, {o[1].item():03.01f}, {o[2].item():07.01f}, imgs({len(o[3])})]"
                )
    finally:
        if env is not None:
            env.close()
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for checking the TrackMania environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--lidar",
        action="store_true",
        help="Check the TrackMania LiDAR interface regardless of config.json.",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="Check the TrackMania full-image interface regardless of config.json.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and launch the TrackMania environment check."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    use_lidar = True if args.lidar else False if args.full else bool(cfg.PRAGMA_LIDAR)
    try:
        _maybe_preinitialize_dxcam()
        if use_lidar:
            check_env_tm20lidar()
        else:
            check_env_tm20full()
    except KeyboardInterrupt:
        return 0
    except (RuntimeError, ImportError, ValueError, OSError) as exc:
        print(f"Environment check failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

