"""Initialize and manage the TMRL data directory (``~/TmrlData``)."""

import argparse
import logging
import platform
from pathlib import Path
from shutil import copy2, copytree
import socket
import sys
import tempfile
from typing import Sequence
import urllib.error
import urllib.request
from zipfile import ZipFile


RESOURCES_URL = "https://github.com/trackmania-rl/tmrl/releases/download/v0.6.0/resources.zip"
TMRL_FOLDER = Path.home() / "TmrlData"

_LEGACY_OPENPLANET_SCRIPTS = (
    "Plugin_GrabData_0_1.as",
    "Plugin_GrabData_0_1.as.sig",
    "Plugin_GrabData_0_2.as",
    "Plugin_GrabData_0_2.as.sig",
)
_OPENPLANET_PLUGINS = (
    "TMRL_GrabData.op",
    "TMRL_SaveGhost.op",
)


def rmdir(directory: Path | str) -> None:
    """Recursively delete ``directory`` and all of its contents."""
    directory = Path(directory)
    for item in directory.iterdir():
        if item.is_dir():
            rmdir(item)
        else:
            item.unlink()
    directory.rmdir()


def url_retrieve(url: str, outfile: Path | str, overwrite: bool = False) -> None:
    """Download ``url`` to ``outfile``, raising ``ConnectionError`` on network failures."""
    outfile = Path(outfile).expanduser().resolve()
    if outfile.is_dir():
        raise ValueError("Please specify full filepath, including filename")
    if overwrite or not outfile.is_file():
        outfile.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(url, str(outfile))
        except (socket.gaierror, urllib.error.URLError) as err:
            raise ConnectionError(f"could not download {url} due to {err}") from err


def _install_openplanet_plugins(
    resources_folder: Path,
    openplanet_folder: Path | None = None,
) -> None:
    """Copy TMRL OpenPlanet plugins on Windows when ``OpenplanetNext`` exists."""
    if platform.system() != "Windows":
        return
    if openplanet_folder is None:
        openplanet_folder = Path.home() / "OpenplanetNext"

    if not openplanet_folder.exists():
        logging.warning(
            "The OpenPlanet folder was not found at %s. "
            "Please copy the OpenPlanet plugins manually for TrackMania 2020 support.",
            openplanet_folder,
        )
        return

    try:
        op_scripts_folder = openplanet_folder / "Scripts"
        if op_scripts_folder.exists():
            for filename in _LEGACY_OPENPLANET_SCRIPTS:
                old_file = op_scripts_folder / filename
                if old_file.exists():
                    old_file.unlink()

        op_plugins_folder = openplanet_folder / "Plugins"
        op_plugins_folder.mkdir(parents=True, exist_ok=True)
        for plugin_name in _OPENPLANET_PLUGINS:
            copy2(resources_folder / "Plugins" / plugin_name, op_plugins_folder)
    except Exception as exc:
        logging.warning(
            "An exception was caught when trying to copy the OpenPlanet plugin automatically. "
            "Please copy the plugin manually for TrackMania 2020 support. Error: %s",
            exc,
        )


def init_tmrl_data(
    tmrl_folder: Path | str | None = None,
    resources_url: str = RESOURCES_URL,
) -> Path:
    """Wipe and re-generate the ``TmrlData`` folder atomically after downloading resources."""
    target_folder = (
        Path(tmrl_folder).expanduser()
        if tmrl_folder is not None
        else Path.home() / "TmrlData"
    )

    with tempfile.TemporaryDirectory(prefix="tmrl_init_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        resources_zip = tmp_dir / "resources.zip"
        url_retrieve(resources_url, resources_zip, overwrite=True)

        extracted_dir = tmp_dir / "extracted"
        with ZipFile(resources_zip, "r") as zip_ref:
            zip_ref.extractall(extracted_dir)

        staged_resources = extracted_dir / "resources"
        if not staged_resources.is_dir():
            raise RuntimeError(
                f"Downloaded archive from {resources_url} is missing the 'resources/' directory."
            )

        if target_folder.exists():
            rmdir(target_folder)
        if target_folder.exists():
            raise RuntimeError(f"Failed to delete {target_folder}")

        checkpoints_folder = target_folder / "checkpoints"
        dataset_folder = target_folder / "dataset"
        reward_folder = target_folder / "reward"
        weights_folder = target_folder / "weights"
        config_folder = target_folder / "config"
        for folder in (
            checkpoints_folder,
            dataset_folder,
            reward_folder,
            weights_folder,
            config_folder,
        ):
            folder.mkdir(parents=True, exist_ok=True)

        resources_folder = target_folder / "resources"
        copytree(staged_resources, resources_folder)

        copy2(resources_folder / "config.json", config_folder)
        copy2(resources_folder / "reward.pkl", reward_folder)
        copy2(resources_folder / "SAC_4_LIDAR_pretrained.tmod", weights_folder)
        copy2(resources_folder / "SAC_4_imgs_pretrained.tmod", weights_folder)

    _install_openplanet_plugins(resources_folder)
    return target_folder


def ensure_tmrl_data(
    tmrl_folder: Path | str | None = None,
    force: bool = False,
) -> Path:
    """Ensure ``TmrlData`` exists without wiping existing data unless ``force=True``."""
    target = (
        Path(tmrl_folder).expanduser()
        if tmrl_folder is not None
        else Path.home() / "TmrlData"
    )
    if target.exists() and not force:
        logging.info(
            "TMRL data folder already exists at %s; skipping download (pass --force to reset).",
            target,
        )
        return target

    logging.info("Initializing TMRL data folder at %s...", target)
    return init_tmrl_data(tmrl_folder=target)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for ``tmrl-init``."""
    parser = argparse.ArgumentParser(
        prog="tmrl-init",
        description="Initialize the TMRL data folder (~/TmrlData).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Wipe and re-download the TMRL data folder if it already exists.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Target TMRL data directory (defaults to ~/TmrlData).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``tmrl-init``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    target = ensure_tmrl_data(tmrl_folder=args.path, force=args.force)
    logging.info("TMRL data folder ready: %s", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
