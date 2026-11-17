"""TMRL: Network-based framework for real-time robot learning."""

from importlib.metadata import PackageNotFoundError, version as _package_version
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "__version__",
    "CONFIG_DICT",
    "GenericGymEnv",
    "TMRL_FOLDER",
    "get_environment",
]

_FALLBACK_VERSION = "0.7.1"


def get_environment():
    """Return the default TMRL Gymnasium environment for TrackMania 2020."""
    import tmrl.config.config_constants as cfg
    from tmrl.config.config_objects import CONFIG_DICT
    from tmrl.core.envs import GenericGymEnv

    return GenericGymEnv(id=cfg.RTGYM_VERSION, gym_kwargs={"config": CONFIG_DICT})


def __getattr__(name: str):
    if name == "__version__":
        try:
            return _package_version("tmrl")
        except PackageNotFoundError:
            return _FALLBACK_VERSION
    if name == "GenericGymEnv":
        from tmrl.core.envs import GenericGymEnv

        return GenericGymEnv
    if name == "CONFIG_DICT":
        from tmrl.config.config_objects import CONFIG_DICT

        return CONFIG_DICT
    if name == "TMRL_FOLDER":
        from tmrl.tools.init_package.init_tmrl import TMRL_FOLDER

        return TMRL_FOLDER
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
