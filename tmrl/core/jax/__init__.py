"""JAX/Flax NNX backend primitives for TMRL."""

from importlib import import_module
import logging
import multiprocessing as mp


__all__ = [
    "NNXActorModule",
    "NNXTrainingOffline",
    "collate_jax",
    "ensure_spawn_start_method",
    "get_rngs",
    "resolve_jax_device",
]

_LAZY_EXPORTS = {
    "NNXActorModule": ("tmrl.core.jax.actor", "NNXActorModule"),
    "NNXTrainingOffline": ("tmrl.core.jax.training_offline", "NNXTrainingOffline"),
    "collate_jax": ("tmrl.core.jax.util", "collate_jax"),
    "get_rngs": ("tmrl.core.jax.util", "get_rngs"),
    "resolve_jax_device": ("tmrl.core.jax.util", "resolve_jax_device"),
}


def ensure_spawn_start_method() -> str:
    """Ensures multiprocessing uses ``'spawn'`` when no start method has been configured.

    JAX is not fork-safe; if another component already selected a non-``'spawn'``
    start method, this logs a warning without overriding the caller's setting.

    Returns:
        str: The active multiprocessing start method.
    """
    current = mp.get_start_method(allow_none=True)
    if current is None:
        try:
            mp.set_start_method("spawn")
            return "spawn"
        except RuntimeError:
            return mp.get_start_method(allow_none=True) or "spawn"
    if current != "spawn":
        logging.warning(
            "Multiprocessing start method is already set to %r; JAX is not fork-safe.",
            current,
        )
    return current


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
