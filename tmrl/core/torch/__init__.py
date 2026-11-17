"""PyTorch backend primitives for TMRL."""

from importlib import import_module

try:
    import torch  # noqa: F401
except ImportError as exc:
    raise ImportError("TMRL must be installed with torch support to use this package.") from exc


__all__ = [
    "TorchActorModule",
    "TorchMemory",
    "TorchTrainingOffline",
    "collate_torch",
    "concat_collated",
]

_LAZY_EXPORTS = {
    "TorchActorModule": ("tmrl.core.torch.actor", "TorchActorModule"),
    "TorchMemory": ("tmrl.core.torch.memory", "TorchMemory"),
    "TorchTrainingOffline": ("tmrl.core.torch.training_offline", "TorchTrainingOffline"),
    "collate_torch": ("tmrl.core.torch.util", "collate_torch"),
    "concat_collated": ("tmrl.core.torch.util", "concat_collated"),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
