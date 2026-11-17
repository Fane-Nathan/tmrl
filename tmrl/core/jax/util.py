from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np


__docformat__ = "google"

__all__ = ["MAX_SEED", "collate_jax", "get_rngs", "resolve_jax_device"]

MAX_SEED = 2**32


def resolve_jax_device(device):
    """Resolves a device specification into a ``jax.Device`` object (or ``None``).

    Args:
        device: ``None``, a ``jax.Device`` instance, a backend name such as
            ``"cpu"`` or ``"cuda"``, or an indexed string such as ``"cuda:0"``.

    Returns:
        jax.Device | None: The resolved JAX device, or ``None`` if ``device`` is ``None``.
    """
    if device is None:
        return None
    if isinstance(device, jax.Device):
        return device
    if isinstance(device, str):
        if ":" in device:
            backend, idx_str = device.split(":", 1)
            try:
                idx = int(idx_str)
            except ValueError as exc:
                raise ValueError(f"Invalid JAX device specification: {device!r}") from exc
            devices = jax.devices(backend)
            if idx < 0 or idx >= len(devices):
                raise ValueError(
                    f"Device index {idx} out of range for backend {backend!r} ({len(devices)} available)."
                )
            return devices[idx]
        return jax.devices(device)[0]
    raise TypeError(f"Unsupported JAX device specification: {type(device).__name__}")


def collate_jax(batch, device=None):
    """Turns a batch of nested structures with numpy arrays as leaves into batched JAX arrays."""
    if not batch:
        raise ValueError("collate_jax requires a non-empty batch.")
    collated = jax.tree.map(lambda *xs: jnp.stack(xs), *batch)
    if device is not None:
        collated = jax.device_put(collated, resolve_jax_device(device))
    return collated


# === randomness =======================================================================================================


# This function is NOT jit-compilable
def get_rngs(
    params_seed=None,
    dropout_seed=None,
    noise_seed=None,
):
    """Returns an ``nnx.Rngs`` with default seeds for ``"params"``, ``"dropout"``, and ``"noise"``."""
    if params_seed is None:
        params_seed = int(np.random.randint(0, MAX_SEED, dtype=np.uint32))
    if dropout_seed is None:
        dropout_seed = int(np.random.randint(0, MAX_SEED, dtype=np.uint32))
    if noise_seed is None:
        noise_seed = int(np.random.randint(0, MAX_SEED, dtype=np.uint32))

    return nnx.Rngs(
        params=jax.random.key(params_seed),
        dropout=jax.random.key(dropout_seed),
        noise=jax.random.key(noise_seed),
    )
