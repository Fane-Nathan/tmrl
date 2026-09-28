"""JAX-facing adapters for the existing episode-safe TrackMania replay."""

from __future__ import annotations

import time

import jax
import numpy as np
import torch

from tmrl.custom.torch.custom_memories import ArrayTorchMemoryTMFullSequence


def _torch_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class ArrayJaxMemoryTMFullSequence(ArrayTorchMemoryTMFullSequence):
    """Reuse compatible replay storage and return JAX sequence batches.

    Storage and episode-safe indexing stay on CPU. Only the sampled batch is
    transferred to JAX's selected trainer device. This is the M1 migration
    bridge; the protected continual archive is introduced later in M3.
    """

    def __init__(self, device=None, **kwargs):
        self.jax_device = device
        self.jax_transfer_time = 0.0
        self.jax_transfer_iterations = 0
        super().__init__(device="cpu", **kwargs)

    def _target_device(self):
        if self.jax_device is None:
            return None
        if hasattr(self.jax_device, "platform"):
            return self.jax_device
        devices = jax.devices(str(self.jax_device))
        if not devices:
            raise RuntimeError(f"No JAX device found for {self.jax_device!r}")
        return devices[0]

    def sample(self):
        torch_batch = super().sample()
        started = time.perf_counter()
        target = self._target_device()
        batch = jax.tree.map(
            lambda value: jax.device_put(_torch_to_numpy(value), target),
            torch_batch,
        )
        self.jax_transfer_time += time.perf_counter() - started
        self.jax_transfer_iterations += 1
        return batch

    def get_benchmarks(self):
        base = tuple(super().get_benchmarks())
        if self.jax_transfer_iterations:
            transfer = self.jax_transfer_time / self.jax_transfer_iterations
        else:
            transfer = 0.0
        self.jax_transfer_time = 0.0
        self.jax_transfer_iterations = 0
        return (*base, transfer)

    def get_benchmarks_names(self):
        return (*super().get_benchmarks_names(), "jax_transfer")

