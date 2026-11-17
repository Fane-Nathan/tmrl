from abc import ABC
from pathlib import Path
import pickle as pkl

from flax import nnx
import gymnasium
import jax

from tmrl.core.actor import ActorModule
from tmrl.core.jax.util import collate_jax, resolve_jax_device


__docformat__ = "google"


class NNXActorModule(ActorModule, nnx.Module, ABC):
    """
    Partial implementation of `ActorModule` as a `nnx.Module`.

    You can implement this instead of `ActorModule` when using Jax.
    `NNXActorModule` is a subclass of `nnx.Module`.

    When using `NNXActorModule`, the device is selected by Jax.

    .. note::
       If overridden, the ``__init__()`` definition must at least take the two following arguments (args or kwargs):
       ``observation_space`` and ``action_space``.
       When overriding ``__init__``, don't forget to call ``super().__init__`` in the subclass.
    """
    def __init__(self,
                 observation_space: gymnasium.spaces.Space,
                 action_space: gymnasium.spaces.Space,
                 device=None):
        """
        Args:
            observation_space (gymnasium.spaces.Space): observation space (here for your convenience)
            action_space (gymnasium.spaces.Space): action space (here for your convenience)
            device: optional device specification (stored as a picklable spec such as ``str`` or ``None``)
        """
        ActorModule.__init__(self, observation_space, action_space)
        # Keep device as a picklable specification (e.g. "cpu", "cuda:0", or None).
        self.device = device

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _, state = nnx.split(self)
        with open(path, 'wb') as f:
            pkl.dump(state, f)

    def load(self, path, device):
        with open(path, 'rb') as f:
            state = pkl.load(f)
        nnx.update(self, state)
        if device is not None:
            self.to_device(device)
        return self

    def act_(self, obs, test=False):
        """Transforms ``obs`` into a tree of batched JAX arrays and calls :meth:`act`."""
        device = resolve_jax_device(self.device)
        obs = collate_jax([obs], device=device)
        action = self.act(obs, test=test)
        return action

    def to_device(self, device: str):
        self.device = device  # store picklable device spec
        if device is not None:
            resolved = resolve_jax_device(device)
            _, state = nnx.split(self)
            state = jax.device_put(state, resolved)
            nnx.update(self, state)
        return self
