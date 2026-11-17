from abc import ABC
from pathlib import Path

import torch

from tmrl.core.actor import ActorModule
from tmrl.core.torch.util import collate_torch


__docformat__ = "google"


class TorchActorModule(ActorModule, torch.nn.Module, ABC):
    """
    Partial implementation of `ActorModule` as a `torch.nn.Module`.

    You can implement this instead of `ActorModule` when using PyTorch.
    `TorchActorModule` is a subclass of `torch.nn.Module` and may implement `forward()`.
    Typically, your implementation of `act()` can call `forward()` with gradients turned off.

    When using `TorchActorModule`, the `act` method receives observations collated on `device`,
    with an additional dimension corresponding to the batch size.

    .. note::
       If overridden, the ``__init__()`` definition must at least take the two following arguments (args or kwargs):
       ``observation_space`` and ``action_space``.
       When overriding ``__init__``, don't forget to call ``super().__init__`` in the subclass.
    """
    def __init__(self, observation_space, action_space, device="cpu"):
        """
        Args:
            observation_space (gymnasium.spaces.Space): observation space (here for your convenience)
            action_space (gymnasium.spaces.Space): action space (here for your convenience)
            device: device where your model should live and where observations for `act` will be collated
        """
        super().__init__(observation_space, action_space)
        self.device = device

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path, device):
        self.device = device
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        if device is not None:
            self.to(device)
        return self

    def act_(self, obs, test=False):
        """Transforms ``obs`` into a tree of batched torch tensors and calls :meth:`act`."""
        obs = collate_torch([obs], device=self.device)
        with torch.no_grad():
            action = self.act(obs, test=test)
        return action

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        if "device" in kwargs and kwargs["device"] is not None:
            self.device = kwargs["device"]
        elif args:
            first = args[0]
            if isinstance(first, torch.Tensor):
                self.device = first.device
            elif isinstance(first, (str, torch.device, int)) and not isinstance(first, bool):
                self.device = first
        return result

    def to_device(self, device):
        return self.to(device)
