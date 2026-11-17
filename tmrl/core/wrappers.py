# standard library imports
from collections.abc import Mapping, Sequence

# third-party imports
import gymnasium
import numpy as np


__docformat__ = "google"


def _to_float32_space(space):
    if isinstance(space, gymnasium.spaces.Box) and space.dtype == np.float64:
        return gymnasium.spaces.Box(
            low=space.low.astype(np.float32),
            high=space.high.astype(np.float32),
            shape=space.shape,
            dtype=np.float32,
        )
    if isinstance(space, gymnasium.spaces.Tuple):
        return gymnasium.spaces.Tuple(tuple(_to_float32_space(s) for s in space.spaces))
    if isinstance(space, gymnasium.spaces.Dict):
        return gymnasium.spaces.Dict({k: _to_float32_space(v) for k, v in space.spaces.items()})
    return space


class AffineObservationWrapper(gymnasium.ObservationWrapper):
    """Applies an affine transformation ``(obs + shift) * scale`` to Box observations."""

    def __init__(self, env, shift, scale):
        super().__init__(env)
        assert isinstance(env.observation_space, gymnasium.spaces.Box)
        self.shift = shift
        self.scale = scale
        low = self.observation(env.observation_space.low)
        high = self.observation(env.observation_space.high)
        self.observation_space = gymnasium.spaces.Box(
            np.minimum(low, high),
            np.maximum(low, high),
            dtype=env.observation_space.dtype,
        )

    def observation(self, obs):
        return (obs + self.shift) * self.scale


class Float64ToFloat32(gymnasium.ObservationWrapper):
    """Converts ``np.float64`` arrays and floats in observations to ``np.float32``."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = _to_float32_space(env.observation_space)

    def observation(self, observation):
        observation = deepmap({np.ndarray: float64_to_float32,
                               float: float_to_float32,
                               np.float32: float_to_float32,
                               np.float64: float_to_float32}, observation)
        return observation


# === Utilities ========================================================================================================


def deepmap(f, m):
    """Apply functions to the leaves of a dictionary or sequence, depending on the leaf value type."""
    for cls in f:
        if isinstance(m, cls):
            return f[cls](m)
    if isinstance(m, Sequence) and not isinstance(m, (str, bytes, bytearray)):
        return type(m)(deepmap(f, x) for x in m)
    elif isinstance(m, Mapping):
        return type(m)((k, deepmap(f, m[k])) for k in m)
    else:
        raise AttributeError(f"m is a {type(m)}, not a Sequence nor a Mapping: {m}")


def float64_to_float32(x):
    """Casts ``np.float64`` arrays to ``np.float32`` without altering array shape."""
    return x.astype(np.float32) if x.dtype == np.float64 else x


def float_to_float32(x):
    """Converts a scalar float into a 1-element ``np.float32`` array."""
    return np.asarray([x, ], np.float32)
