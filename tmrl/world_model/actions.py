import torch
import torch.nn.functional as F


def canonicalize_tm_action(action):
    """Map model controls to the physical TrackMania actuator semantics.

    TMRL exposes a three-dimensional Box action, but the actual gamepad uses
    positive values only for gas and brake. Negative pedal values therefore
    represent the same physical command as zero. We also resolve simultaneous
    gas/brake commands into one net longitudinal command so the world model
    does not have to learn redundant actuator representations.

    Args:
        action: tensor with final dimension ``[gas, brake, steer]``.

    Returns:
        Tensor with gas/brake in [0, 1], steer in [-1, 1], and at most one
        pedal active at a time.
    """
    if action.shape[-1] != 3:
        raise ValueError(f"TrackMania actions must have dimension 3, got {action.shape[-1]}.")

    gas = action[..., 0].clamp(0.0, 1.0)
    brake = action[..., 1].clamp(0.0, 1.0)
    net_longitudinal = gas - brake
    gas = F.relu(net_longitudinal)
    brake = F.relu(-net_longitudinal)
    steer = action[..., 2].clamp(-1.0, 1.0)
    return torch.stack((gas, brake, steer), dim=-1)
