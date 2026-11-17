# TMRL tutorial scripts

For the opt-in recurrent world-model controller, see
[Recurrent Dreamer control](../readme/dreamer_control.md).

:warning: _For simplicity, the "tuto" scripts launch the `Server`, the `Trainer` and a `RolloutWorker` in one go.
In real-world applications, you often want to launch these entities in separate terminals / machines instead.
See for instance [this script](competition/custom_actor_module.py)._

## "Plug-and-play" TMRL pipelines:

These scripts describe straightforward ways of implementing TMRL pipelines with minimal effort.

- [tuto_minimal_drone.py](tuto_minimal_drone.py) shows how to deploy a PyTorch TMRL pipeline on your robot.
  - `GenericTorchMemory` works with arbitrary (nested) observation/action spaces, but sampling is slow due to the collate function.
  - `ArrayTorchMemory` is much faster at sampling, but works only with homogeneous numpy arrays.
- [tuto_minimal_pendulum_torch.py](tuto_minimal_pendulum_torch.py) describes how TMRL can be used for training policies in classic (non-real-time) environments.
  While TMRL primarily targets real-time environments, this script illustrates optional synchronization mechanisms.
- [tuto_minimal_pendulum_jax.py](tuto_minimal_pendulum_jax.py) is an equivalent of the previous script, using the TMRL pipeline for JAX / Flax NNX instead of PyTorch.
  It illustrates how to leverage TMRL JAX support to develop optimized training pipelines.

## Diagnostics, self-play, and world-model scripts:

- [debug_all_neural_networks.py](debug_all_neural_networks.py) runs offline CPU/GPU smoke diagnostics across the PyTorch neural network modules.
- [train_world_model_on_replay.py](train_world_model_on_replay.py) trains the JAX/Flax NNX latent world model and imagination policy on a saved `.tcpt` replay checkpoint (`--checkpoint PATH`).
- [tuto_self_play_jax.py](tuto_self_play_jax.py) verifies self-play trajectory ratcheting and latent world-model components.

The experimental self-play and world-model scripts are described in
[AZR, Latent Imagination, and Zero-Shot Transfer](../readme/azr_latent_imagination.md).
Their smoke tests verify execution only; use the held-out evaluation protocol in
that note before making a zero-shot-transfer claim.

## Ad-hoc optimized TMRL pipelines:

[tuto.py](tuto.py) is the script discussed in the [long TMRL tutorial](../readme/tuto_library.md).
This tutorial is meant to teach you how to implement your own optimized TMRL objects, such as an Internet bandwidth-efficient and memory-efficient pipeline for observation spaces containing histories of images.

Note that, while this tutorial teaches you how to implement a subclass of `TorchMemory`, it is advisable to rather directly implement a subclass of `Memory` in sampling-bound applications (i.e., directly sampling minibatches to avoid collating).
If interested, please read `tmrl/core/memory.py`.
