"""Checkpoint guards for the separately versioned JAX Dreamer pipeline."""

from tmrl.core.util import cloudpickle_dump, cloudpickle_load
from tmrl.custom.jax.continual_dreamer import assert_finite_nnx_state


def dump_continual_jax_run_instance(run_instance, checkpoint_path):
    """Atomically serialize NNX/Optax state, including optimizer closures."""

    agent = getattr(run_instance, "agent", None)
    if agent is None:
        raise RuntimeError("continual JAX checkpoint has no training agent")
    assert_finite_nnx_state(agent, "continual JAX checkpoint agent")
    cloudpickle_dump(run_instance, checkpoint_path)


def load_continual_jax_run_instance(checkpoint_path):
    return cloudpickle_load(checkpoint_path)


def validate_continual_jax_run_instance(run_instance, training_cls):
    """Reject accidental loading of a Torch/SAC checkpoint into JAX Dreamer."""

    del training_cls
    agent = getattr(run_instance, "agent", None)
    mode = getattr(agent, "checkpoint_mode", None)
    if mode != "CONTINUAL_DREAMER_JAX":
        raise RuntimeError(
            "The selected RUN_NAME points to a non-JAX checkpoint. Choose a new "
            "RUN_NAME for CONTINUAL_DREAMER_JAX; the existing checkpoint was not "
            "modified."
        )
    assert_finite_nnx_state(agent, "loaded continual JAX checkpoint agent")
    return run_instance
