#!/usr/bin/env python3
"""Compile and execute one real-replay plus imagination update on JAX GPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from gymnasium import spaces

from tmrl.custom.jax.continual_dreamer import JAXDreamerAgent


def _spaces(image_size: int):
    observation_space = spaces.Tuple(
        (
            spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            spaces.Box(0.0, 1.0, (4, image_size, image_size), dtype=np.float32),
            spaces.Box(-1.0, 1.0, (3,), dtype=np.float32),
        )
    )
    action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
    return observation_space, action_space


def _batch(batch_size: int, sequence_length: int, image_size: int):
    key = jax.random.key(71)

    def observations(offset: int):
        images = jax.random.uniform(
            jax.random.fold_in(key, offset),
            (batch_size, sequence_length, 4, image_size, image_size),
        )
        return (
            jnp.full((batch_size, sequence_length, 1), 0.35, dtype=jnp.float32),
            jnp.full((batch_size, sequence_length, 1), 0.4, dtype=jnp.float32),
            jnp.full((batch_size, sequence_length, 1), 0.55, dtype=jnp.float32),
            images,
            jnp.zeros((batch_size, sequence_length, 3), dtype=jnp.float32),
        )

    is_first = jnp.zeros((batch_size, sequence_length), dtype=jnp.bool_)
    is_first = is_first.at[:, 0].set(True)
    return (
        observations(0),
        jnp.zeros((batch_size, sequence_length, 3), dtype=jnp.float32),
        jnp.linspace(-0.1, 0.2, sequence_length)[None].repeat(batch_size, axis=0),
        observations(1),
        jnp.zeros((batch_size, sequence_length), dtype=jnp.float32),
        jnp.zeros((batch_size, sequence_length), dtype=jnp.float32),
        is_first,
    )


def _parameters(module):
    return [
        np.asarray(jax.device_get(value)).copy()
        for value in jax.tree.leaves(nnx.state(module, nnx.Param))
    ]


def _changed(before, module):
    after = _parameters(module)
    return any(not np.array_equal(old, new) for old, new in zip(before, after))


def _write_json_atomic(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run_smoke(image_size=64, require_gpu=True):
    backend = jax.default_backend()
    devices = jax.devices()
    if require_gpu and backend != "gpu":
        raise RuntimeError(f"JAX GPU backend required; selected backend is {backend!r}")

    observation_space, action_space = _spaces(image_size)
    agent = JAXDreamerAgent(
        observation_space=observation_space,
        action_space=action_space,
        latent_dim=16,
        hidden_dim=32,
        policy_hidden_dim=32,
        encoder_channels=(8, 16, 32, 32),
        burn_in=1,
        imagination_horizon=2,
        imagination_batch_size=4,
        world_model_warmup_steps=0,
        seed=29,
    )
    world_before = _parameters(agent.actor.world_model)
    policy_before = _parameters(agent.actor.policy)
    batch = _batch(batch_size=2, sequence_length=4, image_size=image_size)

    started = time.perf_counter()
    metrics = agent.train(batch)
    jax.tree.map(
        lambda value: value.block_until_ready()
        if hasattr(value, "block_until_ready")
        else value,
        metrics,
    )
    compile_and_train_seconds = time.perf_counter() - started

    warmed_started = time.perf_counter()
    warmed_metrics = agent.train(batch)
    jax.tree.map(
        lambda value: value.block_until_ready()
        if hasattr(value, "block_until_ready")
        else value,
        warmed_metrics,
    )
    warmed_train_seconds = time.perf_counter() - warmed_started
    world_changed = _changed(world_before, agent.actor.world_model)
    policy_changed = _changed(policy_before, agent.actor.policy)
    if not world_changed or not policy_changed:
        raise RuntimeError("JAX Dreamer smoke step did not update deployed parameters")

    return {
        "backend": backend,
        "devices": [str(device) for device in devices],
        "image_size": int(image_size),
        "compile_and_train_seconds": float(compile_and_train_seconds),
        "warmed_train_seconds": float(warmed_train_seconds),
        "world_model_changed": world_changed,
        "deployed_policy_changed": policy_changed,
        "metrics": {
            name: float(np.asarray(jax.device_get(value)))
            for name, value in warmed_metrics.items()
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args(argv)
    result = run_smoke(
        image_size=args.image_size,
        require_gpu=not args.allow_cpu,
    )
    if args.output is not None:
        _write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
