"""Small, dependency-free helpers for hyperbolic replay experiments.

The Dreamer replay buffer uses these functions only as a detached sampling
index.  Model states, gradients, and checkpoint replay tensors remain
Euclidean and unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


REPLAY_SAMPLER_DEFAULTS = {
    "replay_sampler": "uniform",
    "replay_candidate_count": 128,
    "replay_uniform_fraction": 0.25,
    "replay_embedding_dim": 16,
    "replay_curvature": 1.0,
    "replay_tangent_scale": 0.75,
    "replay_max_radius": 0.95,
    "replay_seed": 0,
}


def replay_sampler_kwargs_from_config(dreamer_config):
    """Translate ``DREAMER.REPLAY_MEMORY`` into memory constructor kwargs."""
    dreamer_config = dreamer_config if isinstance(dreamer_config, Mapping) else {}
    replay_config = dreamer_config.get("REPLAY_MEMORY", {})
    if isinstance(replay_config, str):
        replay_config = {"MODE": replay_config}
    elif not isinstance(replay_config, Mapping):
        replay_config = {}

    return {
        "replay_sampler": str(
            replay_config.get("MODE", REPLAY_SAMPLER_DEFAULTS["replay_sampler"])
        ).lower(),
        "replay_candidate_count": int(
            replay_config.get(
                "CANDIDATES",
                REPLAY_SAMPLER_DEFAULTS["replay_candidate_count"],
            )
        ),
        "replay_uniform_fraction": float(
            replay_config.get(
                "UNIFORM_FRACTION",
                REPLAY_SAMPLER_DEFAULTS["replay_uniform_fraction"],
            )
        ),
        "replay_embedding_dim": int(
            replay_config.get(
                "EMBED_DIM",
                REPLAY_SAMPLER_DEFAULTS["replay_embedding_dim"],
            )
        ),
        "replay_curvature": float(
            replay_config.get(
                "CURVATURE",
                REPLAY_SAMPLER_DEFAULTS["replay_curvature"],
            )
        ),
        "replay_tangent_scale": float(
            replay_config.get(
                "TANGENT_SCALE",
                REPLAY_SAMPLER_DEFAULTS["replay_tangent_scale"],
            )
        ),
        "replay_max_radius": float(
            replay_config.get(
                "MAX_RADIUS",
                REPLAY_SAMPLER_DEFAULTS["replay_max_radius"],
            )
        ),
        "replay_seed": int(
            replay_config.get("SEED", REPLAY_SAMPLER_DEFAULTS["replay_seed"])
        ),
    }


def poincare_expmap0(vectors, curvature=1.0, max_radius=0.95):
    """Map tangent vectors at the origin into a Poincare ball.

    ``max_radius`` is expressed as a fraction of the ball radius.  Clipping
    below the boundary keeps subsequent distances finite.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    curvature = float(curvature)
    max_radius = float(max_radius)
    if not np.isfinite(curvature) or curvature <= 0.0:
        raise ValueError("curvature must be finite and positive")
    if not np.isfinite(max_radius) or not 0.0 < max_radius < 1.0:
        raise ValueError("max_radius must be between 0 and 1")

    sqrt_c = np.sqrt(curvature)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    safe_norms = np.maximum(norms, 1e-12)
    factors = np.tanh(sqrt_c * norms) / (sqrt_c * safe_norms)
    mapped = vectors * factors

    mapped_norms = np.linalg.norm(mapped, axis=-1, keepdims=True)
    radius_limit = max_radius / sqrt_c
    clip = np.minimum(1.0, radius_limit / np.maximum(mapped_norms, 1e-12))
    return (mapped * clip).astype(np.float32, copy=False)


def pairwise_euclidean_distance(points):
    """Return a stable all-pairs Euclidean distance matrix."""
    points = np.asarray(points, dtype=np.float64)
    differences = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    distances = np.sqrt(np.maximum(np.sum(differences * differences, axis=-1), 0.0))
    np.fill_diagonal(distances, 0.0)
    return distances.astype(np.float32, copy=False)


def pairwise_poincare_distance(points, curvature=1.0):
    """Return all-pairs geodesic distances inside a Poincare ball."""
    points = np.asarray(points, dtype=np.float64)
    curvature = float(curvature)
    if not np.isfinite(curvature) or curvature <= 0.0:
        raise ValueError("curvature must be finite and positive")

    squared_norms = np.sum(points * points, axis=-1)
    if np.any(curvature * squared_norms >= 1.0):
        raise ValueError("points must lie strictly inside the Poincare ball")
    differences = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    squared_distances = np.sum(differences * differences, axis=-1)
    denominator = (
        (1.0 - curvature * squared_norms)[:, np.newaxis]
        * (1.0 - curvature * squared_norms)[np.newaxis, :]
    )
    acosh_argument = 1.0 + 2.0 * curvature * squared_distances / np.maximum(
        denominator, 1e-15
    )
    distances = np.arccosh(np.maximum(acosh_argument, 1.0)) / np.sqrt(curvature)
    np.fill_diagonal(distances, 0.0)
    return distances.astype(np.float32, copy=False)


def greedy_diverse_subset(distance_matrix, count, rng, initial_count=1):
    """Choose a farthest-first subset, seeded by random uniform anchors."""
    distances = np.asarray(distance_matrix, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be square")
    population = distances.shape[0]
    count = min(max(int(count), 0), population)
    if count == 0:
        return np.empty(0, dtype=np.int64)

    initial_count = min(max(int(initial_count), 1), count)
    selected = list(
        np.asarray(
            rng.choice(population, size=initial_count, replace=False),
            dtype=np.int64,
        )
    )
    selected_mask = np.zeros(population, dtype=bool)
    selected_mask[selected] = True
    nearest_selected = distances[:, selected].min(axis=1)

    while len(selected) < count:
        nearest_selected[selected_mask] = -np.inf
        next_index = int(np.argmax(nearest_selected))
        selected.append(next_index)
        selected_mask[next_index] = True
        nearest_selected = np.minimum(nearest_selected, distances[:, next_index])

    return np.asarray(selected, dtype=np.int64)


def mean_off_diagonal(distance_matrix):
    """Mean pairwise distance excluding the diagonal."""
    distances = np.asarray(distance_matrix, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be square")
    if distances.shape[0] < 2:
        return 0.0
    mask = ~np.eye(distances.shape[0], dtype=bool)
    return float(distances[mask].mean())
