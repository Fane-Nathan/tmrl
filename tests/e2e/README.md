# Opaque-Box End-to-End (E2E) Test Suite (`tests/e2e/`)

This directory contains a 4-tier, opaque-box, requirement-driven end-to-end test suite covering the TMRL repository (`tmrl/`, `experiments/sensor_adaptation_v1/`, `scripts/`, `tutorial/`, documentation, and workspace hygiene).

## Test Tiers & Markers

- **`tier1`** (`test_tier1_core_and_backends.py`, `test_tier1_experiments_docs_hygiene.py`): Fast offline contracts covering public CLI `--help`/validation, headless/lazy package imports without `~/TmrlData` or `pywin32`, config/factory wiring, pure-stdlib `tmrl.tools.*`, TrackMania interfaces/preprocessors/rewards, PyTorch & JAX model instantiation, sensor-adaptation decoupling & schemas, documentation parity, and git/workspace hygiene.
- **`tier2`** (`test_tier2_boundaries.py`): Boundary value analysis, negative/error paths, and local CPU/loopback component integrations across all 15 features.
- **`tier3`** (`test_tier3_cross_feature.py`): 18 pairwise cross-feature integration tests (`T3-01`..`T3-18`).
- **`tier4`** (`test_tier4_scenarios.py`): 10 end-to-end external-researcher workflow scenarios (`T4-01`..`T4-10`).

### Capability Markers
- `requires_torch`, `requires_mujoco`, `requires_jax`, `requires_tlspyo`, `requires_sphinx`, `requires_bash`, `external_runtime`

Tests requiring optional runtimes not installed on the current host (e.g., `jax`/`flax`/`optax`, `sphinx`, working `bash`, live TrackMania 2020 + Openplanet, or WSL2 CUDA 13) skip cleanly via `pytest.importorskip` or capability checks without faking external execution.

## Running the Suite

From the repository root, using a Python 3.12 environment with test dependencies installed (`pytest`, `torch`, `gymnasium[mujoco]`, `mujoco`, `numpy`, `scipy`, `matplotlib`, `psutil`, `packaging`, `pyyaml`, `cloudpickle`, `pandas`, `opencv-python`, `tlspyo`, `rtgym`):

```bash
# Run the full E2E suite
pytest tests/e2e -v

# Run a single tier
pytest tests/e2e -m tier1 -v
pytest tests/e2e -m tier2 -v
pytest tests/e2e -m tier3 -v
pytest tests/e2e -m tier4 -v

# Collect all tests without executing
pytest tests/e2e --collect-only -q
```

### Isolation Guarantees
- Every test isolates `HOME` and `USERPROFILE` to `pytest` `tmp_path` directories.
- No test writes checkpoints, `TmrlData`, figures, or study artifacts into the repository or the user's home directory.
- Network-sensitive import and `--help` checks explicitly block outbound sockets in isolated subprocesses.
