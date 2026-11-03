#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This script must run inside WSL2/Linux." >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
venv_path="${TMRL_JAX_VENV_PATH:-/home/${USER}/.venvs/tmrl-jax-cuda13}"
artifact_dir="${TMRL_CONTINUAL_ARTIFACT_DIR:-/home/${USER}/tmrl-artifacts/continual-foundation}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it before running this bootstrap." >&2
  exit 2
fi

mkdir -p "$(dirname -- "${venv_path}")" "${artifact_dir}/preflight"
if [[ ! -x "${venv_path}/bin/python" ]]; then
  uv venv --python 3.12 "${venv_path}"
else
  echo "Reusing existing virtual environment: ${venv_path}"
fi
# Replay storage currently reuses TMRL's episode-safe Torch CPU memory. Install
# the CPU-only wheel so it cannot compete with JAX for the trainer GPU.
uv pip install \
  --python "${venv_path}/bin/python" \
  --index-url https://download.pytorch.org/whl/cpu \
  'torch>=2.10,<2.12'
uv pip install \
  --python "${venv_path}/bin/python" \
  --requirement "${repo_root}/requirements/continual-jax-wsl2.txt"

# The preflight may run while the native-Windows Torch baseline still owns
# part of the laptop GPU. Avoid JAX's default large up-front allocation.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

"${venv_path}/bin/python" \
  "${repo_root}/tmrl/tools/continual_preflight.py" \
  --require-gpu \
  --output "${artifact_dir}/preflight/jax_environment.json"

echo "JAX WSL2 environment is ready: ${venv_path}"
echo "Preflight artifact: ${artifact_dir}/preflight/jax_environment.json"
