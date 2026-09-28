#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This script must run inside WSL2/Linux." >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
venv_path="${TMRL_JAX_VENV_PATH:-/home/${USER}/.venvs/tmrl-jax-cuda13}"
source_config="${1:-/mnt/c/Users/felix/TmrlData/config/config.continual-jax-m1-stable-v2.json}"
target_config="${2:-/home/${USER}/TmrlData/config/config.json}"
relay_port="${TMRL_RELAY_PORT:-55555}"
run_name="${TMRL_CONTINUAL_RUN_NAME:-Continual_Dreamer_JAX_M1_Stable_v2}"

windows_gateway="$(ip -4 route show default | awk '{print $3; exit}')"
if [[ -z "${windows_gateway}" ]]; then
  echo "Could not determine the Windows host gateway from the WSL route." >&2
  exit 2
fi

cd "${repo_root}"

"${venv_path}/bin/python" \
  -m tmrl.tools.prepare_continual_jax_config \
  --source "${source_config}" \
  --output "${target_config}" \
  --run-name "${run_name}" \
  --trainer-server-ip "${windows_gateway}" \
  --scrub-wandb-key \
  --overwrite

"${venv_path}/bin/python" - "${windows_gateway}" "${relay_port}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
with socket.create_connection((host, port), timeout=3.0):
    pass
print(f"Windows TMRL relay reachable from WSL2 at {host}:{port}.")
PY

echo "Restart the WSL trainer so it reloads: ${target_config}"
