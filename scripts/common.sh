#!/usr/bin/env bash
set -euo pipefail

: "${REPRO_ROOT:?Set REPRO_ROOT to a writable experiment directory}"
: "${MODEL_ROOT:?Set MODEL_ROOT to the public model directory}"
: "${DATA_ROOT:?Set DATA_ROOT to the public dataset directory}"

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_ROOT="${REPRO_ROOT}/upstreams"
OUTPUT_ROOT="${REPRO_ROOT}/outputs"
export PACKAGE_ROOT UPSTREAM_ROOT OUTPUT_ROOT

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 2
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Required directory is missing: $1" >&2
    exit 2
  fi
}

check_budget() {
  case "$1" in
    4|9|16|25) ;;
    *) echo "Budget must be one of: 4, 9, 16, 25" >&2; exit 2 ;;
  esac
}

set_repro_seed() {
  local seed="$1"
  export PYTHONHASHSEED="${seed}"
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
}

record_environment() {
  local destination="$1"
  mkdir -p "$destination"
  python -VV > "${destination}/python.txt"
  python -m pip freeze > "${destination}/pip-freeze.txt"
  nvidia-smi -q > "${destination}/nvidia-smi.txt"
  git -C "${UPSTREAM_ROOT}/llava" rev-parse HEAD > "${destination}/llava-commit.txt"
  cp "${PACKAGE_ROOT}/configs/braco.yaml" "${destination}/resolved_protocol.yaml"
}
