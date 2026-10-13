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

parse_braco_run() {
  if [[ $# -lt 2 ]]; then
    echo "Usage: $0 BUDGET SEED [--pe true|false] [--idct true|false]" >&2
    exit 2
  fi
  BUDGET="$1"
  SEED="$2"
  shift 2
  check_budget "${BUDGET}"
  if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "SEED must be a non-negative integer" >&2
    exit 2
  fi

  case "${BUDGET}" in
    4)  C=1; S=3 ;;
    9)  C=2; S=5 ;;
    16) C=3; S=7 ;;
    25) C=4; S=9 ;;
  esac
  local option value
  USE_PE=True
  LOW_IDCT=False
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --pe|--idct)
        option="$1"
        if [[ $# -lt 2 ]]; then
          echo "${option} requires true or false" >&2
          exit 2
        fi
        value="$2"
        shift 2
        ;;
      --pe=*|--idct=*)
        option="${1%%=*}"
        value="${1#*=}"
        shift
        ;;
      *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
    case "${value,,}" in
      true) value=True ;;
      false) value=False ;;
      *) echo "${option} requires true or false, got: ${value}" >&2; exit 2 ;;
    esac
    if [[ "${option}" == --pe ]]; then USE_PE="${value}"; else LOW_IDCT="${value}"; fi
  done

  RUN_ROOT="${OUTPUT_ROOT}/braco/k${BUDGET}/seed${SEED}"
  if [[ "${USE_PE}" != True || "${LOW_IDCT}" != False ]]; then
    RUN_ROOT+="-pe-${USE_PE,,}-idct-${LOW_IDCT,,}"
  fi
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
  python - "${PACKAGE_ROOT}/configs/braco.yaml" "${destination}/resolved_protocol.yaml" \
    "${BUDGET}" "${SEED}" "${USE_PE}" "${LOW_IDCT}" <<'PY'
import sys
from pathlib import Path
import yaml

source, destination, budget, seed, pe, idct = sys.argv[1:]
config = yaml.safe_load(Path(source).read_text())
config["selected_budget"] = int(budget)
config["seed"] = int(seed)
config["methods"]["braco"]["use_polar_pe"] = pe == "True"
config["methods"]["braco"]["split"][int(budget)]["low_idct"] = idct == "True"
Path(destination).write_text(yaml.safe_dump(config, sort_keys=False))
PY
}
