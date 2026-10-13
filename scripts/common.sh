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

parse_braco_run() {
  if [[ $# -eq 0 ]]; then
    echo "Usage: $0 BUDGET SEED [--c C --s S] [--pe true|false] [--idct true|false]" >&2
    echo "   or: $0 --c C --s S [--seed SEED] [--pe true|false] [--idct true|false]" >&2
    exit 2
  fi
  BUDGET=""
  SEED=0
  C=""
  S=""
  if [[ "$1" != --* ]]; then
    if [[ $# -lt 2 ]]; then
      echo "Provide both BUDGET and SEED, or use --c C --s S." >&2
      exit 2
    fi
    BUDGET="$1"
    SEED="$2"
    shift 2
    if [[ ! "${BUDGET}" =~ ^[1-9][0-9]{0,2}$ ]] || (( BUDGET > 576 )); then
      echo "BUDGET must be an integer from 1 to 576" >&2
      exit 2
    fi
  fi

  local option value split_requested=false
  USE_PE=True
  LOW_IDCT=False
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --pe|--idct|--c|--s|--seed)
        option="$1"
        if [[ $# -lt 2 ]]; then
          echo "${option} requires a value" >&2
          exit 2
        fi
        value="$2"
        shift 2
        ;;
      --pe=*|--idct=*|--c=*|--s=*|--seed=*)
        option="${1%%=*}"
        value="${1#*=}"
        shift
        ;;
      *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
    case "${option}" in
      --c) C="${value}"; split_requested=true ;;
      --s) S="${value}"; split_requested=true ;;
      --seed)
        if [[ -n "${BUDGET}" ]]; then
          echo "Use the positional SEED with BUDGET, or --seed with --c and --s." >&2
          exit 2
        fi
        SEED="${value}"
        ;;
      --pe|--idct)
        case "${value,,}" in
          true) value=True ;;
          false) value=False ;;
          *) echo "${option} requires true or false, got: ${value}" >&2; exit 2 ;;
        esac
        if [[ "${option}" == --pe ]]; then USE_PE="${value}"; else LOW_IDCT="${value}"; fi
        ;;
    esac
  done

  if [[ ! "${SEED}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "SEED must be a non-negative integer" >&2
    exit 2
  fi
  if [[ "${split_requested}" == true ]]; then
    if [[ ! "${C}" =~ ^[1-9][0-9]?$ ]] || (( C > 24 )); then
      echo "--c must be an integer from 1 to 24" >&2
      exit 2
    fi
    if [[ ! "${S}" =~ ^(0|[1-9][0-9]{0,2})$ ]] || (( C*C + S > 576 )); then
      echo "--s must be a non-negative integer with C^2 + S <= 576" >&2
      exit 2
    fi
    local tokens=$((C*C + S))
    if [[ -n "${BUDGET}" && "${BUDGET}" != "${tokens}" ]]; then
      echo "BUDGET=${BUDGET} does not match C^2 + S = ${tokens}" >&2
      exit 2
    fi
    BUDGET="${tokens}"
  fi

  local default_c="" default_s=""
  case "${BUDGET}" in
    4)  default_c=1; default_s=3 ;;
    9)  default_c=2; default_s=5 ;;
    16) default_c=3; default_s=7 ;;
    25) default_c=4; default_s=9 ;;
  esac
  if [[ -z "${C}" ]]; then
    if [[ -z "${default_c}" ]]; then
      echo "Choose a preset budget (4, 9, 16, 25), or provide both --c and --s." >&2
      exit 2
    fi
    C="${default_c}"
    S="${default_s}"
  fi

  RUN_ROOT="${OUTPUT_ROOT}/braco/k${BUDGET}/seed${SEED}"
  if [[ "${C}" != "${default_c}" || "${S}" != "${default_s}" ]]; then
    RUN_ROOT+="-c${C}-s${S}"
  fi
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
    "${BUDGET}" "${SEED}" "${C}" "${S}" "${USE_PE}" "${LOW_IDCT}" <<'PY'
import sys
from pathlib import Path
import yaml

source, destination, budget, seed, c, s, pe, idct = sys.argv[1:]
config = yaml.safe_load(Path(source).read_text())
budget = int(budget)
config["selected_budget"] = budget
config["seed"] = int(seed)
braco = config["methods"]["braco"]
braco["use_polar_pe"] = pe == "True"
braco["split"][budget] = {"C": int(c), "S": int(s), "low_idct": idct == "True"}
for budgets in (config["budgets"], braco["budgets"]):
    if budget not in budgets:
        budgets.append(budget)
Path(destination).write_text(yaml.safe_dump(config, sort_keys=False))
PY
}
