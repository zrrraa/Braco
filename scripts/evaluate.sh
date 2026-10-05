#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

BUDGET="${1:?Usage: scripts/evaluate.sh BUDGET SEED}"
SEED="${2:?Usage: scripts/evaluate.sh BUDGET SEED}"
METHOD=braco
check_budget "${BUDGET}"
set_repro_seed "${SEED}"
CODE_ROOT="${UPSTREAM_ROOT}/llava"
require_dir "${CODE_ROOT}"
RUN_ROOT="${OUTPUT_ROOT}/braco/k${BUDGET}/seed${SEED}"
require_dir "${RUN_ROOT}/stage2"
# LLaVA's evaluation loader identifies the model family from the path name.
MODEL_PATH="${RUN_ROOT}/llava-braco"
if [[ ! -e "${MODEL_PATH}" && ! -L "${MODEL_PATH}" ]]; then
  ln -s stage2 "${MODEL_PATH}"
fi
if [[ "$(readlink -f "${MODEL_PATH}")" != "$(readlink -f "${RUN_ROOT}/stage2")" ]]; then
  echo "Expected ${MODEL_PATH} to refer to ${RUN_ROOT}/stage2" >&2
  exit 2
fi

RESULT_ROOT="${RUN_ROOT}/evaluation"
mkdir -p "${RESULT_ROOT}"

export MODEL_PATH RESULT_ROOT DATA_ROOT BUDGET SEED METHOD
export PYTHONPATH="${CODE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${CODE_ROOT}"

# Run the Braco benchmark manifest.
while IFS= read -r benchmark; do
  [[ -z "${benchmark}" || "${benchmark}" == \#* ]] && continue
  bash "${PACKAGE_ROOT}/scripts/eval_one.sh" "${benchmark}"
done < "${PACKAGE_ROOT}/configs/benchmarks.txt"

touch "${RESULT_ROOT}/COMPLETE"
