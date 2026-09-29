#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

mkdir -p "${UPSTREAM_ROOT}" "${OUTPUT_ROOT}"

clone_at() {
  local name="$1"
  local url="$2"
  local commit="$3"
  local destination="${UPSTREAM_ROOT}/${name}"
  if [[ ! -d "${destination}/.git" ]]; then
    git clone --filter=blob:none "${url}" "${destination}"
  fi
  git -C "${destination}" fetch --depth 1 origin "${commit}"
  git -C "${destination}" checkout --detach "${commit}"
}

clone_at llava https://github.com/haotian-liu/LLaVA.git c121f0432da27facab705978f83c4ada465e46fd

cp -a "${PACKAGE_ROOT}/integrations/overlays/llava/." "${UPSTREAM_ROOT}/llava/"
git -C "${UPSTREAM_ROOT}/llava" rev-parse HEAD > "${REPRO_ROOT}/upstream-commits.txt"
echo "Prepared Braco LLaVA checkout under ${UPSTREAM_ROOT}/llava"
