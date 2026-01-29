#!/usr/bin/env bash
set -eo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=./_common.sh
source "$script_dir/_common.sh"

accelsim_cd_repo
accelsim_ensure_cuda_install_path
accelsim_validate_cuda

export GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN=
source ./gpu-simulator/setup_environment.sh

tmp_out="$(mktemp)"
set +o pipefail
./gpu-simulator/bin/release/accel-sim.out 2>&1 | head -n 40 >"$tmp_out"
set -o pipefail

if grep -q "Accel-Sim \\[build" "$tmp_out"; then
  sed -n '1,5p' "$tmp_out"
  exit 0
fi

cat "$tmp_out"
exit 1
