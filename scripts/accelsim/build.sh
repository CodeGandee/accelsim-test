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

make -j -C ./gpu-simulator

if command -v patchelf >/dev/null 2>&1; then
  patchelf --remove-rpath ./gpu-simulator/bin/release/accel-sim.out || true
fi

