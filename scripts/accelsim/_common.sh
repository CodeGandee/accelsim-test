#!/usr/bin/env bash
set -eo pipefail

accelsim_repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P
}

accelsim_cd_repo() {
  local repo_root
  repo_root="$(accelsim_repo_root)"
  cd "$repo_root/extern/tracked/accel-sim-framework"
}

accelsim_ensure_cuda_install_path() {
  local repo_root cuda_shim
  repo_root="$(accelsim_repo_root)"

  if [[ -n "${CUDA_INSTALL_PATH:-}" ]]; then
    return 0
  fi

  if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "ERROR: set CUDA_INSTALL_PATH (system CUDA) or run inside pixi env (CONDA_PREFIX missing)" >&2
    return 2
  fi

  cuda_shim="$repo_root/.pixi/accelsim-cuda"
  mkdir -p "$cuda_shim/bin"

  ln -sf "$CONDA_PREFIX/bin/nvcc" "$cuda_shim/bin/nvcc"
  ln -sf "$CONDA_PREFIX/bin/ptxas" "$cuda_shim/bin/ptxas"
  ln -sf "$CONDA_PREFIX/bin/cuobjdump" "$cuda_shim/bin/cuobjdump"
  ln -sf "$CONDA_PREFIX/bin/nvdisasm" "$cuda_shim/bin/nvdisasm"
  ln -sf "$CONDA_PREFIX/bin/nvprof" "$cuda_shim/bin/nvprof"

  ln -sfn "$CONDA_PREFIX/targets/x86_64-linux/include" "$cuda_shim/include"
  ln -sfn "$CONDA_PREFIX/targets/x86_64-linux/lib" "$cuda_shim/lib"
  ln -sfn "$CONDA_PREFIX/targets/x86_64-linux/lib" "$cuda_shim/lib64"

  export CUDA_INSTALL_PATH="$cuda_shim"
  export PATH="$CUDA_INSTALL_PATH/bin:$PATH"
}

accelsim_validate_cuda() {
  if [[ -z "${CUDA_INSTALL_PATH:-}" ]]; then
    echo "ERROR: CUDA_INSTALL_PATH is not set" >&2
    return 2
  fi
  if [[ ! -x "$CUDA_INSTALL_PATH/bin/nvcc" ]]; then
    echo "ERROR: nvcc not found at $CUDA_INSTALL_PATH/bin/nvcc" >&2
    return 2
  fi
}

