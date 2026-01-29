#!/usr/bin/env bash
set -eo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=./_common.sh
source "$script_dir/_common.sh"

accelsim_cd_repo
accelsim_ensure_cuda_install_path

./short-tests.sh

