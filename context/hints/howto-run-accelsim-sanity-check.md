# How to Run Accel-Sim Sanity Checks (Pixi)

## HEADER
- **Purpose**: Provide a reproducible Pixi environment to build and sanity-check `extern/tracked/accel-sim-framework`
- **Status**: Active
- **Date**: 2026-01-29
- **Dependencies**: `pixi`, git submodules initialized, network access (first build clones `gpgpu-sim` + `pybind11`)
- **Target**: AI assistants and developers

## What this repo provides

- A dedicated Pixi environment `accelsim` (see `pyproject.toml`) with build tools and CUDA Toolkit 12.8 from conda-forge.
- Pixi tasks to build Accel-Sim and run a quick smoke test without needing a host CUDA install.

## Prerequisites

- Initialize submodules once: `git submodule update --init --recursive`
- You need enough disk space for builds and optional trace downloads (Accel-Sim’s `short-tests.sh` downloads a Rodinia trace bundle).

## Build + smoke test (recommended quick sanity)

1) Install the environment: `pixi install -e accelsim`
2) Build the simulator: `pixi run -e accelsim accelsim-build`
3) Smoke test (verifies the binary loads and prints the banner): `pixi run -e accelsim accelsim-smoke`

Notes:
- The tasks create a local CUDA “shim” directory at `.pixi/accelsim-cuda/` so Accel-Sim sees `bin/` and `include/` in the layout it expects (conda CUDA headers live under `targets/x86_64-linux/include`).
- The build task also removes the binary RPATH so Accel-Sim can use the simulator-provided `libcudart.so` when running (required to avoid symbol lookup errors).

## Full trace-driven functional test (slow, optional)

Accel-Sim upstream provides `short-tests.sh` which builds the simulator, downloads pre-traced Rodinia workloads, and runs trace-driven simulations. Run it via:

- `pixi run -e accelsim accelsim-short-tests`

This may take a while and downloads a trace bundle similar to: `https://engineering.purdue.edu/tgrogers/accel-sim/traces/tesla-v100/latest/rodinia_2.0-ft.tgz`.

## Troubleshooting

- If you want to use a system CUDA instead of the Pixi-provided CUDA, set `CUDA_INSTALL_PATH` before running tasks (must contain `bin/nvcc` and `include/`).
- If you see missing build tools (e.g., `makedepend`) or link errors (e.g., `-lGL`), ensure the `accelsim` environment is installed and you are running via `pixi run -e accelsim ...`.

## References

- Accel-Sim Framework (official): https://github.com/accel-sim/accel-sim-framework
- Accel-Sim Dockerfile repo (lists build deps): https://github.com/accel-sim/Dockerfile
- Accel-Sim container image: https://github.com/accel-sim/Dockerfile/pkgs/container/accel-sim-framework
