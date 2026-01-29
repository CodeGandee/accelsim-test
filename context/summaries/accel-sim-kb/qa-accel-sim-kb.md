# Q&A: accel-sim-kb

## Introduction

This Q&A captures recurring questions about basic Accel-Sim usage in `accelsim-test` for developers (including future maintainers).

**Related docs**
- `context/hints/howto-run-accelsim-sanity-check.md`
- `pyproject.toml`
- `extern/tracked/accel-sim-framework/README.md`

**Key entrypoints and modules**
- `scripts/accelsim/build.sh`
- `scripts/accelsim/smoke.sh`
- `scripts/accelsim/short-tests.sh`
- `extern/tracked/accel-sim-framework/gpu-simulator/bin/release/accel-sim.out`

## Given a `.cu` file, how do I compile it into PTX and simulate it in Accel-Sim?
> Last revised at: `2026-01-29T10:45:47Z` | Last revised base commit: `6b50679aff45452cd1be5d15c0ddfb845e107aaa`

- Compile PTX for inspection/debugging (PTX alone is not enough to “run” unless you also have a host launcher): `pixi run -e accelsim bash -lc 'nvcc -O3 -arch=sm_70 -ptx path/to/app.cu -o app.ptx'`.
- For PTX-mode simulation you normally simulate a CUDA executable (host code + kernels) and let GPGPU-Sim/Accel-Sim extract embedded PTX via `cuobjdump`; build it with: `pixi run -e accelsim bash -lc 'nvcc -O3 -arch=sm_70 path/to/app.cu -o app'`.
- Ensure the simulator is built: `pixi run -e accelsim accelsim-build`.
- Add your executable to the Accel-Sim job launcher by creating `extern/tracked/accel-sim-framework/util/job_launching/apps/define-myapps.yml` (copy the structure from `extern/tracked/accel-sim-framework/util/job_launching/apps/define-all-apps.yml`) and point `exec_dir` at the folder containing your `app` binary (either set `GPUAPPS_ROOT` or use an absolute path).
- Run PTX-mode via the launcher (no traces needed): `cd extern/tracked/accel-sim-framework/util/job_launching && ./run_simulations.py -B <your_suite_name> -C QV100-PTX -N <run_id>`.
- Outputs land under `sim_run_<cuda_version>/...` (see `extern/tracked/accel-sim-framework/util/job_launching/README.md` for the directory layout).
- If repeated runs are slow due to repeated PTX extraction, enable `-save_embedded_ptx 1` once and then set `PTX_SIM_USE_PTX_FILE`, `PTX_SIM_KERNELFILE`, and `CUOBJDUMP_SIM_FILE` as described in `extern/tracked/accel-sim-framework/gpu-simulator/gpgpu-sim/README.md`.

## [question title]
> Last revised at: `2026-01-29T10:41:24Z` | Last revised base commit: `6b50679aff45452cd1be5d15c0ddfb845e107aaa`

- [answer/code]
