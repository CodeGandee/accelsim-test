# Quickstart: Accel-Sim Dummy CUDA PTX Simulation

This quickstart describes the intended end-to-end workflow for `/data1/huangzhe/code/accelsim-test/specs/003-accelsim-dummy-ptx-sim/spec.md`.

## Prerequisites

- Repository is available at: `/data1/huangzhe/code/accelsim-test`
- Submodules initialized:
  - `cd /data1/huangzhe/code/accelsim-test && git submodule update --init --recursive`

## 1) Build Accel-Sim (once)

```bash
cd /data1/huangzhe/code/accelsim-test
pixi install -e accelsim
pixi run -e accelsim accelsim-build
pixi run -e accelsim accelsim-smoke
```

## 2) Run the dummy PTX simulation

Run a single end-to-end compile + simulate workflow (outputs go under `tmp/accelsim_dummy_ptx_sim/<run_id>/`):

```bash
cd /data1/huangzhe/code/accelsim-test
pixi run -e accelsim python -m accelsim_test.accelsim_dummy_ptx_sim run --run-id 2026-02-03T00-00-00Z
```

Alternative (Pixi task wrapper):

```bash
cd /data1/huangzhe/code/accelsim-test
pixi run -e accelsim accelsim-dummy-ptx-sim -- --run-id 2026-02-03T00-00-00Z
```

Expected outcomes:
- The command exits `0` on success.
- The simulator output log contains a simulator banner and the program prints `PASS`.

## 3) Inspect artifacts

Example artifact directory (paths are illustrative; actual layout is defined in `contracts/cli.md`):

- `/data1/huangzhe/code/accelsim-test/tmp/accelsim_dummy_ptx_sim/2026-02-03T00-00-00Z/ptx/matmul.ptx`
- `/data1/huangzhe/code/accelsim-test/tmp/accelsim_dummy_ptx_sim/2026-02-03T00-00-00Z/bin/matmul`
- `/data1/huangzhe/code/accelsim-test/tmp/accelsim_dummy_ptx_sim/2026-02-03T00-00-00Z/run/gpgpusim.config`
- `/data1/huangzhe/code/accelsim-test/tmp/accelsim_dummy_ptx_sim/2026-02-03T00-00-00Z/run/matmul.sim.log`

## Notes / Gotchas

- The simulator preset for this feature is Ampere/A100-like (`SM80_A100`).
- PTX extraction can add overhead on first run. Upstream GPGPU-Sim documentation describes an optional “save embedded PTX” flow to speed up repeated executions by reusing extracted PTX (`-save_embedded_ptx` + `PTX_SIM_USE_PTX_FILE` / `PTX_SIM_KERNELFILE` / `CUOBJDUMP_SIM_FILE`). This is optional and not required by the spec.

## Troubleshooting

### Missing submodules

Symptoms:
- The runner reports `submodule_initialized` failed, or `setup_environment.sh`/`gpgpusim.config` paths are missing.

Fix:
```bash
cd /data1/huangzhe/code/accelsim-test
git submodule update --init --recursive
```

### Simulator not built

Symptoms:
- The runner reports `simulator_built` failed.

Fix:
```bash
cd /data1/huangzhe/code/accelsim-test
pixi install -e accelsim
pixi run -e accelsim accelsim-build
pixi run -e accelsim accelsim-smoke
```

### `nvcc` not found

Symptoms:
- The runner reports `nvcc_available` failed.

Fix (preferred):
```bash
cd /data1/huangzhe/code/accelsim-test
pixi install -e accelsim
```

Fix (system fallback):
- Install the NVIDIA CUDA Toolkit and ensure `nvcc` is on `PATH` (`which nvcc`).

### `tmp/` permission issues

Symptoms:
- The runner reports `tmp_writable` failed.

Fix:
- Ensure the repository `tmp/` directory is writable by your user (and not mounted read-only).

### Runs are very slow / appear hung

Notes:
- PTX-mode simulation can be slow, especially on first run (PTX extraction + initialization).
- If it truly hangs, re-run with a fresh `--run-id` and inspect `run/matmul.sim.log` for the last line printed.
