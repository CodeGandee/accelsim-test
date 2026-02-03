# Contract: CLI Interface (Accel-Sim Dummy PTX Simulation)

This feature is delivered primarily as a Pixi-invoked Python CLI that orchestrates compilation and PTX-mode simulation, and produces run artifacts under `tmp/<run_id>/`.

## 1) Python orchestrator CLI (primary)

Location (planned):
- `/data1/huangzhe/code/accelsim-test/src/accelsim_test/accelsim_dummy_ptx_sim/`

Invocation:
- `pixi run -e accelsim python -m accelsim_test.accelsim_dummy_ptx_sim run [options]`

### `run`

Runs the full workflow: prerequisite checks → compile (emit PTX + executable) → run under PTX-mode simulation → validate output → write artifacts.

Inputs:
- `--run-id <string>` (optional): Defaults to a timestamp-based identifier. Must be filesystem-safe.
- `--compiler <auto|pixi|system>` (optional, default: `auto`): Selects how `nvcc` is located. `auto` prefers `pixi` and falls back to `system`.
- `--config-preset <sm80_a100>` (optional, default: `sm80_a100`): Preset selector. For this feature, only `sm80_a100` is supported.

Outputs (under `/data1/huangzhe/code/accelsim-test/tmp/accelsim_dummy_ptx_sim/<run_id>/`):
- `bin/`:
  - `matmul` (or similar): compiled executable
- `ptx/`:
  - `matmul.ptx`: standalone PTX output (for inspection/debugging)
- `run/`:
  - `gpgpusim.config`: copied from the preset config
  - `matmul.sim.log`: captured simulator stdout/stderr
- `metadata.json`: run metadata (toolchain selection, git info, commands used, PASS/FAIL)

Behavioral constraints:
- Must store artifacts only under `tmp/<run_id>/` (see `FR-009`).
- Must run PTX-mode only; trace-driven modes are out of scope (`FR-006`).
- Must print or otherwise record an unambiguous `PASS`/`FAIL` based on CPU reference (`FR-008`).
- Must fail fast with actionable errors when prerequisites are missing (`FR-005`).

Exit codes:
- `0` only if the run completes and correctness check reports `PASS`.
- Non-zero for any prerequisite failure, compilation failure, simulator failure, missing expected artifacts, or correctness failure.

## 2) Pixi task (wrapper)

Planned (optional) Pixi task name:
- `accelsim-dummy-ptx-sim`: invokes the Python CLI above in the `accelsim` environment.
