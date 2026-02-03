# Data Model: Accel-Sim Dummy CUDA PTX Simulation

This document defines the entities and invariants used by the dummy PTX simulation workflow described in `/data1/huangzhe/code/accelsim-test/specs/003-accelsim-dummy-ptx-sim/spec.md`.

## Entities

### 1) `SimulationRun`

Represents one end-to-end execution attempt (compile → simulate → verify) for the dummy CUDA program.

Fields:
- `run_id` (string, required): Unique identifier for the run (e.g., timestamp-based).
- `started_at` / `finished_at` (RFC3339 string, required).
- `status` (enum, required): `pass` | `fail`.
- `failure_reason` (string, optional): Present when `status=fail` (e.g., missing prerequisite, compilation error, simulation error, correctness mismatch).
- `mode` (enum, required): `ptx` (fixed for this feature).
- `config_preset` (enum, required): `SM80_A100` (fixed for this feature).
- `compiler_source` (enum, required): `pixi` | `system` (how `nvcc` was selected for the run).
- `artifacts_dir` (string, required): Absolute path under `/data1/huangzhe/code/accelsim-test/tmp/accelsim_dummy_ptx_sim/<run_id>/`.
- `git` (object, required): `{ "branch": string, "commit": string, "dirty": bool }`.
- `commands` (object, required): `{ "build_simulator": string, "compile_app": string, "run_simulation": string }`.

Validation rules:
- `mode` must be `ptx`.
- `config_preset` must be `SM80_A100`.
- `artifacts_dir` must be under `/data1/huangzhe/code/accelsim-test/tmp/`.

State transitions:
- `started -> completed(pass|fail)`.

### 2) `RunArtifacts`

Represents the concrete files produced for a `SimulationRun`.

Fields (paths are absolute and must exist when `status=pass`):
- `exe_path` (string, required): The compiled CUDA executable (host code + kernels).
- `ptx_path` (string, required): Standalone emitted PTX used for inspection/debugging.
- `config_path` (string, required): Copied `gpgpusim.config` used for this run.
- `stdout_log_path` (string, required): Captured combined stdout/stderr from the simulation run.
- `metadata_path` (string, required): A single machine-readable metadata file that links the run, settings, and artifacts (e.g., JSON).

Validation rules:
- `config_path` must correspond to the preset `SM80_A100` for this feature.
- `stdout_log_path` must include a simulator banner line and the program’s `PASS`/`FAIL` line.

### 3) `PrerequisiteCheck`

Represents the pre-flight checks executed before attempting compilation/simulation.

Fields:
- `check_name` (string, required): e.g., `submodule_initialized`, `simulator_built`, `nvcc_available`, `config_preset_exists`, `tmp_writable`.
- `status` (enum, required): `pass` | `fail`.
- `details` (string, optional): Actionable guidance when `fail` (e.g., which command to run next).

Validation rules:
- A run must not proceed past prerequisites if any check fails (`FR-005`).

## Invariants / Derived Concepts

- The workflow’s public “success signal” is both:
  - simulator banner presence (basic sanity), and
  - the program’s correctness check result (`PASS`/`FAIL`) (functional sanity).
- All run artifacts are isolated to `tmp/<run_id>/` to avoid polluting the repository and to simplify cleanup.
