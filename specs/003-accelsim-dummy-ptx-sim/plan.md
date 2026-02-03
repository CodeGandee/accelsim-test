# Implementation Plan: Accel-Sim Dummy CUDA PTX Simulation

**Branch**: `[003-accelsim-dummy-ptx-sim]` | **Date**: 2026-02-03 | **Spec**: `/data1/huangzhe/code/accelsim-test/specs/003-accelsim-dummy-ptx-sim/spec.md`  
**Input**: Feature specification from `/data1/huangzhe/code/accelsim-test/specs/003-accelsim-dummy-ptx-sim/spec.md`

## Summary

Implement a small, reproducible PTX-mode Accel-Sim workflow to compile and run a dummy CUDA kernel (e.g., naive matmul) under simulation as a sanity check. The workflow is orchestrated from Python (via Pixi) and produces a self-contained run artifact directory under `tmp/<run_id>/` containing the executable, emitted PTX, simulator configuration copy, and logs. The simulated run prints an unambiguous `PASS`/`FAIL` based on a CPU reference check (small sizes), and the runner fails fast with actionable errors when prerequisites are missing.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.12 (Pixi) + CUDA C++ (nvcc; PTX-mode simulation target)  
**Primary Dependencies**: Pixi tasks (workflow), Accel-Sim submodule (`extern/tracked/accel-sim-framework`), bash runner glue, CUDA compiler (`nvcc`) via Pixi when available (fallback to system `nvcc`)  
**Storage**: Files under `tmp/<run_id>/` (binary, PTX, `gpgpusim.config` copy, logs, metadata)  
**Testing**: `pytest` unit tests for Python orchestration utilities; manual smoke run for end-to-end simulation  
**Target Platform**: Linux x86_64 (simulation does not require a physical GPU, but requires a working Accel-Sim build and CUDA compiler)  
**Project Type**: Monorepo with Python orchestrator + C++ subproject (`cpp/`)  
**Performance Goals**: Keep the dummy workload small enough that the PTX-mode simulation completes in a few minutes and provides deterministic output for repeated runs  
**Constraints**: PTX-mode only (no trace-driven/SASS); simulator config preset is Ampere/A100-like; artifacts must be written only under `tmp/<run_id>/`; fail fast on missing prerequisites; avoid system Python (Pixi is the entrypoint)  
**Scale/Scope**: One minimal CUDA program + one simulator preset + a small CLI/workflow to compile/run and collect artifacts

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Pre-Research Gates (PASS)

- PASS: Python remains the orchestrator and the primary entry point; C++ is invoked as a subsystem artifact (Constitution I).
- PASS: Pixi remains the workflow authority (`pixi run -e accelsim ...`) and system Python is not used for the workflow (Constitution IV).
- PASS: Permanent CUDA/C++ source is kept in `cpp/` (subproject boundary preserved) (Constitution I/II).
- PASS: Testing strategy includes at least unit and manual coverage for orchestration and the end-to-end path (Constitution V).

### Post-Design Re-check (PASS)

- PASS: Design avoids new service tiers and stays within “Python orchestrator + C++ subsystem” architecture.
- PASS: The workflow is file-based (artifacts under `tmp/<run_id>/`) and does not introduce external persistence requirements.

## Project Structure

### Documentation (this feature)

```text
specs/003-accelsim-dummy-ptx-sim/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```text
src/
└── accelsim_test/
    └── accelsim_dummy_ptx_sim/    # Python CLI/workflow to compile + run + collect artifacts

cpp/
└── accelsim_dummy_ptx_sim/        # Minimal CUDA program source (e.g., matmul.cu)

scripts/
└── accelsim/                      # Accel-Sim build/smoke tasks (existing); may add wrapper entrypoints

tests/
├── unit/                          # Unit tests for Python orchestration helpers
└── manual/                        # Manual end-to-end run recipe (optional script)

tmp/
└── accelsim_dummy_ptx_sim/
    └── <run_id>/                  # Run artifacts (binary, PTX, config copy, logs)

```

**Structure Decision**: Keep all workflow orchestration in Python (invoked via Pixi), keep permanent CUDA/C++ sources in the C++ subtree (`cpp/`), and store run artifacts exclusively under `tmp/<run_id>/`. This matches the repository constitution and keeps simulation experiments self-contained and reproducible.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No constitution violations are required for this feature.
