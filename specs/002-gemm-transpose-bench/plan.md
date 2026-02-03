# Implementation Plan: GEMM Transpose Performance Benchmark

**Branch**: `[002-gemm-transpose-bench]` | **Date**: 2026-02-02 | **Spec**: `/data1/huangzhe/code/accelsim-test/specs/002-gemm-transpose-bench/spec.md`  
**Input**: Feature specification from `/data1/huangzhe/code/accelsim-test/specs/002-gemm-transpose-bench/spec.md`

## Summary

Implement a rigorous CUDA GEMM transpose benchmark with two suites (square and non-square), comparing transpose as a view vs a pre-materialized copy, across shapes and dtype pairs. Measurement is done exclusively with NVBench, GEMMs use cuBLASLt, and every executed configuration/case produces a structured export plus per-case Nsight Compute artifacts. Python is the orchestrator for building, running sweeps, dispatching per-case `ncu` runs, and generating stakeholder-facing markdown report tables with required ratios and `flop_count` consistency.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.12 (Pixi) + CUDA C++17 (NVCC via Pixi `cuda13`)  
**Primary Dependencies**: Pixi tasks (workflow), Ruff + Mypy (Python QA), `attrs`/Hydra (Python config), Conan 2 + CMake/Ninja (C++ build), cuBLASLt (GEMM), NVBench (timing; `extern/orphan/nvbench`), Nsight Compute `ncu` (profiling)  
**Storage**: Files (structured JSON/CSV export + Markdown report + per-case `*.ncu-rep` artifacts)  
**Testing**: `pytest` for Python unit/integration tests; manual smoke scripts for benchmark runs; C++ build-only checks and minimal sanity checks (correctness sampled)  
**Target Platform**: Linux x86_64 host with NVIDIA GPU; CUDA toolchain via Pixi `cuda13` environment  
**Project Type**: Monorepo with Python orchestrator + C++ subproject (`cpp/`)  
**Performance Goals**: Stable per-case GPU timing using NVBench (e.g., `--min-time>=1s`, `--max-noise<=0.5%` for timing runs) with clear view-vs-copy ratios and complete profiling coverage  
**Constraints**: Must exclude H2D/D2H and transpose materialization from timed region; must profile every executed configuration/case; must fail overall run if any correctness check fails while still exporting results; avoid system Python (Pixi is build/run entrypoint)  
**Scale/Scope**: Shape/dtype sweeps per `/data1/huangzhe/code/accelsim-test/context/tasks/req-cuda-gemm-test.md`, covering cache-friendly and cache-stressing cases; multiple dtype pairs (fp16/bf16/fp32/tf32/int8) and two suites (square + non-square)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Pre-Research Gates (PASS)

- PASS: Python is the orchestrator; C++ stays in `cpp/` and is invoked/managed by Python workflows (Constitution I).
- PASS: Benchmark timing uses NVBench (no ad-hoc `cudaEventRecord` loops) and profiling uses Nsight tools (Constitution VI; requirement mandates NVBench).
- PASS: Pixi remains the master workflow entrypoint; C++ builds are exposed as Pixi tasks that call Conan/CMake (Constitution IV).
- PASS: Python code is type-annotated and kept orchestration-focused; lint/type checks remain `ruff` + `mypy` (Constitution III).
- PASS: C++ code follows modern C++17 idioms and subproject conventions (Constitution II).
- PASS: Testing strategy includes manual smoke runs + Python unit/integration tests for export/report logic (Constitution V).

### Post-Design Re-check (PASS)

- PASS: Design keeps performance-critical code in C++ (cuBLASLt + NVBench) and keeps sweep/profiling/report orchestration in Python.
- PASS: Export/report contracts are file-based (no new service tiers) and keep the repo architecture within the existing two-layer model.

## Project Structure

### Documentation (this feature)

```text
specs/002-gemm-transpose-bench/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
src/accelsim_test/
├── gemm_transpose_bench/
│   ├── __init__.py
│   ├── config.py           # Sweep definitions (shapes/dtypes/cases), CLI config
│   ├── runner.py           # Orchestrates timing runs (NVBench CLI) and collects outputs
│   ├── profiling.py        # Orchestrates per-case `ncu` runs and artifact management
│   ├── export.py           # Normalizes NVBench JSON into project result schema
│   └── report.py           # Generates stakeholder markdown tables + conclusions
└── __init__.py

cpp/
├── conanfile.py
├── CMakeLists.txt
└── src/
    ├── gemm_transpose_bench.cu   # NVBench benchmark(s) calling cuBLASLt
    ├── gemm_transpose_bench.h
    └── cublaslt_gemm.h/.cu       # Minimal cuBLASLt wrapper for the benchmark

scripts/
└── gemm_transpose/               # Optional helper scripts (build/run wrappers; invoked via Pixi tasks)

tests/
├── unit/
│   └── gemm_transpose_bench/      # Unit tests for export/report schema + ratio math
├── integration/
│   └── test_gemm_transpose_smoke.py
└── manual/
    └── run_gemm_transpose_smoke.py
```

**Structure Decision**: Keep CUDA benchmarking in the C++ subproject (`cpp/`) and keep all workflow, sweep selection, profiling dispatch, export normalization, and report generation in Python (`src/accelsim_test/…`). This matches the repository constitution (Python orchestrator, C++ performance subsystem) and keeps build authority centralized via Pixi tasks.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No constitution violations are required for this feature.
