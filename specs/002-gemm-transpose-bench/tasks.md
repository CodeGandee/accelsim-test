# Tasks: GEMM Transpose Performance Benchmark

**Input**: Design documents from `/data1/huangzhe/code/accelsim-test/specs/002-gemm-transpose-bench/`  
**Prerequisites**: `/data1/huangzhe/code/accelsim-test/specs/002-gemm-transpose-bench/plan.md`, `/data1/huangzhe/code/accelsim-test/specs/002-gemm-transpose-bench/spec.md`, `/data1/huangzhe/code/accelsim-test/specs/002-gemm-transpose-bench/research.md`, `/data1/huangzhe/code/accelsim-test/specs/002-gemm-transpose-bench/data-model.md`, `/data1/huangzhe/code/accelsim-test/specs/002-gemm-transpose-bench/contracts/`

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: User story mapping: US1..US5 (from spec.md)

---

## Phase 1: Setup (Shared Infrastructure)

- [X] T001 [P] [US1] Create Python package skeleton in `/data1/huangzhe/code/accelsim-test/src/accelsim_test/gemm_transpose_bench/` (`__init__.py`, `config.py`, `runner.py`, `export.py`, `report.py`, `profiling.py`)
- [X] T002 [P] [US1] Add Pixi tasks for build/run/report in `/data1/huangzhe/code/accelsim-test/pyproject.toml` (target Pixi env: `cuda13` for benchmark execution)
- [X] T003 [P] [US1] Ensure Conan is available via Pixi (add dependency in `/data1/huangzhe/code/accelsim-test/pyproject.toml` and document usage) to avoid system Python usage (Constitution IV)
- [X] T004 [P] [US1] Add C++ NVBench benchmark executable target in `/data1/huangzhe/code/accelsim-test/cpp/CMakeLists.txt` and source in `/data1/huangzhe/code/accelsim-test/cpp/src/gemm_transpose_bench.cu`
- [X] T005 [US1] Integrate NVBench from `/data1/huangzhe/code/accelsim-test/extern/orphan/nvbench` into the C++ build (fail-fast with a clear error if missing)

---

## Phase 2: Foundational (Blocking Prerequisites)

- [X] T006 [US4] Implement normalized export writer/validator in `/data1/huangzhe/code/accelsim-test/src/accelsim_test/gemm_transpose_bench/export.py` that emits `results.json` conforming to `/data1/huangzhe/code/accelsim-test/specs/002-gemm-transpose-bench/contracts/results.schema.json`
- [X] T007 [US4] Implement NVBench JSON ingestion + normalization mapping in `/data1/huangzhe/code/accelsim-test/src/accelsim_test/gemm_transpose_bench/export.py` (add `flop_count=2*m*n*k`, attach suite/case semantics, capture environment metadata)
- [X] T008 [US4] Implement ratio computation rules in `/data1/huangzhe/code/accelsim-test/src/accelsim_test/gemm_transpose_bench/report.py` (enforce per-row `flop_count` consistency; compute slowdowns vs `AB` and copy-over-view ratios where applicable)
- [X] T009 [US3] Implement correctness verification (sampled dot-product check + optional full check for small shapes) in `/data1/huangzhe/code/accelsim-test/src/accelsim_test/gemm_transpose_bench/runner.py` (or a dedicated `verify.py`)
- [X] T010 [US2] Implement shape/dtype/case registry in `/data1/huangzhe/code/accelsim-test/src/accelsim_test/gemm_transpose_bench/config.py` using the shape sets and dtype requirements in `/data1/huangzhe/code/accelsim-test/context/tasks/req-cuda-gemm-test.md`
- [X] T011 [US2] Implement “single configuration/case” invocation builder for NVBench axis overrides in `/data1/huangzhe/code/accelsim-test/src/accelsim_test/gemm_transpose_bench/runner.py`
- [X] T012 [US4] Add `pytest` execution task and basic test configuration (if needed) in `/data1/huangzhe/code/accelsim-test/pyproject.toml` and `/data1/huangzhe/code/accelsim-test/tests/`

Checkpoint: At this point, it should be possible to run a tiny sweep, export `results.json`, and validate schema.

---

## Phase 3: User Story 1 - Run square-suite benchmark (Priority: P1)

Goal: Produce square-suite results for all five required cases with stable timing and required comparison fields.

Independent Test: Run one small square config and verify export includes all square-suite cases with timings and derived ratios.

### Tests (US1)

- [X] T013 [P] [US1] Add integration smoke test skeleton in `/data1/huangzhe/code/accelsim-test/tests/integration/test_gemm_transpose_square_smoke.py` (skips if no GPU)
- [X] T014 [P] [US1] Add unit test for ratio math in `/data1/huangzhe/code/accelsim-test/tests/unit/test_gemm_transpose_ratios.py`

### Implementation (US1)

- [X] T015 [US1] Implement NVBench axes and square-suite cases in `/data1/huangzhe/code/accelsim-test/cpp/src/gemm_transpose_bench.cu` (cases: `AB`, `ATB_view`, `ABT_view`, `ATB_copyA`, `ABT_copyB`)
- [X] T016 [US1] Implement cuBLASLt call wrapper + handle/workspace reuse in `/data1/huangzhe/code/accelsim-test/cpp/src/cublaslt_gemm.cu` (ensure handle creation and allocations are outside timed region)
- [X] T017 [US1] Implement transpose materialization kernels for `*_copy*` cases in `/data1/huangzhe/code/accelsim-test/cpp/src/gemm_transpose_bench.cu` with materialization outside NVBench timing (FR-008a)
- [X] T018 [US1] Implement Python `timing` workflow in `/data1/huangzhe/code/accelsim-test/src/accelsim_test/gemm_transpose_bench/runner.py` to run square suite and write `raw/nvbench_timing.json` + `results.json`

Checkpoint: Square-suite benchmark works end-to-end for at least one configuration, and failing correctness causes non-zero exit while still exporting results (FR-011a).

---

## Phase 4: User Story 2 - Run non-square-suite benchmark (Priority: P2)

Goal: Produce transpose-A and transpose-B non-square results with matched FLOP counts.

Independent Test: Run one non-square config and verify both view and copy results exist for the selected direction.

### Tests (US2)

- [X] T019 [P] [US2] Add integration smoke test in `/data1/huangzhe/code/accelsim-test/tests/integration/test_gemm_transpose_nonsquare_smoke.py` (skips if no GPU)

### Implementation (US2)

- [X] T020 [US2] Extend NVBench benchmark to support `nonsquare_atb` and `nonsquare_abt` suites in `/data1/huangzhe/code/accelsim-test/cpp/src/gemm_transpose_bench.cu` (validate shapes and only run valid cases per suite)
- [X] T021 [US2] Extend Python sweep selection in `/data1/huangzhe/code/accelsim-test/src/accelsim_test/gemm_transpose_bench/config.py` to include required non-square shape sets and safety control set

Checkpoint: Non-square suites produce expected outputs and ratios without invalid-shape crashes.

---

## Phase 5: User Story 3 - Validate numerical correctness (Priority: P3)

Goal: Every configuration/case has a verification pass/fail and an error summary; any failure fails the overall run (FR-011/FR-011a).

Independent Test: Introduce an intentional mismatch path and confirm it is flagged and fails the run while exporting results.

### Tests (US3)

- [X] T022 [P] [US3] Add unit tests for verification logic in `/data1/huangzhe/code/accelsim-test/tests/unit/test_gemm_transpose_verification.py`

### Implementation (US3)

- [X] T023 [US3] Implement per-record verification result fields and run-level failure aggregation in `/data1/huangzhe/code/accelsim-test/src/accelsim_test/gemm_transpose_bench/export.py`

---

## Phase 6: User Story 4 - Export structured results for analysis (Priority: P4)

Goal: Export is machine-readable, complete, and stable; includes required metadata and derived ratios where applicable (FR-009).

Independent Test: Validate schema for a small sweep; ensure required fields exist for all records.

### Tests (US4)

- [X] T024 [P] [US4] Add schema validation test in `/data1/huangzhe/code/accelsim-test/tests/unit/test_gemm_transpose_schema.py`

### Implementation (US4)

- [X] T025 [US4] Capture environment metadata (GPU/driver/CUDA/NVBench settings) in `/data1/huangzhe/code/accelsim-test/src/accelsim_test/gemm_transpose_bench/export.py`
- [X] T026 [US4] Ensure int8 rules are respected in report generation (no per-second throughput reporting) in `/data1/huangzhe/code/accelsim-test/src/accelsim_test/gemm_transpose_bench/report.py` (FR-008b)

---

## Phase 7: User Story 5 - Generate stakeholder report with comparison tables (Priority: P5)

Goal: Produce a concise markdown report with square-suite and non-square-suite tables including all executed configs/cases by default and the required ratios (FR-010 / FR-010a).

Independent Test: Generate report from `results.json` and verify table structure and ratio columns.

### Tests (US5)

- [X] T027 [P] [US5] Add unit test for markdown table generation in `/data1/huangzhe/code/accelsim-test/tests/unit/test_gemm_transpose_report.py`

### Implementation (US5)

- [X] T028 [US5] Implement report generator in `/data1/huangzhe/code/accelsim-test/src/accelsim_test/gemm_transpose_bench/report.py` (two tables + conclusions section)

---

## Phase 8: Profiling Mode (Cross-cutting)

- [X] T029 [US4] Implement `profile` workflow in `/data1/huangzhe/code/accelsim-test/src/accelsim_test/gemm_transpose_bench/profiling.py` to run `ncu` once per configuration/case and link artifacts into `results.json` (FR-012 / FR-012a)
- [X] T030 [US4] Add a minimal “profiling smoke” test harness in `/data1/huangzhe/code/accelsim-test/tests/manual/` that runs one configuration under `ncu` when the tool is present

---

## Phase 9: Documentation & Validation (Cross-cutting)

- [X] T031 [US5] Validate and update `/data1/huangzhe/code/accelsim-test/specs/002-gemm-transpose-bench/quickstart.md` to match the final CLI/task names
- [X] T032 [US5] Add README pointer(s) in `/data1/huangzhe/code/accelsim-test/README.md` to the new benchmark quickstart and report outputs
