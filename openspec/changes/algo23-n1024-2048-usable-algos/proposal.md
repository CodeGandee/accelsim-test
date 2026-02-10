## Why

We need a targeted, reproducible experiment that shows how cuBLASLt’s usable algorithm space performs for int8 GEMM when we switch between `matmul(A,B)` and `matmul(A,B.T)` (view transpose). This will let us quantify whether `ABT_view` has a consistent advantage or only wins for specific sizes/configurations.

## What Changes

- Add an experiment workflow that compares only `AB` and `ABT_view` for square int8 GEMM with row-major layouts.
- Cover `N=1024` and `N=2048`, plus `N=1000` as a reference row, under identical run settings.
- Enumerate usable cuBLASLt algorithm/config candidates via discovery APIs and classify them as usable/non-usable via `cublasLtMatmulAlgoCheck` for each case.
- Time each usable candidate and produce per-`N` comparison tables for `AB` vs `ABT_view`.
- Export raw machine-readable artifacts (full candidate list, usability status, timing stats, selected heuristic result) plus a concise markdown summary for report integration.

## Capabilities

### New Capabilities
- `cublaslt-usable-algo-sweep`: Enumerate, validate, and benchmark usable cuBLASLt algorithms for row-major int8 `AB` vs `ABT_view` across selected square sizes (`N=1000/1024/2048`) and emit reproducible analysis artifacts.

### Modified Capabilities
- None.

## Impact

- Affected code: experiment runner and/or repro utility under `scripts/` and `cpp/src/`, plus report-generation glue under `src/accelsim_test/` if needed.
- Affected outputs: new report directory under `reports/transpose_matmul/` containing markdown summary and raw CSV/JSON artifacts.
- Dependencies/tools: cuBLASLt (`AlgoGetHeuristic`, `AlgoCheck`, possible ID/cap enumeration path), existing Pixi CUDA environment, optional Nsight tools for follow-up profiling.
