## Why

Current evidence explains the `N=1000` int8 outlier (`ABT_view` selecting `algo_id=23`), but it does not yet prove why the same does not happen at `N=1024` and `N=2048`. We need a targeted, reproducible experiment that enumerates all usable cuBLASLt algorithms per case so we can separate heuristic ranking effects from hard eligibility constraints.

## What Changes

- Add an experiment workflow that compares only `AB` and `ABT_view` for square int8 GEMM with row-major layouts.
- Cover `N=1000` (control), `N=1024`, and `N=2048` under identical run settings.
- Enumerate candidate cuBLASLt algorithm/config combinations and classify them as usable/non-usable via `cublasLtMatmulAlgoCheck` for each case.
- Time each usable algorithm and produce per-`N` comparison tables for `AB` vs `ABT_view`.
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
