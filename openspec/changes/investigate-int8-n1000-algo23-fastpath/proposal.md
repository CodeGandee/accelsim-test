## Why

In `reports/transpose_matmul/gemm_transpose_full_sweep_20260205_041151/stakeholder_report.md`, the square `N=1000` int8 case shows an abnormal behavior: `ABT_view` heuristically selects cuBLASLt `algo_id=23` and runs ~2-3x faster than the other transpose modes (which select `algo_id=64`). We need to explain what is different about the kernel/workload when algo 23 is selected, and why that algorithm appears ineligible for `AB` / `ATB_view`.

## What Changes

- Add a structured profiling investigation using Nsight Compute (`ncu`) for the standalone repro case `N=1000` int8.
- Use a comparison set that isolates "transpose flag/layout" vs "algorithm choice":
  - `AB` (heuristic; expected algo 64)
  - `ABT_view` (heuristic; expected algo 23, fast path)
  - `ABT_view` forced to algo 64 (control; same transpose mode, different algo)
  - (optional) `ABT_view` forced to algo 23 (sanity; should match heuristic)
  - (optional) attempt forcing algo 23 for `AB` and `ATB_view`; record `NA` when `cublasLtMatmulAlgoCheck` rejects it
- Capture and persist the `ncu` report artifacts (per-case `.ncu-rep` plus derived exports as needed) so results are reproducible and diffable.
- Produce a concise analysis write-up answering:
  - What kernels are launched in each case (kernel identity/count, launch configuration, register/shared usage)
  - What differs in achieved utilization (tensor core pipe activity, occupancy, memory traffic, stall reasons)
  - Why algo 23 is only eligible for `ABT_view` (or why the specific algo-23 tactic/config is rejected for `AB` / `ATB_view`)
- Update the stakeholder report with a profiling-backed explanation and pointers to the captured artifacts.

## Capabilities

### New Capabilities

- `ncu-profile-int8-n1000-algo23-fastpath`: Provide a reproducible, artifacted `ncu` profiling workflow and analysis for the `N=1000` int8 case. The workflow must (1) confirm the `ABT_view` algo-23 fast path, (2) include an apples-to-apples control by running `ABT_view` under algo 64, and (3) document eligibility constraints when attempting to force algo 23 for other transpose modes.

### Modified Capabilities

- (none)

## Impact

- Code:
  - `cpp/src/repro_algo23_int8_n1000.cu` (standalone repro target used as the profiling driver).
  - Potentially `scripts/` for a repeatable "run ncu + export reports" wrapper and standardized output directories.
- Tooling/dependencies:
  - Requires Nsight Compute `ncu` (location and counter-permission setup can be handled separately from the profiling plan).
- Artifacts:
  - New `.ncu-rep` files and derived exports (CSV/JSON where appropriate) stored alongside the experiment report under `reports/transpose_matmul/`.
  - Stakeholder-facing explanation added to `reports/transpose_matmul/gemm_transpose_full_sweep_20260205_041151/stakeholder_report.md`.
