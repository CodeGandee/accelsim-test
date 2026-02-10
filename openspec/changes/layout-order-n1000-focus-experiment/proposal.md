## Why

The current N=1000 int8 results show `ABT_view` can select a much faster cuBLASLt/CUTLASS kernel than `AB`, and one plausible driver is **memory access contiguity in the GEMM mainloop** (especially whether B is read unit-stride along K for the chosen `trans_b` + storage order). We need a focused experiment that varies **A/B matrix storage order** (including mixed row/col cases) to test whether the “fast path” moves between `ABT_view` and `ATB_view` as predicted.

## What Changes

- Add a focused N=1000 experiment that runs `AB`, `ATB_view`, and `ABT_view` in **view-only** mode (no copy/pack cases), for all A/B order pairs:
  - `row/row`, `row/col`, `col/row`, `col/col` (where `row = CUBLASLT_ORDER_ROW`, `col = CUBLASLT_ORDER_COL`).
- Additionally, vary the output layout order (`order_c`) only for the baseline `A=row, B=row` cases to sanity-check whether output storage order affects algo/kernel selection or timing.
- Ensure the experiment can enforce **math equivalence** across the three transpose modes (e.g., by generating symmetric A and B) so performance comparisons are not confounded by “different GEMM” concerns.
- Emit a small, reproducible artifact set per run directory: selected algo/config, kernel names, and optional `nsys`/`ncu` captures for the hottest GEMM kernel.

## Capabilities

### New Capabilities

- `layout-order-focus-experiment`: A focused benchmark/profiling run that compares `AB` vs `ATB_view` vs `ABT_view` across A/B row/col storage-order combinations for N=1000, and captures kernel-level evidence to validate (or falsify) the “K-contiguity drives fast-kernel selection” hypothesis.

### Modified Capabilities

<!-- none -->

## Impact

- C++: add a small focused executable (or extend the existing repro) to parameterize `cublasLtMatrixLayout` order and optionally generate symmetric inputs.
- Python/scripts: add a thin orchestrator to run the A/B order matrix and the limited `order_c` sweep for `A=row, B=row`, and optionally collect `nsys`/`ncu` artifacts into a user-specified output directory (e.g., `tmp/<subdir>`).
- Reports: add a short note describing the experiment and how to interpret outcomes (kernel name + algo selection vs layout order).
