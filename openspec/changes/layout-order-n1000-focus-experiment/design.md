## Context

We have an observed performance anomaly for `M=N=K=1000` int8 GEMM variants in which `ABT_view` (i.e., `trans_b=T`) selects a different cuBLASLt algorithm/kernel and is much faster than `AB` under row-major layouts. A plausible mechanism is that `trans_b=T` changes how B is traversed along K during the GEMM mainloop, enabling a more efficient CUTLASS TensorOp kernel family (better coalescing/vectorization and shared-memory staging).

This repository already contains:

- a standalone cuBLASLt repro (`cpp/src/repro_algo23_int8_n1000.cu`) and helpers (`cpp/src/cublaslt_gemm.cu`), and
- profiling wrappers/scripts (`src/accelsim_test/profiling/cublaslt_profiling.py`, `scripts/cublaslt_kernel_discovery.py`, `scripts/cublaslt_ncu_profile.py`) that can capture `nsys`/`ncu` artifacts into a user-specified directory.

We want a focused experiment that flips only **storage order** (row-major vs column-major) to test whether the “fast path” shifts from `ABT_view` to `ATB_view`, consistent with a “K-contiguity drives kernel selection/efficiency” hypothesis.

## Goals / Non-Goals

**Goals:**
- Provide a focused N=1000 experiment that runs the view-only cases `AB`, `ATB_view`, and `ABT_view` for both `CUBLASLT_ORDER_ROW` and `CUBLASLT_ORDER_COL`.
- Provide an option to make the three cases mathematically equivalent (symmetric A and B) so that comparisons are not dismissed as “different GEMM”.
- Make outputs reproducible and portable by writing everything under a user-specified `--out-dir`.
- Optionally capture kernel evidence (`nsys` kernel list, `ncu` report + CSV exports) for each case.

**Non-Goals:**
- Optimizing cuBLASLt kernel selection itself (this change is measurement/investigation, not a performance fix).
- Supporting the full sweep harness / all shapes / all dtypes (this is intentionally narrow: `N=1000` int8).
- Adding copy/pack cases (`ATB_copyA`, `ABT_copyB`) for this focused experiment.

## Decisions

1) **Reuse the existing standalone repro as the execution engine**

Decision: extend `repro_algo23_int8_n1000` (or factor shared code) to support:
- `--order {row|col}` to set `CUBLASLT_MATRIX_LAYOUT_ORDER` for A/B/C layouts, and
- `--symmetric-inputs` (or similar) to generate symmetric A and symmetric B.

Rationale:
- We already have a minimal cuBLASLt harness with deterministic matrix generation, algorithm reporting, and NVTX support.
- Adding order + symmetric inputs keeps changes localized and avoids duplicating cuBLASLt setup code.

Alternatives considered:
- Create a new standalone binary dedicated to layout-order experiments. This is cleaner naming-wise, but adds build plumbing and risks drifting from the already validated repro.

2) **Orchestrate the 2×3 experiment matrix in Python and reuse the existing profiling wrappers**

Decision: add a thin script (under `scripts/`) that:
- runs the 6 cases (2 orders × 3 transpose-view variants),
- records per-case stdout + selected algo/config in a machine-readable file (JSON/CSV), and
- optionally invokes `scripts/cublaslt_kernel_discovery.py` / `scripts/cublaslt_ncu_profile.py` to capture kernel evidence into `<out-dir>/profiles/...`.

Rationale:
- The repo’s convention is “Python orchestrates; C++ is the runner”.
- The profiling wrappers already implement deterministic artifact layout and metadata files.
- Keeping `--out-dir` as the top-level control matches the “standalone investigation” requirement (tests put artifacts under `tmp/<subdir>`).

3) **Use symmetric-input mode to ensure “same math” without changing transpose flags**

Decision: keep transpose flags as `NN`, `TN`, `NT` (so we exercise the real access patterns), but optionally generate symmetric A and B so that `AB == AᵀB == ABᵀ` mathematically.

Rationale:
- This preserves the performance-relevant differences (transpose flags and associated memory access) while removing correctness/interpretation objections.

## Risks / Trade-offs

- **[cuBLASLt internal transforms]** cuBLASLt may internally pack/transform operands for int8, weakening the “contiguity flips” story. → Mitigation: rely on kernel-name evidence (`nsys`) and instruction/memory metrics (`ncu`) rather than assumptions about raw pointer access.
- **[Heuristic variability]** the heuristic-selected algo may differ across driver/toolkit versions. → Mitigation: record exact algo/config and provide an option to force a chosen algo ID for controlled comparisons.
- **[Symmetric inputs change data distribution]** symmetric generation may alter values. → Mitigation: keep the same value distribution and only mirror elements; run both symmetric and non-symmetric modes if needed.
