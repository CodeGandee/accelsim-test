## Context

We have an observed performance anomaly for `M=N=K=1000` int8 GEMM variants in which `ABT_view` (i.e., `trans_b=T`) selects a different cuBLASLt algorithm/kernel and is much faster than `AB` under row-major layouts. A plausible mechanism is that `trans_b=T` changes how B is traversed along K during the GEMM mainloop, enabling a more efficient CUTLASS TensorOp kernel family (better coalescing/vectorization and shared-memory staging).

This repository already contains:

- a standalone cuBLASLt repro (`cpp/src/repro_algo23_int8_n1000.cu`) and helpers (`cpp/src/cublaslt_gemm.cu`), and
- profiling wrappers/scripts (`src/accelsim_test/profiling/cublaslt_profiling.py`, `scripts/cublaslt_kernel_discovery.py`, `scripts/cublaslt_ncu_profile.py`) that can capture `nsys`/`ncu` artifacts into a user-specified directory.

We want a focused experiment that varies **A/B storage order** (including mixed row/col cases) to test whether the “fast path” shifts between `ABT_view` and `ATB_view`, consistent with a “K-contiguity drives kernel selection/efficiency” hypothesis.

## Goals / Non-Goals

**Goals:**
- Provide a focused N=1000 experiment that runs the view-only cases `AB`, `ATB_view`, and `ABT_view` across A/B layout-order combinations (`row/row`, `row/col`, `col/row`, `col/col`).
- Include a limited output-layout sensitivity check by varying the output order (`order_c`) for the baseline `order_a=row, order_b=row` cases.
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
- `--order {row|col}` (shorthand) and/or per-matrix flags (e.g., `--order-a {row|col}`, `--order-b {row|col}`, `--order-c {row|col}`) to set `CUBLASLT_MATRIX_LAYOUT_ORDER` for A/B/C layouts, and
- `--symmetric-inputs` (or similar) to generate symmetric A and symmetric B.

Rationale:
- We already have a minimal cuBLASLt harness with deterministic matrix generation, algorithm reporting, and NVTX support.
- Adding order + symmetric inputs keeps changes localized and avoids duplicating cuBLASLt setup code.

Alternatives considered:
- Create a new standalone binary dedicated to layout-order experiments. This is cleaner naming-wise, but adds build plumbing and risks drifting from the already validated repro.

2) **Orchestrate the 4×3 experiment matrix in Python and reuse the existing profiling wrappers**

Decision: add a thin script (under `scripts/`) that:
- runs the 12 cases (4 A/B order pairs × 3 transpose-view variants), plus a limited output-order sweep for `order_a=row, order_b=row` (`order_c=row` and `order_c=col`),
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

## Reporting (manual template)

The experiment outputs are intended to be *referenced* by a human-written report (not program-generated narrative). A suggested report template:

```markdown
# Layout-order focus experiment (N=1000 int8): Results & Analysis

## TL;DR
- Winner(s):
- Does the winner flip with layout order?
- One-sentence conclusion:

## Hypothesis
K-contiguity / layout-order drives algo+kernel selection and thus performance.

## Environment
- Date:
- Host:
- GPU:
- Driver:
- CUDA / Pixi env:
- Git SHA (and dirty?):
- Tooling: `nsys --version`, `ncu --version`

## Experiment Design
- Cases: order_a ∈ {row, col} × order_b ∈ {row, col} × variant ∈ {AB, ATB_view, ABT_view}
- Output order: keep `order_c=row` for the full A/B matrix; additionally sweep `order_c ∈ {row, col}` for `order_a=row, order_b=row`.
- Inputs: symmetric? (yes/no)
- Timing: warmup=?, iters=?, min-time/noise criteria?

## Results (A/B order matrix; `order_c=row`)
| order_a | order_b | variant  | time (us) | algo_id | tile | stages | main kernel (nsys) | grid/block |
|--------:|--------:|----------|----------:|---------|------|--------|--------------------|------------|
| row     | row     | AB       |           |         |      |        |                    |            |
| row     | row     | ATB_view |           |         |      |        |                    |            |
| row     | row     | ABT_view |           |         |      |        |                    |            |
| row     | col     | AB       |           |         |      |        |                    |            |
| row     | col     | ATB_view |           |         |      |        |                    |            |
| row     | col     | ABT_view |           |         |      |        |                    |            |
| col     | row     | AB       |           |         |      |        |                    |            |
| col     | row     | ATB_view |           |         |      |        |                    |            |
| col     | row     | ABT_view |           |         |      |        |                    |            |
| col     | col     | AB       |           |         |      |        |                    |            |
| col     | col     | ATB_view |           |         |      |        |                    |            |
| col     | col     | ABT_view |           |         |      |        |                    |            |

## Output order check (baseline `order_a=row, order_b=row`)
| order_c | variant  | time (us) | algo_id | tile | stages | main kernel (nsys) | grid/block |
|--------:|----------|----------:|---------|------|--------|--------------------|------------|
| row     | AB       |           |         |      |        |                    |            |
| col     | AB       |           |         |      |        |                    |            |
| row     | ATB_view |           |         |      |        |                    |            |
| col     | ATB_view |           |         |      |        |                    |            |
| row     | ABT_view |           |         |      |        |                    |            |
| col     | ABT_view |           |         |      |        |                    |            |

Artifacts to reference per case (when captured):
- `profiles/<case_id>/nsys/kernel_list.csv`
- `profiles/<case_id>/ncu/details.csv` (plus `raw.csv` and `profile.ncu-rep`)

## Analysis
### 1) Does the “winner” flip with layout order?
- Expected if hypothesis holds:
- Observed:

### 2) Algo/kernel deltas (what changed?)
- algo_id / tile / stages / splitK differences:
- kernel family differences (e.g., `tensorop_i16832...` vs `wmma_tensorop_i161616...`):

### 3) ncu evidence (why faster?)
- Tensor utilization / instruction shape:
- Memory throughput / stalls:
- Occupancy / waves / launch config implications:

## Repro (commands)
- Full command lines used (or point to each `meta.json` / `invocation.txt`)

## Appendix
- Links to raw captures and exported CSVs
```

Notes:
- Prefer linking to per-case `meta.json` / `invocation.txt` under `profiles/<case_id>/...` rather than copy/pasting long commands into the narrative.
- Keep the report portable: reference artifacts relative to the output root directory used for the experiment.
