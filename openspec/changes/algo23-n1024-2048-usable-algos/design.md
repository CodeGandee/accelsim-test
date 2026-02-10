## Context

We want a direct, per-case view of the usable cuBLASLt algorithm space and its performance when switching between `AB` and `ABT_view` for int8 GEMM. The focus is square GEMM with row-major layouts at `N=1024` and `N=2048`, with `N=1000` included as a reference row.

Current runner behavior uses a top-1 cuBLASLt heuristic result, which is insufficient for understanding the broader performance landscape. We need an experiment that enumerates candidate algorithms/configurations and validates each candidate with `cublasLtMatmulAlgoCheck` under fixed layout, dtype, shape, and workspace constraints.

Stakeholders: experiment/report owners working under `reports/transpose_matmul/`, and future contributors who need reproducible evidence for algorithm-selection claims.

## Goals / Non-Goals

**Goals:**
- Produce a reproducible experiment for int8 row-major square GEMM that compares only `AB` and `ABT_view`.
- Cover `N=1024` and `N=2048`, with `N=1000` included as a reference row, using identical run policy.
- Enumerate candidate cuBLASLt algorithm/config combinations and classify each as usable/non-usable via `cublasLtMatmulAlgoCheck`.
- Benchmark usable candidates and output machine-readable + markdown artifacts suitable for stakeholder reporting.
- Make the resulting tables directly answer how performance differs between `AB` and `ABT_view` across the usable algorithm space.

**Non-Goals:**
- Expanding to non-row-major layouts in this change.
- Expanding to dtypes other than `int8,int8->int32`.
- Full-kernel attribution (nsys/ncu) for every candidate; that remains a follow-up deep profiling step.
- Changing cuBLASLt behavior or implementing custom kernels.

## Decisions

### Decision: Scope to `AB` and `ABT_view` only for row-major matrices
- Rationale: this isolates the exact question raised by current evidence while minimizing confounders from other transpose/copy modes.
- Alternative considered: include `ATB_view` and copy variants in the same run.
- Why rejected: broader scope would dilute the key AB-vs-ABT eligibility signal and significantly increase run volume.

### Decision: Include `N=1000` as in-run control next to `1024/2048`
- Rationale: `N=1000` provides a reference row to compare against the power-of-two sizes (`1024/2048`) and helps detect small-shape behavior that may not extrapolate.
- Alternative considered: run only `1024/2048`.
- Why rejected: without the control, null results could be attributed to tooling issues rather than true shape effects.

### Decision: Enumerate `algo_id` via GetIds; pick best config per `algo_id` per case
- Rationale: the experiment’s output table is keyed by `algo_id`, so enumeration should start from the complete `algo_id` list for the dtype/compute tuple (via `cublasLtMatmulAlgoGetIds`), not from a heuristic subset.
- For each `(N, variant, algo_id)`, request the best configuration for that `algo_id` using heuristic search constrained to that ID, then validate with `cublasLtMatmulAlgoCheck` (including workspace requirement under the fixed workspace policy).
- Alternative considered: enumerate only a large global `cublasLtMatmulAlgoGetHeuristic` list.
- Why rejected: a global heuristic list is not guaranteed to include all `algo_id` values and cannot support “all usable algorithms” claims in the `algo_id` sense.

### Decision: Define "usable" strictly as `cublasLtMatmulAlgoCheck` success under fixed workspace policy
- Rationale: this aligns with cuBLASLt’s own compatibility gate and provides a clear binary eligibility criterion.
- Alternative considered: infer usability from heuristic output only.
- Why rejected: heuristic rank does not cover the full candidate space and cannot distinguish unsupported vs simply unselected.

### Decision: Persist both candidate-level raw stats and concise summary tables
- Rationale: raw artifacts are needed for auditability; summary tables are needed for communication in stakeholder reports.
- Alternative considered: summary-only markdown.
- Why rejected: summary-only output prevents independent validation and follow-up slicing.

### Decision: Report one best config per `algo_id` per case
- Rationale: cuBLASLt “algorithms” are identified by `algo_id`, but each `algo_id` has many possible config knobs (tile/stages/splitK/etc). Reporting one best usable config per `algo_id` per `(N,variant)` keeps the merged table interpretable and bounded.
- Alternative considered: report every distinct config as a separate row.
- Why rejected: config-level enumeration can explode in size and obscures the key question (how each `algo_id` family behaves for `AB` vs `ABT_view`).

### Decision: Two-pass timing (fast scan + stability rerun)
- Rationale: timing all `algo_id`s can be expensive, and some kernels are short enough to be noisy with small iteration counts.
- Pass 1: time all usable `algo_id`s with a low warmup/iters setting; produce the merged table.
- Pass 2 (optional): rerun the top-K (per case) with higher iters for stability and record both pass-1 and pass-2 timing in raw artifacts.

## Risks / Trade-offs

- [Candidate-space explosion] Enumerating too many algorithm/config combinations can increase runtime substantially.
  - Mitigation: constrain scope to two variants and three square sizes; allow bounded candidate limits when needed while recording the limit in metadata.

- [Heuristic/API version variability] Candidate ordering and availability can vary across CUDA versions.
  - Mitigation: store environment metadata (CUDA toolkit/runtime, GPU, driver where available) and complete invocation commands.

- [False interpretation of non-selection] An algorithm can be usable but still not selected because of ranking.
  - Mitigation: report both usability and measured timing so selection-vs-performance distinctions are explicit.

- [Noisy timing for tiny kernels] Short kernels can be sensitive to measurement configuration.
  - Mitigation: use fixed warmup/iteration policy and report run parameters in metadata.

## Migration Plan

1. Add/extend experiment runner logic to support the required shapes, variants, and candidate evaluation flow.
2. Produce artifacts under a dedicated output directory in `reports/transpose_matmul/`.
3. Validate output schema and summary table content against the requirements.
4. Integrate summary findings into the stakeholder report section discussing `algo_id=23` vs alternatives.

Rollback strategy: if the new runner path is unstable, keep existing report content and preserve this change as experiment-only scaffolding without replacing prior conclusions.

## Defaults / Flags

- Default shapes: `N in {1000, 1024, 2048}`; variants: `AB` and `ABT_view`; dtype: `int8,int8->int32`; orders: A/B/C row-major.
- Enumeration: `cublasLtMatmulAlgoGetIds` for `algo_id` list; per-`algo_id` best-config selection via constrained heuristic + `AlgoCheck`.
- Workspace policy: fixed `max_workspace_bytes` (default 64 MiB) applied consistently across all cases.
- Timing policy:
  - Pass 1 defaults: warmup 10, iters 50 (tuneable).
  - Pass 2 defaults: rerun top-K (e.g., 10) per case with warmup 50, iters 500 (tuneable).
- Safety cutoff: allow `--max-algo-ids K` to cap runtime; always record both requested and returned algo-id counts in metadata.
- Correctness: perform one correctness check per case using the heuristic-selected configuration (or a documented baseline). Do not do full verification per `algo_id` in v1; treat runtime failures as `NA` with an error reason.

## Final Report Template

This experiment is intended to produce a single stakeholder-facing markdown report that can be regenerated from raw artifacts. The report SHOULD be written under `reports/transpose_matmul/<run_id>/stakeholder_report.md` (or an equivalent stable report path) and reference raw CSV/JSON artifacts by relative path.

### 1) Executive Summary

- What was tested: int8 square GEMM, row-major, `AB` vs `ABT_view`, `N in {1000,1024,2048}`.
- Main findings: 2-5 bullets summarizing (a) whether `ABT_view` tends to be faster, (b) which `algo_id`s dominate for each case, and (c) any notable outliers.

### 2) Environment

- GPU model / SM, CUDA version, cuBLASLt version (if available)
- Host OS and repo commit SHA (dirty/clean)
- Workspace policy: `max_workspace_bytes`

### 3) Methodology

- Definition of "algorithm": table rows are `algo_id`.
- Definition of "best config": for each `(N, variant, algo_id)`, choose the best configuration produced by the per-`algo_id` constrained heuristic, then require `AlgoCheck` to pass (and workspace requirement to be <= policy).
- Timing policy: pass 1 (scan all usable), optional pass 2 (rerun top-K for stability).
- Note: `AB` and `ABT_view` are different math unless inputs are constrained; this report is about performance and algorithm selection, not numerical equivalence.

### 4) Results (By N)

Organize results by `N` so each section is self-contained. Within each `N`, include:

- a timing table (rows: `algo_id`, columns: `AB` vs `ABT_view`)
- a selected-config table (the chosen best config per `algo_id` for that `N` and variant)

Use `NA` to indicate "no usable selected config" for that `(N, variant, algo_id)` cell.

Recommended units: `time_us` (average per GEMM call).

#### N=1000

Timing:

| algo_id | AB time_us | ABT_view time_us |
|---:|---:|---:|
| 23 | NA | 12.3 |
| 64 | 33.4 | 22.9 |

Selected config details (AB):

| algo_id | tile_id | stages_id | splitk_num | required_workspace_bytes | notes |
|---:|---:|---:|---:|---:|---|
| 64 | 20 | 8 | 1 | 0 | chosen best for this algo_id and case |

Selected config details (ABT_view):

| algo_id | tile_id | stages_id | splitk_num | required_workspace_bytes | notes |
|---:|---:|---:|---:|---:|---|
| 23 | 18 | 21 | 1 | 0 | chosen best for this algo_id and case |
| 64 | 20 | 8 | 1 | 0 | usable but slower |

#### N=1024

Timing:

| algo_id | AB time_us | ABT_view time_us |
|---:|---:|---:|
| 71 | 18.4 | 18.6 |

Selected config details (AB):

| algo_id | tile_id | stages_id | splitk_num | required_workspace_bytes | notes |
|---:|---:|---:|---:|---:|---|
| 71 | 23 | 36 | 1 | 0 | chosen best for this algo_id and case |

Selected config details (ABT_view):

| algo_id | tile_id | stages_id | splitk_num | required_workspace_bytes | notes |
|---:|---:|---:|---:|---:|---|
| 71 | 23 | 36 | 1 | 0 | chosen best for this algo_id and case |

#### N=2048

Timing:

| algo_id | AB time_us | ABT_view time_us |
|---:|---:|---:|
| 71 | 22.0 | 22.3 |

Selected config details (AB):

| algo_id | tile_id | stages_id | splitk_num | required_workspace_bytes | notes |
|---:|---:|---:|---:|---:|---|
| 71 | 23 | 36 | 1 | 0 | chosen best for this algo_id and case |

Selected config details (ABT_view):

| algo_id | tile_id | stages_id | splitk_num | required_workspace_bytes | notes |
|---:|---:|---:|---:|---:|---|
| 71 | 23 | 36 | 1 | 0 | chosen best for this algo_id and case |

### 6) Analysis

Suggested subsections:

- `AB` vs `ABT_view` consistency: which `algo_id`s are competitive in both? which are case-specific?
- `algo_id=23` presence/absence: for each case, state whether it is usable and whether it is competitive when usable.
- Sensitivity checks: show whether results change materially with pass-2 stable timing (if performed).

### 7) Reproducibility

- Command lines used to run the experiment (verbatim).
- Output directory layout and key files:
  - `meta.json`
  - `results.json` / `results.csv` (candidate-level artifacts)
  - `merged_table.csv` (optional; if emitted, keep it as an index/union view)
  - `report.md` (generated summary, if any)

### 8) Appendix (Optional)

- Top-K per case (sorted by `time_us`), including `algo_id` and key config knobs.
- Notes about any `AlgoCheck` failures or runtime errors (distinct from `not returned`).
