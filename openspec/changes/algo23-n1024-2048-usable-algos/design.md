## Context

The existing investigation established a strong explanation for the `N=1000` int8 outlier (`ABT_view` selecting `algo_id=23`) and showed that larger square sizes (`N=1024`, `N=2048`) heuristically select `algo_id=71` for both `AB` and `ABT_view`. What is still missing is a direct, per-case answer to: (1) whether `algo_id=23` is even usable at `N=1024/2048`, and (2) if usable, why it is not selected.

Current runner behavior uses a top-1 cuBLASLt heuristic result, which is insufficient for eligibility analysis. We need an experiment that explicitly enumerates candidate algorithms/configurations and validates each candidate with `cublasLtMatmulAlgoCheck` under fixed layout, dtype, shape, and workspace constraints.

Stakeholders: experiment/report owners working under `reports/transpose_matmul/`, and future contributors who need reproducible evidence for algorithm-selection claims.

## Goals / Non-Goals

**Goals:**
- Produce a reproducible experiment for int8 row-major square GEMM that compares only `AB` and `ABT_view`.
- Cover `N=1000` (control), `N=1024`, and `N=2048` with identical run policy.
- Enumerate candidate cuBLASLt algorithm/config combinations and classify each as usable/non-usable via `cublasLtMatmulAlgoCheck`.
- Benchmark usable candidates and output machine-readable + markdown artifacts suitable for stakeholder reporting.
- Make the resulting tables directly answer why `algo_id=23` appears only at `N=1000` in current observations.

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
- Rationale: `N=1000` provides a known positive control where `ABT_view` can select `algo_id=23`; this validates that the experiment pipeline can reproduce prior behavior.
- Alternative considered: run only `1024/2048`.
- Why rejected: without the control, null results could be attributed to tooling issues rather than true shape effects.

### Decision: Define "usable" strictly as `cublasLtMatmulAlgoCheck` success under fixed workspace policy
- Rationale: this aligns with cuBLASLt’s own compatibility gate and provides a clear binary eligibility criterion.
- Alternative considered: infer usability from heuristic output only.
- Why rejected: heuristic rank does not cover the full candidate space and cannot distinguish unsupported vs simply unselected.

### Decision: Persist both candidate-level raw stats and concise summary tables
- Rationale: raw artifacts are needed for auditability; summary tables are needed for communication in stakeholder reports.
- Alternative considered: summary-only markdown.
- Why rejected: summary-only output prevents independent validation and follow-up slicing.

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

## Open Questions

- Should candidate enumeration include only a large heuristic list, or also explicit `MatmulAlgoGetIds` expansion with tile/stage combinations in this first version?
- What candidate upper bound keeps runtime practical on B200 while still being representative?
- Do we need per-candidate verification checks beyond performance timing for this analysis scope?
