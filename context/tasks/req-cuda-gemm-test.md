# Requirement: CUDA GEMM transpose benchmark (`matmul(A,B)` vs `matmul(A.T,B)` vs `matmul(A,B.T)`)

## Background
Matrix multiplication performance on GPUs depends heavily on operand layouts, transpose flags, and which kernel path the backend selects (e.g., cuBLASLt tensor-core kernels vs fallback kernels). In Python frameworks, `A.T` / `B.T` are typically *views* with swapped strides (not a materialized transpose), which can turn a fast GEMM into a slower strided-access GEMM, or trigger an implicit copy/contiguous conversion.

This task defines a benchmark to quantify performance differences between:

- **Square suite (A and B are `N×N`)**:
  - `matmul(A, B)`
  - `matmul(A.T, B)`
  - `matmul(A, B.T)`
  - `matmul(copy(A.T), B)`
  - `matmul(A, copy(B.T))`
- **Non-square suite (transpose-A and transpose-B; FLOP-matched)**:
  - given `A_atb` of size `K×M` and `B_atb` of size `K×N`:
    - `matmul(A_atb.T, B_atb)`
    - `matmul(copy(A_atb.T), B_atb)`
  - given `A_abt` of size `M×K` and `B_abt` of size `N×K`:
    - `matmul(A_abt, B_abt.T)`
    - `matmul(A_abt, copy(B_abt.T))`

on CUDA, under controlled conditions.

## Goal
Produce a reproducible benchmark (microbenchmark + reporting) that measures and explains the performance differences among the specified matmul cases across a representative set of shapes and dtypes on NVIDIA GPUs.

## Scope
### In scope
- CUDA GPU benchmarking for the matmul cases above.
- Use cuBLASLt for GEMM (no custom GEMM kernels).
- Multiple operand **layouts**:
  - contiguous (`contiguous()` in row-major frameworks)
  - transposed views (stride-swapped, non-contiguous)
  - optionally: explicitly materialized transposes (`A_T = A.T.contiguous()`)
- Multiple **dtypes** and math modes (as supported by the chosen backend):
  - `fp16`, `bf16`, `fp32` (and `tf32` when applicable)
  - int8 and mixed-type matmuls where supported (see “Dtypes” below)
- Shape sweep for GEMM sizes (M, N, K) relevant to deep learning and HPC.
- Collection of timing and achieved throughput (e.g., TFLOP/s), plus minimal correctness checks.

### Out of scope (for this requirement)
- Multi-GPU, distributed GEMM, pipeline parallelism.
- End-to-end model benchmarks (training/inference).
- Exhaustive autotuning across all possible kernels/algorithms.
- Non-NVIDIA backends (ROCm, CPU BLAS).
- Writing/optimizing a custom GEMM kernel (the benchmark must rely on cuBLASLt).

## Definitions
- **GEMM**: General matrix-matrix multiply. For `C[M,N] = A[M,K] @ B[K,N]`.
- **Transpose view**: `A.T` or `B.T` without a copy; same storage, different strides (framework-level concept).
- **Transpose flag (cuBLASLt)**: requesting transpose via matmul descriptors (e.g., `op(A)=T` / `op(B)=T`) without explicitly transposing data.
- **Materialized transpose**: `A_T = A.T.contiguous()`; creates a new contiguous buffer.
- **Variant naming** (assuming `A` is `[M,K]` and `B` is `[K,N]`):
  - Square suite (`A[N,N]`, `B[N,N]`):
    - `AB`         := `A @ B`
    - `ATB_view`   := `A.T @ B`
    - `ABT_view`   := `A @ B.T`
    - `ATB_copyA`  := `copy(A.T) @ B`
    - `ABT_copyB`  := `A @ copy(B.T)`
  - Non-square suite:
    - transpose-A (`A_atb[K,M]`, `B_atb[K,N]`):
      - `ATB_view`  := `A_atb.T @ B_atb`
      - `ATB_copyA` := `copy(A_atb.T) @ B_atb`
    - transpose-B (`A_abt[M,K]`, `B_abt[N,K]`):
      - `ABT_view`  := `A_abt @ B_abt.T`
      - `ABT_copyB` := `A_abt @ copy(B_abt.T)`

## Important constraint (non-square matrices)
If `AB` is defined as `A[M,K] @ B[K,N]`, then:
- `A.T @ B` is only defined when `B` has shape `[M,N]` (because `A.T` is `[K,M]`).
- `A @ B.T` is only defined when `A` has shape `[M,N]` (because `B.T` is `[N,K]`).

Therefore, for **rectangular** matrices, it is impossible to compare `A @ B` vs `A.T @ B` vs `A @ B.T` while keeping the *exact same stored* `A[M,K]` and `B[K,N]` for all three expressions. The only case where all three expressions are simultaneously well-defined with the same `A` and `B` is the fully square case `M=N=K`.

## Primary questions to answer
1. In the square suite, how do `ATB_view` and `ABT_view` compare to `AB`?
2. Is slowdown primarily due to:
   - strided memory access in the GEMM kernel, and/or
   - implicit materialization (hidden copies), and/or
   - different algorithm selection (tensor core usage, tile shapes, etc.)?
3. What is the overhead of explicit transpose materialization (compare `*_view` vs `*_copy*`)?
4. How do these effects vary with shape (square vs non-square, aspect ratios), dtype pair, and alignment?

## Requirements
### Functional requirements
- The benchmark MUST run in the Pixi environment `cuda13` (CUDA 13 toolchain stack), so results are reproducible.
  - Example: `pixi run -e cuda13 -- <benchmark-cli> ...`
- **Benchmarking Library:** MUST use **NVBench** (located in `extern/orphan/nvbench`) for all kernel timing, warmup, and statistical aggregation.
  - DO NOT write manual timing loops with `cudaEventRecord`.
  - Use NVBench's `state.exec(nvbench::exec_tag::sync, ...)` pattern for kernel execution.
- Provide a CLI or scriptable entry point (via NVBench's built-in CLI or a custom wrapper) that supports:
  - device selection (`-d`)
  - axis definition for:
    - square suite: `N` (and optional Batch Size)
    - non-square suite: `M`, `N`, `K` (and optional Batch Size)
  - dtype-pair selection: `(int8, int8)`, `(int8, fp16)`, `(fp16, int8)`, `(fp16, fp16)`, `(fp32, fp32)`, `(fp16, fp32)`, `(fp32, fp16)`
  - case selection:
    - square suite: `AB`, `ATB_view`, `ABT_view`, `ATB_copyA`, `ABT_copyB`
    - non-square suite: `ATB_view`, `ATB_copyA`, `ABT_view`, `ABT_copyB`
- **Reporting:** Use NVBench's `--json` or `--csv` output features to generate machine-readable results.
- GEMM implementation MUST call cuBLASLt (e.g., `cublasLtMatmul`) and MUST NOT use a hand-written GEMM kernel.
- The benchmark MUST clearly separate and label:
  - GEMM-with-transpose-flags (e.g., `op(A)=T` / `op(B)=T` in cuBLASLt) which should not require data movement
  - explicit transpose materialization + GEMM (to measure copy cost and cache effects) - implement as separate benchmark cases.
- Implement the benchmark cases (square and non-square suites) in a way that:
  - guarantees the intended operand shapes for each case
  - guarantees “no materialization” cases are implemented via transpose flags (not an explicit transpose copy)
  - avoids accidental extra allocations or copies in the timed region
- Validate correctness minimally:
  - MUST validate GPU results against a CPU reference computed with Eigen for every configuration/variant (sanity check).
    - The CPU reference MUST be computed outside the timed region (or in a separate "validation" pass).
    - The benchmark MUST support `--verify={full,sample,off}` (or equivalent):
      - `full`: compute full `C_ref = A @ B` on CPU via Eigen and compare elementwise.
      - `sample`: for configurations where full CPU GEMM is impractically slow, compare a deterministic subset (e.g., selected rows/cols) computed via Eigen.
      - `off`: disabled (for debugging only; not allowed for official benchmark runs).
    - Default behavior:
      - MUST run `full` verification for the safety control set (dims <= 1000).
      - MAY fall back to `sample` for larger configurations, but MUST still perform some Eigen-based check per configuration.
  - Tolerances:
    - For `fp32×fp32` (and TF32): compare in `fp32` with tight tolerances (e.g., `rtol~1e-4`, `atol~1e-4`), documenting exact values used.
    - For `fp16`/`bf16` and mixed-float cases: allow relaxed tolerances and/or compare against an `fp32`-accumulated CPU reference.
    - For `int8×int8`: define and document the accumulation/output type (often int32 accumulate); use exact comparison when the API semantics permit.

### Non-functional requirements
- Reproducibility:
  - fixed random seeds for input generation
  - report environment metadata (GPU model, driver, CUDA version, framework version)
  - ability to rerun with identical parameters and get stable results (within expected variance)
- Measurement quality:
  - **Relies on NVBench** for:
    - Warmup iterations.
    - Automatic scaling of iteration counts based on stability.
    - Accurate GPU timing using CUDA Events.
  - isolate the measured region from allocations where possible (pre-allocate inputs/outputs)
- Documentation:
  - describe how transpose views differ from materialized transposes in the chosen framework
  - explain the chosen shape/dtype matrix and how to extend it

## Benchmark matrix
The benchmark must cover at least:

- Shapes:
  - “square-ish”: `M=N=K` in {512, 1024, 2048, 4096}
  - “tall-skinny / wide”: include at least two non-square patterns (e.g., `M=4096,K=1024,N=1024` and `M=1024,K=4096,N=1024`)
  - one “LLM-like” pattern (e.g., large K with moderate M/N), to be refined per target workloads
- Dtypes:
  - fp16 and/or bf16 (tensor core path)
  - fp32 (with a toggle for tf32 if relevant)
  - dtype pairs to test (if supported by the backend/framework):
    - `matmul(int8, int8)` (typically int32 accumulate; output type depends on API)
    - `matmul(int8, fp16)` and `matmul(fp16, int8)`
    - `matmul(fp16, fp16)`
    - `matmul(fp32, fp32)` (with optional TF32 toggle)
    - `matmul(fp16, fp32)` and `matmul(fp32, fp16)` (may imply casting or a specific mixed-type path; MUST record what actually happened)
- Layout modes (minimum):
  - contiguous (baseline)
  - transpose-without-materialization (framework transpose view and/or cuBLASLt transpose flag)
  - materialized transpose (explicit copy) — optional but strongly recommended to separate “layout penalty” vs “copy penalty”

### Suggested initial sweep (A100-SXM4, L2 = 40 MiB)
These concrete configurations are chosen to highlight two regimes:
1) **Cache-resident**: `A + B + C (+ one extra transpose materialization)` fits in L2.
2) **Cache-spill**: `A + B + C` may fit in L2, but `+ materialized transpose copy` does **not**, so any implicit/explicit transpose duplication is impossible for L2 to hold.

Implementation note (what we compare, and how we keep FLOPs comparable):

- **Square suite (`N×N`)** is the strict comparison: same `A` and `B`, same output shape, same FLOPs.
- **Non-square suite** intentionally omits `A @ B` (because it is not defined for `A[K,M]` and `B[K,N]`) and instead measures:
  - transpose-A (`A_atb.T @ B_atb`) and transpose-B (`A_abt @ B_abt.T`) on the same `(M,N,K)` so all measured GEMMs have identical FLOP count `2*M*N*K`.
  - view/flag (“no materialization”) vs explicit `copy(·)` (“materialize then GEMM”) within each transpose direction.

#### Cache-resident (L2-fit, even if a transpose is materialized)
- fp16/bf16 square: `M=N=K` in {1024, 1536, 2048}
- fp32 square: `M=N=K` in {768, 1024, 1280, 1536}
- fp16/bf16 aspect-ratio probes:
  - tall: `(M,N,K) = (4096, 1024, 1024)`
  - wide: `(M,N,K) = (1024, 4096, 1024)`
  - large-K: `(M,N,K) = (1024, 1024, 4096)`

#### Cache-spill (transpose duplication cannot fit in L2)
- fp16/bf16 square boundary: `(M,N,K) = (2304, 2304, 2304)`
- fp16/bf16 non-square (separate “copy A” vs “copy B” stress):
  - copy-A stress: `(M,N,K) = (3072, 2048, 2048)`
  - copy-B stress: `(M,N,K) = (2048, 3072, 2048)`
  - definitely spill: `(M,N,K) = (8192, 1024, 1024)`
- fp32 square boundary: `(M,N,K) = (1664, 1664, 1664)`
- fp32 non-square:
  - copy-A stress: `(M,N,K) = (2304, 1536, 1536)`
  - copy-B stress: `(M,N,K) = (1536, 2304, 1536)`

#### Safety control set (dims <= 1000; resilient to “someone else used cache”)
Also include a small-dimension set to reduce sensitivity to external cache pressure and make comparisons safer.

- Square-ish: `M=N=K` in {512, 768, 896, 960, 992, 1000}
  - Note: for int8 Tensor Core-friendly runs, prefer multiples of 32 (e.g., 992) over “awkward” sizes like 1000.
- Non-square (aspect-ratio probes, all dims <= 1000):
  - tall: `(M,N,K) = (992, 256, 256)`
  - wide: `(M,N,K) = (256, 992, 256)`
  - large-K: `(M,N,K) = (256, 256, 992)`
  - mixed: `(M,N,K) = (960, 320, 640)`

## Measurement methodology
Timing MUST use **NVBench**.

- Two versions MUST be run for every configuration:
  - **Timing run (no profiler)**: normal execution via NVBench CLI (e.g., `./bench --json=results.json`).
  - **Profiling run (with `ncu`)**: executed under Nsight Compute to capture kernel behavior.

- Warmup & Timing:
  - Handled automatically by NVBench.
  - Configuration of min_time / min_iterations should be set to ensure stability (e.g., `--min_time=1s`).
- Synchronization:
  - ensure inputs are on device before timing (H2D copies excluded)
  - NVBench handles synchronization via `state.exec(nvbench::exec_tag::sync, ...)`.
- Allocation discipline:
  - pre-create tensors/buffers outside the timed loop (in the benchmark setup phase).
  - MUST NOT include H2D/D2H transfers (`cudaMemcpy*`) inside the timed region
  - avoid measuring transpose materialization unless explicitly benchmarking it as a separate step
  - avoid timing cuBLASLt handle creation/workspace allocation; create handles once per run and reuse

## Kernel profiling (Nsight Compute / `ncu`)
In addition to timing, **all configurations in the sweep MUST be profiled** with Nsight Compute to capture what kernels actually run and how they behave.

Requirements:
- Run `ncu` as a **separate invocation per configuration** (to keep reports attributable and avoid cross-talk).
- Profile the same `cuda13` environment execution (e.g., `ncu ... -- pixi run -e cuda13 -- <benchmark-cli> ...`).
- Apply the same warmup discipline:
  - NVBench's default behavior is usually sufficient, but for profiling you may want to restrict iterations (e.g., `--min_iterations=1`).
- Timing under `ncu`:
  - MUST record kernel duration from `ncu` output (e.g., kernel duration / `gpu__time_duration.sum`), and treat it as the authoritative “with-profiler” timing.
  - MUST NOT compare or report application wall-clock time while profiling as performance.
- Capture at minimum:
  - kernel name(s) and launch configuration
  - memory traffic indicators (HBM/DRAM bytes, L2 bytes/hit rate where available)
  - compute utilization signals (SM throughput, tensor core usage/pipe activity when applicable)
- Save artifacts per configuration:
  - `*.ncu-rep` report file
  - an exported summary (CSV or text) sufficient to diff view vs copy cases (and, for square suite, compare `AB` vs transpose cases)

Implementation guidance (not prescriptive):
- Use NVTX ranges (if supported by the framework) to bracket the measured matmul region so `ncu` can filter/attribute kernels cleanly.
- Use `ncu` kernel filters (e.g., by regex) to focus on GEMM kernels and exclude unrelated runtime kernels when possible.

## Outputs and reporting
Minimum outputs:
- A table (Markdown and/or CSV) showing for each configuration:
  - per-case time (ms) and TFLOP/s
  - in square suite: slowdown factors vs `AB`
  - in non-square suite: materialization overhead factors (`ATB_copyA / ATB_view`, `ABT_copyB / ABT_view`)
- A short analysis section per dtype explaining observed patterns:
  - when non-contiguous views cause fallback/slowdown
  - when explicit `.contiguous()` helps (and the cost of making it contiguous)
- For each configuration, references to the corresponding `ncu` report artifacts used to interpret the behavior.

### Final comparison table (stakeholder-facing)
The final report MUST include a stakeholder-facing comparison table (in Markdown) derived from the structured export.

**Key principle:** show results **per configuration** (shape + dtype pair + suite) with **all relevant cases on one row**, plus explicit ratios.

**FLOP consistency requirement:** within a single table row, all compared cases MUST have the same theoretical `flop_count`. If a set of cases would have different FLOP counts, they MUST be split into separate rows (or excluded from direct ratio comparisons).

#### Table A: Square suite summary (`A[N,N]`, `B[N,N]`)
One row per `(N, dtype_pair, layout_mode, math_mode)`:

| suite | N | dtype_pair | flop_count | timed_ms_AB | timed_ms_ATB_view | timed_ms_ABT_view | timed_ms_ATB_copyA | timed_ms_ABT_copyB | slow_ATB_view_vs_AB | slow_ABT_view_vs_AB | over_ATB_copyA_vs_view | over_ABT_copyB_vs_view | verify |
|------:|--:|------------|----------:|------------:|------------------:|------------------:|-------------------:|-------------------:|--------------------:|--------------------:|-----------------------:|-----------------------:|--------|
| square |  |            |           |             |                   |                   |                    |                    |                     |                     |                        |                        |        |

Definitions:
- `flop_count`: theoretical GEMM FLOP count for the case (`2 * N * N * N`).
- `timed_ms_*`: average per-matmul time from the **no-profiler** run.
- `slow_*_vs_AB`: `timed_ms_case / timed_ms_AB`.
- `over_*_vs_view`: `timed_ms_copy / timed_ms_view` (materialization overhead factor).
- `verify`: pass/fail (+ optional error summary).

#### Table B: Non-square suite summary (FLOP-matched; transpose-A and transpose-B)
One row per `(M,N,K, dtype_pair, layout_mode, math_mode)`:

| suite | M | N | K | dtype_pair | flop_count | timed_ms_ATB_view | timed_ms_ATB_copyA | over_ATB_copyA_vs_view | timed_ms_ABT_view | timed_ms_ABT_copyB | over_ABT_copyB_vs_view | verify |
|------:|--:|--:|--:|------------|----------:|------------------:|-------------------:|-----------------------:|------------------:|-------------------:|-----------------------:|--------|
| non_square |  |  |  |           |           |                   |                    |                        |                   |                    |                        |        |

Notes:
- The non-square table intentionally does not include `AB` because the non-square suite is defined around `A_atb[K,M] @ B_atb[K,N]` (transpose-A) and `A_abt[M,K] @ B_abt[N,K]` (transpose-B).
- `flop_count` is `2 * M * N * K` for all non-square cases in this table.
- The report SHOULD also include TFLOP/s columns (computed using `flop_count / time_seconds`) for each case when comparing across shapes.

## Acceptance criteria
- Running the benchmark on a CUDA machine produces:
  - results for all required cases across the required shape/dtype matrix (square and non-square suites)
  - exported CSV/JSONL with metadata and timing stats
  - a summary report that clearly highlights when/why transpose-without-materialization vs transpose-with-materialization differs, and when transpose changes performance in the square suite
  - `ncu` reports for all configurations, enabling kernel-level explanation of observed slowdowns
- Results are stable enough that reruns differ by no more than ~5–10% for the same configuration (excluding known noisy environments).

## Risks and notes
- Framework behavior varies:
  - Some frameworks may transparently call a GEMM with transpose flags (fast) *or* may materialize to contiguous (copy) depending on strides and backend.
- Kernel selection can change with driver/CUDA/framework versions; metadata capture is mandatory.
- For small matrices, launch overhead dominates; prioritize sufficiently large sizes for meaningful GEMM throughput.

## References
- **NVBench**: https://github.com/NVIDIA/nvbench (Source code: `extern/orphan/nvbench`)
- **cuBLASLt Documentation**: https://docs.nvidia.com/cuda/cublas/index.html#cublaslt-api-reference
- **Nsight Compute CLI (ncu)**: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
- **Eigen Documentation**: https://eigen.tuxfamily.org/dox/
- **Google Benchmark**: https://github.com/google/benchmark

## Appendix: Configuration Checklist

This list consolidates the "Benchmark Matrix" and "Suggested Initial Sweep" for the A100-SXM4 (40MB L2).

### 1. Safety Control Set (Small, Resilient)
*Goal: Sanity check & low-noise baseline.*
- **Shapes (M=N=K):** 512, 768, 896, 960, 992, 1000
- **Shapes (Non-Square):** 
  - (992, 256, 256)
  - (256, 992, 256)
  - (256, 256, 992)
  - (960, 320, 640)
- **Cases:**
  - Square suite: `AB`, `ATB_view`, `ABT_view`, `ATB_copyA`, `ABT_copyB`
  - Non-square suite: `ATB_view`, `ATB_copyA`, `ABT_view`, `ABT_copyB`

### 2. Cache-Resident (Fits in L2)
*Goal: High throughput, minimal DRAM traffic.*
- **FP16/BF16 Square:** 1024, 1536, 2048
- **FP32 Square:** 768, 1024, 1280, 1536
- **FP16/BF16 Non-Square:**
  - Tall: (4096, 1024, 1024)
  - Wide: (1024, 4096, 1024)
  - Large-K: (1024, 1024, 4096)

### 3. Cache-Spill (Data fits, Extra Transpose Copy doesn't)
*Goal: Expose cost of materialization.*
- **FP16/BF16 Square:** 2304
- **FP32 Square:** 1664
- **FP16/BF16 Non-Square:**
  - Copy-A Stress: (3072, 2048, 2048)
  - Copy-B Stress: (2048, 3072, 2048)
- **FP32 Non-Square:**
  - Copy-A Stress: (2304, 1536, 1536)
  - Copy-B Stress: (1536, 2304, 1536)

### 4. L2 Capacity Spill (Definitely Spills)
*Goal: DRAM bandwidth bound.*
- **FP16/BF16:** (8192, 1024, 1024)

### 5. Dtypes & Layouts (Apply to above)
- **Dtypes:**
  - `fp16` (Main Tensor Core test)
  - `bf16` (Alternative Tensor Core test)
  - `fp32` (Simt/TF32 path)
  - `int8` (Optional/Advanced: `int8` inputs, `int32` accum)
- **Layouts:**
  - Square suite operands: `A[N,N]`, `B[N,N]`
  - Non-square transpose-A operands: `A_atb[K,M]`, `B_atb[K,N]`
  - Non-square transpose-B operands: `A_abt[M,K]`, `B_abt[N,K]`
  - Transpose Views: `A.T`, `B.T`
  - Materialized: `A_new = Copy(A.T)` (Separate Benchmark Case)
