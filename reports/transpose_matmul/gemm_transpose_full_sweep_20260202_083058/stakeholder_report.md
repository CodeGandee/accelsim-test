# GEMM Transpose Sweep — Stakeholder Report

- Run ID: `gemm_transpose_full_sweep_20260202_083058`
- Artifacts: `reports/transpose_matmul/gemm_transpose_full_sweep_20260202_083058/`
- Git: `002-gemm-transpose-bench` @ `27511017a4fd709d3c22f9fe8b7897121276a667` (dirty: `True`)
- GPU: `NVIDIA A100-SXM4-80GB` (sm_800), driver `580.95.05`
- CUDA toolkit: `13.0` (nvcc `V13.0.88`)
- Pixi env: `cuda13`
- NVBench: `0.1.0` (source: `/data1/huangzhe/code/accelsim-test/extern/orphan/nvbench`)
- NVBench settings: `{'max_noise_pct': 0.3, 'min_samples': 20, 'min_time_s': 0.5, 'stopping_criterion': 'stdrel'}`
- Sweep size: 525 records (`square`: 325, `nonsquare_atb`: 100, `nonsquare_abt`: 100)
- Measurement scope: **GEMM-only** timing; transpose materialization for `*_copy*` cases is intentionally **outside** the timed region.

## Executive Summary

- Full sweep completed successfully on A100 (`results.json.run.status=pass`; verification passed for all 525 records).
- Transpose-as-view vs transpose-as-copy timing deltas are driven by **cuBLASLt algorithm/config selection** and operand layout (not by transpose cost, which is excluded from timing for `*_copy*`).
- **Int8 transpose-B is the standout:** `ABT_view` consistently selects cuBLASLt `algo_id=21` while `ABT_copyB` selects `algo_id=0`, producing large gaps (2.06×–4.89× faster for `ABT_view` across all tested non-square shapes).
- Interpretation caveat: if your real workload includes an explicit transpose + GEMM, this report does **not** include the transpose cost; it isolates GEMM kernel/layout effects.

## Key Results (curated)

### Square Suite snapshots (N=4096, 2048, 1024, 512)

From `report.md` (times in ms; `algo_id` is cuBLASLt-selected for that specific case):

#### N=4096

| dtype_pair | A@B(ms) (algo_id) | A.T@B(ms) (algo_id) | A@B.T(ms) (algo_id) | copy(A.T)@B(ms) (algo_id) | A@copy(B.T)(ms) (algo_id) | note |
|---|---:|---:|---:|---:|---:|---|
| `bf16,bf16->bf16 (fp32,default)` | 0.566 (6) | 0.562 (6) | 0.569 (6) | 0.564 (6) | 0.564 (6) | similar across transpose modes |
| `fp16,fp16->fp16 (fp32,default)` | 0.578 (6) | 0.573 (6) | 0.619 (21) | 0.581 (6) | 0.582 (6) | transpose-B view flips algorithm here |
| `fp32,fp32->fp32 (fp32,default)` | 7.238 (0) | 7.219 (20) | 7.299 (0) | 7.236 (0) | 7.236 (0) | expected slower fp32 |
| `fp32,fp32->fp32 (tf32,tf32)` | 1.163 (21) | 1.176 (21) | 1.147 (21) | 1.159 (21) | 1.160 (21) | TF32 is a large speedup vs fp32 |
| `int8,int8->int32 (int32,default)` | 1.924 (0) | 1.935 (0) | 0.357 (21) | 1.923 (0) | 1.924 (0) | **~5.4× faster** with transpose-B view |

#### N=2048

| dtype_pair | A@B(ms) (algo_id) | A.T@B(ms) (algo_id) | A@B.T(ms) (algo_id) | copy(A.T)@B(ms) (algo_id) | A@copy(B.T)(ms) (algo_id) |
|---|---:|---:|---:|---:|---:|
| `bf16,bf16->bf16 (fp32,default)` | 0.112 (6) | 0.111 (6) | 0.113 (6) | 0.112 (6) | 0.112 (6) |
| `fp16,fp16->fp16 (fp32,default)` | 0.077 (34) | 0.082 (34) | 0.079 (34) | 0.077 (34) | 0.077 (34) |
| `fp32,fp32->fp32 (fp32,default)` | 0.978 (0) | 0.953 (1) | 0.976 (0) | 0.978 (0) | 0.978 (0) |
| `fp32,fp32->fp32 (tf32,tf32)` | 0.179 (21) | 0.181 (21) | 0.180 (21) | 0.179 (21) | 0.180 (21) |
| `int8,int8->int32 (int32,default)` | 0.300 (0) | 0.302 (0) | 0.061 (21) | 0.299 (0) | 0.299 (0) |

#### N=1024

| dtype_pair | A@B(ms) (algo_id) | A.T@B(ms) (algo_id) | A@B.T(ms) (algo_id) | copy(A.T)@B(ms) (algo_id) | A@copy(B.T)(ms) (algo_id) |
|---|---:|---:|---:|---:|---:|
| `bf16,bf16->bf16 (fp32,default)` | 0.023 (6) | 0.023 (6) | 0.023 (6) | 0.023 (6) | 0.023 (6) |
| `fp16,fp16->fp16 (fp32,default)` | 0.023 (34) | 0.023 (34) | 0.024 (6) | 0.023 (34) | 0.023 (34) |
| `fp32,fp32->fp32 (fp32,default)` | 0.135 (0) | 0.135 (20) | 0.137 (0) | 0.135 (0) | 0.135 (0) |
| `fp32,fp32->fp32 (tf32,tf32)` | 0.036 (21) | 0.037 (21) | 0.037 (21) | 0.036 (21) | 0.036 (21) |
| `int8,int8->int32 (int32,default)` | 0.069 (0) | 0.069 (0) | 0.018 (21) | 0.069 (0) | 0.069 (0) |

#### N=512

| dtype_pair | A@B(ms) (algo_id) | A.T@B(ms) (algo_id) | A@B.T(ms) (algo_id) | copy(A.T)@B(ms) (algo_id) | A@copy(B.T)(ms) (algo_id) |
|---|---:|---:|---:|---:|---:|
| `bf16,bf16->bf16 (fp32,default)` | 0.011 (31) | 0.012 (31) | 0.011 (31) | 0.011 (31) | 0.011 (31) |
| `fp16,fp16->fp16 (fp32,default)` | 0.011 (6) | 0.011 (6) | 0.011 (6) | 0.011 (6) | 0.011 (6) |
| `fp32,fp32->fp32 (fp32,default)` | 0.033 (1) | 0.032 (1) | 0.036 (1) | 0.033 (1) | 0.033 (1) |
| `fp32,fp32->fp32 (tf32,tf32)` | 0.015 (21) | 0.015 (21) | 0.015 (21) | 0.015 (21) | 0.015 (21) |
| `int8,int8->int32 (int32,default)` | 0.024 (0) | 0.028 (0) | 0.010 (21) | 0.024 (0) | 0.023 (0) |

### Non-square Suite (FLOP-matched; view vs copy)

Non-square timing is split into two suites with **different storage shapes** (so the transpose-view expressions are well-defined for non-square matrices):

- `nonsquare_atb`: store `A` as `(K,M)` and `B` as `(K,N)`, then measure `A.T@B` (view) and `copy(A.T)@B` (copy).
- `nonsquare_abt`: store `A` as `(M,K)` and `B` as `(N,K)`, then measure `A@B.T` (view) and `A@copy(B.T)` (copy).

For copy cases, the transposed operand is materialized **outside** the timed region, so the copied buffer’s `.shape` matches what appears in the expression (e.g. `copy(B.T).shape=(K,N)`).

All times below are **GEMM-only** (transpose materialization for `*_copy*` is intentionally outside timing).

| suite | expr | A.shape | B.shape | M | N | K | dtype_pair | time(ms) (algo_id) | note |
|---|---|---|---|---:|---:|---:|---|---:|---|
| `nonsquare_atb` | `A.T@B` | (256,256) | (256,992) | 256 | 992 | 256 | `int8,int8->int32 (int32,default)` | 0.020327 (0) |  |
| `nonsquare_atb` | `copy(A.T)@B` | (256,256) | (256,992) | 256 | 992 | 256 | `int8,int8->int32 (int32,default)` | 0.022409 (0) |  |
| `nonsquare_abt` | `A@B.T` | (256,256) | (992,256) | 256 | 992 | 256 | `int8,int8->int32 (int32,default)` | 0.009295 (21) |  |
| `nonsquare_abt` | `A@copy(B.T)` | (256,256) | (256,992) | 256 | 992 | 256 | `int8,int8->int32 (int32,default)` | 0.023144 (0) | transpose-B copy/view ≈ 2.49× |
| `nonsquare_atb` | `A.T@B` | (4096,1024) | (4096,1024) | 1024 | 1024 | 4096 | `int8,int8->int32 (int32,default)` | 0.173452 (0) |  |
| `nonsquare_atb` | `copy(A.T)@B` | (1024,4096) | (4096,1024) | 1024 | 1024 | 4096 | `int8,int8->int32 (int32,default)` | 0.172443 (0) |  |
| `nonsquare_abt` | `A@B.T` | (1024,4096) | (1024,4096) | 1024 | 1024 | 4096 | `int8,int8->int32 (int32,default)` | 0.040784 (21) |  |
| `nonsquare_abt` | `A@copy(B.T)` | (1024,4096) | (4096,1024) | 1024 | 1024 | 4096 | `int8,int8->int32 (int32,default)` | 0.171977 (0) | transpose-B copy/view ≈ 4.22× |
| `nonsquare_atb` | `A.T@B` | (2048,3072) | (2048,2048) | 3072 | 2048 | 2048 | `int8,int8->int32 (int32,default)` | 0.398203 (0) |  |
| `nonsquare_atb` | `copy(A.T)@B` | (3072,2048) | (2048,2048) | 3072 | 2048 | 2048 | `int8,int8->int32 (int32,default)` | 0.393961 (0) |  |
| `nonsquare_abt` | `A@B.T` | (2048,2048) | (3072,2048) | 2048 | 3072 | 2048 | `int8,int8->int32 (int32,default)` | 0.080541 (21) |  |
| `nonsquare_abt` | `A@copy(B.T)` | (2048,2048) | (2048,3072) | 2048 | 3072 | 2048 | `int8,int8->int32 (int32,default)` | 0.393818 (0) | transpose-B copy/view ≈ 4.89× |
| `nonsquare_atb` | `A.T@B` | (640,960) | (640,320) | 960 | 320 | 640 | `fp32,fp32->fp32 (tf32,tf32)` | 0.022024 (21) | same algo_id, different config |
| `nonsquare_atb` | `copy(A.T)@B` | (960,640) | (640,320) | 960 | 320 | 640 | `fp32,fp32->fp32 (tf32,tf32)` | 0.016271 (21) | same algo_id, different config |

Notes:
- For int8 transpose-B (`A@B.T` vs `A@copy(B.T)`), the large gaps correlate with **different cuBLASLt algos** (`algo_id=21` vs `algo_id=0`).
- “Same `algo_id`” does **not** imply the same underlying configuration. Example at `M=960, N=320, K=640`, TF32 (`nonsquare_atb`): `A.T@B` and `copy(A.T)@B` both use `algo_id=21` but differ in `splitk_num`, `stages_id`, and `required_workspace_bytes` (see `all_timings.md` / `results.json`).

## Analysis

### What is actually being compared?

Each suite/case measures GEMM only:

- `*_view` cases use cuBLASLt transpose flags (`trans_a`/`trans_b`) and pass the original device buffer.
- `*_copy*` cases materialize the transpose via a CUDA kernel **before timing**, then run GEMM with `trans_* = N`.

So, for example, “`ABT_view` vs `ABT_copyB`” is primarily comparing **operand layout + cuBLASLt algo/config selection**, *not* the cost of transposing B.

### Why can `*_copy*` be faster (or slower)?

cuBLASLt’s heuristic picks different kernels/configs depending on:

- operand layouts (leading dimensions, order),
- transpose flags,
- shape (M/N/K),
- datatype/compute mode,
- allowed workspace.

Even when `algo_id` matches, the actual configuration can differ (e.g., `splitk_num`, `stages_id`, workspace requirements), which can change performance materially (see the TF32 example above).

### Int8 transpose-B: why is `ABT_view` much faster than `ABT_copyB`?

In this sweep, `ABT_view` selected `algo_id=21` for all int8 non-square shapes, while `ABT_copyB` selected `algo_id=0`. The resulting gap is large (2.06×–4.89×).

This is not specific to `M=256, N=992, K=256`; it occurs across all tested non-square shapes. The key correlates in our artifacts are:

- `algo_id` changes (21 vs 0),
- plus other algo fields (`tile_id`, `splitk_num`, `stages_id`, workspace) that can further shift results.

The most plausible root cause is that the **effective memory layout** presented to cuBLASLt differs between `trans_b=T` (view) and “pre-transposed contiguous buffer + `trans_b=N`” (copy), so the heuristic chooses different kernels.

### Can we pin the cuBLASLt algorithm by hand?

Yes in principle: cuBLASLt supports providing an explicit `cublasLtMatmulAlgo_t` to `cublasLtMatmul(...)`. Our current implementation still uses heuristics to select an algo per record, then calls `cublasLtMatmul` with the selected algo.

What is missing today is an **override path** (CLI/env) to bypass heuristics and force a specific algorithm/config (not just `algo_id`, but the full config). If we add that, we can:

- reduce run-to-run variance due to heuristic changes,
- make “view vs copy” comparisons isolate layout effects under the *same* kernel.

## Correctness & Verification (critical path)

### Timed region and kernel launch

The single most important line that launches the matmul is:

`plan.Run(launch.get_stream(), a_used, b_used, c_dev, &alpha, &beta);` (`cpp/src/gemm_transpose_bench.cu:958`)

That ultimately dispatches cuBLASLt:

`cublasLtMatmul(..., &m_algo, ..., stream)` (`cpp/src/cublaslt_gemm.cu:149`)

### View vs copy semantics (what buffers are passed into cuBLASLt)

Case selection happens via transpose flags and optional materialization:

- View cases set `trans_a`/`trans_b` (e.g., `ABT_view` sets `trans_b=CUBLAS_OP_T`) (`cpp/src/gemm_transpose_bench.cu:709`).
- Copy cases run a transpose CUDA kernel prior to timing:
  - `transpose_kernel<<<...>>>(...)` (`cpp/src/gemm_transpose_bench.cu:57`)
  - then set `a_used` / `b_used` to the copied buffer (`cpp/src/gemm_transpose_bench.cu:714`, `cpp/src/gemm_transpose_bench.cu:765`).

### Algorithm recording

We persist the selected cuBLASLt algorithm/config into NVBench summaries:

`state.add_summary(\"accelsim/cublaslt/algo\")` (`cpp/src/gemm_transpose_bench.cu:805`)

The Python exporter lifts this into `results.json` under `record.cublaslt.algo`.

### Verification policy

- One **untimed** GEMM run is performed for correctness checks (`cpp/src/gemm_transpose_bench.cu:820`).
- Full verification when `max(M,N,K) <= 1000`; otherwise sampled indices (`cpp/src/gemm_transpose_bench.cu:836`).
- For fp16/bf16/tf32, the reference path quantizes inputs to match compute precision (avoids false mismatches):
  - fp16/bf16 quantization via round-trip conversion (`cpp/src/gemm_transpose_bench.cu:164`)
  - TF32 quantization via mantissa truncation (`cpp/src/gemm_transpose_bench.cu:174`)
- Verification summaries are exported into `results.json` as:
  - `record.verification.{status,mode,max_abs_error,max_rel_error,details}`.

## Reproduction

- Build: `pixi run -e cuda13 gemm-transpose-build`
- Sweep (experiments): `pixi run -e cuda13 gemm-transpose sweep --out-dir /data1/huangzhe/code/accelsim-test/reports/transpose_matmul/gemm_transpose_full_sweep_20260202_083058`
- Report (generated Markdown): `pixi run -e cuda13 gemm-transpose report --out-dir /data1/huangzhe/code/accelsim-test/reports/transpose_matmul/gemm_transpose_full_sweep_20260202_083058`

## Appendix

### Artifact map

- Generated summary: `report.md`
- Full timing table: `all_timings.md`
- Normalized export: `results.json`
- Raw NVBench JSON: `raw/`

### Column definitions (generated artifacts)

#### `report.md`

- `suite`: `square` or `non_square` (the latter aggregates both non-square suites into one FLOP-matched row).
- `N`: square dimension (`M=N=K=N`).
- `M,N,K`: GEMM dimensions for `C[M,N] = A[M,K] @ B[K,N]`.
- `dtype_pair`: `A,B->C (compute,math_mode)`.
- `A@B(ms)`, `A.T@B(ms)`, `A@B.T(ms)`, `copy(A.T)@B(ms)`, `A@copy(B.T)(ms)`: mean GPU time in ms (NVBench cold GPU mean; GEMM-only).
- `...(algo_id)`: cuBLASLt algorithm ID selected for that case (heuristic-selected; see `results.json`).
- `verify`: `pass` iff all records contributing to the row passed verification.

#### `all_timings.md`

- `suite`: `square`, `nonsquare_atb`, or `nonsquare_abt`.
- `case`: one of `AB`, `ATB_view`, `ABT_view`, `ATB_copyA`, `ABT_copyB` (suite-dependent).
- `M,N,K`: GEMM dimensions.
- `dtype_pair`: `A,B->C (compute,math_mode)`.
- `time(ms)`: mean GPU time in ms (`nv/cold/time/gpu/mean`).
- `samples`: NVBench cold sample size (`nv/cold/sample_size`) used for the reported mean.
- `verify`: `pass|fail` from the benchmark’s untimed correctness check.
- `algo_id`, `tile_id`, `splitk_num`, `stages_id`: selected cuBLASLt algo/config (subset; see full `record.cublaslt.algo` in `results.json`).

### NVBench flags used (and what they mean)

These are recorded in `results.json.run.settings.nvbench`:

- `--stopping-criterion stdrel`: stop when relative standard deviation converges (`stdev/mean`), after minimum samples/time.
- `--min-time 0.5`: require at least 0.5s of accumulated GPU time before allowing the run to stop (`extern/orphan/nvbench/nvbench/detail/stdrel_criterion.cxx:58`).
- `--min-samples 20`: require at least 20 samples before checking criteria (NVBench CLI help: `cpp/build/Release/gemm_transpose_bench --help`).
- `--max-noise 0.3`: target 0.3% relative stdev; NVBench may still stop if noise stabilizes above the target (see `extern/orphan/nvbench/nvbench/detail/stdrel_criterion.cxx:69`).
- `--devices 0`: run on GPU device 0.

### NVBench metrics (what it outputs vs what we consume)

- The raw NVBench JSON under `raw/` contains summary statistics for timings (GPU and CPU), including mean and other aggregates + noise, as well as the full axis values used to produce each record.
- Our normalized export (`results.json`) currently consumes:
  - `nv/cold/time/gpu/mean` → `record.timing.gpu_time_ms`
  - `nv/cold/time/cpu/mean` → `record.timing.cpu_time_ms` (if present)
  - `nv/cold/sample_size` → `record.timing.samples`
  - custom summaries emitted by the benchmark:
    - `accelsim/verification/*`
    - `accelsim/cublaslt/algo`

### Profiling note

This sweep did **not** run Nsight Compute (`ncu`). Profiling is a separate stage (`gemm-transpose profile`) and was intentionally omitted for this experiment.
