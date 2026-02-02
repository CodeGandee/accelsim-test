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

### Square Suite (N=4096)

From `report.md` (times in ms; `algo_id` is cuBLASLt-selected for that specific case):

| dtype_pair | A@B(ms) (algo_id) | A@B.T(ms) (algo_id) | note |
|---|---:|---:|---|
| `bf16,bf16->bf16 (fp32,default)` | 0.566 (6) | 0.569 (6) | similar across transpose modes |
| `fp16,fp16->fp16 (fp32,default)` | 0.578 (6) | 0.619 (21) | transpose-B view flips algorithm here |
| `fp32,fp32->fp32 (fp32,default)` | 7.238 (0) | 7.299 (0) | expected slower fp32 |
| `fp32,fp32->fp32 (tf32,tf32)` | 1.163 (21) | 1.147 (21) | TF32 is a large speedup vs fp32 |
| `int8,int8->int32 (int32,default)` | 1.924 (0) | 0.357 (21) | **~5.4× faster** with transpose-B view |

### Non-square Suite: int8 transpose-B (`nonsquare_abt`)

This compares `ABT_view` (A@B.T) vs `ABT_copyB` (A@copy(B.T)). Both rows time **only** the GEMM; the `copy(B.T)` materialization is outside timing.

| MxNxK | ABT_view: A@B.T(ms) (algo_id) | ABT_copyB: A@copy(B.T)(ms) (algo_id) | speedup (copy/view) |
|---|---:|---:|---:|
| 256×992×256 | 0.009295 (21) | 0.023144 (0) | 2.49× |
| 1024×1024×4096 | 0.040784 (21) | 0.171977 (0) | 4.22× |
| 2048×3072×2048 | 0.080541 (21) | 0.393818 (0) | 4.89× |

Across *all* tested non-square shapes for int8, `ABT_view` was faster than `ABT_copyB` by **2.06×–4.89×** in this run.

### Non-square Suite: algorithm config can change even when `algo_id` is the same

Example: `fp32,fp32->fp32 (tf32,tf32)` at `M×N×K = 960×320×640` in `nonsquare_atb`:

| case | time(ms) | algo_id | splitk_num | stages_id | required_workspace_bytes |
|---|---:|---:|---:|---:|---:|
| `ATB_view` | 0.022024 | 21 | 4 | 6 | 4915200 |
| `ATB_copyA` | 0.016271 | 21 | 1 | 12 | 0 |

So “same `algo_id`” does **not** guarantee the same underlying configuration; use the full `record.cublaslt.algo.*` fields in `results.json`/`all_timings.md`.

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
