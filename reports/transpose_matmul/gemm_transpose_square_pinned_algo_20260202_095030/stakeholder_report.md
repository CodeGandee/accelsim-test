# GEMM Transpose Sweep — Stakeholder Report (Square, Pinned cuBLASLt Algorithms)

- Run ID: `gemm_transpose_square_pinned_algo_20260202_095030`
- Scope: **square suite only** (`M=N=K`, 13 shapes) × 5 dtypes × 5 cases = **325 records**
- Git: `002-gemm-transpose-bench` @ `27511017a4fd709d3c22f9fe8b7897121276a667` (dirty: `True`)
- GPU: `NVIDIA A100-SXM4-80GB` (sm_800), driver `580.95.05`
- CUDA toolkit: `unknown`
- Pixi env: `cuda13`
- NVBench: `0.1.0` (source: `/data1/huangzhe/code/accelsim-test/extern/orphan/nvbench`)
- NVBench settings: `{'max_noise_pct': 0.3, 'min_samples': 20, 'min_time_s': 0.5, 'stopping_criterion': 'stdrel'}`
- cuBLASLt algo selection: `pinned`
  - Algo-map: `algo_map_square.json` (sha256: `cde618d9dd0c5ec8d3dd31b90f717871a8dc9006f461d6f1d3722e3a29f2bb5c`)
  - Algo-map source run: `tmp/gemm_transpose_full_sweep_20260202_083058/results.json`

## Executive Summary

- This run **replays the square suite with pinned cuBLASLt algorithm configs** (per `(suite, case, dtype_key, shape)`), so we can compare transpose variants without re-running cuBLASLt heuristic selection.
- The largest effect remains `int8,int8->int32`: `A@B.T` (`ABT_view`) uses a different algorithm family (algo_id `21`/`23`) and is **much faster** than `A@B` (algo_id `0`) for all tested square sizes (e.g., **~5.4× faster at N=4096**).
- For `bf16` and `tf32`, the same algorithm is typically used across all cases at a given shape, and the timings are within a few percent.
- Run-to-run timing drift vs the heuristic-selected source run is small (median <0.3% across square records), suggesting the earlier findings are not explained by short-term benchmark noise.

## Key Results (curated)

### N=4096 summary (all dtypes)

| dtype_pair | A@B (ms) | algo | A.T@B (ms) | algo | A@B.T (ms) | algo |
|---|---:|---:|---:|---:|---:|---:|
| `bf16,bf16->bf16 (fp32,default)` | 0.564910 | 6 | 0.560488 | 6 | 0.566518 | 6 |
| `fp16,fp16->fp16 (fp32,default)` | 0.580794 | 6 | 0.572189 | 6 | 0.621058 | 21 |
| `fp32,fp32->fp32 (fp32,default)` | 7.246365 | 0 | 7.266098 | 20 | 7.305854 | 0 |
| `fp32,fp32->fp32 (tf32,tf32)` | 1.161301 | 21 | 1.177271 | 21 | 1.149671 | 21 |
| `int8,int8->int32 (int32,default)` | 1.922978 | 0 | 1.938666 | 0 | 0.356208 | 21 |

### `int8,int8->int32` across all square sizes (timings + pinned algo_id)

| N | A@B (ms) | algo | A.T@B (ms) | algo | A@B.T (ms) | algo | copy(A.T)@B (ms) | algo | A@copy(B.T) (ms) | algo |
|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 0.023301 | 0 | 0.027652 | 0 | 0.010134 | 21 | 0.023789 | 0 | 0.023391 | 0 |
| 768 | 0.043387 | 0 | 0.043201 | 0 | 0.013382 | 21 | 0.043403 | 0 | 0.043425 | 0 |
| 896 | 0.051938 | 0 | 0.061686 | 0 | 0.016391 | 21 | 0.051953 | 0 | 0.051924 | 0 |
| 960 | 0.066950 | 0 | 0.067120 | 0 | 0.017489 | 21 | 0.066787 | 0 | 0.066710 | 0 |
| 992 | 0.067360 | 0 | 0.067784 | 0 | 0.018833 | 21 | 0.067065 | 0 | 0.067247 | 0 |
| 1000 | 0.069087 | 0 | 0.069160 | 0 | 0.037736 | 23 | 0.069063 | 0 | 0.068834 | 0 |
| 1024 | 0.069203 | 0 | 0.068673 | 0 | 0.018353 | 21 | 0.069189 | 0 | 0.068785 | 0 |
| 1280 | 0.100203 | 0 | 0.104113 | 0 | 0.027312 | 21 | 0.100259 | 0 | 0.100472 | 0 |
| 1536 | 0.153684 | 0 | 0.154551 | 0 | 0.036703 | 21 | 0.153928 | 0 | 0.153829 | 0 |
| 1664 | 0.200429 | 0 | 0.207295 | 0 | 0.037776 | 21 | 0.199823 | 0 | 0.199790 | 0 |
| 2048 | 0.299155 | 0 | 0.301864 | 0 | 0.061061 | 21 | 0.299808 | 0 | 0.299868 | 0 |
| 2304 | 0.334924 | 0 | 0.337923 | 0 | 0.090026 | 21 | 0.334762 | 0 | 0.334419 | 0 |
| 4096 | 1.922978 | 0 | 1.938666 | 0 | 0.356208 | 21 | 1.920118 | 0 | 1.921067 | 0 |

## Analysis

### What was pinned (and what wasn't)

- We pin cuBLASLt algorithm configs **per record** key: `suite|case|dtype_key|shape`.
  - Example key: `square|ABT_view|int8_int8_int32|4096x4096x4096`
- Mechanism: the orchestrator passes `--algo-map ...`, which sets `ACCELSIM_TEST_CUBLASLT_ALGO_MAP=<path>` for the benchmark process.
- This run uses `results.json.run.settings.cublaslt.algo_selection_mode = pinned`, and records:
  - `algo_map_path`
  - `algo_map_sha256`
- With pinning enabled, we **bypass** `cublasLtMatmulAlgoGetHeuristic` and instead:
  - initialize by ID (`cublasLtMatmulAlgoInit`)
  - apply config attrs (`cublasLtMatmulAlgoConfigSetAttribute`)
  - validate and obtain required workspace (`cublasLtMatmulAlgoCheck`)

### Interpreting “A@B.T faster than A@B” (int8)

- **Important:** `A@B` and `A@B.T` compute different results unless `B` is symmetric. This benchmark is measuring layout/transpose-flag effects, not claiming a drop-in optimization for `A@B`.
- The performance gap correlates strongly with **different pinned cuBLASLt algorithms** for different transpose flags.
  - Example (N=4096, int8): `AB` uses algo_id `0`, while `ABT_view` uses algo_id `21`.
- The conceptual trap and next-step experiments are tracked in:
  - `context/issues/issue-int8-abt-view-faster-than-ab.md`

### View vs copy cases (what is measured)

- `*_view`: uses cuBLASLt transpose flags (`trans_a` / `trans_b`) while data remains in the original contiguous row-major buffers.
- `*_copy*`: materializes an explicit transpose into a new contiguous device buffer via our own `transpose_kernel`, and synchronizes **outside** the timed region.
- Therefore: **all reported times are GEMM-only**, not end-to-end “transpose + GEMM”.

### NVBench timing settings (what the flags mean)

These flags are recorded under `results.json.run.settings.nvbench`:

- `--stopping-criterion stdrel`: stop sampling when the “relative standard deviation” criterion is met (per NVBench’s criterion implementation).
- `--min-time 0.5`: require at least 0.5 seconds of total GPU time per benchmark state before the run can stop.
- `--max-noise 0.3`: maximum allowed relative noise; specified as a **percentage** (0.3%), and NVBench converts it internally to a ratio (0.003).
- `--min-samples 20`: take at least 20 timed samples per benchmark state.
- `--devices 0`: run only on CUDA device 0.

Per-record sample count is exported as `records[].timing.samples`, and the reported time is the mean GPU time `records[].timing.gpu_time_ms`.

### Reproducibility vs the source heuristic run (timing drift)

This pinned run was compared against the square records from the source run `tmp/gemm_transpose_full_sweep_20260202_083058/`:

- Median relative drift (pinned/source) is typically **< 0.3%** across dtype+case groups.
- 95th-percentile drift is usually **< ~3%**; the largest observed p95 in this comparison was **~5.2%** (fp16 `ATB_view`).

## Correctness & Verification (critical path)

### The timed “matmul launch” line (what actually gets timed)

The GEMM is invoked inside NVBench’s timed region via:

```cpp
state.exec(nvbench::exec_tag::sync,
           [&](nvbench::launch &launch) { plan.Run(launch.get_stream(), a_used, b_used, c_dev, &alpha, &beta); });
```

See: `cpp/src/gemm_transpose_bench.cu` (around lines 1075–1089).

### cuBLASLt call site

The cuBLASLt matmul call (which dispatches the GEMM kernel corresponding to the chosen/pinned algo) is:

```cpp
cublasLtMatmul(m_handle, m_matmul_desc,
               alpha, a, m_a_layout, b, m_b_layout,
               beta, c, m_c_layout, c, m_c_layout,
               &m_algo, m_workspace, m_workspace_bytes, stream);
```

See: `cpp/src/cublaslt_gemm.cu` (around lines 191–215).

Note: the exact internal kernel name is not captured in this timing-only run; you would need a profiler run (e.g., Nsight Compute) to attribute the dispatched kernel(s) by name.

### Verification strategy (excluded from timing)

- For each benchmark state, we run one untimed `plan.Run(...)` and synchronize, then verify against a CPU reference.
- Verification mode:
  - `full` when `max(M,N,K) <= 1000` (exact check of all `C` elements).
  - `sampled` otherwise (check a fixed set of indices).
- For lower precision modes, the CPU reference is adjusted to match effective precision:
  - fp16/bf16: quantize inputs to fp16/bf16 before reference GEMM
  - tf32: quantize inputs to TF32 mantissa before reference GEMM and use a relaxed tolerance

## Reproduction

- Build: `pixi run -e cuda13 gemm-transpose-build`
- Generate algo-map (from the heuristic source run): `pixi run -e cuda13 gemm-transpose algo-map --results /data1/huangzhe/code/accelsim-test/tmp/gemm_transpose_full_sweep_20260202_083058/results.json --suite square --out /data1/huangzhe/code/accelsim-test/reports/transpose_matmul/gemm_transpose_square_pinned_algo_20260202_095030/algo_map_square.json`
- Run pinned square timing: `pixi run -e cuda13 gemm-transpose timing --out-dir /data1/huangzhe/code/accelsim-test/reports/transpose_matmul/gemm_transpose_square_pinned_algo_20260202_095030 --suite square --dtype all --shape-set full_sweep_required --algo-map /data1/huangzhe/code/accelsim-test/reports/transpose_matmul/gemm_transpose_square_pinned_algo_20260202_095030/algo_map_square.json`
- Reporting (no re-run): `pixi run -e cuda13 gemm-transpose report --out-dir /data1/huangzhe/code/accelsim-test/reports/transpose_matmul/gemm_transpose_square_pinned_algo_20260202_095030`

## Appendix

- Generated summary: `report.md`
- Full timing table: `all_timings.md`
- Normalized export: `results.json`
- Raw NVBench JSON: `raw/`
