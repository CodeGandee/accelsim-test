# Issue: `int8` `A@B.T` Much Faster Than `A@B` (Square GEMM)

## Summary

In the GEMM transpose benchmark, for **square** matrices (`M=N=K`) and dtype `int8,int8->int32`, we observed a large performance gap where **`A@B.T` is much faster than `A@B`** (e.g., ~5.4× at `N=4096` on A100). This confuses readers because, for square matrices, one can always rewrite:

`A@B = A@(B.T).T`

which leads to the question: *If `A@B.T` is faster, why not always do that (and recover `A@B` by transposing B twice)?*

This issue documents the observation, clarifies the conceptual trap (math vs storage/layout vs kernel eligibility), and proposes follow-up actions to make the benchmark and stakeholder narrative less misleading.

---

## Where It Was Observed

**Run / artifacts**

- Run dir: `tmp/gemm_transpose_full_sweep_20260202_083058/`
- Key files:
  - `tmp/gemm_transpose_full_sweep_20260202_083058/report.md`
  - `tmp/gemm_transpose_full_sweep_20260202_083058/all_timings.md`
  - `tmp/gemm_transpose_full_sweep_20260202_083058/results.json`

**Pinned-algo confirmation run (square-only, algorithms pinned)**

- Run dir: `tmp/gemm_transpose_square_pinned_algo_20260202_095030/`
- Key files:
  - `tmp/gemm_transpose_square_pinned_algo_20260202_095030/report.md`
  - `tmp/gemm_transpose_square_pinned_algo_20260202_095030/all_timings.md`
  - `tmp/gemm_transpose_square_pinned_algo_20260202_095030/results.json`

**Environment**

- GPU: `NVIDIA A100-SXM4-80GB` (`sm_800`)
- NVBench settings: `stdrel`, `min_time_s=0.5`, `min_samples=20`, `max_noise_pct=0.3`, `devices=0`

---

## Concrete Example (Square `N=4096`, int8)

From `results.json`:

- `AB` (`A@B`):
  - `time_ms ≈ 1.9235`
  - `algo_id = 0`
- `ABT_view` (`A@B.T`):
  - `time_ms ≈ 0.3569`
  - `algo_id = 21`

So, for this run: `A@B.T` is ~**5.4× faster** than `A@B`.

Quick inspection command:

```bash
out_dir=/data1/huangzhe/code/accelsim-test/tmp/gemm_transpose_full_sweep_20260202_083058
jq -r '
  .records[]
  | select(.suite=="square" and .shape.m==4096 and .dtype.a=="int8" and (.case=="AB" or .case=="ABT_view"))
  | [.case, (.timing.gpu_time_ms|tostring), (.cublaslt.algo.id|tostring)]
  | @tsv
' "$out_dir/results.json" | sort
```

---

## Why This Confuses People

### The “structurally same” reasoning

For square matrices, a reader may argue:

1) `A@B` can be rewritten as `A@(B.T).T` (true).
2) If `A@B.T` is always faster than `A@B`, then by storing `B.T` and calling the “fast” path, we could get `A@B` faster too.

### The hidden mismatch

In our benchmark as written, **`A@B` and `A@B.T` are not computing the same result** (unless `B` happens to be symmetric). So this benchmark is not (yet) showing a drop-in optimization.

To turn the “`A@B = A@(B.T).T`” identity into an optimization, you must:

- **physically store** `B_pre = B.T` (or produce it upstream in that layout), and then
- compute `A@B_pre.T` (which equals `A@B`).

That changes:

- memory layout of `B`,
- the `trans_b` flag,
- and therefore which cuBLASLt kernels are eligible / preferred.

It also introduces a real cost: producing `B_pre` (a transpose) unless you can amortize it.

---

## What We Learned from External References (cuBLAS/cuBLASLt docs)

### Int8 IMMA (Tensor Core) kernels have strict transpose/layout constraints

NVIDIA’s cuBLAS documentation for cuBLASLt `cublasLtMatmul()` states that **to use IMMA (int8 Tensor Core) kernels**, specific conditions must be met, including constraints on transpose formats and (optionally) special packed memory orders:

- With *regular* row/column-major ordering, the docs state: **only “TN” is supported** (A transposed, B non-transposed).  
  Source: cuBLAS v13.0.2 docs, `cublasLtMatmul()` section (“To use IMMA kernels…”, bullet list).  
  Ref: https://docs.nvidia.com/cuda/archive/13.0.2/cublas/index.html

- With IMMA-specific packed ordering (Ampere/Turing): A/C/D in `CUBLASLT_ORDER_COL32` and B in `CUBLASLT_ORDER_COL32_2R_4R4` (Ampere) / `CUBLASLT_ORDER_COL4_4R2_8C` (Turing/Ampere), the docs include the requirement that the **matmul descriptor specifies `op(B)=T` and `op(A)=N`**.  
  The docs also state for this packed-layout path: **only “NT” is supported** (A non-transposed, B transposed).  
  Source: same `cublasLtMatmul()` IMMA-specific ordering bullets.  
  Ref: https://docs.nvidia.com/cuda/archive/13.0.2/cublas/index.html

This is directly relevant to our observation, because:

- our `ABT_view` case sets `trans_b=CUBLAS_OP_T`, and
- our `AB` case sets `trans_b=CUBLAS_OP_N`.

Even though we do not (yet) use `COL32_2R_4R4` in this benchmark, the docs strongly suggest that for int8, **transpose flags can gate eligibility for the fast IMMA kernel families**, and `op(B)=T` is explicitly called out as a requirement in the high-performance packed-layout path.

### Row-major vs column-major (and why transpose flags can change “what’s contiguous”)

cuBLASLt makes the matrix storage order explicit via the matrix layout descriptor:

- `CUBLASLT_MATRIX_LAYOUT_ORDER` controls the memory order, and the documented default is `CUBLASLT_ORDER_COL`.  
  Ref: cuBLAS v13.0.2 docs, `cublasLtMatrixLayoutAttribute_t`: https://docs.nvidia.com/cuda/archive/13.0.2/cublas/index.html
- `cublasLtOrder_t` defines `CUBLASLT_ORDER_ROW`, `CUBLASLT_ORDER_COL`, and IMMA-friendly packed orders like `CUBLASLT_ORDER_COL32_2R_4R4`.  
  Ref: cuBLAS v13.1.0 docs, `cublasLtOrder_t`: https://docs.nvidia.com/cuda/archive/13.1.0/cublas/index.html

Practical implication:

- Changing `trans_b` changes which logical dimension is “the contiguous one” for how the kernel will traverse B, *and* it can change which kernel families are legal/fast.
- “B.T is faster because it uses contiguous B rows in row-major” is not sufficient as a standalone explanation in our data (see next section).

### Weight prepacking is a known technique for int8 GEMM

It is common in inference stacks to **prepack int8 weights** into cuBLASLt’s IMMA-friendly layouts to unlock the fastest kernels (e.g., `CUBLASLT_ORDER_COL32_2R_4R4` on Ampere), which is consistent with the packed-layout requirements described above. This also shows up as an ecosystem discussion point (TensorRT issue thread; informational, not normative):  
Ref: https://github.com/NVIDIA/TensorRT/issues/3233

---

## Likely Root Cause (Updated Working Hypothesis)

cuBLASLt uses heuristics that depend on:

- transpose flags (`trans_a`, `trans_b`),
- matrix layout metadata (order, leading dimensions),
- dtype/compute type,
- and allowed workspace.

In this run, the heuristic selected **different algorithm families** for the two cases:

- `AB` → `algo_id=0` (slow here)
- `ABT_view` → `algo_id=21` (fast here)

This suggests that, for our current **row-major** layouts, the “transpose-B” view path unlocks a higher-performance int8 kernel/config than the non-transpose path.

### Evidence that it’s not just “B is contiguous”

If “`B.T` is faster because it makes B contiguous for the kernel” were the whole story, then **materializing** `B.T` into a contiguous buffer (`ABT_copyB`) should be similarly fast. But in our measurements:

- `ABT_copyB` is ~as slow as `AB` and uses the same algo_id as `AB` (e.g., at `N=4096`, `algo_id=0`),
- while `ABT_view` remains much faster and uses `algo_id=21`.

So the dominant effect is that **`trans_b=CUBLAS_OP_T` itself changes which cuBLASLt algorithms are valid/selected**, consistent with the IMMA constraints called out in NVIDIA docs (especially the “packed ordering” bullet that requires `op(B)=T`).

### Evidence from manual pinning attempts

We tried to “force” `AB` to use the `ABT_view` int8 algorithm (`algo_id=21`) by pinning it, and cuBLASLt rejected it with `CUBLAS_STATUS_NOT_SUPPORTED` during `cublasLtMatmulAlgoCheck`. This aligns with the idea that an algorithm ID is not universally valid across transpose/layout variants.

Important nuance:

- The benchmark times **GEMM only**; any explicit transpose for `*_copy*` is done outside the timed region.
- Therefore, even if “store `B.T` and use `trans_b=T`” is a valid optimization for repeated GEMMs, it must be evaluated with a clear accounting of **transpose cost and amortization**.

---

## What We Should Do Next (Proposed Actions)

### A) Add an “equivalent math” comparison for square int8

Add a case that computes **the same math as `A@B`** but uses a different storage/layout:

- Create `B_pre = B.T` once (outside timing).
- Time `A@B_pre.T` (which is mathematically `A@B`).
- Report both:
  - GEMM-only time (current style), and
  - end-to-end time including the transpose (new optional metric), at least for representative sizes.

This directly answers: *Can we always “convert `A@B` into a transpose-B call” and win? Under what reuse assumptions?*

### B) Improve the stakeholder-facing explanation

In stakeholder docs, explicitly state:

- `A@B` vs `A@B.T` are **different computations** unless `B` is symmetric.
- The observed “`A@B.T` is faster” is an algorithm/layout effect, **not** a free win for `A@B`.

### C) Algorithm pinning / override support (future)

Expose an “algorithm override” mechanism to:

- pin `cublasLtMatmulAlgo_t` (or a stable subset of its config),
- or at least constrain to a set of algorithms,

so we can run controlled experiments:

- Same math, same layout, different algo
- Same math, different layout, pinned algo

This will help separate “layout effect” vs “heuristic choice” vs “kernel family difference”.

---

## Relevant Code Pointers

- Case selection (`AB` vs `ABT_view`) and transpose flags:
  - `cpp/src/gemm_transpose_bench.cu:699` (case dispatch)
  - `cpp/src/gemm_transpose_bench.cu:709` (`ABT_view` sets `trans_b=CUBLAS_OP_T`)
- Timed GEMM launch:
  - `cpp/src/gemm_transpose_bench.cu:958`
- cuBLASLt dispatch:
  - `cpp/src/cublaslt_gemm.cu:149` (`cublasLtMatmul(...)`)
- Selected algorithm export:
  - `cpp/src/gemm_transpose_bench.cu:805` (`accelsim/cublaslt/algo` summary)

---

## Definition of Done

- We can answer, with data and a clear narrative:
  1) When `A@B.T` is faster than `A@B` (by dtype/shape), and which `algo_id`/config differences correlate.
  2) Whether an “equivalent math” transformation (`B_pre=B.T`, run transpose-B GEMM) is actually a win **including transpose cost**, under reasonable reuse assumptions.
- Stakeholder report and generated reports are explicit about “math difference vs storage/layout transformation”, avoiding the misleading interpretation.
