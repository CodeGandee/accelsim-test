# N=1000 int8 (ABT_view): Why `algo_id=23` Is Faster Than `algo_id=64`

This note explains the large kernel-duration difference observed in the B200 / CUDA 13.0 sweep for:

- Shape: `M=N=K=1000`
- Dtype: `int8,int8->int32 (int32,default)`
- Case: `ABT_view` (i.e., `A @ B.T`, transpose flags `trans_a=N`, `trans_b=T`)
- Comparison: `ABT_view` forced `algo_id=23` vs `ABT_view` forced `algo_id=64`

This is intentionally **not** a program-generated report. It references the profiling artifacts captured under this run directory.

## Artifacts (evidence)

Kernel discovery (Nsight Systems):

- `profiles/n1000_int8_abt_view_algo23/nsys/kernel_list.csv`
- `profiles/n1000_int8_abt_view_algo64/nsys/kernel_list.csv`

Kernel profiling (Nsight Compute, `--set basic`):

- `profiles/n1000_int8_abt_view_algo23/ncu/details.csv`
- `profiles/n1000_int8_abt_view_algo64/ncu/details.csv`

## What kernels were used?

From `profiles/.../nsys/kernel_list.csv`, the timed GEMM kernel differs between the two algos:

### `algo_id=23`

- Kernel name (abridged): `cutlass::Kernel2<cutlass_80_tensorop_i16832gemm_s8_128x64_128x3_tn_align4>(...)`
- Launch config (from `nsys` / `ncu`):
  - Grid: `128x1x1` blocks
  - Block: `128x1x1` threads

### `algo_id=64`

- Kernel name (abridged): `cutlass::Kernel2<cutlass_80_wmma_tensorop_i161616gemm_s8_forwardCompat_128x128_32x2_tn_align4>(...)`
- Launch config (from `nsys` / `ncu`):
  - Grid: `64x1x1` blocks
  - Block: `256x1x1` threads

The most important takeaway: **the algorithm flip changes the executed CUTLASS kernel family and its tiling**, not just a minor knob.

## Decode the CUTLASS name (what it implies)

CUTLASS kernel names encode (roughly):

`cutlass_<family>_<opcode_class>_<mma_shape>gemm_<dtype>_<TB_MxTB_N>_<TB_K x pipeline_stages>_<layout>_align<bytes>`

Two key fields differ here:

1) **Opcode class:** `tensorop` vs `wmma_tensorop`

- `TensorOp` kernels use Tensor Core MMA directly.
- `WmmaTensorOp` kernels use the CUDA WMMA abstraction (still on Tensor Cores, but with additional constraints and potentially different kernel variants).

2) **MMA instruction shape:** `i16832` vs `i161616`

- `i161616` corresponds to a WMMA 16×16×16 instruction shape.
- `i16832` corresponds to a TensorOp 16×8×32 instruction shape.

These are fundamentally different warp-level math primitives, which cascades into different data movement and scheduling at the threadblock level.

3) **Threadblock tile + K tile:** `..._128x64_128x3_...` vs `..._128x128_32x2_...`

It is consistent with CUTLASS naming conventions that:

- `128x64_128x3` indicates a threadblock tile of roughly `(TB_M,TB_N,TB_K)=(128,64,128)` with `stages=3`.
- `128x128_32x2` indicates `(TB_M,TB_N,TB_K)=(128,128,32)` with `stages=2`.

If this interpretation holds (and it matches the conventional encoding used in CUTLASS GEMM names), then for `K=1000`:

- `TB_K=128` → ~`ceil(1000/128)=8` mainloop iterations
- `TB_K=32`  → ~`ceil(1000/32)=32` mainloop iterations

That is a **4× difference** in the number of pipeline/mainloop “steps” per CTA, which can translate into a large runtime delta for small/medium GEMMs where fixed overheads matter.

## What do the `ncu` numbers say?

From `profiles/.../ncu/details.csv`:

| metric | `algo_id=23` | `algo_id=64` |
|---|---:|---:|
| Duration | 22.59 us | 49.95 us |
| Compute (SM) Throughput | 15.01% | 12.49% |
| Memory Throughput | 44.98% | 21.70% |
| L1/TEX Cache Throughput | 71.75% | 59.24% |
| L2 Cache Throughput | 9.14% | 5.03% |
| DRAM Throughput | 1.36% | 0.61% |
| Grid size | 128 blocks | 64 blocks |
| Block size | 128 threads | 256 threads |
| Waves per SM | 0.29 | 0.22 |
| Dynamic SMEM per block | 73.73 KB | 34.82 KB |

Observations grounded in these measurements:

- **Kernel time:** `algo_id=23`’s kernel is ~2.2× shorter (22.6us vs 50.0us).
- **Underfill dominates:** both launches show **waves per SM << 1**, meaning this problem is too small to fully occupy B200. In that regime, kernel choice can swing runtime significantly.
- **Grid size difference matters:** `algo_id=23` launches 128 CTAs vs 64 CTAs, which reduces underutilization on a 148-SM GPU.
- **Different tiling/instruction shape:** the kernel families differ (`TensorOp i16832` vs `WmmaTensorOp i161616`), and the threadblock tiling hints imply a large difference in K-iteration count (and therefore loop overhead).

## Why can this create such a large duration gap? (most likely drivers)

Based on the kernel names + the above metrics, the large duration difference is plausibly driven by a combination of:

1) **Fewer K mainloop iterations (likely `TB_K=128` vs `TB_K=32`)**

For `K=1000`, a smaller K tile requires more mainloop steps. Each step typically includes global→shared movement (or equivalent), synchronization/pipeline control, and MMA issue. If one kernel does ~4× more steps, its duration can be substantially larger even if each step is efficient.

2) **Less underutilization due to more CTAs**

This GEMM is small enough that *grid size* is a major factor. Doubling CTAs (64 → 128) improves SM coverage, which is consistent with the higher `Waves per SM` reported for `algo_id=23`.

3) **Different warp-level MMA shape / kernel family (`TensorOp` vs `WmmaTensorOp`)**

The `WmmaTensorOp ... forwardCompat` naming suggests this kernel is a more compatibility-oriented WMMA path. In practice, it can carry additional constraints that lead to less favorable schedules for this exact TN int8 problem, compared to the `TensorOp i16832` family.

## Caveats / what this does not prove yet

- The exact mapping from name fields to `(TB_M,TB_N,TB_K,stages)` is inferred from typical CUTLASS naming conventions. It is a strong hint, but it is not a formal guarantee without inspecting the exact CUTLASS template instantiation in the library build.
- The `cutlass_80_*` prefix is a kernel family tag, not a statement about the physical GPU (this run is on B200). Forward-compatible kernels can run on newer architectures.
- `ncu` adds overhead; use it for kernel metrics and relative comparisons, not for “real” end-to-end timing.

## How to reproduce (minimal)

The captures in `profiles/` were created from the standalone repro binary plus the profiling wrappers:

Kernel discovery:

```bash
pixi run python scripts/cublaslt_kernel_discovery.py \
  --out-dir tmp/algo23_investigation \
  --case-id n1000_int8_abt_view_algo23 \
  --pixi-env cuda13 \
  -- ./cpp/build/Release/repro_algo23_int8_n1000 \
       --variant ABT_view --force-algo 23 --tile-id 18 --stages-id 21 --splitk 1 \
       --nvtx --iters 1 --warmup 0
```

Kernel profiling (forced `algo_id=23` vs forced `algo_id=64`):

```bash
pixi run python scripts/cublaslt_ncu_profile.py \
  --out-dir tmp/algo23_investigation \
  --case-prefix n1000_int8_abt_view \
  --compare-abt23-vs-abt64 \
  --pixi-env cuda13 \
  --set basic \
  -- ./cpp/build/Release/repro_algo23_int8_n1000
```

