# N=1000 int8 (ABT_view): Why `algo_id=23` Is Faster Than `algo_id=64`

This note explains the large kernel-duration difference observed in the B200 / CUDA 13.0 sweep for:

- Shape: `M=N=K=1000`
- Dtype: `int8,int8->int32 (int32,default)`
- Case: `ABT_view` (i.e., `A @ B.T`, transpose flags `trans_a=N`, `trans_b=T`)
- Comparison: `ABT_view` forced `algo_id=23` vs `ABT_view` forced `algo_id=64`

This is intentionally **not** a program-generated report. It references the profiling artifacts captured under this run directory.

## Terminology (shortnames used in this note)

- **GEMM**: General Matrix Multiply, `C = A * B` (with optional transposes).
- **`ABT_view`**: The benchmark case `A @ B.T` (transpose flags `trans_a=N`, `trans_b=T`). This is a *view/layout choice* at the benchmark level; internally, a kernel may still choose a specific data movement strategy.
- **`algo_id` / “algo23”, “algo64”**: cuBLASLt algorithm IDs as selected/forced in the run. Changing `algo_id` can change the entire kernel family (not just a small tuning knob).
- **CUTLASS**: NVIDIA’s open-source CUDA template library for GEMM/Conv. cuBLASLt often uses CUTLASS-derived kernels under the hood.
- **Kernel name**: The `cutlass_...` string reported by `nsys` / `ncu`. It encodes tiling, instruction class, layouts, and more.

GPU execution:

- **Grid**: The full set of blocks launched for a kernel.
- **Block / Threadblock / TB**: A CUDA thread block. Size shown as `block=(threads_x, threads_y, threads_z)`.
- **CTA** (*Cooperative Thread Array*): NVIDIA’s term for a thread block. In practice, **CTA ≈ threadblock**.
- **SM** (*Streaming Multiprocessor*): The basic execution unit on NVIDIA GPUs. A kernel runs by scheduling CTAs onto SMs.
- **Wave**: One “pass” where at most one CTA per SM (or more, depending on resources) is resident; for small grids, you may see `< 1` wave per SM, meaning the GPU is underfilled.
- **Warp**: 32 threads scheduled together.

Tensor Core math:

- **MMA**: Matrix Multiply-Accumulate instruction family on Tensor Cores (PTX: `mma.sync.*`).
- **WMMA**: Warp Matrix Multiply-Accumulate API/instruction family (PTX: `wmma.*`). Still uses Tensor Cores, but via the WMMA abstraction.
- **Opcode class `tensorop` vs `wmma_tensorop`**: CUTLASS shorthand for “MMA-sync TensorOp path” vs “WMMA path”.
- **`i16832` / `i161616`**: Encoded MMA tile shapes for int8:
  - `i16832` aligns with `m16n8k32` (typical for int8 MMA-sync paths).
  - `i161616` aligns with `m16n16k16` (typical WMMA shape).

Tiling / pipelining:

- **`TB_MxTB_N`**: Threadblock tile shape in M and N (e.g., `128x64`).
- **`TB_K`**: Threadblock tile in K (depth). Smaller `TB_K` generally means more mainloop iterations for a fixed `K`.
- **Stages / pipeline stages**: How many mainloop stages are used to overlap memory movement and compute (e.g., `..._128x3` suggests `TB_K=128`, `stages=3` in typical CUTLASS naming).

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
- `WmmaTensorOp` kernels use the CUDA WMMA abstraction (still on Tensor Cores, but with additional constraints and potentially different kernel variants). CUTLASS explicitly distinguishes these opcode classes in its documentation.

2) **MMA instruction shape:** `i16832` vs `i161616`

- `i161616` aligns with the WMMA shape `m16n16k16` (i.e., 16×16×16) described in the CUDA PTX ISA documentation for `wmma.*` operations.
- `i16832` aligns with the dense Tensor Core MMA shape `m16n8k32` (i.e., 16×8×32) described in the CUDA PTX ISA documentation for `mma.sync.*` operations (integer `s8/u8` supports `m16n8k32`).

These are fundamentally different warp-level math primitives, which cascades into different data movement and scheduling at the threadblock level.

3) **Threadblock tile + K tile:** `..._128x64_128x3_...` vs `..._128x128_32x2_...`

It is consistent with CUTLASS naming conventions that:

- `128x64_128x3` indicates a threadblock tile of roughly `(TB_M,TB_N,TB_K)=(128,64,128)` with `stages=3`.
- `128x128_32x2` indicates `(TB_M,TB_N,TB_K)=(128,128,32)` with `stages=2`.

If this interpretation holds (and it matches the conventional encoding used in CUTLASS GEMM names), then for `K=1000`:

- `TB_K=128` → ~`ceil(1000/128)=8` mainloop iterations
- `TB_K=32`  → ~`ceil(1000/32)=32` mainloop iterations

That is a **4× difference** in the number of pipeline/mainloop “steps” per CTA, which can translate into a large runtime delta for small/medium GEMMs where fixed overheads matter.

4) **Threadblock tile implies different “shape decomposition” of the GEMM**

For `M=N=1000` and the threadblock tiles implied by the names:

- `TB_MxTB_N = 128x64` → `ceil(1000/128) * ceil(1000/64) = 8 * 16 = 128` CTAs (matches the observed `grid=128`)
- `TB_MxTB_N = 128x128` → `ceil(1000/128) * ceil(1000/128) = 8 * 8 = 64` CTAs (matches the observed `grid=64`)

This matters on B200 because **both kernels are “underfilled”** (grid smaller than SM count), so the entire kernel duration is essentially the duration of a single wave of CTAs. In that regime, *per-CTA efficiency* dominates: a kernel that does fewer mainloop steps and overlaps memory better can be much faster even if it uses more shared memory.

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

4) **Different implementation “era”: `wmma_tensorop ... forwardCompat` tends to be a more conservative kernel family**

This is the key “kernel name → CUTLASS intuition” point:

- `tensorop_i16832...` kernels typically map to CUTLASS’s lower-level Tensor Core MMA pathway (`mma.sync.*`) plus architecture-specific data movement patterns (e.g., `ldmatrix`-style shared-memory matrix loads and multi-stage pipelining). The name’s `..._128x3...` staging hint is consistent with “deeper” pipelining.
- `wmma_tensorop_i161616...forwardCompat...` kernels typically map to CUTLASS’s WMMA abstraction pathway (`wmma.*`). The “forwardCompat” hint usually means “pick a WMMA-compatible variant that will run broadly”, which often correlates with more conservative tile-K and staging choices (here: `..._32x2...`), and less aggressive overlap.

You can treat this as a hypothesis until verified, but it matches the measured symptom: **`algo_id=64` has ~2× lower Memory Throughput (21.7% vs 45.0%) and ~2.2× longer Duration**, consistent with a kernel that is less effective at feeding Tensor Core math on this problem.

## How to validate the above with “hard” evidence (SASS / instruction mix)

If you want a definitive answer beyond inference from names/metrics:

1) Open `profiles/.../ncu/profile.ncu-rep` in Nsight Compute GUI and inspect **Source / SASS**:
   - For `algo_id=23`, look for `mma.sync.aligned.m16n8k32...s8...` (or corresponding SASS) and `cp.async` / `ldmatrix` patterns.
   - For `algo_id=64`, look for `wmma.mma.sync.aligned.m16n16k16...s8...` (or corresponding SASS) and whether the pipeline relies on classic `ld.global` → `st.shared` → `ld.shared` rather than `cp.async`.

2) In `ncu` CLI exports, add additional sections/metrics if needed:
   - Instruction mix: Tensor Core pipe utilization, total instruction counts, barrier/sync counts
   - Memory pipeline: `cp.async` usage (if available), shared-memory bank conflict indicators, L1/L2 hit rates

Those checks will tell you whether the real root cause is (a) mainloop step count, (b) memory pipeline differences, (c) barrier overhead, or (d) instruction-level efficiency differences between WMMA and MMA-sync kernels on this architecture.

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
