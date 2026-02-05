# About GPU Memory Alignment

**Memory Alignment** refers to the practice of ensuring that the starting address of a memory block is a multiple of a specific byte size (e.g., 16, 128, or 256 bytes). In GPU programming, proper alignment is critical for both **correctness** and **performance**.

## Why It Matters

### 1. Performance: Memory Coalescing
GPU memory controllers read data in transactions (chunks), typically **32 bytes** or **128 bytes** (L1/L2 cache line size).
*   **Aligned:** If a data array starts at address `0`, a single 128-byte transaction fetches indices `0-31` (assuming 4-byte `float`).
*   **Misaligned:** If the array starts at address `4`, the same indices span addresses `4-131`. This crosses a 128-byte boundary, potentially forcing the memory controller to issue **two** transactions (one for `0-127` and one for `128-255`) to serve a single warp request. This cuts effective bandwidth in half.

### 2. Correctness: Vectorized Instructions
GPUs use vectorized load/store instructions (e.g., `LD.E.128`, `ST.E.128`) to move data efficiently.
*   Instructions like `L.128` (loading `float4` or `int4`) **require** the memory address to be naturally aligned to the size of the vector (16 bytes).
*   Accessing a `float4` at a non-16-byte aligned address leads to **Illegal Address** errors (kernel crashes) or compiler fallback to slower scalar loads.

### 3. Tensor Cores
Hardware accelerators like Tensor Cores often require strict alignment (usually 16 bytes) for input pointers (`m`, `k`, `n` dimensions) to function at peak efficiency.

## Standard Alignments

*   **256 Bytes:** The guaranteed alignment returned by `cudaMalloc`. Safe for all standard types and texture/surface memory.
*   **128 Bytes:** Cache line size on most NVIDIA GPUs. Ideal for manual tiling to avoid cache line straddling.
*   **16 Bytes:** Minimum requirement for `float4`, `int4`, and Tensor Core operands.

## How to Specify Alignment in CUDA

### 1. Global Memory (Host Side)
Use `cudaMalloc`, which guarantees 256-byte alignment by default.
```cpp
float* d_ptr;
// Guaranteed to be at least 256-byte aligned
cudaMalloc(&d_ptr, N * sizeof(float));
```

### 2. Custom Structures (Device Side)
Use the `__align__(N)` qualifier or C++11 `alignas(N)` to ensure structures have the correct padding and alignment requirements.

**CUDA Specific:**
```cpp
struct __align__(16) MyFloat4 {
    float x, y, z, w;
};
```

**Standard C++11 (Preferred for portability):**
```cpp
struct alignas(16) MyFloat4 {
    float x, y, z, w;
};
```

### 3. Static Shared Memory
The CUDA compiler handles alignment for standard types. For byte arrays reinterpreted as other types, explicitly align the declaration.

```cpp
__shared__ alignas(16) uint8_t shared_buffer[1024];
// Now safe to cast to float4*
float4* f4_ptr = reinterpret_cast<float4*>(shared_buffer);
```

### 4. Dynamic Shared Memory
Dynamic shared memory is just a `void*` pointer (technically `extern __shared__ int[]`). It is guaranteed to be aligned to a generic standard (usually 4 or 8 bytes), but you may need to manually align pointers if you are partitioning it for larger types (like `double` or `float4`).

```cpp
extern __shared__ char smem[];
// Manual alignment adjustment might be needed if you have mixed types
// but usually the base pointer is sufficiently aligned (16+ bytes).
```

## References
*   [NVIDIA CUDA C++ Programming Guide - Device Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory)
*   [NVIDIA CUDA Best Practices Guide - Memory Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)

## Side Note: Is this unique to GPUs?

No, this concept applies to **all** modern computing (CPUs, RAM, SSDs), but the consequences differ:

*   **CPUs (x86/ARM):** CPU memory controllers also read in "chunks" (Cache Lines, typically **64 bytes**).
    *   **Misalignment:** If a value straddles two cache lines, the CPU must fetch both lines. On modern x86, this is handled automatically with a small performance penalty. On older ARM or strict architectures, unaligned access can cause a crash (`SIGBUS`).
    *   **Vector Units (AVX/NEON):** Like GPUs, CPU SIMD instructions often require alignment (e.g., 32-byte for AVX2) to work or perform optimally.
*   **The Difference:** GPUs are throughput-oriented. A "small" 2x penalty on a CPU is often masked by out-of-order execution and caches. On a GPU, where thousands of threads do the same thing simultaneously, a 2x bandwidth penalty (fetching 2 lines instead of 1 for every thread) effectively **halves the performance** of the entire kernel.
