# About CUTLASS vs cuBLAS

## The Core Difference

**cuBLAS** is a **pre-compiled binary library** of kernels provided by NVIDIA. It is the standard, closed-source solution for linear algebra on GPUs.
**CUTLASS** (CUDA Templates for Linear Algebra Subroutines) is an **open-source C++ template header-only library** that allows you to *build* high-performance kernels.

Think of **cuBLAS** as a finished sports car you can drive but not modify.
Think of **CUTLASS** as a crate of engine parts you can assemble into a standard engine or a custom one perfectly tuned for your specific chassis (workload).

## Comparison Table

| Feature | cuBLAS | CUTLASS |
| :--- | :--- | :--- |
| **Type** | Closed-source Shared Library (`.so`/`.dll`) | Open-source C++ Template Headers |
| **Integration** | Link against it, call C-style APIs | `#include` headers, compile with `nvcc` |
| **Flexibility** | **Low**: Fixed APIs (e.g., `cublasSgemm`) | **High**: Customize data types, tiling, layouts |
| **Transparency** | **None**: Black box assembly | **Full**: Visible C++/PTX hierarchy |
| **Key Advantage** | Ease of use, guaranteed baseline performance | Kernel Fusion, Customization, Research |

## The "Killer Feature": Kernel Fusion

The main reason to choose CUTLASS over cuBLAS is the ability to fuse operations (Epilogues).

**cuBLAS Workflow (Two Kernels):**
1. `GEMM(A, B)` -> Write to Global Memory
2. `ReLU(Add(Result, C))` -> Read Global Memory -> Write Result
*Bottleneck:* Double Global Memory traffic.

**CUTLASS Workflow (One Kernel):**
Using a custom **Epilogue**, CUTLASS keeps the GEMM result in registers and applies the bias/activation *before* writing to global memory.

```cpp
// Conceptual CUTLASS Epilogue definition
// This defines an operation: D = alpha * acc + beta * C
// Where 'acc' is the result of GEMM held in registers.
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,      // Data type
    128 / cutlass::sizeof_bits<ElementOutput>::value, // Vector length
    ElementAccumulator, // Accumulator type
    ElementCompute,     // Compute type
    cutlass::epilogue::thread::ScaleType::NoBetaScaling // Scaling/Activation
>;

// The GEMM kernel now includes this operation automatically at the end of the tile computation,
// avoiding the round-trip to global memory for the intermediate result.
```

## Relevance to Accel-Sim

For Accel-Sim users, CUTLASS is often superior because:
1.  **Transparency:** You can see the exact source code generating the SASS, allowing precise correlation between simulation traces and C++ code.
2.  **Experimentation:** You can change tiling strategies (block sizes, warp shapes) to stress specific parts of the simulated architecture, which is impossible with the fixed binaries of cuBLAS.

## Sources
- [NVIDIA CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
- [NVIDIA cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/index.html)
