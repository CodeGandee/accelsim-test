# How to Perform Matmul and Transpose with cuBLASLt

## Overview

`cuBLASLt` (cuBLAS Light) provides a flexible, descriptor-based API for matrix multiplication (`D = alpha * op(A) * op(B) + beta * C`). Unlike standard cuBLAS, it does not have separate functions for every transpose combination. Instead, you configure the operation via descriptors.

This guide covers:
1.  **Standard:** `Matmul(A, B)`
2.  **Transpose A:** `Matmul(A.T, B)`
3.  **Transpose B:** `Matmul(A, B.T)`
4.  **Efficiency:** Using heuristics to find the best algorithm.

## Prerequisites

- Link against `libcublasLt.so` / `cublasLt.lib`.
- Include `<cublasLt.h>`.
- Initialize a `cublasLtHandle_t`.

## Step-by-Step Implementation

### 1. Basic Setup (Common)

Create the handle and define matrix layouts. Note that cuBLAS typically uses **Column-Major** storage by default.

```cpp
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

void run_matmul(int m, int n, int k, bool trans_a, bool trans_b) {
    cublasLtHandle_t handle;
    cublasLtCreate(&handle);

    // Define data types
    cudaDataType_t type_a = CUDA_R_32F;
    cudaDataType_t type_b = CUDA_R_32F;
    cudaDataType_t type_c = CUDA_R_32F;
    cudaDataType_t type_r = CUDA_R_32F; // Compute type

    // Create Matrix Layouts
    // NOTE: cuBLAS is Column-Major. 
    // If A is MxK (no trans), lda = M.
    // If A is KxM (trans), lda = K.
    cublasLtMatrixLayout_t layout_a, layout_b, layout_c;
    
    // Layout logic changes based on transpose if you want to strictly follow logical dims
    // But typically you describe the PHYSICAL memory layout here.
    // Assume we have physical buffers:
    // A_buf: (M x K) if no trans, else (K x M)
    // B_buf: (K x N) if no trans, else (N x K)
    // C_buf: (M x N)
    
    // Simplified for demonstration (Standard Column Major):
    int lda = trans_a ? k : m;
    int ldb = trans_b ? n : k;
    int ldc = m;

    cublasLtMatrixLayoutCreate(&layout_a, type_a, trans_a ? k : m, trans_a ? m : k, lda);
    cublasLtMatrixLayoutCreate(&layout_b, type_b, trans_b ? n : k, trans_b ? k : n, ldb);
    cublasLtMatrixLayoutCreate(&layout_c, type_c, m, n, ldc);

    // ... continued below
}
```

### 2. Configure Operation Descriptor (The Critical Part)

This is where you tell cuBLASLt to transpose A or B.

```cpp
    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    // Handle Transpose A
    cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a));

    // Handle Transpose B
    cublasOperation_t op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b));
```

### 3. Algorithm Selection (Efficiency)

Don't guess the algorithm. Use the heuristic API to ask the driver for the fastest one for your specific GPU and problem size.

```cpp
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulPreferenceCreate(&preference);
    
    // Allow using workspace memory for faster kernels
    size_t workspace_size = 1024 * 1024 * 4; // 4MB
    cublasLtMatmulPreferenceSetAttribute(preference, 
                                         CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_SIZE, 
                                         &workspace_size, 
                                         sizeof(workspace_size));

    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedAlgoCount = 0;
    
    cublasLtMatmulAlgoGetHeuristic(handle, operationDesc, layout_a, layout_b, layout_c, layout_c, 
                                   preference, 1, &heuristicResult, &returnedAlgoCount);

    if (returnedAlgoCount == 0) {
        std::cerr << "No valid algorithm found!" << std::endl;
        return;
    }
```

### 4. Execute

```cpp
    float alpha = 1.0f;
    float beta = 0.0f;
    void* workspace_dev; // Allocate this via cudaMalloc
    
    // Pointers to device memory (d_A, d_B, d_C)
    // ...
    
    cublasLtMatmul(handle,
                   operationDesc,
                   &alpha,
                   d_A, layout_a,
                   d_B, layout_b,
                   &beta,
                   d_C, layout_c,
                   d_C, layout_c, // D = C usually
                   &heuristicResult.algo,
                   workspace_dev,
                   workspace_size,
                   0); // CUDA stream

    // Cleanup
    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtMatrixLayoutDestroy(layout_a);
    cublasLtMatrixLayoutDestroy(layout_b);
    cublasLtMatrixLayoutDestroy(layout_c);
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtDestroy(handle);
```

## Summary of Configurations

| Goal | `CUBLASLT_MATMUL_DESC_TRANSA` | `CUBLASLT_MATMUL_DESC_TRANSB` | Layout A Dims | Layout B Dims |
| :--- | :--- | :--- | :--- | :--- |
| **A * B** | `CUBLAS_OP_N` | `CUBLAS_OP_N` | M x K | K x N |
| **A.T * B** | `CUBLAS_OP_T` | `CUBLAS_OP_N` | K x M | K x N |
| **A * B.T** | `CUBLAS_OP_N` | `CUBLAS_OP_T` | M x K | N x K |

## Critical Efficiency Note: Zero-Copy Transpose

**DO NOT** manually transpose your matrices (e.g., using `cudaMemcpy` or a custom kernel) before calling `cublasLt`. This defeats the purpose of the library and wastes huge amounts of memory bandwidth.

1.  **Leave Data As-Is:** Keep your source matrix `A` in its original memory layout.
2.  **Describe Reality:** Configure `cublasLtMatrixLayout_t` to match the *physical* dimensions of that existing data.
3.  **Request Math:** Use `CUBLASLT_MATMUL_DESC_TRANSA` to tell the math engine to treat that data as transposed during the multiply.

The `cublasLtMatmul` kernel will read the data in the transposed order directly from the source buffer.

**Note:** The matrix layout (`cublasLtMatrixLayout_t`) describes the *physical* storage in memory. The operation descriptor (`cublasLtMatmulDesc_t`) describes the *mathematical* operation. If you have a physically transposed matrix, you just tell the layout it is KxM, and then tell the operation to transpose it (or not) depending on what math you want.

## Sources
- [NVIDIA cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/index.html)
