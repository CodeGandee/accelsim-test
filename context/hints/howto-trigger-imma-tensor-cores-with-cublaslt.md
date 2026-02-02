# How-to: Trigger IMMA Tensor Cores with cuBLASLt (int8)

## Overview

Achieving peak `int8` performance on NVIDIA GPUs (Turing/Ampere/Hopper) requires using **IMMA (Integer Matrix Multiply Accumulate)** Tensor Core instructions. However, these kernels have strict requirements on data layout and alignment.

Simply passing `int8` pointers to `cublasLtMatmul` might fallback to slower implementations (e.g., CUDA Core `IDP4` or non-optimized TC paths) if the memory layout doesn't match what the hardware or kernel expects.

As observed in `context/issues/issue-int8-abt-view-faster-than-ab.md`, `trans_b=T` (transpose B) is often a hard requirement for the highest-performance kernels because it allows `B` (the weights) to be accessed in a K-major contiguous fashion (or a specific packed format).

To **guarantee** usage of optimal IMMA kernels, you often need to **pre-pack** your data (usually the weights matrix `B`) into a hardware-friendly layout using `cublasLtMatrixTransform`.

## The "Opaque" Packed Layouts

cuBLASLt defines several specialized layouts that are "opaque" (you shouldn't manually write them) but optimized for Tensor Core access:

- `CUBLASLT_ORDER_COL32`: often for A/C matrices.
- `CUBLASLT_ORDER_COL4_4R2_8C`: for weights (Turing/Ampere).
- `CUBLASLT_ORDER_COL32_2R_4R4`: for weights (Ampere+).

### Understanding "A/C Matrices" vs "Weights"

In the GEMM equation $C = \alpha \times (A \times B) + \beta \times C$:

-   **A (Input / Activation):** The dynamic input data (e.g., batch of tokens). Since this changes every request, we often keep it in standard layouts or simpler aligned layouts like `COL32`.
-   **B (Weights):** The static model parameters. Since these are constant, we can afford the expensive one-time transformation into complex, read-optimized layouts like `COL32_2R_4R4`.
-   **C (Output):** The result destination. Like A, it is dynamic and often uses `COL32` or standard layouts to align with warp memory transactions.

**Key Strategy:**
1.  Keep inputs in standard layout (Row/Col Major) if possible.
2.  **Transform** the static weight matrix (`B`) into a packed layout using `cublasLtMatrixTransform` *before* the inference loop.
3.  Pass the packed layout descriptor to `cublasLtMatmul`.

## Workflow Example (C++)

This example shows how to transform a standard Column-Major matrix `B` into `CUBLASLT_ORDER_COL32_2R_4R4` for use as a pre-packed weight matrix.

```cpp
#include <cublasLt.h>
#include <vector>

void transform_weights_to_packed(cublasLtHandle_t lt_handle,
                                 int n, int k,
                                 const void* d_B_standard, // Source: Standard layout (e.g., COL)
                                 void* d_B_packed,         // Dest:   Packed layout
                                 cudaStream_t stream) {

    // 1. Define Standard Source Layout (e.g., Column Major)
    cublasLtMatrixLayout_t layout_src;
    cublasLtMatrixLayoutCreate(&layout_src, CUDA_R_8I, k, n, k); // LD depends on orientation
    cublasLtOrder_t order_src = CUBLASLT_ORDER_COL;
    cublasLtMatrixLayoutSetAttribute(layout_src, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_src, sizeof(order_src));

    // 2. Define Packed Destination Layout
    // Note: Dimensions match the logical shape, but the 'order' changes physical storage.
    cublasLtMatrixLayout_t layout_dst;
    cublasLtMatrixLayoutCreate(&layout_dst, CUDA_R_8I, k, n, k); // LDC is often ignored for packed, but good practice to set
    cublasLtOrder_t order_dst = CUBLASLT_ORDER_COL32_2R_4R4;     // Ampere+ specialized layout
    cublasLtMatrixLayoutSetAttribute(layout_dst, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_dst, sizeof(order_dst));

    // 3. Define Transform Descriptor
    cublasLtMatrixTransformDesc_t transform_desc;
    cublasLtMatrixTransformDescCreate(&transform_desc, CUDA_R_32F); // Scale type float

    // 4. Execute Transformation
    // C_packed = alpha * A_standard + beta * B (unused)
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasLtMatrixTransform(lt_handle,
                            transform_desc,
                            &alpha,
                            d_B_standard, layout_src,
                            &beta,
                            nullptr,      nullptr,    // No "beta" input
                            d_B_packed,   layout_dst, // Output
                            stream);

    // Cleanup descriptors...
    cublasLtMatrixLayoutDestroy(layout_src);
    cublasLtMatrixLayoutDestroy(layout_dst);
    cublasLtMatrixTransformDescDestroy(transform_desc);
}
```

## Using the Packed Weights

Once `d_B_packed` is populated, use it in your matmul. Crucially, the **layout descriptor** you pass to `cublasLtMatmul` must match the packed format.

```cpp
// In your inference loop:
cublasLtMatrixLayout_t B_packed_desc;
cublasLtMatrixLayoutCreate(&B_packed_desc, CUDA_R_8I, k, n, k);
cublasLtOrder_t packed_order = CUBLASLT_ORDER_COL32_2R_4R4;
cublasLtMatrixLayoutSetAttribute(B_packed_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &packed_order, sizeof(packed_order));

cublasLtMatmul(lt_handle,
               matmul_desc,
               &alpha,
               d_A, A_desc,
               d_B_packed, B_packed_desc, // <--- Use the packed pointer AND descriptor
               &beta,
               d_C, C_desc,
               d_C, C_desc,
               &algo,
               workspace, workspaceSize,
               stream);
```

## Troubleshooting & Constraints

1.  **Dimensions:** IMMA kernels typically require M, N, K to be multiples of 16 (or even larger powers of 2 for best performance). Padding may be required.
2.  **Transpose Flags:** Even with packed layouts, the `cublasOperation_t` flags in `cublasLtMatmulDesc` might need to be specific (e.g., `CUBLAS_OP_T` for B is common for int8 weights). Check `cublasLtMatmulAlgoGetHeuristic` results.
3.  **Architecture:** `COL32_2R_4R4` is Ampere specific. Use `COL4_4R2_8C` for Turing compatibility.

## References

- [NVIDIA cuBLASLt Documentation - Matrix Layouts](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatrixlayoutattribute-t)
- [NVIDIA TensorRT Developer Guide - Int8](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#working-with-int8)
