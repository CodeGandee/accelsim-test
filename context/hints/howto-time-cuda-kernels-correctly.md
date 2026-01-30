# How to Time CUDA Kernels Correctly

## Goal
Accurately measure the execution time of a CUDA kernel (e.g., `matmul(A, B)`) while **excluding** Host-to-Device (H2D) and Device-to-Host (D2H) memory transfer overhead.

## The Problem with CPU Timers
CUDA kernel launches are **asynchronous**. If you use standard CPU timers (like `std::chrono` or Python's `time.time()`), you will only measure the time it takes the CPU to *submit* the command to the GPU driver (microseconds), not the time the GPU spends *executing* it (milliseconds).

## The Solution: CUDA Events
Use `cudaEvent_t` to place markers directly into the GPU's command stream. This measures exactly how long the GPU worked, regardless of what the CPU was doing.

## Correct Implementation Pattern

1.  **Prepare Data First:** Move all data to the GPU *before* starting the timer.
2.  **Record Start:** Place the start event.
3.  **Launch Kernel:** Call your kernel or library function (cuBLAS, CUTLASS, etc.).
4.  **Record Stop:** Place the stop event.
5.  **Synchronize:** Wait for the stop event to complete.
6.  **Retrieve Data:** Move results back to Host *after* the timer stops.

### Code Example (C++)

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

void time_kernel_execution(cublasLtHandle_t handle, ...) {
    // --- Step 1: Preparation (Excluded from timing) ---
    // Allocations and H2D copies happen here.
    // ... cudaMemcpy(d_A, h_A, ...); 
    
    // Create events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup (Optional but recommended to load driver/caches)
    cublasLtMatmul(handle, ...);

    // --- Step 2: Start Timing ---
    // Record the start event in the stream
    cudaEventRecord(start, 0);

    // --- Step 3: Kernel Execution ---
    // Launch the kernel. This returns immediately to the CPU.
    cublasLtMatmul(handle, ...);

    // --- Step 4: Stop Timing ---
    // Record the stop event. This won't "happen" until the kernel is finished.
    cudaEventRecord(stop, 0);

    // --- Step 5: Synchronization ---
    // Block CPU execution until the stop event is recorded by the GPU.
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel Execution Time: %.3f ms\n", milliseconds);

    // --- Step 6: Retrieval (Excluded from timing) ---
    // cudaMemcpy(h_C, d_C, ...);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
```

## Key Takeaways
- **Never** include `cudaMemcpy` inside the measured block if you only want compute time.
- **Never** use CPU wall-clock time unless you explicitly call `cudaDeviceSynchronize()` before stopping the clock (which is less precise than Events).
- **Always** use `cudaEventRecord` and `cudaEventElapsedTime`.

## Robust Benchmarking Pattern (Warmup + Averaging)

For kernels that are very fast or to ensure consistent GPU clock states, you should measure the average of many iterations.

```cpp
void benchmark_kernel(cublasLtHandle_t handle, int num_iterations = 100) {
    // 1. Warmup (Untimed)
    // Run a few times to wake up the GPU and stabilize clocks.
    for (int i = 0; i < 10; ++i) {
        cublasLtMatmul(handle, ...);
    }
    cudaDeviceSynchronize(); // Ensure warmup is done

    // 2. Setup Events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 3. Record Start
    cudaEventRecord(start, 0);

    // 4. Batch Execution
    // Launch many times to saturate the GPU and minimize CPU launch noise
    for (int i = 0; i < num_iterations; ++i) {
        cublasLtMatmul(handle, ...);
    }

    // 5. Record Stop
    cudaEventRecord(stop, 0);

    // 6. Synchronize and Calculate
    cudaEventSynchronize(stop);
    
    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / num_iterations;

    printf("Avg Kernel Time: %.5f ms (over %d iterations)\n", avg_ms, num_iterations);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
```

## Advanced Note: What exactly is being measured?

When using CUDA Events as shown above, you are measuring the time interval on the **GPU Command Processor**.

*   **Included:**
    *   **Kernel Execution:** The actual math.
    *   **GPU Grid Launch Latency:** The hardware time required for the GPU to parse the command, allocate resources, and dispatch the first wave of thread blocks. This is an intrinsic part of the GPU's work and *should* be measured.
*   **Excluded:**
    *   **CPU Driver Latency:** The time the CPU takes to format the command and push it to the queue.
    *   **PCIe Transfer Time:** (If `cudaMemcpy` is placed outside the events).

If you strictly need "Arithmetic Execution Time Only" (Start of first warp to End of last warp, ignoring dispatch overhead), you cannot do this via C++ APIs. You must use a hardware profiler like **NVIDIA Nsight Compute (`ncu`)**.

### Using `ncu` for Pure Compute Time

To isolate the time the execution units were actually active:

```bash
ncu --metrics gpu__time_duration.sum ./your_app
```

*   **`gpu__time_duration.sum`**: Measures the nanoseconds the kernel was executing on the GPU.
*   **Warning:** `ncu` slows down application execution significantly. **Ignore** the wall-clock time of your app while profiling; only trust the numbers reported in the `ncu` output.

## Professional Libraries

For rigorous scientific benchmarking, consider using dedicated libraries instead of writing your own loops.

### 1. NVBench (Recommended for CUDA)
[NVBench](https://github.com/NVIDIA/nvbench) is a library built by NVIDIA on top of Google Benchmark. It automatically handles:
*   CUDA Event timing.
*   Throughput calculations (items/sec, memory bandwidth).
*   Parameter sweeping (testing matrix sizes 64, 128, 256...).

```cpp
#include <nvbench/nvbench.cuh>

void bm_matmul(nvbench::state& state) {
    // Setup...
    
    // Automatically handles timing, warmup, and averaging
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        cublasLtMatmul(handle, ...);
    });
}
NVBENCH_BENCH(bm_matmul);
```

### 2. Google Benchmark
[Google Benchmark](https://github.com/google/benchmark) is the C++ industry standard. To use it with CUDA, you must use the `UseManualTime()` feature and manually report `cudaEvent` times to avoid measuring CPU synchronization overhead.
