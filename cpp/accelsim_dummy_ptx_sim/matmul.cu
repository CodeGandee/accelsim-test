// Minimal CUDA program for Accel-Sim PTX-mode simulation.
//
// This program runs a tiny matmul and prints an unambiguous PASS/FAIL based on a CPU reference.

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>

__global__ void matmul_naive(const float* A, const float* B, float* C, int n) {
    int row = static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;
    int col = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) sum += A[row * n + k] * B[k * n + col];
    C[row * n + col] = sum;
}

static bool check_cuda(cudaError_t st, const char* what) {
    if (st == cudaSuccess) return true;
    std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(st));
    return false;
}

int main() {
    const int n = 16;
    const size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(float);

    std::vector<float> hA(static_cast<size_t>(n) * static_cast<size_t>(n));
    std::vector<float> hB(static_cast<size_t>(n) * static_cast<size_t>(n));
    std::vector<float> hC(static_cast<size_t>(n) * static_cast<size_t>(n));
    std::vector<float> hRef(static_cast<size_t>(n) * static_cast<size_t>(n));

    for (int i = 0; i < n * n; ++i) {
        hA[static_cast<size_t>(i)] = static_cast<float>((i % 13) - 6) * 0.1f;
        hB[static_cast<size_t>(i)] = static_cast<float>((i % 7) - 3) * 0.2f;
    }
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) sum += hA[static_cast<size_t>(row * n + k)] * hB[static_cast<size_t>(k * n + col)];
            hRef[static_cast<size_t>(row * n + col)] = sum;
        }
    }

    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;
    if (!check_cuda(cudaMalloc(&dA, bytes), "cudaMalloc(dA)")) return 1;
    if (!check_cuda(cudaMalloc(&dB, bytes), "cudaMalloc(dB)")) return 1;
    if (!check_cuda(cudaMalloc(&dC, bytes), "cudaMalloc(dC)")) return 1;

    if (!check_cuda(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy(H2D dA)")) return 1;
    if (!check_cuda(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy(H2D dB)")) return 1;
    if (!check_cuda(cudaMemset(dC, 0, bytes), "cudaMemset(dC)")) return 1;

    dim3 block(16, 16, 1);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y, 1);
    matmul_naive<<<grid, block>>>(dA, dB, dC, n);
    if (!check_cuda(cudaGetLastError(), "kernel launch")) return 1;
    if (!check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) return 1;

    if (!check_cuda(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(D2H dC)")) return 1;

    const float atol = 1e-3f;
    for (size_t i = 0; i < hC.size(); ++i) {
        if (std::fabs(hC[i] - hRef[i]) > atol) {
            std::puts("FAIL");
            std::printf("Mismatch at %zu: got=%f ref=%f\n", i, hC[i], hRef[i]);
            cudaFree(dA);
            cudaFree(dB);
            cudaFree(dC);
            return 1;
        }
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    std::puts("PASS");
    return 0;
}
