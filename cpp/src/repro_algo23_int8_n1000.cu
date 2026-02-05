#include "cublaslt_gemm.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace accelsim::gemm
{
namespace
{

constexpr int kN = 1000;

/**
 * @brief Create a deterministic int8 host matrix with small integer values.
 *
 * The distribution is intentionally narrow to reduce overflow risk in int32
 * accumulation while still exercising real int8 data paths.
 *
 * Storage is row-major contiguous.
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param seed RNG seed to make runs reproducible.
 * @return Host buffer of size rows*cols in row-major order.
 */
std::vector<std::int8_t> make_host_matrix_int8(int rows, int cols, int seed)
{
  std::vector<std::int8_t> out(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(-5, 5);
  for (auto &v : out)
  {
    v = static_cast<std::int8_t>(dist(rng));
  }
  return out;
}

/**
 * @brief Allocate a raw device buffer of @p bytes using cudaMalloc.
 *
 * This is intentionally minimal: the repro wants cudaMalloc semantics (alignment,
 * placement) and fails fast on any allocation error.
 *
 * @param bytes Number of bytes to allocate on the current CUDA device.
 * @return Pointer to device memory.
 * @throws std::runtime_error on CUDA failure.
 */
void *device_alloc_bytes(std::size_t bytes)
{
  void *ptr = nullptr;
  CublasLtGemmPlan::CheckCuda(cudaMalloc(&ptr, bytes), "cudaMalloc");
  return ptr;
}

struct TimedRun
{
  bool ok{false};
  std::string label;
  std::string notes;
  float avg_ms{0.0f};
  CublasLtAlgoConfig algo{};
};

/**
 * @brief Build a cuBLASLt plan and time repeated executions of a GEMM on a single CUDA stream.
 *
 * Timing uses CUDA events around the loop of @p iters calls to plan.Run(). A warmup phase
 * (@p warmup_iters) is executed and synchronized before measurement.
 *
 * If @p opts has an enabled algo override, plan construction will attempt to force that
 * algorithm/config. If the forced algorithm is incompatible for the given problem
 * (shapes/types/layout/transpose), plan creation fails and this function returns ok=false
 * with the exception string in notes (reported as "NA" in the output table).
 *
 * @param label Human-readable label for this run (used in the printed report).
 * @param dims GEMM dimensions (M,N,K).
 * @param types cuBLASLt datatype + compute configuration.
 * @param a_dims A matrix layout (rows, cols, ld) matching the storage buffer passed in.
 * @param b_dims B matrix layout (rows, cols, ld) matching the storage buffer passed in.
 * @param c_dims C matrix layout (rows, cols, ld) matching the storage buffer passed in.
 * @param trans_a cuBLAS op for A (N or T).
 * @param trans_b cuBLAS op for B (N or T).
 * @param opts Plan options (workspace and optional forced algo config).
 * @param stream CUDA stream used for all launches and timing.
 * @param a_dev Device pointer to A storage buffer.
 * @param b_dev Device pointer to B storage buffer.
 * @param c_dev Device pointer to C storage buffer.
 * @param alpha Pointer to alpha scalar (host pointer in this repro).
 * @param beta Pointer to beta scalar (host pointer in this repro).
 * @param warmup_iters Number of warmup iterations before timing.
 * @param iters Number of timed iterations.
 * @return Timing + selected algorithm metadata (or NA on failure).
 */
TimedRun time_plan(const std::string &label,
                   const GemmDims &dims,
                   const GemmTypes &types,
                   const MatrixDims &a_dims,
                   const MatrixDims &b_dims,
                   const MatrixDims &c_dims,
                   cublasOperation_t trans_a,
                   cublasOperation_t trans_b,
                   const GemmPlanOptions &opts,
                   cudaStream_t stream,
                   const void *a_dev,
                   const void *b_dev,
                   void *c_dev,
                   const void *alpha,
                   const void *beta,
                   int warmup_iters,
                   int iters)
{
  TimedRun out{};
  out.label = label;

  try
  {
    const CublasLtGemmPlan plan(dims, types, a_dims, b_dims, c_dims, trans_a, trans_b, opts);
    out.algo = plan.SelectedAlgo();

    // Warmup.
    for (int i = 0; i < warmup_iters; ++i)
    {
      plan.Run(stream, a_dev, b_dev, c_dev, alpha, beta);
    }
    CublasLtGemmPlan::CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(warmup)");

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CublasLtGemmPlan::CheckCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    CublasLtGemmPlan::CheckCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    CublasLtGemmPlan::CheckCuda(cudaEventRecord(start, stream), "cudaEventRecord(start)");
    for (int i = 0; i < iters; ++i)
    {
      plan.Run(stream, a_dev, b_dev, c_dev, alpha, beta);
    }
    CublasLtGemmPlan::CheckCuda(cudaEventRecord(stop, stream), "cudaEventRecord(stop)");
    CublasLtGemmPlan::CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float total_ms = 0.0f;
    CublasLtGemmPlan::CheckCuda(cudaEventElapsedTime(&total_ms, start, stop), "cudaEventElapsedTime");
    out.avg_ms = total_ms / static_cast<float>(iters);
    out.ok     = true;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  catch (const std::exception &e)
  {
    out.ok    = false;
    out.notes = e.what();
  }

  return out;
}

/**
 * @brief Print a single row of the results table.
 *
 * If @p r.ok is false, prints "NA" plus the failure reason (typically a cuBLASLt status
 * from cublasLtMatmulAlgoCheck when an algo override is incompatible).
 *
 * @param r Result of a timed run.
 */
void print_row(const TimedRun &r)
{
  std::cout << std::left << std::setw(24) << r.label;
  if (!r.ok)
  {
    std::cout << " NA"
              << "  (reason: " << r.notes << ")\n";
    return;
  }

  std::cout << " " << std::right << std::setw(8) << std::fixed << std::setprecision(4) << r.avg_ms << " ms"
            << "  algo=" << r.algo.id << " tile=" << r.algo.tile_id << " stages=" << r.algo.stages_id
            << " splitk=" << r.algo.splitk_num << "\n";
}

} // namespace
} // namespace accelsim::gemm

/**
 * @brief Reproduce the "square N=1000 int8" algo flip where ABT_view selects algo 23 and is much faster.
 *
 * What this program does:
 * - Builds three baseline plans using heuristic selection: AB, ATB_view, ABT_view.
 * - Confirms ABT_view heuristically selects algo_id=23 (as observed in the sweep report)
 *   and prints the measured timing and speedup.
 * - Attempts to force the exact algo 23 config into AB and ATB_view; prints NA if cuBLASLt
 *   rejects it (via cublasLtMatmulAlgoCheck), otherwise reports timing.
 * - As a control, tries forcing the "slow" algo 64 config into ABT_view to show that ABT_view
 *   can run with algo 64 but is slower than algo 23.
 *
 * Configuration:
 * - Default iterations are chosen to reduce noise on fast kernels.
 * - Override with environment variables:
 *   - ACCELSIM_REPRO_ITERS
 *   - ACCELSIM_REPRO_WARMUP
 * - Optionally pass a CUDA device index as argv[1].
 *
 * @return 0 on success. Returns 2 if the key phenomenon does not reproduce (ABT_view heuristic does not pick algo 23).
 */
int main(int argc, char **argv)
{
  using namespace accelsim::gemm;

  int iters        = 2000;
  int warmup_iters = 200;
  if (const char *s = std::getenv("ACCELSIM_REPRO_ITERS"))
    iters = std::max(1, std::atoi(s));
  if (const char *s = std::getenv("ACCELSIM_REPRO_WARMUP"))
    warmup_iters = std::max(0, std::atoi(s));

  // Optionally select device via argv[1] as an integer device index.
  if (argc >= 2)
  {
    const int dev = std::atoi(argv[1]);
    cudaSetDevice(dev);
  }

  int dev_id = 0;
  CublasLtGemmPlan::CheckCuda(cudaGetDevice(&dev_id), "cudaGetDevice");
  cudaDeviceProp prop{};
  CublasLtGemmPlan::CheckCuda(cudaGetDeviceProperties(&prop, dev_id), "cudaGetDeviceProperties");
  std::cout << "Repro: N=1000 int8 GEMM variants (focus: ABT_view algo 23)\n";
  std::cout << "GPU: " << prop.name << " (cc " << prop.major << "." << prop.minor << ")\n";
  std::cout << "iters=" << iters << " warmup=" << warmup_iters << "\n\n";

  const int n = kN;
  const GemmDims dims{n, n, n};

  const GemmTypes types{CUDA_R_8I, CUDA_R_8I, CUDA_R_32I, CUBLAS_COMPUTE_32I, CUDA_R_32I};

  // Storage shapes match the square suite in gemm_transpose_bench:
  // A: (M,K), B: (K,N), C: (M,N), row-major.
  const MatrixDims a_dims{n, n, n};
  const MatrixDims b_dims{n, n, n};
  const MatrixDims c_dims{n, n, n};

  const auto a_host = make_host_matrix_int8(n, n, /*seed=*/123);
  const auto b_host = make_host_matrix_int8(n, n, /*seed=*/124);

  const std::size_t a_bytes = static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * sizeof(std::int8_t);
  const std::size_t b_bytes = static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * sizeof(std::int8_t);
  const std::size_t c_bytes = static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * sizeof(std::int32_t);

  void *a_dev = nullptr;
  void *b_dev = nullptr;
  void *c_dev = nullptr;
  cudaStream_t stream{};

  try
  {
    CublasLtGemmPlan::CheckCuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    a_dev = device_alloc_bytes(a_bytes);
    b_dev = device_alloc_bytes(b_bytes);
    c_dev = device_alloc_bytes(c_bytes);

    CublasLtGemmPlan::CheckCuda(cudaMemcpy(a_dev, a_host.data(), a_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(A)");
    CublasLtGemmPlan::CheckCuda(cudaMemcpy(b_dev, b_host.data(), b_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(B)");
    CublasLtGemmPlan::CheckCuda(cudaMemsetAsync(c_dev, 0, c_bytes, stream), "cudaMemsetAsync(C)");
    CublasLtGemmPlan::CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(setup)");

    const std::int32_t alpha = 1;
    const std::int32_t beta  = 0;

    // Algo configs taken from the sweep run:
    // reports/transpose_matmul/gemm_transpose_full_sweep_20260205_041151/results.json
    // - AB/ATB/... used algo 64: tile_id=20, stages_id=8
    // - ABT_view used algo 23: tile_id=18, stages_id=21
    CublasLtAlgoConfig cfg64{};
    cfg64.id               = 64;
    cfg64.tile_id          = 20;
    cfg64.splitk_num       = 1;
    cfg64.reduction_scheme = 0;
    cfg64.cta_swizzling    = 0;
    cfg64.custom_option    = 0;
    cfg64.stages_id        = 8;
    cfg64.inner_shape_id   = 0;
    cfg64.cluster_shape_id = 0;
    cfg64.waves_count      = 0;

    CublasLtAlgoConfig cfg23{};
    cfg23.id               = 23;
    cfg23.tile_id          = 18;
    cfg23.splitk_num       = 1;
    cfg23.reduction_scheme = 0;
    cfg23.cta_swizzling    = 0;
    cfg23.custom_option    = 0;
    cfg23.stages_id        = 21;
    cfg23.inner_shape_id   = 0;
    cfg23.cluster_shape_id = 0;
    cfg23.waves_count      = 0;

    GemmPlanOptions heuristic_opts{};
    heuristic_opts.max_workspace_bytes = 64ull * 1024ull * 1024ull;
    heuristic_opts.order               = CUBLASLT_ORDER_ROW;

    auto forced_opts = [&](const CublasLtAlgoConfig &cfg) {
      GemmPlanOptions o{};
      o.max_workspace_bytes         = heuristic_opts.max_workspace_bytes;
      o.order                       = heuristic_opts.order;
      o.algo_override.enabled       = true;
      o.algo_override.config        = cfg;
      return o;
    };

    std::cout << "Heuristic selections (baseline):\n";
    const auto r_ab = time_plan("AB (heuristic)",
                                dims, types, a_dims, b_dims, c_dims,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                heuristic_opts,
                                stream, a_dev, b_dev, c_dev, &alpha, &beta, warmup_iters, iters);
    const auto r_atb = time_plan("ATB_view (heuristic)",
                                 dims, types, a_dims, b_dims, c_dims,
                                 CUBLAS_OP_T, CUBLAS_OP_N,
                                 heuristic_opts,
                                 stream, a_dev, b_dev, c_dev, &alpha, &beta, warmup_iters, iters);
    const auto r_abt = time_plan("ABT_view (heuristic)",
                                 dims, types, a_dims, b_dims, c_dims,
                                 CUBLAS_OP_N, CUBLAS_OP_T,
                                 heuristic_opts,
                                 stream, a_dev, b_dev, c_dev, &alpha, &beta, warmup_iters, iters);
    print_row(r_ab);
    print_row(r_atb);
    print_row(r_abt);
    if (r_ab.ok && r_abt.ok)
    {
      std::cout << "ABT_view/AB speedup: " << std::fixed << std::setprecision(2) << (r_ab.avg_ms / r_abt.avg_ms) << "x\n";
    }
    std::cout << "\n";

    std::cout << "Force algo 23 into other transpose modes (if supported):\n";
    const auto r_ab_force23 = time_plan("AB forced algo23",
                                        dims, types, a_dims, b_dims, c_dims,
                                        CUBLAS_OP_N, CUBLAS_OP_N,
                                        forced_opts(cfg23),
                                        stream, a_dev, b_dev, c_dev, &alpha, &beta, warmup_iters, iters);
    const auto r_atb_force23 = time_plan("ATB forced algo23",
                                         dims, types, a_dims, b_dims, c_dims,
                                         CUBLAS_OP_T, CUBLAS_OP_N,
                                         forced_opts(cfg23),
                                         stream, a_dev, b_dev, c_dev, &alpha, &beta, warmup_iters, iters);
    const auto r_abt_force23 = time_plan("ABT forced algo23",
                                         dims, types, a_dims, b_dims, c_dims,
                                         CUBLAS_OP_N, CUBLAS_OP_T,
                                         forced_opts(cfg23),
                                         stream, a_dev, b_dev, c_dev, &alpha, &beta, warmup_iters, iters);
    print_row(r_ab_force23);
    print_row(r_atb_force23);
    print_row(r_abt_force23);
    std::cout << "\n";

    std::cout << "Control: try forcing algo 64 into ABT_view (if supported):\n";
    const auto r_abt_force64 = time_plan("ABT forced algo64",
                                         dims, types, a_dims, b_dims, c_dims,
                                         CUBLAS_OP_N, CUBLAS_OP_T,
                                         forced_opts(cfg64),
                                         stream, a_dev, b_dev, c_dev, &alpha, &beta, warmup_iters, iters);
    print_row(r_abt_force64);
    std::cout << "\n";

    // Exit with failure only if the key phenomenon does not reproduce.
    if (!r_abt.ok || r_abt.algo.id != 23)
    {
      std::cerr << "ERROR: ABT_view heuristic did not select algo 23; cannot reproduce.\n";
      return 2;
    }
  }
  catch (const std::exception &e)
  {
    std::cerr << "FATAL: " << e.what() << "\n";
    return 1;
  }

  if (c_dev)
    cudaFree(c_dev);
  if (b_dev)
    cudaFree(b_dev);
  if (a_dev)
    cudaFree(a_dev);
  if (stream)
    cudaStreamDestroy(stream);

  return 0;
}
