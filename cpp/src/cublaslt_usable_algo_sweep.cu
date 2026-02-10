#include "cublaslt_gemm.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace accelsim::gemm
{
namespace
{

constexpr const char *kSummaryPrefix = "ACCELSIM_USABLE_ALGO_CASE ";

struct CliOptions
{
  int device_id{0};
  int n{1000};
  std::string variant{"AB"};
  int iters{50};
  int warmup{10};
  std::size_t max_workspace_bytes{64ull * 1024ull * 1024ull};
  int max_algo_ids{-1}; // <0 => no cap (request a large default)
  bool summary_json{false};
};

enum class Variant
{
  kAB,
  kABTView,
};

Variant parse_variant(const std::string &s)
{
  if (s == "AB")
    return Variant::kAB;
  if (s == "ABT_view")
    return Variant::kABTView;
  throw std::runtime_error("Invalid --variant (expected one of: AB, ABT_view): '" + s + "'");
}

void trans_from_variant(Variant v, cublasOperation_t &ta, cublasOperation_t &tb)
{
  ta = CUBLAS_OP_N;
  tb = (v == Variant::kABTView) ? CUBLAS_OP_T : CUBLAS_OP_N;
}

std::string order_to_string(cublasLtOrder_t order)
{
  return order == CUBLASLT_ORDER_ROW ? "row" : "col";
}

const char *cublas_status_to_string(cublasStatus_t status)
{
  switch (status)
  {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
    default:
      return "CUBLAS_STATUS_<unknown>";
  }
}

int parse_int(const std::string &arg, const char *what)
{
  try
  {
    return std::stoi(arg);
  }
  catch (...)
  {
    throw std::runtime_error(std::string("Invalid ") + what + ": '" + arg + "'");
  }
}

std::size_t parse_size(const std::string &arg, const char *what)
{
  const long long v = std::stoll(arg);
  if (v < 0)
  {
    throw std::runtime_error(std::string("Invalid ") + what + " (must be >=0): '" + arg + "'");
  }
  return static_cast<std::size_t>(v);
}

CliOptions parse_cli(int argc, char **argv)
{
  CliOptions cli{};
  for (int i = 1; i < argc; ++i)
  {
    const std::string a = argv[i];
    auto need = [&](const char *flag) -> std::string {
      if (i + 1 >= argc)
      {
        throw std::runtime_error(std::string("Missing value for ") + flag);
      }
      return argv[++i];
    };

    if (a == "--device")
    {
      cli.device_id = parse_int(need("--device"), "--device");
      continue;
    }
    if (a == "--n")
    {
      cli.n = parse_int(need("--n"), "--n");
      continue;
    }
    if (a == "--variant")
    {
      cli.variant = need("--variant");
      (void)parse_variant(cli.variant);
      continue;
    }
    if (a == "--iters")
    {
      cli.iters = parse_int(need("--iters"), "--iters");
      continue;
    }
    if (a == "--warmup")
    {
      cli.warmup = parse_int(need("--warmup"), "--warmup");
      continue;
    }
    if (a == "--max-workspace-bytes")
    {
      cli.max_workspace_bytes = parse_size(need("--max-workspace-bytes"), "--max-workspace-bytes");
      continue;
    }
    if (a == "--max-algo-ids")
    {
      cli.max_algo_ids = parse_int(need("--max-algo-ids"), "--max-algo-ids");
      continue;
    }
    if (a == "--summary-json")
    {
      cli.summary_json = true;
      continue;
    }
    if (a == "--help" || a == "-h")
    {
      throw std::runtime_error("help");
    }
    throw std::runtime_error("Unknown flag: " + a);
  }

  if (cli.n <= 0)
  {
    throw std::runtime_error("--n must be >0");
  }
  if (cli.iters <= 0)
  {
    throw std::runtime_error("--iters must be >0");
  }
  if (cli.warmup < 0)
  {
    throw std::runtime_error("--warmup must be >=0");
  }
  if (cli.max_algo_ids == 0)
  {
    throw std::runtime_error("--max-algo-ids must be omitted, <0, or >0");
  }
  return cli;
}

void print_usage(const char *prog)
{
  std::cerr << "Usage: " << prog << " --n <N> --variant <AB|ABT_view> [options]\n\n"
            << "Options:\n"
            << "  --device <id>              CUDA device index (default: 0)\n"
            << "  --iters <count>            Timed iterations per candidate (default: 50)\n"
            << "  --warmup <count>           Warmup iterations per candidate (default: 10)\n"
            << "  --max-workspace-bytes <B>  Fixed workspace policy for AlgoCheck (default: 64MiB)\n"
            << "  --max-algo-ids <K>          Optional cap on evaluated algo IDs per case\n"
            << "  --summary-json             Emit one JSON summary line (for Python parsing)\n";
}

MatrixDims make_matrix_dims(int rows, int cols, cublasLtOrder_t order)
{
  const int ld = (order == CUBLASLT_ORDER_COL) ? rows : cols;
  return MatrixDims{rows, cols, ld};
}

std::vector<std::int8_t> make_host_matrix_int8(int rows, int cols, int seed)
{
  std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
  std::uniform_int_distribution<int> dist(-4, 4);
  std::vector<std::int8_t> out(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
  for (auto &v : out)
  {
    v = static_cast<std::int8_t>(dist(rng));
  }
  return out;
}

template <typename T>
bool try_get_algo_attr(const cublasLtMatmulAlgo_t &algo, cublasLtMatmulAlgoConfigAttributes_t attr, T &out)
{
  std::size_t written = 0;
  const cublasStatus_t st = cublasLtMatmulAlgoConfigGetAttribute(&algo, attr, &out, sizeof(out), &written);
  return st == CUBLAS_STATUS_SUCCESS && written == sizeof(out);
}

nlohmann::json algo_config_to_json(const cublasLtMatmulAlgo_t &algo, std::size_t required_workspace_bytes, float waves_count)
{
  CublasLtAlgoConfig cfg{};
  (void)try_get_algo_attr(algo, CUBLASLT_ALGO_CONFIG_ID, cfg.id);
  (void)try_get_algo_attr(algo, CUBLASLT_ALGO_CONFIG_TILE_ID, cfg.tile_id);
  (void)try_get_algo_attr(algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, cfg.splitk_num);
  (void)try_get_algo_attr(algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, cfg.reduction_scheme);
  (void)try_get_algo_attr(algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, cfg.cta_swizzling);
  (void)try_get_algo_attr(algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, cfg.custom_option);
  (void)try_get_algo_attr(algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, cfg.stages_id);
  (void)try_get_algo_attr(algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, cfg.inner_shape_id);
  (void)try_get_algo_attr(algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, cfg.cluster_shape_id);
  cfg.required_workspace_bytes = required_workspace_bytes;
  cfg.waves_count              = static_cast<std::int32_t>(waves_count);

  return {
    {"algo_id", cfg.id},
    {"tile_id", cfg.tile_id},
    {"stages_id", cfg.stages_id},
    {"splitk_num", cfg.splitk_num},
    {"reduction_scheme", cfg.reduction_scheme},
    {"cta_swizzling", cfg.cta_swizzling},
    {"custom_option", cfg.custom_option},
    {"inner_shape_id", cfg.inner_shape_id},
    {"cluster_shape_id", cfg.cluster_shape_id},
    {"required_workspace_bytes", cfg.required_workspace_bytes},
    {"waves_count", waves_count},
  };
}

struct TimedResult
{
  bool ok{false};
  double time_us{0.0};
  std::string error;
};

TimedResult time_algo(cudaStream_t stream,
                      cublasLtHandle_t handle,
                      cublasLtMatmulDesc_t op_desc,
                      cublasLtMatrixLayout_t a_layout,
                      cublasLtMatrixLayout_t b_layout,
                      cublasLtMatrixLayout_t c_layout,
                      const cublasLtMatmulAlgo_t &algo,
                      void *workspace,
                      std::size_t workspace_bytes,
                      const void *a_dev,
                      const void *b_dev,
                      void *c_dev,
                      const void *alpha,
                      const void *beta,
                      int warmup,
                      int iters)
{
  for (int i = 0; i < warmup; ++i)
  {
    const cublasStatus_t st = cublasLtMatmul(handle,
                                            op_desc,
                                            alpha,
                                            a_dev,
                                            a_layout,
                                            b_dev,
                                            b_layout,
                                            beta,
                                            c_dev,
                                            c_layout,
                                            c_dev,
                                            c_layout,
                                            &algo,
                                            workspace,
                                            workspace_bytes,
                                            stream);
    if (st != CUBLAS_STATUS_SUCCESS)
    {
      return TimedResult{false, 0.0, std::string("cublasLtMatmul(warmup): ") + cublas_status_to_string(st)};
    }
  }

  cudaEvent_t start{};
  cudaEvent_t stop{};
  CublasLtGemmPlan::CheckCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
  CublasLtGemmPlan::CheckCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

  CublasLtGemmPlan::CheckCuda(cudaEventRecord(start, stream), "cudaEventRecord(start)");
  for (int i = 0; i < iters; ++i)
  {
    const cublasStatus_t st = cublasLtMatmul(handle,
                                            op_desc,
                                            alpha,
                                            a_dev,
                                            a_layout,
                                            b_dev,
                                            b_layout,
                                            beta,
                                            c_dev,
                                            c_layout,
                                            c_dev,
                                            c_layout,
                                            &algo,
                                            workspace,
                                            workspace_bytes,
                                            stream);
    if (st != CUBLAS_STATUS_SUCCESS)
    {
      (void)cudaEventDestroy(start);
      (void)cudaEventDestroy(stop);
      return TimedResult{false, 0.0, std::string("cublasLtMatmul(timed): ") + cublas_status_to_string(st)};
    }
  }
  CublasLtGemmPlan::CheckCuda(cudaEventRecord(stop, stream), "cudaEventRecord(stop)");
  CublasLtGemmPlan::CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

  float elapsed_ms = 0.0f;
  CublasLtGemmPlan::CheckCuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
  (void)cudaEventDestroy(start);
  (void)cudaEventDestroy(stop);

  const double avg_ms = static_cast<double>(elapsed_ms) / static_cast<double>(iters);
  return TimedResult{true, avg_ms * 1000.0, ""};
}

nlohmann::json run_case(const CliOptions &cli)
{
  // Fixed experiment settings: int8,int8->int32, row-major A/B/C, square M=N=K.
  const int n = cli.n;
  const GemmDims dims{n, n, n};
  const GemmTypes types{CUDA_R_8I, CUDA_R_8I, CUDA_R_32I, CUBLAS_COMPUTE_32I, CUDA_R_32I};
  const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  const MatrixDims a_dims     = make_matrix_dims(n, n, order);
  const MatrixDims b_dims     = make_matrix_dims(n, n, order);
  const MatrixDims c_dims     = make_matrix_dims(n, n, order);

  const Variant variant_enum = parse_variant(cli.variant);
  cublasOperation_t trans_a{};
  cublasOperation_t trans_b{};
  trans_from_variant(variant_enum, trans_a, trans_b);

  // Device selection + basic info.
  CublasLtGemmPlan::CheckCuda(cudaSetDevice(cli.device_id), "cudaSetDevice");
  int dev_id = 0;
  CublasLtGemmPlan::CheckCuda(cudaGetDevice(&dev_id), "cudaGetDevice");
  cudaDeviceProp prop{};
  CublasLtGemmPlan::CheckCuda(cudaGetDeviceProperties(&prop, dev_id), "cudaGetDeviceProperties");

  const auto a_host = make_host_matrix_int8(n, n, /*seed=*/123);
  const auto b_host = make_host_matrix_int8(n, n, /*seed=*/124);

  const std::size_t a_bytes = static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * sizeof(std::int8_t);
  const std::size_t b_bytes = static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * sizeof(std::int8_t);
  const std::size_t c_bytes = static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * sizeof(std::int32_t);

  void *a_dev = nullptr;
  void *b_dev = nullptr;
  void *c_dev = nullptr;
  void *workspace = nullptr;
  cudaStream_t stream{};

  cublasLtHandle_t handle{};
  cublasLtMatmulDesc_t op_desc{};
  cublasLtMatrixLayout_t a_layout{};
  cublasLtMatrixLayout_t b_layout{};
  cublasLtMatrixLayout_t c_layout{};

  auto destroy_all = [&]() {
    if (workspace)
    {
      cudaFree(workspace);
      workspace = nullptr;
    }
    if (a_dev)
    {
      cudaFree(a_dev);
      a_dev = nullptr;
    }
    if (b_dev)
    {
      cudaFree(b_dev);
      b_dev = nullptr;
    }
    if (c_dev)
    {
      cudaFree(c_dev);
      c_dev = nullptr;
    }
    if (stream)
    {
      cudaStreamDestroy(stream);
      stream = nullptr;
    }
    if (a_layout)
    {
      cublasLtMatrixLayoutDestroy(a_layout);
      a_layout = nullptr;
    }
    if (b_layout)
    {
      cublasLtMatrixLayoutDestroy(b_layout);
      b_layout = nullptr;
    }
    if (c_layout)
    {
      cublasLtMatrixLayoutDestroy(c_layout);
      c_layout = nullptr;
    }
    if (op_desc)
    {
      cublasLtMatmulDescDestroy(op_desc);
      op_desc = nullptr;
    }
    if (handle)
    {
      cublasLtDestroy(handle);
      handle = nullptr;
    }
  };

  try
  {
    CublasLtGemmPlan::CheckCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    CublasLtGemmPlan::CheckCuda(cudaMalloc(&a_dev, a_bytes), "cudaMalloc(A)");
    CublasLtGemmPlan::CheckCuda(cudaMalloc(&b_dev, b_bytes), "cudaMalloc(B)");
    CublasLtGemmPlan::CheckCuda(cudaMalloc(&c_dev, c_bytes), "cudaMalloc(C)");
    CublasLtGemmPlan::CheckCuda(cudaMalloc(&workspace, cli.max_workspace_bytes), "cudaMalloc(workspace)");

    CublasLtGemmPlan::CheckCuda(cudaMemcpy(a_dev, a_host.data(), a_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(A)");
    CublasLtGemmPlan::CheckCuda(cudaMemcpy(b_dev, b_host.data(), b_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(B)");
    CublasLtGemmPlan::CheckCuda(cudaMemsetAsync(c_dev, 0, c_bytes, stream), "cudaMemsetAsync(C)");
    CublasLtGemmPlan::CheckCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(setup)");

    CublasLtGemmPlan::CheckLt(cublasLtCreate(&handle), "cublasLtCreate");

    CublasLtGemmPlan::CheckLt(cublasLtMatmulDescCreate(&op_desc, types.compute_type, types.scale_type),
                              "cublasLtMatmulDescCreate");
    CublasLtGemmPlan::CheckLt(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)),
                              "cublasLtMatmulDescSetAttribute(TRANSA)");
    CublasLtGemmPlan::CheckLt(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)),
                              "cublasLtMatmulDescSetAttribute(TRANSB)");

    CublasLtGemmPlan::CheckLt(cublasLtMatrixLayoutCreate(&a_layout, types.a_type, a_dims.rows, a_dims.cols, a_dims.ld),
                              "cublasLtMatrixLayoutCreate(A)");
    CublasLtGemmPlan::CheckLt(cublasLtMatrixLayoutCreate(&b_layout, types.b_type, b_dims.rows, b_dims.cols, b_dims.ld),
                              "cublasLtMatrixLayoutCreate(B)");
    CublasLtGemmPlan::CheckLt(cublasLtMatrixLayoutCreate(&c_layout, types.c_type, c_dims.rows, c_dims.cols, c_dims.ld),
                              "cublasLtMatrixLayoutCreate(C)");

    // Set row-major order explicitly.
    CublasLtGemmPlan::CheckLt(
      cublasLtMatrixLayoutSetAttribute(a_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
      "cublasLtMatrixLayoutSetAttribute(A, ORDER)");
    CublasLtGemmPlan::CheckLt(
      cublasLtMatrixLayoutSetAttribute(b_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
      "cublasLtMatrixLayoutSetAttribute(B, ORDER)");
    CublasLtGemmPlan::CheckLt(
      cublasLtMatrixLayoutSetAttribute(c_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
      "cublasLtMatrixLayoutSetAttribute(C, ORDER)");

    // Baseline heuristic-selected algorithm.
    nlohmann::json heuristic = nlohmann::json::object();
    {
      cublasLtMatmulPreference_t pref{};
      CublasLtGemmPlan::CheckLt(cublasLtMatmulPreferenceCreate(&pref), "cublasLtMatmulPreferenceCreate");
      CublasLtGemmPlan::CheckLt(
        cublasLtMatmulPreferenceSetAttribute(pref,
                                             CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &cli.max_workspace_bytes,
                                             sizeof(cli.max_workspace_bytes)),
        "cublasLtMatmulPreferenceSetAttribute(MAX_WORKSPACE_BYTES)");

      cublasLtMatmulHeuristicResult_t heur{};
      int found = 0;
      const cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(handle,
                                                               op_desc,
                                                               a_layout,
                                                               b_layout,
                                                               c_layout,
                                                               c_layout,
                                                               pref,
                                                               1,
                                                               &heur,
                                                               &found);
      (void)cublasLtMatmulPreferenceDestroy(pref);

      heuristic["status"] = cublas_status_to_string(st);
      heuristic["status_code"] = static_cast<int>(st);
      heuristic["found"] = found;
      heuristic["state"] = cublas_status_to_string(heur.state);
      heuristic["state_code"] = static_cast<int>(heur.state);

      if (st == CUBLAS_STATUS_SUCCESS && found > 0 && heur.state == CUBLAS_STATUS_SUCCESS)
      {
        cublasLtMatmulHeuristicResult_t chk{};
        const cublasStatus_t st_chk = cublasLtMatmulAlgoCheck(handle,
                                                              op_desc,
                                                              a_layout,
                                                              b_layout,
                                                              c_layout,
                                                              c_layout,
                                                              &heur.algo,
                                                              &chk);
        heuristic["check_status"] = cublas_status_to_string(st_chk);
        heuristic["check_status_code"] = static_cast<int>(st_chk);
        heuristic["required_workspace_bytes"] = static_cast<std::size_t>(chk.workspaceSize);
        heuristic["waves_count"] = static_cast<double>(chk.wavesCount);

        heuristic["config"] = algo_config_to_json(heur.algo,
                                                  static_cast<std::size_t>(chk.workspaceSize),
                                                  static_cast<float>(chk.wavesCount));

        const std::int32_t alpha = 1;
        const std::int32_t beta  = 0;
        const TimedResult timed = time_algo(stream,
                                            handle,
                                            op_desc,
                                            a_layout,
                                            b_layout,
                                            c_layout,
                                            heur.algo,
                                            workspace,
                                            cli.max_workspace_bytes,
                                            a_dev,
                                            b_dev,
                                            c_dev,
                                            &alpha,
                                            &beta,
                                            cli.warmup,
                                            cli.iters);
        heuristic["timing_ok"] = timed.ok;
        if (timed.ok)
        {
          heuristic["time_us"] = timed.time_us;
        }
        else
        {
          heuristic["time_us"] = nullptr;
          heuristic["timing_error"] = timed.error;
        }
      }
      else
      {
        heuristic["config"] = nullptr;
        heuristic["time_us"] = nullptr;
      }
    }

    // Candidate algo-id enumeration.
    const int requested_ids = (cli.max_algo_ids > 0) ? cli.max_algo_ids : 10000;
    std::vector<int> algo_ids(static_cast<std::size_t>(requested_ids));
    int returned_ids = 0;
    const cublasStatus_t st_ids = cublasLtMatmulAlgoGetIds(handle,
                                                           types.compute_type,
                                                           types.scale_type,
                                                           types.a_type,
                                                           types.b_type,
                                                           types.c_type,
                                                           types.c_type,
                                                           requested_ids,
                                                           algo_ids.data(),
                                                           &returned_ids);
    if (st_ids != CUBLAS_STATUS_SUCCESS)
    {
      throw std::runtime_error(std::string("cublasLtMatmulAlgoGetIds failed: ") + cublas_status_to_string(st_ids));
    }
    if (returned_ids < 0)
    {
      throw std::runtime_error("cublasLtMatmulAlgoGetIds returned negative count");
    }
    algo_ids.resize(static_cast<std::size_t>(returned_ids));

    // Evaluate one best config per algo_id via limited-by-algo-id heuristic + AlgoCheck.
    nlohmann::json candidates = nlohmann::json::array();

    cublasLtMatmulPreference_t pref_limited{};
    CublasLtGemmPlan::CheckLt(cublasLtMatmulPreferenceCreate(&pref_limited), "cublasLtMatmulPreferenceCreate(limited)");
    const std::uint32_t search_mode = CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID;
    CublasLtGemmPlan::CheckLt(
      cublasLtMatmulPreferenceSetAttribute(pref_limited,
                                           CUBLASLT_MATMUL_PREF_SEARCH_MODE,
                                           &search_mode,
                                           sizeof(search_mode)),
      "cublasLtMatmulPreferenceSetAttribute(SEARCH_MODE)");
    CublasLtGemmPlan::CheckLt(
      cublasLtMatmulPreferenceSetAttribute(pref_limited,
                                           CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                           &cli.max_workspace_bytes,
                                           sizeof(cli.max_workspace_bytes)),
      "cublasLtMatmulPreferenceSetAttribute(MAX_WORKSPACE_BYTES)");

    const std::int32_t alpha = 1;
    const std::int32_t beta  = 0;

    for (const int algo_id : algo_ids)
    {
      nlohmann::json cand = nlohmann::json::object();
      cand["algo_id"] = algo_id;

      cublasLtMatmulHeuristicResult_t heur{};
      const cublasStatus_t st_init = cublasLtMatmulAlgoInit(handle,
                                                            types.compute_type,
                                                            types.scale_type,
                                                            types.a_type,
                                                            types.b_type,
                                                            types.c_type,
                                                            types.c_type,
                                                            algo_id,
                                                            &heur.algo);
      if (st_init != CUBLAS_STATUS_SUCCESS)
      {
        cand["usable"] = false;
        cand["time_us"] = nullptr;
        cand["notes"] = std::string("AlgoInit failed: ") + cublas_status_to_string(st_init);
        candidates.push_back(cand);
        continue;
      }

      int found = 0;
      const cublasStatus_t st_heur = cublasLtMatmulAlgoGetHeuristic(handle,
                                                                    op_desc,
                                                                    a_layout,
                                                                    b_layout,
                                                                    c_layout,
                                                                    c_layout,
                                                                    pref_limited,
                                                                    1,
                                                                    &heur,
                                                                    &found);
      cand["heuristic_status"] = cublas_status_to_string(st_heur);
      cand["heuristic_status_code"] = static_cast<int>(st_heur);
      cand["heuristic_found"] = found;
      cand["heuristic_state"] = cublas_status_to_string(heur.state);
      cand["heuristic_state_code"] = static_cast<int>(heur.state);

      if (!(st_heur == CUBLAS_STATUS_SUCCESS && found > 0 && heur.state == CUBLAS_STATUS_SUCCESS))
      {
        cand["usable"] = false;
        cand["time_us"] = nullptr;
        cand["notes"] = "No heuristic config for this algo_id";
        candidates.push_back(cand);
        continue;
      }

      cublasLtMatmulHeuristicResult_t chk{};
      const cublasStatus_t st_chk = cublasLtMatmulAlgoCheck(handle,
                                                            op_desc,
                                                            a_layout,
                                                            b_layout,
                                                            c_layout,
                                                            c_layout,
                                                            &heur.algo,
                                                            &chk);
      cand["check_status"] = cublas_status_to_string(st_chk);
      cand["check_status_code"] = static_cast<int>(st_chk);
      cand["required_workspace_bytes"] = static_cast<std::size_t>(chk.workspaceSize);
      cand["waves_count"] = static_cast<double>(chk.wavesCount);

      const bool workspace_ok = static_cast<std::size_t>(chk.workspaceSize) <= cli.max_workspace_bytes;
      const bool usable = (st_chk == CUBLAS_STATUS_SUCCESS) && workspace_ok;
      cand["usable"] = usable;
      cand["config"] = algo_config_to_json(heur.algo,
                                           static_cast<std::size_t>(chk.workspaceSize),
                                           static_cast<float>(chk.wavesCount));

      if (!usable)
      {
        if (!workspace_ok)
        {
          cand["notes"] = "required workspace exceeds policy";
        }
        candidates.push_back(cand);
        continue;
      }

      const TimedResult timed = time_algo(stream,
                                          handle,
                                          op_desc,
                                          a_layout,
                                          b_layout,
                                          c_layout,
                                          heur.algo,
                                          workspace,
                                          cli.max_workspace_bytes,
                                          a_dev,
                                          b_dev,
                                          c_dev,
                                          &alpha,
                                          &beta,
                                          cli.warmup,
                                          cli.iters);
      if (timed.ok)
      {
        cand["time_us"] = timed.time_us;
      }
      else
      {
        cand["time_us"] = nullptr;
        cand["notes"] = timed.error;
      }
      candidates.push_back(cand);
    }
    (void)cublasLtMatmulPreferenceDestroy(pref_limited);

    // Assemble final JSON.
    nlohmann::json out = nlohmann::json::object();
    out["n"] = n;
    out["variant"] = cli.variant;
    out["orders"] = {{"a", order_to_string(order)}, {"b", order_to_string(order)}, {"c", order_to_string(order)}};
    out["dtype"] = {{"a", "int8"}, {"b", "int8"}, {"c", "int32"}, {"compute", "int32"}};
    out["device"] = {{"id", dev_id}, {"name", prop.name}, {"cc", std::to_string(prop.major) + "." + std::to_string(prop.minor)}};
    out["dims"] = {{"m", dims.m}, {"n", dims.n}, {"k", dims.k}};
    out["workspace_policy"] = {{"max_workspace_bytes", cli.max_workspace_bytes}};
    out["timing_policy"] = {{"warmup", cli.warmup}, {"iters", cli.iters}};
    out["algo_ids"] = {
      {"requested", requested_ids},
      {"returned", returned_ids},
      {"truncated", (returned_ids == requested_ids)},
    };
    out["heuristic"] = heuristic;
    out["candidates"] = candidates;
    destroy_all();
    return out;
  }
  catch (...)
  {
    destroy_all();
    throw;
  }
}

} // namespace
} // namespace accelsim::gemm

int main(int argc, char **argv)
{
  using namespace accelsim::gemm;

  CliOptions cli{};
  try
  {
    cli = parse_cli(argc, argv);
  }
  catch (const std::exception &e)
  {
    if (std::string(e.what()) != "help")
    {
      std::cerr << "ERROR: " << e.what() << "\n";
    }
    print_usage(argv[0]);
    return 2;
  }

  try
  {
    const nlohmann::json j = run_case(cli);
    if (cli.summary_json)
    {
      std::cout << kSummaryPrefix << j.dump() << "\n";
    }
    else
    {
      std::cout << j.dump(2) << "\n";
    }
    return 0;
  }
  catch (const std::exception &e)
  {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}

