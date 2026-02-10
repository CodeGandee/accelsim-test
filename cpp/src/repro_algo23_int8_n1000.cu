#include "cublaslt_gemm.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace accelsim::gemm
{
namespace
{

constexpr int kN = 1000;

/**
 * @brief Simple RAII wrapper for an optional NVTX range.
 *
 * NVTX is used to make profiler filtering deterministic (Nsight Systems / Nsight Compute)
 * without relying on fragile kernel-name matching or invocation-index counting.
 */
class NvtxRange
{
public:
  /**
   * @brief Begin an NVTX range if enabled.
   *
   * @param enabled Whether to emit NVTX markers.
   * @param name Range label (copied by NVTX).
   */
  NvtxRange(bool enabled, const std::string &name) : m_enabled{enabled}
  {
    if (m_enabled)
    {
      nvtxRangePushA(name.c_str());
    }
  }

  NvtxRange(const NvtxRange &)            = delete;
  NvtxRange &operator=(const NvtxRange &) = delete;
  NvtxRange(NvtxRange &&)                 = delete;
  NvtxRange &operator=(NvtxRange &&)      = delete;

  /** @brief End the NVTX range if enabled. */
  ~NvtxRange()
  {
    if (m_enabled)
    {
      nvtxRangePop();
    }
  }

private:
  bool m_enabled{false};
};

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
std::vector<std::int8_t> make_logical_matrix_int8(int rows, int cols, int seed)
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
 * @brief Convert a logical row-major matrix into a specified storage order.
 *
 * The input is interpreted as row-major (index = i*cols + j). The output is packed
 * as either row-major or column-major based on @p order.
 */
std::vector<std::int8_t> pack_matrix_int8(const std::vector<std::int8_t> &logical, int rows, int cols, cublasLtOrder_t order)
{
  std::vector<std::int8_t> out(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      const std::size_t src = static_cast<std::size_t>(i) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(j);
      const std::size_t dst = (order == CUBLASLT_ORDER_ROW)
                                ? (static_cast<std::size_t>(i) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(j))
                                : (static_cast<std::size_t>(j) * static_cast<std::size_t>(rows) + static_cast<std::size_t>(i));
      out[dst] = logical[src];
    }
  }
  return out;
}

/**
 * @brief Symmetrize a square logical matrix in-place (A := A^T) using row-major indexing.
 */
void symmetrize_logical_square(std::vector<std::int8_t> &logical, int n)
{
  for (int i = 0; i < n; ++i)
  {
    for (int j = i + 1; j < n; ++j)
    {
      logical[static_cast<std::size_t>(j) * static_cast<std::size_t>(n) + static_cast<std::size_t>(i)] =
        logical[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j)];
    }
  }
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

enum class Variant
{
  kAll,
  kAB,
  kATBView,
  kABTView,
};

struct ForceAlgo
{
  bool enabled{false};
  CublasLtAlgoConfig cfg{};
};

struct CliOptions
{
  int device_id{-1};
  Variant variant{Variant::kAll};
  int iters{2000};
  int warmup_iters{200};
  bool enable_nvtx{false};
  bool enable_cuda_profiler_gating{false};
  bool symmetric_inputs{false};
  bool summary_json{false};
  cublasLtOrder_t order_a{CUBLASLT_ORDER_ROW};
  cublasLtOrder_t order_b{CUBLASLT_ORDER_ROW};
  cublasLtOrder_t order_c{CUBLASLT_ORDER_ROW};
  ForceAlgo force_algo{};
};

/**
 * @brief Return true if @p s is a non-empty string of digits.
 */
bool is_digits(const std::string &s)
{
  if (s.empty())
    return false;
  for (const unsigned char c : s)
  {
    if (c < '0' || c > '9')
      return false;
  }
  return true;
}

/**
 * @brief Parse a CLI integer argument.
 *
 * @param arg Argument string.
 * @param what Human-readable label used for errors.
 * @return Parsed integer value.
 */
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

/**
 * @brief Parse a CLI unsigned integer argument.
 *
 * @param arg Argument string.
 * @param what Human-readable label used for errors.
 * @return Parsed unsigned integer value.
 */
std::uint32_t parse_u32(const std::string &arg, const char *what)
{
  const int v = parse_int(arg, what);
  if (v < 0)
  {
    throw std::runtime_error(std::string("Invalid ") + what + " (must be >=0): '" + arg + "'");
  }
  return static_cast<std::uint32_t>(v);
}

/**
 * @brief Convert a user-facing string into a Variant enum.
 */
Variant parse_variant(const std::string &s)
{
  if (s == "all")
    return Variant::kAll;
  if (s == "AB")
    return Variant::kAB;
  if (s == "ATB_view")
    return Variant::kATBView;
  if (s == "ABT_view")
    return Variant::kABTView;
  throw std::runtime_error("Invalid --variant (expected one of: all, AB, ATB_view, ABT_view): '" + s + "'");
}

/**
 * @brief Convert a user-facing string into a cuBLASLt matrix layout order.
 */
cublasLtOrder_t parse_order(const std::string &s, const char *flag)
{
  if (s == "row")
    return CUBLASLT_ORDER_ROW;
  if (s == "col")
    return CUBLASLT_ORDER_COL;
  throw std::runtime_error(std::string("Invalid ") + flag + " (expected one of: row, col): '" + s + "'");
}

const char *order_to_string(cublasLtOrder_t order)
{
  return (order == CUBLASLT_ORDER_COL) ? "col" : "row";
}

const char *op_to_string(cublasOperation_t op)
{
  return (op == CUBLAS_OP_T) ? "T" : "N";
}

/**
 * @brief Format an algo config as a short human-readable string.
 */
std::string algo_to_string(const CublasLtAlgoConfig &cfg)
{
  return "algo=" + std::to_string(cfg.id) + " tile=" + std::to_string(cfg.tile_id)
         + " stages=" + std::to_string(cfg.stages_id) + " splitk=" + std::to_string(cfg.splitk_num);
}

/**
 * @brief Fill defaults for known algorithm IDs if fields are unset.
 *
 * This repro focuses on two observed configs from the sweep:
 * - algo 23: tile=18, stages=21, splitk=1
 * - algo 64: tile=20, stages=8,  splitk=1
 */
void apply_known_algo_defaults(CublasLtAlgoConfig &cfg)
{
  if (cfg.id == 23)
  {
    if (cfg.tile_id == 0)
      cfg.tile_id = 18;
    if (cfg.stages_id == 0)
      cfg.stages_id = 21;
    if (cfg.splitk_num == 0)
      cfg.splitk_num = 1;
  }
  if (cfg.id == 64)
  {
    if (cfg.tile_id == 0)
      cfg.tile_id = 20;
    if (cfg.stages_id == 0)
      cfg.stages_id = 8;
    if (cfg.splitk_num == 0)
      cfg.splitk_num = 1;
  }
}

/**
 * @brief Print usage information to stdout.
 */
void print_usage(const char *argv0)
{
  std::cout
    << "Usage: " << argv0 << " [options]\n"
    << "\n"
    << "Options:\n"
    << "  --variant {all|AB|ATB_view|ABT_view}   Select which GEMM variant(s) to run (default: all)\n"
    << "  --iters N                              Timed iterations per plan (default: 2000)\n"
    << "  --warmup N                             Warmup iterations per plan (default: 200)\n"
    << "  --device ID                            CUDA device index to use (default: current)\n"
    << "  --order {row|col}                      Shorthand: set --order-a/--order-b/--order-c\n"
    << "  --order-a {row|col}                    Storage order for A layout (default: row)\n"
    << "  --order-b {row|col}                    Storage order for B layout (default: row)\n"
    << "  --order-c {row|col}                    Storage order for C layout (default: row)\n"
    << "  --symmetric-inputs                     Generate symmetric A and B (A=A^T, B=B^T)\n"
    << "  --nvtx                                 Emit NVTX ranges around timed GEMM loops\n"
    << "  --cuda-profiler-gating                 Call cudaProfilerStart/Stop around timed GEMM loops\n"
    << "  --force-algo ID                        Force a cuBLASLt algorithm ID (enables AlgoCheck)\n"
    << "  --tile-id ID                           Force tile_id (optional; defaults for algo 23/64)\n"
    << "  --stages-id ID                         Force stages_id (optional; defaults for algo 23/64)\n"
    << "  --splitk N                             Force splitk_num (optional; defaults for algo 23/64)\n"
    << "  --summary-json                         Print one JSON summary line per run (for scripts)\n"
    << "  --help                                 Show this help\n"
    << "\n"
    << "Notes:\n"
    << "  - For deterministic profiling, prefer running a single variant with --nvtx\n"
    << "    and/or --cuda-profiler-gating, and set low --iters/--warmup.\n";
}

/**
 * @brief Parse command line arguments into CliOptions.
 *
 * @throws std::runtime_error on invalid arguments.
 */
CliOptions parse_cli(int argc, char **argv)
{
  CliOptions out{};

  if (const char *s = std::getenv("ACCELSIM_REPRO_ITERS"))
    out.iters = std::max(1, std::atoi(s));
  if (const char *s = std::getenv("ACCELSIM_REPRO_WARMUP"))
    out.warmup_iters = std::max(0, std::atoi(s));

  // Backwards-compatible positional device ID: ./repro <device_id>
  // If argv[1] is a plain integer and not an option, treat it as --device.
  int i = 1;
  if (argc >= 2)
  {
    const std::string a1 = argv[1];
    if (!a1.empty() && a1[0] != '-' && is_digits(a1))
    {
      out.device_id = parse_int(a1, "device id");
      i             = 2;
    }
  }

  for (; i < argc; ++i)
  {
    const std::string arg = argv[i];
    auto require_value = [&](const char *flag) -> std::string {
      if (i + 1 >= argc)
      {
        throw std::runtime_error(std::string("Missing value for ") + flag);
      }
      return std::string(argv[++i]);
    };

    if (arg == "--help" || arg == "-h")
    {
      print_usage(argv[0]);
      std::exit(0);
    }
    if (arg == "--variant")
    {
      out.variant = parse_variant(require_value("--variant"));
      continue;
    }
    if (arg == "--iters")
    {
      out.iters = std::max(1, parse_int(require_value("--iters"), "iters"));
      continue;
    }
    if (arg == "--warmup")
    {
      out.warmup_iters = std::max(0, parse_int(require_value("--warmup"), "warmup"));
      continue;
    }
    if (arg == "--device")
    {
      out.device_id = parse_int(require_value("--device"), "device id");
      continue;
    }
    if (arg == "--order")
    {
      const auto o = parse_order(require_value("--order"), "--order");
      out.order_a  = o;
      out.order_b  = o;
      out.order_c  = o;
      continue;
    }
    if (arg == "--order-a")
    {
      out.order_a = parse_order(require_value("--order-a"), "--order-a");
      continue;
    }
    if (arg == "--order-b")
    {
      out.order_b = parse_order(require_value("--order-b"), "--order-b");
      continue;
    }
    if (arg == "--order-c")
    {
      out.order_c = parse_order(require_value("--order-c"), "--order-c");
      continue;
    }
    if (arg == "--symmetric-inputs")
    {
      out.symmetric_inputs = true;
      continue;
    }
    if (arg == "--nvtx")
    {
      out.enable_nvtx = true;
      continue;
    }
    if (arg == "--cuda-profiler-gating")
    {
      out.enable_cuda_profiler_gating = true;
      continue;
    }
    if (arg == "--force-algo")
    {
      out.force_algo.enabled = true;
      out.force_algo.cfg.id  = parse_int(require_value("--force-algo"), "algo id");
      continue;
    }
    if (arg == "--tile-id")
    {
      out.force_algo.enabled    = true;
      out.force_algo.cfg.tile_id = parse_u32(require_value("--tile-id"), "tile id");
      continue;
    }
    if (arg == "--stages-id")
    {
      out.force_algo.enabled      = true;
      out.force_algo.cfg.stages_id = parse_u32(require_value("--stages-id"), "stages id");
      continue;
    }
    if (arg == "--splitk")
    {
      out.force_algo.enabled        = true;
      out.force_algo.cfg.splitk_num = std::max(1, parse_int(require_value("--splitk"), "splitk"));
      continue;
    }
    if (arg == "--summary-json")
    {
      out.summary_json = true;
      continue;
    }

    throw std::runtime_error("Unknown argument: '" + arg + "'. Use --help.");
  }

  if (out.force_algo.enabled)
  {
    apply_known_algo_defaults(out.force_algo.cfg);
    // Default "unset" fields to 0 for non-specified knobs (valid for many algos).
    out.force_algo.cfg.reduction_scheme = 0;
    out.force_algo.cfg.cta_swizzling    = 0;
    out.force_algo.cfg.custom_option    = 0;
    out.force_algo.cfg.inner_shape_id   = 0;
    out.force_algo.cfg.cluster_shape_id = 0;
    out.force_algo.cfg.waves_count      = 0;
  }

  return out;
}

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
                   bool enable_cuda_profiler_gating,
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

    // Timed region.
    cudaEvent_t start{};
    cudaEvent_t stop{};
    CublasLtGemmPlan::CheckCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    CublasLtGemmPlan::CheckCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    if (enable_cuda_profiler_gating)
    {
      // ncu can be configured with --profile-from-start off to profile only between start/stop.
      cudaProfilerStart();
    }
    CublasLtGemmPlan::CheckCuda(cudaEventRecord(start, stream), "cudaEventRecord(start)");
    for (int i = 0; i < iters; ++i)
    {
      plan.Run(stream, a_dev, b_dev, c_dev, alpha, beta);
    }
    CublasLtGemmPlan::CheckCuda(cudaEventRecord(stop, stream), "cudaEventRecord(stop)");
    CublasLtGemmPlan::CheckCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
    if (enable_cuda_profiler_gating)
    {
      cudaProfilerStop();
    }

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

std::string json_escape(const std::string &s)
{
  std::string out;
  out.reserve(s.size());
  for (const unsigned char c : s)
  {
    switch (c)
    {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (c < 0x20)
        {
          std::ostringstream oss;
          oss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
          out += oss.str();
        }
        else
        {
          out += static_cast<char>(c);
        }
        break;
    }
  }
  return out;
}

void print_summary_json_line(const TimedRun &r,
                             cublasOperation_t ta,
                             cublasOperation_t tb,
                             const GemmPlanOptions &opts,
                             const CliOptions &cli)
{
  std::ostringstream oss;
  oss << "ACCELSIM_GEMM_RUN {"
      << "\"label\":\"" << json_escape(r.label) << "\""
      << ",\"ok\":" << (r.ok ? "true" : "false")
      << ",\"avg_ms\":" << std::fixed << std::setprecision(6) << r.avg_ms
      << ",\"algo_id\":" << r.algo.id
      << ",\"tile_id\":" << r.algo.tile_id
      << ",\"stages_id\":" << r.algo.stages_id
      << ",\"splitk_num\":" << r.algo.splitk_num
      << ",\"trans_a\":\"" << op_to_string(ta) << "\""
      << ",\"trans_b\":\"" << op_to_string(tb) << "\""
      << ",\"order_a\":\"" << order_to_string(opts.orders.a) << "\""
      << ",\"order_b\":\"" << order_to_string(opts.orders.b) << "\""
      << ",\"order_c\":\"" << order_to_string(opts.orders.c) << "\""
      << ",\"iters\":" << cli.iters
      << ",\"warmup\":" << cli.warmup_iters;
  if (cli.symmetric_inputs)
  {
    oss << ",\"symmetric_inputs\":true";
  }
  if (!r.ok)
  {
    oss << ",\"reason\":\"" << json_escape(r.notes) << "\"";
  }
  oss << "}\n";
  std::cout << oss.str();
}

MatrixDims make_matrix_dims(int rows, int cols, cublasLtOrder_t order)
{
  const int ld = (order == CUBLASLT_ORDER_COL) ? rows : cols;
  return MatrixDims{rows, cols, ld};
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

  CliOptions cli{};
  try
  {
    cli = parse_cli(argc, argv);
  }
  catch (const std::exception &e)
  {
    std::cerr << "ERROR: " << e.what() << "\n";
    print_usage(argv[0]);
    return 2;
  }

  int dev_id = 0;
  if (cli.device_id >= 0)
  {
    cudaSetDevice(cli.device_id);
  }
  CublasLtGemmPlan::CheckCuda(cudaGetDevice(&dev_id), "cudaGetDevice");
  cudaDeviceProp prop{};
  CublasLtGemmPlan::CheckCuda(cudaGetDeviceProperties(&prop, dev_id), "cudaGetDeviceProperties");
  std::cout << "Repro: N=1000 int8 GEMM variants (focus: ABT_view algo 23)\n";
  std::cout << "GPU: " << prop.name << " (cc " << prop.major << "." << prop.minor << ")\n";
  std::cout << "iters=" << cli.iters << " warmup=" << cli.warmup_iters << "\n";
  std::cout << "order_a=" << order_to_string(cli.order_a) << " order_b=" << order_to_string(cli.order_b)
            << " order_c=" << order_to_string(cli.order_c) << "\n";
  if (cli.symmetric_inputs)
  {
    std::cout << "symmetric_inputs=true\n";
  }
  if (cli.variant != Variant::kAll)
  {
    std::cout << "variant=" << (cli.variant == Variant::kAB ? "AB" : (cli.variant == Variant::kATBView ? "ATB_view" : "ABT_view")) << "\n";
  }
  if (cli.force_algo.enabled)
  {
    std::cout << "forced " << algo_to_string(cli.force_algo.cfg) << "\n";
  }
  std::cout << "\n";

  const int n = kN;
  const GemmDims dims{n, n, n};

  const GemmTypes types{CUDA_R_8I, CUDA_R_8I, CUDA_R_32I, CUBLAS_COMPUTE_32I, CUDA_R_32I};

  // Storage shapes match the square suite in gemm_transpose_bench:
  // A: (M,K), B: (K,N), C: (M,N), row-major.
  const MatrixDims a_dims = make_matrix_dims(n, n, cli.order_a);
  const MatrixDims b_dims = make_matrix_dims(n, n, cli.order_b);
  const MatrixDims c_dims = make_matrix_dims(n, n, cli.order_c);

  auto a_logical = make_logical_matrix_int8(n, n, /*seed=*/123);
  auto b_logical = make_logical_matrix_int8(n, n, /*seed=*/124);
  if (cli.symmetric_inputs)
  {
    symmetrize_logical_square(a_logical, n);
    symmetrize_logical_square(b_logical, n);
  }
  const auto a_host = pack_matrix_int8(a_logical, n, n, cli.order_a);
  const auto b_host = pack_matrix_int8(b_logical, n, n, cli.order_b);

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
    heuristic_opts.orders.a            = cli.order_a;
    heuristic_opts.orders.b            = cli.order_b;
    heuristic_opts.orders.c            = cli.order_c;

    auto forced_opts = [&](const CublasLtAlgoConfig &cfg) {
      GemmPlanOptions o{};
      o.max_workspace_bytes         = heuristic_opts.max_workspace_bytes;
      o.orders                      = heuristic_opts.orders;
      o.algo_override.enabled       = true;
      o.algo_override.config        = cfg;
      return o;
    };

    auto run_one = [&](const std::string &label,
                       cublasOperation_t ta,
                       cublasOperation_t tb,
                       const GemmPlanOptions &opts) -> TimedRun {
      const std::string range_name = label;
      const NvtxRange range(cli.enable_nvtx, range_name);
      const auto r = time_plan(label, dims, types, a_dims, b_dims, c_dims, ta, tb, opts,
                               stream, a_dev, b_dev, c_dev, &alpha, &beta, cli.enable_cuda_profiler_gating, cli.warmup_iters, cli.iters);
      if (cli.summary_json)
      {
        print_summary_json_line(r, ta, tb, opts, cli);
      }
      return r;
    };

    auto make_opts = [&](const CublasLtAlgoConfig &cfg) -> GemmPlanOptions {
      if (!cli.force_algo.enabled)
      {
        return heuristic_opts;
      }
      return forced_opts(cfg);
    };

    const auto forced_cfg = [&]() -> CublasLtAlgoConfig {
      if (!cli.force_algo.enabled)
      {
        return CublasLtAlgoConfig{};
      }
      return cli.force_algo.cfg;
    }();

    const auto run_variant = [&](Variant v) -> TimedRun {
      if (v == Variant::kAB)
      {
        return run_one(cli.force_algo.enabled ? "AB (forced)" : "AB (heuristic)",
                       CUBLAS_OP_N, CUBLAS_OP_N,
                       make_opts(forced_cfg));
      }
      if (v == Variant::kATBView)
      {
        return run_one(cli.force_algo.enabled ? "ATB_view (forced)" : "ATB_view (heuristic)",
                       CUBLAS_OP_T, CUBLAS_OP_N,
                       make_opts(forced_cfg));
      }
      return run_one(cli.force_algo.enabled ? "ABT_view (forced)" : "ABT_view (heuristic)",
                     CUBLAS_OP_N, CUBLAS_OP_T,
                     make_opts(forced_cfg));
    };

    if (cli.variant != Variant::kAll)
    {
      const auto r = run_variant(cli.variant);
      if (r.ok && cli.force_algo.enabled)
      {
        std::cout << "Selected " << algo_to_string(r.algo) << "\n";
      }
      print_row(r);
      return r.ok ? 0 : 1;
    }

    // Default "all" mode (kept for backward compatibility and quick sanity checks).
    std::cout << "Heuristic selections (baseline):\n";
    const auto r_ab  = run_one("AB (heuristic)", CUBLAS_OP_N, CUBLAS_OP_N, heuristic_opts);
    const auto r_atb = run_one("ATB_view (heuristic)", CUBLAS_OP_T, CUBLAS_OP_N, heuristic_opts);
    const auto r_abt = run_one("ABT_view (heuristic)", CUBLAS_OP_N, CUBLAS_OP_T, heuristic_opts);
    print_row(r_ab);
    print_row(r_atb);
    print_row(r_abt);
    if (r_ab.ok && r_abt.ok)
    {
      std::cout << "ABT_view/AB speedup: " << std::fixed << std::setprecision(2) << (r_ab.avg_ms / r_abt.avg_ms) << "x\n";
    }
    std::cout << "\n";

    std::cout << "Force algo 23 into other transpose modes (if supported):\n";
    const auto r_ab_force23  = run_one("AB forced algo23", CUBLAS_OP_N, CUBLAS_OP_N, forced_opts(cfg23));
    const auto r_atb_force23 = run_one("ATB forced algo23", CUBLAS_OP_T, CUBLAS_OP_N, forced_opts(cfg23));
    const auto r_abt_force23 = run_one("ABT forced algo23", CUBLAS_OP_N, CUBLAS_OP_T, forced_opts(cfg23));
    print_row(r_ab_force23);
    print_row(r_atb_force23);
    print_row(r_abt_force23);
    std::cout << "\n";

    std::cout << "Control: try forcing algo 64 into ABT_view (if supported):\n";
    const auto r_abt_force64 = run_one("ABT forced algo64", CUBLAS_OP_N, CUBLAS_OP_T, forced_opts(cfg64));
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
