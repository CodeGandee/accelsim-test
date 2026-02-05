#include <cublasLt.h>
#include <cuda_runtime.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using json = nlohmann::json;

[[noreturn]] void die(const std::string &msg)
{
  throw std::runtime_error(msg);
}

void checkCuda(cudaError_t st, const char *what)
{
  if (st != cudaSuccess)
  {
    die(std::string(what) + ": " + cudaGetErrorString(st));
  }
}

void checkLt(cublasStatus_t st, const char *what)
{
  if (st != CUBLAS_STATUS_SUCCESS)
  {
    die(std::string(what) + ": cublasStatus=" + std::to_string(static_cast<int>(st)));
  }
}

std::string read_file(const std::string &path)
{
  std::ifstream f(path);
  if (!f)
  {
    die("Failed to open file for read: " + path);
  }
  std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  return s;
}

void write_file(const std::string &path, const std::string &contents)
{
  std::ofstream f(path);
  if (!f)
  {
    die("Failed to open file for write: " + path);
  }
  f << contents;
}

// Mirrors the mapping used in cpp/src/gemm_transpose_bench.cu:resolve_dtype().
cudaDataType_t parse_cuda_dtype(const std::string &s)
{
  if (s == "fp16")
    return CUDA_R_16F;
  if (s == "bf16")
    return CUDA_R_16BF;
  if (s == "fp32")
    return CUDA_R_32F;
  if (s == "int8")
    return CUDA_R_8I;
  if (s == "int32")
    return CUDA_R_32I;
  die("Unknown dtype string: " + s);
}

struct AlgoKey
{
  int algo_id{};
  cudaDataType_t a_type{};
  cudaDataType_t b_type{};
  cudaDataType_t c_type{};
  cublasComputeType_t compute_type{};
  cudaDataType_t scale_type{};
};

bool operator<(const AlgoKey &x, const AlgoKey &y)
{
  return std::tie(x.algo_id, x.a_type, x.b_type, x.c_type, x.compute_type, x.scale_type) <
         std::tie(y.algo_id, y.a_type, y.b_type, y.c_type, y.compute_type, y.scale_type);
}

std::string algo_key_string(const AlgoKey &k)
{
  // Keep as a stable-ish string key for JSON maps.
  return "id=" + std::to_string(k.algo_id) + "|a=" + std::to_string(static_cast<int>(k.a_type)) +
         "|b=" + std::to_string(static_cast<int>(k.b_type)) + "|c=" + std::to_string(static_cast<int>(k.c_type)) +
         "|compute=" + std::to_string(static_cast<int>(k.compute_type)) +
         "|scale=" + std::to_string(static_cast<int>(k.scale_type));
}

template <typename T>
json cap_scalar(const cublasLtMatmulAlgo_t &algo, cublasLtMatmulAlgoCapAttributes_t attr)
{
  T v{};
  std::size_t written = 0;
  const cublasStatus_t st =
    cublasLtMatmulAlgoCapGetAttribute(&algo, attr, &v, sizeof(v), &written);
  if (st != CUBLAS_STATUS_SUCCESS)
  {
    return json{{"status", "error"}, {"cublas_status", static_cast<int>(st)}};
  }
  if (written != sizeof(v))
  {
    return json{{"status", "error"}, {"error", "unexpected_size"}, {"size_written", written}, {"expected", sizeof(v)}};
  }
  return json{{"status", "ok"}, {"value", v}};
}

json cap_u32_list(const cublasLtMatmulAlgo_t &algo, cublasLtMatmulAlgoCapAttributes_t attr)
{
  std::size_t needed = 0;
  {
    const cublasStatus_t st = cublasLtMatmulAlgoCapGetAttribute(&algo, attr, nullptr, 0, &needed);
    if (st != CUBLAS_STATUS_SUCCESS)
    {
      return json{{"status", "error"}, {"cublas_status", static_cast<int>(st)}};
    }
  }

  std::vector<std::uint32_t> buf;
  if (needed % sizeof(std::uint32_t) != 0)
  {
    return json{{"status", "error"}, {"error", "unexpected_size"}, {"needed_bytes", needed}};
  }
  buf.resize(needed / sizeof(std::uint32_t));

  std::size_t written = 0;
  const cublasStatus_t st = cublasLtMatmulAlgoCapGetAttribute(
    &algo, attr, buf.data(), buf.size() * sizeof(std::uint32_t), &written);
  if (st != CUBLAS_STATUS_SUCCESS)
  {
    return json{{"status", "error"}, {"cublas_status", static_cast<int>(st)}};
  }
  if (written % sizeof(std::uint32_t) != 0)
  {
    return json{{"status", "error"}, {"error", "unexpected_size"}, {"size_written", written}};
  }
  buf.resize(written / sizeof(std::uint32_t));

  return json{{"status", "ok"}, {"values", buf}, {"bytes_written", written}};
}

json dump_caps(const cublasLtMatmulAlgo_t &algo)
{
  json caps = json::object();

  // Scalar caps.
  caps["splitk_support"] = cap_scalar<std::int32_t>(algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT);
  caps["reduction_scheme_mask"] = cap_scalar<std::uint32_t>(algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK);
  caps["cta_swizzling_support"] = cap_scalar<std::uint32_t>(algo, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT);
  caps["strided_batch_support"] = cap_scalar<std::int32_t>(algo, CUBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT);
  caps["out_of_place_result_support"] = cap_scalar<std::int32_t>(algo, CUBLASLT_ALGO_CAP_OUT_OF_PLACE_RESULT_SUPPORT);
  caps["uplo_support"] = cap_scalar<std::int32_t>(algo, CUBLASLT_ALGO_CAP_UPLO_SUPPORT);
  caps["custom_option_max"] = cap_scalar<std::int32_t>(algo, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX);
  caps["custom_memory_order"] = cap_scalar<std::int32_t>(algo, CUBLASLT_ALGO_CAP_CUSTOM_MEMORY_ORDER);
  caps["pointer_mode_mask"] = cap_scalar<std::uint32_t>(algo, CUBLASLT_ALGO_CAP_POINTER_MODE_MASK);
  caps["epilogue_mask"] = cap_scalar<std::uint32_t>(algo, CUBLASLT_ALGO_CAP_EPILOGUE_MASK);
  caps["ld_negative"] = cap_scalar<std::int32_t>(algo, CUBLASLT_ALGO_CAP_LD_NEGATIVE);
  caps["numerical_impl_flags"] = cap_scalar<std::uint64_t>(algo, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS);
  caps["min_alignment_a_bytes"] = cap_scalar<std::uint32_t>(algo, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES);
  caps["min_alignment_b_bytes"] = cap_scalar<std::uint32_t>(algo, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES);
  caps["min_alignment_c_bytes"] = cap_scalar<std::uint32_t>(algo, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES);
  caps["min_alignment_d_bytes"] = cap_scalar<std::uint32_t>(algo, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES);
  caps["atomic_sync"] = cap_scalar<std::int32_t>(algo, CUBLASLT_ALGO_CAP_ATOMIC_SYNC);

  // List caps.
  caps["tile_ids"] = cap_u32_list(algo, CUBLASLT_ALGO_CAP_TILE_IDS);
  caps["stages_ids"] = cap_u32_list(algo, CUBLASLT_ALGO_CAP_STAGES_IDS);

  return caps;
}

struct Args
{
  std::string results_path;
  std::string out_path;
};

Args parse_args(int argc, char **argv)
{
  Args a{};
  for (int i = 1; i < argc; ++i)
  {
    const std::string s = argv[i];
    if (s == "--results" && i + 1 < argc)
    {
      a.results_path = argv[++i];
      continue;
    }
    if (s == "--out" && i + 1 < argc)
    {
      a.out_path = argv[++i];
      continue;
    }
    if (s == "-h" || s == "--help")
    {
      std::cout << "Usage: " << argv[0] << " --results <results.json> --out <algo_caps.json>\n";
      std::exit(0);
    }
    die("Unknown arg: " + s);
  }
  if (a.results_path.empty() || a.out_path.empty())
  {
    die("Missing required args. Use --help.");
  }
  return a;
}

} // namespace

int main(int argc, char **argv)
{
  try
  {
    const Args args = parse_args(argc, argv);
    const json results = json::parse(read_file(args.results_path));

    std::set<AlgoKey> unique{};
    std::map<std::string, std::vector<std::string>> algo_to_records;

    for (const auto &rec : results.at("records"))
    {
      const auto &dtype = rec.at("dtype");
      const auto &algo = rec.at("cublaslt").at("algo");

      const std::string a_s = dtype.at("a").get<std::string>();
      const std::string b_s = dtype.at("b").get<std::string>();
      const std::string c_s = dtype.at("c").get<std::string>();
      const std::string compute_s = dtype.at("compute").get<std::string>();

      AlgoKey k{};
      k.algo_id = algo.at("id").get<int>();
      k.a_type = parse_cuda_dtype(a_s);
      k.b_type = parse_cuda_dtype(b_s);
      k.c_type = parse_cuda_dtype(c_s);

      if (compute_s == "fp32")
      {
        k.compute_type = CUBLAS_COMPUTE_32F;
        k.scale_type = CUDA_R_32F;
      }
      else if (compute_s == "tf32")
      {
        k.compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
        k.scale_type = CUDA_R_32F;
      }
      else if (compute_s == "int32")
      {
        k.compute_type = CUBLAS_COMPUTE_32I;
        k.scale_type = CUDA_R_32I;
      }
      else
      {
        die("Unsupported compute type in results.json: " + compute_s);
      }

      unique.insert(k);

      // Attach an example record key for traceability (suite/case/shape/dtype).
      const auto &shape = rec.at("shape");
      const std::string record_key =
        rec.at("suite").get<std::string>() + "/" + rec.at("case").get<std::string>() + "/" +
        std::to_string(shape.at("m").get<int>()) + "x" + std::to_string(shape.at("n").get<int>()) + "x" +
        std::to_string(shape.at("k").get<int>()) + "/" +
        a_s + "," + b_s + "->" + c_s + "(" + compute_s + ")";
      algo_to_records[algo_key_string(k)].push_back(record_key);
    }

    // Initialize cuBLASLt once.
    cublasLtHandle_t handle{};
    checkLt(cublasLtCreate(&handle), "cublasLtCreate");

    json out = json::object();
    out["meta"] = {
      {"results_path", args.results_path},
      {"unique_algo_count", unique.size()},
    };

    cudaDeviceProp prop{};
    int dev = 0;
    checkCuda(cudaGetDevice(&dev), "cudaGetDevice");
    checkCuda(cudaGetDeviceProperties(&prop, dev), "cudaGetDeviceProperties");
    out["environment"] = {
      {"cuda_device", dev},
      {"gpu_name", prop.name},
      {"sm", std::to_string(prop.major) + std::to_string(prop.minor)},
    };

    json algos = json::array();
    for (const auto &k : unique)
    {
      cublasLtMatmulAlgo_t algo{};
      // D type is the output type; this benchmark uses C==D.
      const cudaDataType_t d_type = k.c_type;

      json entry = json::object();
      entry["algo_key"] = algo_key_string(k);
      entry["algo_id"] = k.algo_id;
      entry["types"] = {
        {"a_type", static_cast<int>(k.a_type)},
        {"b_type", static_cast<int>(k.b_type)},
        {"c_type", static_cast<int>(k.c_type)},
        {"d_type", static_cast<int>(d_type)},
        {"compute_type", static_cast<int>(k.compute_type)},
        {"scale_type", static_cast<int>(k.scale_type)},
      };
      entry["example_records"] = algo_to_records[algo_key_string(k)];

      const cublasStatus_t init_st = cublasLtMatmulAlgoInit(handle,
                                                           k.compute_type,
                                                           k.scale_type,
                                                           k.a_type,
                                                           k.b_type,
                                                           k.c_type,
                                                           d_type,
                                                           k.algo_id,
                                                           &algo);
      if (init_st != CUBLAS_STATUS_SUCCESS)
      {
        entry["status"] = "init_failed";
        entry["init_cublas_status"] = static_cast<int>(init_st);
        algos.push_back(entry);
        continue;
      }

      entry["status"] = "ok";
      entry["caps"] = dump_caps(algo);
      algos.push_back(entry);
    }

    out["algos"] = algos;

    write_file(args.out_path, out.dump(2) + "\n");

    checkLt(cublasLtDestroy(handle), "cublasLtDestroy");
    return 0;
  }
  catch (const std::exception &e)
  {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}

