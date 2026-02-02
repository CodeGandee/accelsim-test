#include "cublaslt_gemm.h"

#include <nvbench/nvbench.cuh>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace accelsim::gemm
{

namespace
{

struct Shape
{
  int m{};
  int n{};
  int k{};
};

Shape parse_shape(const std::string &s)
{
  const auto p1 = s.find('x');
  const auto p2 = s.find('x', p1 == std::string::npos ? 0 : p1 + 1);
  if (p1 == std::string::npos || p2 == std::string::npos)
  {
    throw std::runtime_error("Invalid shape axis value: " + s);
  }
  return Shape{std::stoi(s.substr(0, p1)), std::stoi(s.substr(p1 + 1, p2 - p1 - 1)), std::stoi(s.substr(p2 + 1))};
}

template <typename T>
__global__ void transpose_kernel(const T *in, T *out, int rows, int cols)
{
  const int r = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int c = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (r < rows && c < cols)
  {
    out[c * rows + r] = in[r * cols + c];
  }
}

template <typename T>
void transpose_device(cudaStream_t stream, const T *in, T *out, int rows, int cols)
{
  dim3 block(16, 16);
  dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
  transpose_kernel<T><<<grid, block, 0, stream>>>(in, out, rows, cols);
  CublasLtGemmPlan::CheckCuda(cudaGetLastError(), "transpose_kernel launch");
}

struct Dtype
{
  std::string key;
  GemmTypes types{};
  bool is_int{};
};

Dtype resolve_dtype(const std::string &dtype_key, const std::string &math_mode)
{
  if (dtype_key == "fp16_fp16_fp16")
  {
    return Dtype{dtype_key,
                 GemmTypes{CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F, CUDA_R_32F},
                 false};
  }
  if (dtype_key == "bf16_bf16_bf16")
  {
    return Dtype{dtype_key,
                 GemmTypes{CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F},
                 false};
  }
  if (dtype_key == "fp32_fp32_fp32")
  {
    return Dtype{dtype_key,
                 GemmTypes{CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F},
                 false};
  }
  if (dtype_key == "fp32_fp32_fp32_tf32")
  {
    if (math_mode != "tf32")
    {
      // Keep key canonical; math_mode axis still selects behavior.
    }
    return Dtype{dtype_key,
                 GemmTypes{CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F},
                 false};
  }
  if (dtype_key == "int8_int8_int32")
  {
    return Dtype{dtype_key,
                 GemmTypes{CUDA_R_8I, CUDA_R_8I, CUDA_R_32I, CUBLAS_COMPUTE_32I, CUDA_R_32I},
                 true};
  }

  throw std::runtime_error("Unsupported dtype key: " + dtype_key);
}

template <typename HostT>
std::vector<HostT> make_host_matrix(int rows, int cols, int seed)
{
  std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<HostT> out(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
  for (auto &v : out)
  {
    v = static_cast<HostT>(dist(rng));
  }
  return out;
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
std::vector<T> convert_to(const std::vector<float> &in)
{
  std::vector<T> out(in.size());
  std::transform(in.begin(), in.end(), out.begin(), [](float v) { return static_cast<T>(v); });
  return out;
}

template <>
std::vector<__half> convert_to<__half>(const std::vector<float> &in)
{
  std::vector<__half> out(in.size());
  std::transform(in.begin(), in.end(), out.begin(), [](float v) { return __float2half(v); });
  return out;
}

template <>
std::vector<__nv_bfloat16> convert_to<__nv_bfloat16>(const std::vector<float> &in)
{
  std::vector<__nv_bfloat16> out(in.size());
  std::transform(in.begin(), in.end(), out.begin(), [](float v) { return __float2bfloat16(v); });
  return out;
}

template <typename T>
float to_float(T v)
{
  return static_cast<float>(v);
}

template <>
float to_float(__half v)
{
  return __half2float(v);
}

template <>
float to_float(__nv_bfloat16 v)
{
  return __bfloat162float(v);
}

struct VerificationResult
{
  bool pass{true};
  float max_abs{0.0f};
  float max_rel{0.0f};
  std::string details;
};

template <typename AHostT, typename BHostT, typename CHostT>
VerificationResult verify_sampled(const Shape &shape,
                                  const std::string &suite,
                                  const std::string &case_name,
                                  const std::vector<AHostT> &a,
                                  int a_rows,
                                  int a_cols,
                                  const std::vector<BHostT> &b,
                                  int b_rows,
                                  int b_cols,
                                  const std::vector<CHostT> &c_samples,
                                  const std::vector<std::pair<int, int>> &indices,
                                  float abs_tol,
                                  float rel_tol)
{
  (void)suite;
  (void)a_rows;
  (void)b_rows;
  VerificationResult vr{};

  const int k = shape.k;

  auto a_at = [&](int r, int c) -> float {
    return to_float(a[static_cast<std::size_t>(r) * static_cast<std::size_t>(a_cols) + static_cast<std::size_t>(c)]);
  };
  auto b_at = [&](int r, int c) -> float {
    return to_float(b[static_cast<std::size_t>(r) * static_cast<std::size_t>(b_cols) + static_cast<std::size_t>(c)]);
  };

  auto ref_at = [&](int i, int j) -> float {
    float acc = 0.0f;
    if (case_name == "AB")
    {
      for (int kk = 0; kk < k; ++kk)
      {
        acc += a_at(i, kk) * b_at(kk, j);
      }
      return acc;
    }
    if (case_name == "ATB_view" || case_name == "ATB_copyA")
    {
      // op(A)=T, op(B)=N:
      // Square: A is NxN so this is valid; non-square ATB suite stores A as KxM.
      for (int kk = 0; kk < k; ++kk)
      {
        acc += a_at(kk, i) * b_at(kk, j);
      }
      return acc;
    }
    if (case_name == "ABT_view" || case_name == "ABT_copyB")
    {
      // op(A)=N, op(B)=T:
      for (int kk = 0; kk < k; ++kk)
      {
        acc += a_at(i, kk) * b_at(j, kk);
      }
      return acc;
    }
    throw std::runtime_error("Unknown case for reference: " + case_name);
  };

  for (std::size_t idx = 0; idx < indices.size(); ++idx)
  {
    const auto [i, j] = indices[idx];
    const float got   = to_float(c_samples[idx]);
    const float ref   = ref_at(i, j);
    const float abs_e = std::abs(got - ref);
    const float rel_e = (std::abs(ref) > 0.0f) ? abs_e / std::abs(ref) : abs_e;
    vr.max_abs        = std::max(vr.max_abs, abs_e);
    vr.max_rel        = std::max(vr.max_rel, rel_e);
    if (!(abs_e <= abs_tol || rel_e <= rel_tol))
    {
      vr.pass    = false;
      vr.details = "sample mismatch at (" + std::to_string(i) + "," + std::to_string(j) + ")";
      break;
    }
  }

  return vr;
}

template <typename AHostT, typename BHostT, typename CHostT>
VerificationResult verify_full(const Shape &shape,
                               const std::string &suite,
                               const std::string &case_name,
                               const std::vector<AHostT> &a,
                               int a_rows,
                               int a_cols,
                               const std::vector<BHostT> &b,
                               int b_rows,
                               int b_cols,
                               const std::vector<CHostT> &c_full,
                               float abs_tol,
                               float rel_tol)
{
  (void)suite;
  (void)a_rows;
  (void)b_rows;
  VerificationResult vr{};

  const int m = shape.m;
  const int n = shape.n;
  const int k = shape.k;

  auto a_at = [&](int r, int c) -> float {
    return to_float(a[static_cast<std::size_t>(r) * static_cast<std::size_t>(a_cols) + static_cast<std::size_t>(c)]);
  };
  auto b_at = [&](int r, int c) -> float {
    return to_float(b[static_cast<std::size_t>(r) * static_cast<std::size_t>(b_cols) + static_cast<std::size_t>(c)]);
  };

  auto ref_at = [&](int i, int j) -> float {
    float acc = 0.0f;
    if (case_name == "AB")
    {
      for (int kk = 0; kk < k; ++kk)
      {
        acc += a_at(i, kk) * b_at(kk, j);
      }
      return acc;
    }
    if (case_name == "ATB_view" || case_name == "ATB_copyA")
    {
      for (int kk = 0; kk < k; ++kk)
      {
        acc += a_at(kk, i) * b_at(kk, j);
      }
      return acc;
    }
    if (case_name == "ABT_view" || case_name == "ABT_copyB")
    {
      for (int kk = 0; kk < k; ++kk)
      {
        acc += a_at(i, kk) * b_at(j, kk);
      }
      return acc;
    }
    throw std::runtime_error("Unknown case for reference: " + case_name);
  };

  for (int i = 0; i < m; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      const std::size_t offset = static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j);
      const float got          = to_float(c_full[offset]);
      const float ref          = ref_at(i, j);
      const float abs_e        = std::abs(got - ref);
      const float rel_e        = (std::abs(ref) > 0.0f) ? abs_e / std::abs(ref) : abs_e;
      vr.max_abs               = std::max(vr.max_abs, abs_e);
      vr.max_rel               = std::max(vr.max_rel, rel_e);
      if (!(abs_e <= abs_tol || rel_e <= rel_tol))
      {
        vr.pass    = false;
        vr.details = "full mismatch at (" + std::to_string(i) + "," + std::to_string(j) + ")";
        return vr;
      }
    }
  }

  return vr;
}

std::vector<std::pair<int, int>> pick_sample_indices(int m, int n)
{
  std::vector<std::pair<int, int>> idx;
  idx.emplace_back(0, 0);
  idx.emplace_back(m - 1, n - 1);
  idx.emplace_back(m / 2, n / 2);
  idx.emplace_back(0, n - 1);
  idx.emplace_back(m - 1, 0);
  idx.emplace_back(m / 3, n / 3);
  idx.emplace_back(m / 4, n / 4);
  idx.emplace_back(m / 5, n / 5);
  idx.erase(std::remove_if(idx.begin(),
                           idx.end(),
                           [&](const auto &p) { return p.first < 0 || p.first >= m || p.second < 0 || p.second >= n; }),
            idx.end());
  return idx;
}

template <typename T>
std::vector<T> gather_device_samples(const T *c_dev, int ld, const std::vector<std::pair<int, int>> &indices)
{
  std::vector<T> out(indices.size());
  for (std::size_t i = 0; i < indices.size(); ++i)
  {
    const auto [r, c] = indices[i];
    const std::size_t offset = static_cast<std::size_t>(r) * static_cast<std::size_t>(ld) + static_cast<std::size_t>(c);
    CublasLtGemmPlan::CheckCuda(cudaMemcpy(&out[i], c_dev + offset, sizeof(T), cudaMemcpyDeviceToHost),
                               "cudaMemcpy(sample)");
  }
  return out;
}

template <typename T>
std::vector<T> gather_device_matrix(const T *c_dev, std::size_t elements)
{
  std::vector<T> out(elements);
  CublasLtGemmPlan::CheckCuda(cudaMemcpy(out.data(), c_dev, elements * sizeof(T), cudaMemcpyDeviceToHost),
                             "cudaMemcpy(C full)");
  return out;
}

void *device_alloc(std::size_t bytes)
{
  void *ptr = nullptr;
  CublasLtGemmPlan::CheckCuda(cudaMalloc(&ptr, bytes), "cudaMalloc");
  return ptr;
}

} // namespace

} // namespace accelsim::gemm

namespace accelsim::gemm
{

void gemm_transpose_bench(nvbench::state &state)
{
  const std::string suite     = state.get_string("suite");
  const std::string case_name = state.get_string("case");
  const std::string dtype_key = state.get_string("dtype");
  const std::string math_mode = state.get_string_or_default("math_mode", "default");
  const std::string shape_str = state.get_string("shape");

  if (suite == "nonsquare_atb" && !(case_name == "ATB_view" || case_name == "ATB_copyA"))
  {
    state.skip("case not valid for nonsquare_atb");
    return;
  }
  if (suite == "nonsquare_abt" && !(case_name == "ABT_view" || case_name == "ABT_copyB"))
  {
    state.skip("case not valid for nonsquare_abt");
    return;
  }
  if (suite == "square")
  {
    // All cases valid.
  }
  else if (!(suite == "nonsquare_atb" || suite == "nonsquare_abt"))
  {
    state.skip("unknown suite");
    return;
  }

  const Shape shape = parse_shape(shape_str);
  if (suite == "square" && !(shape.m == shape.n && shape.n == shape.k))
  {
    state.skip("square suite requires M=N=K");
    return;
  }

  if (math_mode == "tf32" && dtype_key != "fp32_fp32_fp32_tf32")
  {
    state.skip("tf32 math_mode only valid for fp32_tf32 dtype");
    return;
  }

  const Dtype dtype = resolve_dtype(dtype_key, math_mode);

  // Define storage shapes per suite.
  const int m = shape.m;
  const int n = shape.n;
  const int k = shape.k;

  int a_rows = 0;
  int a_cols = 0;
  int b_rows = 0;
  int b_cols = 0;

  if (suite == "nonsquare_atb")
  {
    a_rows = k;
    a_cols = m;
    b_rows = k;
    b_cols = n;
  }
  else if (suite == "nonsquare_abt")
  {
    a_rows = m;
    a_cols = k;
    b_rows = n;
    b_cols = k;
  }
  else
  {
    // square
    a_rows = m;
    a_cols = k;
    b_rows = k;
    b_cols = n;
  }

  // Allocate and initialize host matrices.
  const int seed = 123;
  std::vector<float> a_host_f;
  std::vector<float> b_host_f;
  std::vector<std::int8_t> a_host_i8;
  std::vector<std::int8_t> b_host_i8;

  if (dtype.is_int)
  {
    a_host_i8 = make_host_matrix_int8(a_rows, a_cols, seed);
    b_host_i8 = make_host_matrix_int8(b_rows, b_cols, seed + 1);
  }
  else
  {
    a_host_f = make_host_matrix<float>(a_rows, a_cols, seed);
    b_host_f = make_host_matrix<float>(b_rows, b_cols, seed + 1);
  }

  // Device buffers (base)
  void *a_dev = nullptr;
  void *b_dev = nullptr;
  void *c_dev = nullptr;
  void *a_copy_dev = nullptr;
  void *b_copy_dev = nullptr;

  const std::size_t a_elems = static_cast<std::size_t>(a_rows) * static_cast<std::size_t>(a_cols);
  const std::size_t b_elems = static_cast<std::size_t>(b_rows) * static_cast<std::size_t>(b_cols);
  const std::size_t c_elems = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);

  auto free_all = [&]() {
    if (a_copy_dev)
      cudaFree(a_copy_dev);
    if (b_copy_dev)
      cudaFree(b_copy_dev);
    if (a_dev)
      cudaFree(a_dev);
    if (b_dev)
      cudaFree(b_dev);
    if (c_dev)
      cudaFree(c_dev);
  };

  try
  {
    const std::size_t a_bytes = a_elems * (dtype.is_int ? sizeof(std::int8_t)
                                                        : (dtype.types.a_type == CUDA_R_16F ? sizeof(__half)
                                                           : dtype.types.a_type == CUDA_R_16BF ? sizeof(__nv_bfloat16)
                                                                                                : sizeof(float)));
    const std::size_t b_bytes = b_elems * (dtype.is_int ? sizeof(std::int8_t)
                                                        : (dtype.types.b_type == CUDA_R_16F ? sizeof(__half)
                                                           : dtype.types.b_type == CUDA_R_16BF ? sizeof(__nv_bfloat16)
                                                                                                : sizeof(float)));
    const std::size_t c_bytes = c_elems * (dtype.types.c_type == CUDA_R_32I ? sizeof(std::int32_t)
                                                                            : dtype.types.c_type == CUDA_R_16F ? sizeof(__half)
                                                                              : dtype.types.c_type == CUDA_R_16BF ? sizeof(__nv_bfloat16)
                                                                                                                  : sizeof(float));

    a_dev = device_alloc(a_bytes);
    b_dev = device_alloc(b_bytes);
    c_dev = device_alloc(c_bytes);

    // Copy inputs to device.
    if (dtype.is_int)
    {
      CublasLtGemmPlan::CheckCuda(cudaMemcpy(a_dev, a_host_i8.data(), a_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(A)");
      CublasLtGemmPlan::CheckCuda(cudaMemcpy(b_dev, b_host_i8.data(), b_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(B)");
    }
    else if (dtype.types.a_type == CUDA_R_16F)
    {
      auto a_half = convert_to<__half>(a_host_f);
      auto b_half = convert_to<__half>(b_host_f);
      CublasLtGemmPlan::CheckCuda(cudaMemcpy(a_dev, a_half.data(), a_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(A)");
      CublasLtGemmPlan::CheckCuda(cudaMemcpy(b_dev, b_half.data(), b_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(B)");
    }
    else if (dtype.types.a_type == CUDA_R_16BF)
    {
      auto a_bf16 = convert_to<__nv_bfloat16>(a_host_f);
      auto b_bf16 = convert_to<__nv_bfloat16>(b_host_f);
      CublasLtGemmPlan::CheckCuda(cudaMemcpy(a_dev, a_bf16.data(), a_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(A)");
      CublasLtGemmPlan::CheckCuda(cudaMemcpy(b_dev, b_bf16.data(), b_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(B)");
    }
    else
    {
      CublasLtGemmPlan::CheckCuda(cudaMemcpy(a_dev, a_host_f.data(), a_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(A)");
      CublasLtGemmPlan::CheckCuda(cudaMemcpy(b_dev, b_host_f.data(), b_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(B)");
    }

    // Create copy buffers outside timed region if needed.
    cudaStream_t setup_stream = state.get_cuda_stream();

    void *a_used = a_dev;
    void *b_used = b_dev;

    int a_used_rows = a_rows;
    int a_used_cols = a_cols;
    int b_used_rows = b_rows;
    int b_used_cols = b_cols;

    cublasOperation_t trans_a = CUBLAS_OP_N;
    cublasOperation_t trans_b = CUBLAS_OP_N;

    if (case_name == "AB")
    {
      trans_a = CUBLAS_OP_N;
      trans_b = CUBLAS_OP_N;
    }
    else if (case_name == "ATB_view")
    {
      trans_a = CUBLAS_OP_T;
      trans_b = CUBLAS_OP_N;
    }
    else if (case_name == "ABT_view")
    {
      trans_a = CUBLAS_OP_N;
      trans_b = CUBLAS_OP_T;
    }
    else if (case_name == "ATB_copyA")
    {
      // materialize A^T into a new contiguous buffer with shape MxK
      trans_a = CUBLAS_OP_N;
      trans_b = CUBLAS_OP_N;
      a_used_rows = m;
      a_used_cols = k;
      a_copy_dev  = device_alloc(static_cast<std::size_t>(a_used_rows) * static_cast<std::size_t>(a_used_cols)
                                * (dtype.is_int ? sizeof(std::int8_t)
                                                : (dtype.types.a_type == CUDA_R_16F ? sizeof(__half)
                                                   : dtype.types.a_type == CUDA_R_16BF ? sizeof(__nv_bfloat16)
                                                                                        : sizeof(float))));

      if (dtype.types.a_type == CUDA_R_16F)
      {
        transpose_device(setup_stream, static_cast<const __half *>(a_dev), static_cast<__half *>(a_copy_dev), a_rows, a_cols);
      }
      else if (dtype.types.a_type == CUDA_R_16BF)
      {
        transpose_device(setup_stream,
                         static_cast<const __nv_bfloat16 *>(a_dev),
                         static_cast<__nv_bfloat16 *>(a_copy_dev),
                         a_rows,
                         a_cols);
      }
      else if (dtype.types.a_type == CUDA_R_8I)
      {
        transpose_device(setup_stream,
                         static_cast<const std::int8_t *>(a_dev),
                         static_cast<std::int8_t *>(a_copy_dev),
                         a_rows,
                         a_cols);
      }
      else
      {
        transpose_device(setup_stream, static_cast<const float *>(a_dev), static_cast<float *>(a_copy_dev), a_rows, a_cols);
      }
      CublasLtGemmPlan::CheckCuda(cudaStreamSynchronize(setup_stream), "cudaStreamSynchronize(transpose A)");
      a_used = a_copy_dev;
    }
    else if (case_name == "ABT_copyB")
    {
      // materialize B^T into a new contiguous buffer with shape KxN
      trans_a = CUBLAS_OP_N;
      trans_b = CUBLAS_OP_N;
      b_used_rows = k;
      b_used_cols = n;
      b_copy_dev  = device_alloc(static_cast<std::size_t>(b_used_rows) * static_cast<std::size_t>(b_used_cols)
                                * (dtype.is_int ? sizeof(std::int8_t)
                                                : (dtype.types.b_type == CUDA_R_16F ? sizeof(__half)
                                                   : dtype.types.b_type == CUDA_R_16BF ? sizeof(__nv_bfloat16)
                                                                                        : sizeof(float))));

      if (dtype.types.b_type == CUDA_R_16F)
      {
        transpose_device(setup_stream, static_cast<const __half *>(b_dev), static_cast<__half *>(b_copy_dev), b_rows, b_cols);
      }
      else if (dtype.types.b_type == CUDA_R_16BF)
      {
        transpose_device(setup_stream,
                         static_cast<const __nv_bfloat16 *>(b_dev),
                         static_cast<__nv_bfloat16 *>(b_copy_dev),
                         b_rows,
                         b_cols);
      }
      else if (dtype.types.b_type == CUDA_R_8I)
      {
        transpose_device(setup_stream,
                         static_cast<const std::int8_t *>(b_dev),
                         static_cast<std::int8_t *>(b_copy_dev),
                         b_rows,
                         b_cols);
      }
      else
      {
        transpose_device(setup_stream, static_cast<const float *>(b_dev), static_cast<float *>(b_copy_dev), b_rows, b_cols);
      }
      CublasLtGemmPlan::CheckCuda(cudaStreamSynchronize(setup_stream), "cudaStreamSynchronize(transpose B)");
      b_used = b_copy_dev;
    }

    const MatrixDims a_dims{a_used_rows, a_used_cols, a_used_cols};
    const MatrixDims b_dims{b_used_rows, b_used_cols, b_used_cols};
    const MatrixDims c_dims{m, n, n};

    const GemmPlanOptions plan_opts{};
    const CublasLtGemmPlan plan(GemmDims{m, n, k}, dtype.types, a_dims, b_dims, c_dims, trans_a, trans_b, plan_opts);

    // Persist the selected cuBLASLt algorithm configuration for JSON export/reporting.
    {
      const auto &algo = plan.SelectedAlgo();
      auto &s          = state.add_summary("accelsim/cublaslt/algo");
      s.set_string("hide", "");
      s.set_int64("id", static_cast<std::int64_t>(algo.id));
      s.set_int64("tile_id", static_cast<std::int64_t>(algo.tile_id));
      s.set_int64("splitk_num", static_cast<std::int64_t>(algo.splitk_num));
      s.set_int64("reduction_scheme", static_cast<std::int64_t>(algo.reduction_scheme));
      s.set_int64("cta_swizzling", static_cast<std::int64_t>(algo.cta_swizzling));
      s.set_int64("custom_option", static_cast<std::int64_t>(algo.custom_option));
      s.set_int64("stages_id", static_cast<std::int64_t>(algo.stages_id));
      s.set_int64("inner_shape_id", static_cast<std::int64_t>(algo.inner_shape_id));
      s.set_int64("cluster_shape_id", static_cast<std::int64_t>(algo.cluster_shape_id));
      s.set_int64("required_workspace_bytes", static_cast<std::int64_t>(algo.required_workspace_bytes));
      s.set_int64("waves_count", static_cast<std::int64_t>(algo.waves_count));
    }

    // One untimed run for correctness verification (avoid polluting profiler runs).
    if (!state.get_run_once())
    {
      VerificationResult vr{};
      if (dtype.types.scale_type == CUDA_R_32F)
      {
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        plan.Run(setup_stream, a_used, b_used, c_dev, &alpha, &beta);
      }
      else
      {
        const std::int32_t alpha = 1;
        const std::int32_t beta  = 0;
        plan.Run(setup_stream, a_used, b_used, c_dev, &alpha, &beta);
      }
      CublasLtGemmPlan::CheckCuda(cudaStreamSynchronize(setup_stream), "cudaStreamSynchronize(verify gemm)");

      const bool full_verify = std::max({m, n, k}) <= 1000;
      const std::string verify_mode = full_verify ? "full" : "sampled";

      if (dtype.types.c_type == CUDA_R_32I)
      {
        if (full_verify)
        {
          auto c_full = gather_device_matrix(static_cast<const std::int32_t *>(c_dev), static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
          vr          = verify_full<std::int8_t, std::int8_t, std::int32_t>(
            shape, suite, case_name, a_host_i8, a_rows, a_cols, b_host_i8, b_rows, b_cols, c_full, 0.0f, 0.0f);
        }
        else
        {
          const auto idx = pick_sample_indices(m, n);
          auto samples   = gather_device_samples(static_cast<const std::int32_t *>(c_dev), n, idx);
          vr             = verify_sampled<std::int8_t, std::int8_t, std::int32_t>(
            shape, suite, case_name, a_host_i8, a_rows, a_cols, b_host_i8, b_rows, b_cols, samples, idx, 0.0f, 0.0f);
        }
      }
      else if (dtype.types.c_type == CUDA_R_16F)
      {
        if (full_verify)
        {
          auto c_full = gather_device_matrix(static_cast<const __half *>(c_dev), static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
          vr          = verify_full<float, float, __half>(
            shape, suite, case_name, a_host_f, a_rows, a_cols, b_host_f, b_rows, b_cols, c_full, 5e-2f, 5e-2f);
        }
        else
        {
          const auto idx = pick_sample_indices(m, n);
          auto samples   = gather_device_samples(static_cast<const __half *>(c_dev), n, idx);
          vr             = verify_sampled<float, float, __half>(
            shape, suite, case_name, a_host_f, a_rows, a_cols, b_host_f, b_rows, b_cols, samples, idx, 5e-2f, 5e-2f);
        }
      }
      else if (dtype.types.c_type == CUDA_R_16BF)
      {
        if (full_verify)
        {
          auto c_full = gather_device_matrix(static_cast<const __nv_bfloat16 *>(c_dev), static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
          vr          = verify_full<float, float, __nv_bfloat16>(
            shape, suite, case_name, a_host_f, a_rows, a_cols, b_host_f, b_rows, b_cols, c_full, 1e-1f, 1e-1f);
        }
        else
        {
          const auto idx = pick_sample_indices(m, n);
          auto samples   = gather_device_samples(static_cast<const __nv_bfloat16 *>(c_dev), n, idx);
          vr             = verify_sampled<float, float, __nv_bfloat16>(
            shape, suite, case_name, a_host_f, a_rows, a_cols, b_host_f, b_rows, b_cols, samples, idx, 1e-1f, 1e-1f);
        }
      }
      else
      {
        if (full_verify)
        {
          auto c_full = gather_device_matrix(static_cast<const float *>(c_dev), static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
          vr          = verify_full<float, float, float>(
            shape, suite, case_name, a_host_f, a_rows, a_cols, b_host_f, b_rows, b_cols, c_full, 1e-3f, 1e-3f);
        }
        else
        {
          const auto idx = pick_sample_indices(m, n);
          auto samples   = gather_device_samples(static_cast<const float *>(c_dev), n, idx);
          vr             = verify_sampled<float, float, float>(
            shape, suite, case_name, a_host_f, a_rows, a_cols, b_host_f, b_rows, b_cols, samples, idx, 1e-3f, 1e-3f);
        }
      }

      {
        auto &s = state.add_summary("accelsim/verification/pass");
        s.set_string("name", "Verify");
        s.set_int64("value", vr.pass ? 1 : 0);
      }
      {
        auto &s = state.add_summary("accelsim/verification/mode");
        s.set_string("name", "Verify Mode");
        s.set_string("value", verify_mode);
      }
      {
        auto &s = state.add_summary("accelsim/verification/max_abs_error");
        s.set_string("name", "Max Abs Error");
        s.set_float64("value", static_cast<double>(vr.max_abs));
      }
      {
        auto &s = state.add_summary("accelsim/verification/max_rel_error");
        s.set_string("name", "Max Rel Error");
        s.set_float64("value", static_cast<double>(vr.max_rel));
      }
      if (!vr.details.empty())
      {
        auto &s = state.add_summary("accelsim/verification/details");
        s.set_string("name", "Verify Details");
        s.set_string("value", vr.details);
      }
    }

    // Timed execution (GEMM only).
    if (dtype.types.scale_type == CUDA_R_32F)
    {
      const float alpha = 1.0f;
      const float beta  = 0.0f;
      state.exec(nvbench::exec_tag::sync,
                 [&](nvbench::launch &launch) { plan.Run(launch.get_stream(), a_used, b_used, c_dev, &alpha, &beta); });
    }
    else
    {
      const std::int32_t alpha = 1;
      const std::int32_t beta  = 0;
      state.exec(nvbench::exec_tag::sync,
                 [&](nvbench::launch &launch) { plan.Run(launch.get_stream(), a_used, b_used, c_dev, &alpha, &beta); });
    }
  }
  catch (const std::exception &e)
  {
    state.skip(std::string("exception: ") + e.what());
  }

  free_all();
}

NVBENCH_BENCH(gemm_transpose_bench)
  .set_name("gemm_transpose_bench")
  .add_string_axis("suite", {"square"})
  .add_string_axis("case", {"AB"})
  .add_string_axis("dtype", {"fp16_fp16_fp16"})
  .add_string_axis("math_mode", {"default"})
  .add_string_axis("shape", {"512x512x512"});

} // namespace accelsim::gemm
