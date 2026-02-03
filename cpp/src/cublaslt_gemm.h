#pragma once

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace accelsim::gemm
{

struct GemmDims
{
  int m{};
  int n{};
  int k{};
};

struct MatrixDims
{
  int rows{};
  int cols{};
  int ld{}; // leading dimension (row-major: cols)
};

struct GemmTypes
{
  cudaDataType_t a_type{};
  cudaDataType_t b_type{};
  cudaDataType_t c_type{};
  cublasComputeType_t compute_type{};
  cudaDataType_t scale_type{};
};

struct CublasLtAlgoConfig
{
  std::int32_t id{};
  std::uint32_t tile_id{};
  std::int32_t splitk_num{};
  std::uint32_t reduction_scheme{};
  std::uint32_t cta_swizzling{};
  std::uint32_t custom_option{};
  std::uint32_t stages_id{};
  std::uint16_t inner_shape_id{};
  std::uint16_t cluster_shape_id{};
  std::size_t required_workspace_bytes{};
  std::int32_t waves_count{};
};

struct GemmAlgoOverride
{
  bool enabled{false};
  CublasLtAlgoConfig config{};
};

struct GemmPlanOptions
{
  std::size_t max_workspace_bytes{64ull * 1024ull * 1024ull};
  cublasLtOrder_t order{CUBLASLT_ORDER_ROW};
  GemmAlgoOverride algo_override{};
};

class CublasLtGemmPlan
{
public:
  CublasLtGemmPlan(const GemmDims &dims,
                   const GemmTypes &types,
                   const MatrixDims &a_dims,
                   const MatrixDims &b_dims,
                   const MatrixDims &c_dims,
                   cublasOperation_t trans_a,
                   cublasOperation_t trans_b,
                   const GemmPlanOptions &opts);
  ~CublasLtGemmPlan();

  CublasLtGemmPlan(const CublasLtGemmPlan &)            = delete;
  CublasLtGemmPlan &operator=(const CublasLtGemmPlan &) = delete;
  CublasLtGemmPlan(CublasLtGemmPlan &&)                 = delete;
  CublasLtGemmPlan &operator=(CublasLtGemmPlan &&)      = delete;

  void Run(cudaStream_t stream,
           const void *a,
           const void *b,
           void *c,
           const void *alpha,
           const void *beta) const;

  [[nodiscard]] std::size_t WorkspaceBytes() const { return m_workspace_bytes; }
  [[nodiscard]] const CublasLtAlgoConfig &SelectedAlgo() const { return m_algo_config; }

  static void CheckCuda(cudaError_t status, const char *what);
  static void CheckLt(cublasStatus_t status, const char *what);

private:
  cublasLtHandle_t m_handle{};
  cublasLtMatmulDesc_t m_matmul_desc{};
  cublasLtMatrixLayout_t m_a_layout{};
  cublasLtMatrixLayout_t m_b_layout{};
  cublasLtMatrixLayout_t m_c_layout{};
  cublasLtMatmulAlgo_t m_algo{};
  CublasLtAlgoConfig m_algo_config{};

  void *m_workspace{};
  std::size_t m_workspace_bytes{};
};

} // namespace accelsim::gemm
