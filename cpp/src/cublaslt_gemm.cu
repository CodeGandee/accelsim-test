#include "cublaslt_gemm.h"

#include <stdexcept>

namespace accelsim::gemm
{

namespace
{

void set_layout_order(cublasLtMatrixLayout_t layout, cublasLtOrder_t order)
{
  CublasLtGemmPlan::CheckLt(cublasLtMatrixLayoutSetAttribute(
                              layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
                            "cublasLtMatrixLayoutSetAttribute(ORDER)");
}

template <typename T>
bool try_get_algo_attr(const cublasLtMatmulAlgo_t &algo, cublasLtMatmulAlgoConfigAttributes_t attr, T &out)
{
  std::size_t written = 0;
  const cublasStatus_t st =
    cublasLtMatmulAlgoConfigGetAttribute(&algo, attr, &out, sizeof(out), &written);
  return st == CUBLAS_STATUS_SUCCESS && written == sizeof(out);
}

} // namespace

CublasLtGemmPlan::CublasLtGemmPlan(const GemmDims &dims,
                                   const GemmTypes &types,
                                   const MatrixDims &a_dims,
                                   const MatrixDims &b_dims,
                                   const MatrixDims &c_dims,
                                   cublasOperation_t trans_a,
                                   cublasOperation_t trans_b,
                                   const GemmPlanOptions &opts)
    : m_workspace_bytes{opts.max_workspace_bytes}
{
  (void)dims;
  CheckLt(cublasLtCreate(&m_handle), "cublasLtCreate");

  CheckLt(cublasLtMatmulDescCreate(&m_matmul_desc, types.compute_type, types.scale_type),
          "cublasLtMatmulDescCreate");
  CheckLt(cublasLtMatmulDescSetAttribute(
            m_matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)),
          "cublasLtMatmulDescSetAttribute(TRANSA)");
  CheckLt(cublasLtMatmulDescSetAttribute(
            m_matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)),
          "cublasLtMatmulDescSetAttribute(TRANSB)");

  CheckLt(cublasLtMatrixLayoutCreate(&m_a_layout, types.a_type, a_dims.rows, a_dims.cols, a_dims.ld),
          "cublasLtMatrixLayoutCreate(A)");
  CheckLt(cublasLtMatrixLayoutCreate(&m_b_layout, types.b_type, b_dims.rows, b_dims.cols, b_dims.ld),
          "cublasLtMatrixLayoutCreate(B)");
  CheckLt(cublasLtMatrixLayoutCreate(&m_c_layout, types.c_type, c_dims.rows, c_dims.cols, c_dims.ld),
          "cublasLtMatrixLayoutCreate(C)");

  set_layout_order(m_a_layout, opts.order);
  set_layout_order(m_b_layout, opts.order);
  set_layout_order(m_c_layout, opts.order);

  cublasLtMatmulPreference_t pref{};
  CheckLt(cublasLtMatmulPreferenceCreate(&pref), "cublasLtMatmulPreferenceCreate");
  CheckLt(cublasLtMatmulPreferenceSetAttribute(pref,
                                               CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &m_workspace_bytes,
                                               sizeof(m_workspace_bytes)),
          "cublasLtMatmulPreferenceSetAttribute(MAX_WORKSPACE_BYTES)");

  cublasLtMatmulHeuristicResult_t heur{};
  int found = 0;
  CheckLt(cublasLtMatmulAlgoGetHeuristic(m_handle,
                                        m_matmul_desc,
                                        m_a_layout,
                                        m_b_layout,
                                        m_c_layout,
                                        m_c_layout,
                                        pref,
                                        1,
                                        &heur,
                                        &found),
          "cublasLtMatmulAlgoGetHeuristic");
  CheckLt(cublasLtMatmulPreferenceDestroy(pref), "cublasLtMatmulPreferenceDestroy");

  if (found <= 0)
  {
    throw std::runtime_error("cublasLtMatmulAlgoGetHeuristic returned no algorithms.");
  }
  m_algo = heur.algo;

  // Record selected algorithm config for downstream reporting.
  // Note: Some attributes may not be queryable for every algorithm; keep best-effort defaults.
  (void)try_get_algo_attr(m_algo, CUBLASLT_ALGO_CONFIG_ID, m_algo_config.id);
  (void)try_get_algo_attr(m_algo, CUBLASLT_ALGO_CONFIG_TILE_ID, m_algo_config.tile_id);
  (void)try_get_algo_attr(m_algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, m_algo_config.splitk_num);
  (void)try_get_algo_attr(m_algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, m_algo_config.reduction_scheme);
  (void)try_get_algo_attr(m_algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, m_algo_config.cta_swizzling);
  (void)try_get_algo_attr(m_algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, m_algo_config.custom_option);
  (void)try_get_algo_attr(m_algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, m_algo_config.stages_id);
  (void)try_get_algo_attr(m_algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, m_algo_config.inner_shape_id);
  (void)try_get_algo_attr(m_algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, m_algo_config.cluster_shape_id);
  m_algo_config.required_workspace_bytes = static_cast<std::size_t>(heur.workspaceSize);
  m_algo_config.waves_count              = static_cast<std::int32_t>(heur.wavesCount);

  CheckCuda(cudaMalloc(&m_workspace, m_workspace_bytes), "cudaMalloc(workspace)");
}

CublasLtGemmPlan::~CublasLtGemmPlan()
{
  if (m_workspace)
  {
    cudaFree(m_workspace);
    m_workspace = nullptr;
  }
  if (m_a_layout)
  {
    cublasLtMatrixLayoutDestroy(m_a_layout);
    m_a_layout = nullptr;
  }
  if (m_b_layout)
  {
    cublasLtMatrixLayoutDestroy(m_b_layout);
    m_b_layout = nullptr;
  }
  if (m_c_layout)
  {
    cublasLtMatrixLayoutDestroy(m_c_layout);
    m_c_layout = nullptr;
  }
  if (m_matmul_desc)
  {
    cublasLtMatmulDescDestroy(m_matmul_desc);
    m_matmul_desc = nullptr;
  }
  if (m_handle)
  {
    cublasLtDestroy(m_handle);
    m_handle = nullptr;
  }
}

void CublasLtGemmPlan::Run(cudaStream_t stream,
                           const void *a,
                           const void *b,
                           void *c,
                           const void *alpha,
                           const void *beta) const
{
  CheckLt(cublasLtMatmul(m_handle,
                        m_matmul_desc,
                        alpha,
                        a,
                        m_a_layout,
                        b,
                        m_b_layout,
                        beta,
                        c,
                        m_c_layout,
                        c,
                        m_c_layout,
                        &m_algo,
                        m_workspace,
                        m_workspace_bytes,
                        stream),
          "cublasLtMatmul");
}

void CublasLtGemmPlan::CheckCuda(cudaError_t status, const char *what)
{
  if (status != cudaSuccess)
  {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

void CublasLtGemmPlan::CheckLt(cublasStatus_t status, const char *what)
{
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    throw std::runtime_error(std::string(what) + ": cublas status " + std::to_string(static_cast<int>(status)));
  }
}

} // namespace accelsim::gemm
