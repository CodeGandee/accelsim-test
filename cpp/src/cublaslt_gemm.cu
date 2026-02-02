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
