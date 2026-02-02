# GEMM Transpose Benchmark Report

- Branch: `002-gemm-transpose-bench`
- Commit: `27511017a4fd709d3c22f9fe8b7897121276a667`
- Status: `pass`

## Square Suite

| suite | N | dtype_pair | flop_count | A@B(ms) | A@B(algo_id) | A.T@B(ms) | A.T@B(algo_id) | A@B.T(ms) | A@B.T(algo_id) | copy(A.T)@B(ms) | copy(A.T)@B(algo_id) | A@copy(B.T)(ms) | A@copy(B.T)(algo_id) | verify |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| square | 512 | `bf16,bf16->bf16 (fp32,default)` | 268435456 | 0.012 | 31 | 0.012 | 31 | 0.011 | 31 | 0.011 | 31 | 0.011 | 31 | pass |
| square | 512 | `fp16,fp16->fp16 (fp32,default)` | 268435456 | 0.011 | 6 | 0.011 | 6 | 0.011 | 6 | 0.011 | 6 | 0.011 | 6 | pass |
| square | 512 | `fp32,fp32->fp32 (fp32,default)` | 268435456 | 0.033 | 1 | 0.032 | 1 | 0.036 | 1 | 0.033 | 1 | 0.033 | 1 | pass |
| square | 512 | `fp32,fp32->fp32 (tf32,tf32)` | 268435456 | 0.015 | 21 | 0.015 | 21 | 0.015 | 21 | 0.015 | 21 | 0.015 | 21 | pass |
| square | 512 | `int8,int8->int32 (int32,default)` | 268435456 | 0.023 | 0 | 0.028 | 0 | 0.010 | 21 | 0.024 | 0 | 0.023 | 0 | pass |
| square | 768 | `bf16,bf16->bf16 (fp32,default)` | 905969664 | 0.015 | 6 | 0.015 | 6 | 0.015 | 6 | 0.015 | 6 | 0.015 | 6 | pass |
| square | 768 | `fp16,fp16->fp16 (fp32,default)` | 905969664 | 0.015 | 6 | 0.015 | 6 | 0.015 | 6 | 0.015 | 6 | 0.015 | 6 | pass |
| square | 768 | `fp32,fp32->fp32 (fp32,default)` | 905969664 | 0.068 | 1 | 0.070 | 1 | 0.069 | 0 | 0.068 | 1 | 0.068 | 1 | pass |
| square | 768 | `fp32,fp32->fp32 (tf32,tf32)` | 905969664 | 0.023 | 21 | 0.023 | 21 | 0.023 | 21 | 0.023 | 21 | 0.023 | 21 | pass |
| square | 768 | `int8,int8->int32 (int32,default)` | 905969664 | 0.043 | 0 | 0.043 | 0 | 0.013 | 21 | 0.043 | 0 | 0.043 | 0 | pass |
| square | 896 | `bf16,bf16->bf16 (fp32,default)` | 1438646272 | 0.017 | 6 | 0.017 | 6 | 0.018 | 6 | 0.017 | 6 | 0.017 | 6 | pass |
| square | 896 | `fp16,fp16->fp16 (fp32,default)` | 1438646272 | 0.021 | 34 | 0.021 | 34 | 0.021 | 34 | 0.021 | 34 | 0.021 | 34 | pass |
| square | 896 | `fp32,fp32->fp32 (fp32,default)` | 1438646272 | 0.103 | 0 | 0.098 | 1 | 0.135 | 0 | 0.103 | 0 | 0.103 | 0 | pass |
| square | 896 | `fp32,fp32->fp32 (tf32,tf32)` | 1438646272 | 0.026 | 21 | 0.026 | 21 | 0.034 | 21 | 0.026 | 21 | 0.026 | 21 | pass |
| square | 896 | `int8,int8->int32 (int32,default)` | 1438646272 | 0.052 | 0 | 0.062 | 0 | 0.016 | 21 | 0.052 | 0 | 0.052 | 0 | pass |
| square | 960 | `bf16,bf16->bf16 (fp32,default)` | 1769472000 | 0.023 | 6 | 0.023 | 6 | 0.022 | 6 | 0.023 | 6 | 0.023 | 6 | pass |
| square | 960 | `fp16,fp16->fp16 (fp32,default)` | 1769472000 | 0.022 | 34 | 0.021 | 34 | 0.022 | 34 | 0.022 | 34 | 0.022 | 34 | pass |
| square | 960 | `fp32,fp32->fp32 (fp32,default)` | 1769472000 | 0.122 | 0 | 0.119 | 20 | 0.127 | 0 | 0.122 | 0 | 0.122 | 0 | pass |
| square | 960 | `fp32,fp32->fp32 (tf32,tf32)` | 1769472000 | 0.035 | 21 | 0.035 | 21 | 0.036 | 21 | 0.035 | 21 | 0.035 | 21 | pass |
| square | 960 | `int8,int8->int32 (int32,default)` | 1769472000 | 0.067 | 0 | 0.067 | 0 | 0.017 | 21 | 0.067 | 0 | 0.067 | 0 | pass |
| square | 992 | `bf16,bf16->bf16 (fp32,default)` | 1952382976 | 0.023 | 6 | 0.024 | 6 | 0.023 | 6 | 0.023 | 6 | 0.023 | 6 | pass |
| square | 992 | `fp16,fp16->fp16 (fp32,default)` | 1952382976 | 0.024 | 6 | 0.024 | 34 | 0.023 | 6 | 0.024 | 6 | 0.024 | 6 | pass |
| square | 992 | `fp32,fp32->fp32 (fp32,default)` | 1952382976 | 0.132 | 0 | 0.136 | 20 | 0.133 | 0 | 0.137 | 0 | 0.131 | 0 | pass |
| square | 992 | `fp32,fp32->fp32 (tf32,tf32)` | 1952382976 | 0.035 | 21 | 0.037 | 21 | 0.036 | 21 | 0.037 | 21 | 0.035 | 21 | pass |
| square | 992 | `int8,int8->int32 (int32,default)` | 1952382976 | 0.067 | 0 | 0.068 | 0 | 0.019 | 21 | 0.067 | 0 | 0.067 | 0 | pass |
| square | 1000 | `bf16,bf16->bf16 (fp32,default)` | 2000000000 | 0.024 | 6 | 0.024 | 6 | 0.023 | 6 | 0.024 | 6 | 0.024 | 6 | pass |
| square | 1000 | `fp16,fp16->fp16 (fp32,default)` | 2000000000 | 0.030 | 6 | 0.028 | 34 | 0.029 | 6 | 0.030 | 6 | 0.029 | 6 | pass |
| square | 1000 | `fp32,fp32->fp32 (fp32,default)` | 2000000000 | 0.132 | 0 | 0.139 | 0 | 0.133 | 0 | 0.142 | 0 | 0.132 | 0 | pass |
| square | 1000 | `fp32,fp32->fp32 (tf32,tf32)` | 2000000000 | 0.037 | 21 | 0.041 | 21 | 0.037 | 21 | 0.039 | 21 | 0.037 | 21 | pass |
| square | 1000 | `int8,int8->int32 (int32,default)` | 2000000000 | 0.069 | 0 | 0.069 | 0 | 0.038 | 23 | 0.069 | 0 | 0.069 | 0 | pass |
| square | 1024 | `bf16,bf16->bf16 (fp32,default)` | 2147483648 | 0.023 | 6 | 0.023 | 6 | 0.023 | 6 | 0.023 | 6 | 0.023 | 6 | pass |
| square | 1024 | `fp16,fp16->fp16 (fp32,default)` | 2147483648 | 0.023 | 34 | 0.023 | 34 | 0.024 | 6 | 0.023 | 34 | 0.023 | 34 | pass |
| square | 1024 | `fp32,fp32->fp32 (fp32,default)` | 2147483648 | 0.136 | 0 | 0.135 | 20 | 0.137 | 0 | 0.135 | 0 | 0.135 | 0 | pass |
| square | 1024 | `fp32,fp32->fp32 (tf32,tf32)` | 2147483648 | 0.036 | 21 | 0.037 | 21 | 0.037 | 21 | 0.036 | 21 | 0.036 | 21 | pass |
| square | 1024 | `int8,int8->int32 (int32,default)` | 2147483648 | 0.069 | 0 | 0.069 | 0 | 0.018 | 21 | 0.069 | 0 | 0.069 | 0 | pass |
| square | 1280 | `bf16,bf16->bf16 (fp32,default)` | 4194304000 | 0.026 | 6 | 0.026 | 6 | 0.026 | 6 | 0.026 | 6 | 0.026 | 6 | pass |
| square | 1280 | `fp16,fp16->fp16 (fp32,default)` | 4194304000 | 0.026 | 6 | 0.029 | 34 | 0.029 | 34 | 0.026 | 6 | 0.026 | 6 | pass |
| square | 1280 | `fp32,fp32->fp32 (fp32,default)` | 4194304000 | 0.256 | 20 | 0.255 | 1 | 0.257 | 0 | 0.255 | 20 | 0.255 | 20 | pass |
| square | 1280 | `fp32,fp32->fp32 (tf32,tf32)` | 4194304000 | 0.044 | 21 | 0.045 | 21 | 0.044 | 21 | 0.044 | 21 | 0.044 | 21 | pass |
| square | 1280 | `int8,int8->int32 (int32,default)` | 4194304000 | 0.100 | 0 | 0.104 | 0 | 0.027 | 21 | 0.100 | 0 | 0.100 | 0 | pass |
| square | 1536 | `bf16,bf16->bf16 (fp32,default)` | 7247757312 | 0.047 | 6 | 0.047 | 6 | 0.048 | 6 | 0.047 | 6 | 0.047 | 6 | pass |
| square | 1536 | `fp16,fp16->fp16 (fp32,default)` | 7247757312 | 0.043 | 34 | 0.044 | 34 | 0.042 | 34 | 0.043 | 34 | 0.043 | 34 | pass |
| square | 1536 | `fp32,fp32->fp32 (fp32,default)` | 7247757312 | 0.428 | 1 | 0.417 | 1 | 0.452 | 0 | 0.427 | 1 | 0.427 | 1 | pass |
| square | 1536 | `fp32,fp32->fp32 (tf32,tf32)` | 7247757312 | 0.082 | 21 | 0.083 | 21 | 0.082 | 21 | 0.082 | 21 | 0.082 | 21 | pass |
| square | 1536 | `int8,int8->int32 (int32,default)` | 7247757312 | 0.154 | 0 | 0.155 | 0 | 0.037 | 21 | 0.154 | 0 | 0.154 | 0 | pass |
| square | 1664 | `bf16,bf16->bf16 (fp32,default)` | 9214885888 | 0.051 | 6 | 0.050 | 6 | 0.051 | 6 | 0.050 | 6 | 0.051 | 6 | pass |
| square | 1664 | `fp16,fp16->fp16 (fp32,default)` | 9214885888 | 0.051 | 21 | 0.051 | 21 | 0.051 | 21 | 0.051 | 21 | 0.051 | 21 | pass |
| square | 1664 | `fp32,fp32->fp32 (fp32,default)` | 9214885888 | 0.542 | 1 | 0.528 | 1 | 0.589 | 0 | 0.542 | 1 | 0.541 | 1 | pass |
| square | 1664 | `fp32,fp32->fp32 (tf32,tf32)` | 9214885888 | 0.096 | 21 | 0.095 | 21 | 0.097 | 21 | 0.096 | 21 | 0.096 | 21 | pass |
| square | 1664 | `int8,int8->int32 (int32,default)` | 9214885888 | 0.200 | 0 | 0.207 | 0 | 0.038 | 21 | 0.200 | 0 | 0.200 | 0 | pass |
| square | 2048 | `bf16,bf16->bf16 (fp32,default)` | 17179869184 | 0.112 | 6 | 0.111 | 6 | 0.114 | 6 | 0.112 | 6 | 0.112 | 6 | pass |
| square | 2048 | `fp16,fp16->fp16 (fp32,default)` | 17179869184 | 0.077 | 34 | 0.082 | 34 | 0.079 | 34 | 0.077 | 34 | 0.077 | 34 | pass |
| square | 2048 | `fp32,fp32->fp32 (fp32,default)` | 17179869184 | 0.978 | 0 | 0.952 | 1 | 0.975 | 0 | 0.978 | 0 | 0.977 | 0 | pass |
| square | 2048 | `fp32,fp32->fp32 (tf32,tf32)` | 17179869184 | 0.179 | 21 | 0.181 | 21 | 0.179 | 21 | 0.180 | 21 | 0.179 | 21 | pass |
| square | 2048 | `int8,int8->int32 (int32,default)` | 17179869184 | 0.299 | 0 | 0.302 | 0 | 0.061 | 21 | 0.300 | 0 | 0.300 | 0 | pass |
| square | 2304 | `bf16,bf16->bf16 (fp32,default)` | 24461180928 | 0.124 | 6 | 0.123 | 6 | 0.126 | 6 | 0.124 | 6 | 0.124 | 6 | pass |
| square | 2304 | `fp16,fp16->fp16 (fp32,default)` | 24461180928 | 0.103 | 21 | 0.112 | 6 | 0.105 | 34 | 0.104 | 21 | 0.104 | 21 | pass |
| square | 2304 | `fp32,fp32->fp32 (fp32,default)` | 24461180928 | 1.289 | 0 | 1.319 | 20 | 1.310 | 0 | 1.289 | 0 | 1.289 | 0 | pass |
| square | 2304 | `fp32,fp32->fp32 (tf32,tf32)` | 24461180928 | 0.215 | 21 | 0.214 | 21 | 0.217 | 21 | 0.214 | 21 | 0.215 | 21 | pass |
| square | 2304 | `int8,int8->int32 (int32,default)` | 24461180928 | 0.335 | 0 | 0.338 | 0 | 0.090 | 21 | 0.335 | 0 | 0.334 | 0 | pass |
| square | 4096 | `bf16,bf16->bf16 (fp32,default)` | 137438953472 | 0.565 | 6 | 0.560 | 6 | 0.567 | 6 | 0.565 | 6 | 0.563 | 6 | pass |
| square | 4096 | `fp16,fp16->fp16 (fp32,default)` | 137438953472 | 0.581 | 6 | 0.572 | 6 | 0.621 | 21 | 0.581 | 6 | 0.581 | 6 | pass |
| square | 4096 | `fp32,fp32->fp32 (fp32,default)` | 137438953472 | 7.246 | 0 | 7.266 | 20 | 7.306 | 0 | 7.241 | 0 | 7.240 | 0 | pass |
| square | 4096 | `fp32,fp32->fp32 (tf32,tf32)` | 137438953472 | 1.161 | 21 | 1.177 | 21 | 1.150 | 21 | 1.162 | 21 | 1.161 | 21 | pass |
| square | 4096 | `int8,int8->int32 (int32,default)` | 137438953472 | 1.923 | 0 | 1.939 | 0 | 0.356 | 21 | 1.920 | 0 | 1.921 | 0 | pass |

## Non-square Suite

| suite | M | N | K | dtype_pair | flop_count | A.T@B(ms) | A.T@B(algo_id) | copy(A.T)@B(ms) | copy(A.T)@B(algo_id) | A@B.T(ms) | A@B.T(algo_id) | A@copy(B.T)(ms) | A@copy(B.T)(algo_id) | verify |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

## Column Definitions

### Square Suite

- `suite`: Suite identifier (`square`).
- `N`: Matrix size for square shapes (`M=N=K=N`).
- `dtype_pair`: Normalized dtype description `A,B->C (compute,math_mode)`.
- `flop_count`: Theoretical GEMM FLOPs (`2*N*N*N`).
- `A@B(ms)` etc: Mean GPU time in milliseconds from the NVBench timing run (GEMM-only; transpose materialization is outside timing).
- `A@B(algo_id)` etc: cuBLASLt algorithm ID used for that case (heuristic-selected or pinned via algo-map).
- `verify`: `pass` if all cases in the row passed verification; otherwise `fail`.

### Non-square Suite

- `suite`: Suite identifier (`non_square`) summarizing both transpose directions for the same `(M,N,K)` and dtype.
- `M,N,K`: Logical GEMM dimensions for `C[M,N] = A[M,K] @ B[K,N]` (FLOP-matched across non-square cases).
- `dtype_pair`: Normalized dtype description `A,B->C (compute,math_mode)`.
- `flop_count`: Theoretical GEMM FLOPs (`2*M*N*K`) used for row-consistency across compared cases.
- `A.T@B(ms)` / `copy(A.T)@B(ms)`: Times for transpose-A suite (`nonsquare_atb`).
- `A@B.T(ms)` / `A@copy(B.T)(ms)`: Times for transpose-B suite (`nonsquare_abt`).
- `...(algo_id)`: cuBLASLt algorithm ID used for that record; full per-record config lives in `results.json` under `record.cublaslt.algo`.
- `verify`: `pass` if all present non-square records for the row passed verification; otherwise `fail`.

Notes:
- `NA` means the value is missing (e.g., a case was not run).
- `flop_count` is always `2*M*N*K` even for integer cases; this report intentionally does not compute throughput columns.

## Conclusions

TBD: Populate after collecting results on target GPU(s).

