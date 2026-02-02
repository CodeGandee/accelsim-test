# GEMM Transpose Benchmark Report

- Branch: `002-gemm-transpose-bench`
- Commit: `27511017a4fd709d3c22f9fe8b7897121276a667`
- Status: `pass`

## Square Suite

| suite | N | dtype_pair | flop_count | A@B(ms) | A@B(algo_id) | A.T@B(ms) | A.T@B(algo_id) | A@B.T(ms) | A@B.T(algo_id) | copy(A.T)@B(ms) | copy(A.T)@B(algo_id) | A@copy(B.T)(ms) | A@copy(B.T)(algo_id) | verify |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| square | 512 | `bf16,bf16->bf16 (fp32,default)` | 268435456 | 0.011 | 31 | 0.012 | 31 | 0.011 | 31 | 0.011 | 31 | 0.011 | 31 | pass |
| square | 512 | `fp16,fp16->fp16 (fp32,default)` | 268435456 | 0.011 | 6 | 0.011 | 6 | 0.011 | 6 | 0.011 | 6 | 0.011 | 6 | pass |
| square | 512 | `fp32,fp32->fp32 (fp32,default)` | 268435456 | 0.033 | 1 | 0.032 | 1 | 0.036 | 1 | 0.033 | 1 | 0.033 | 1 | pass |
| square | 512 | `fp32,fp32->fp32 (tf32,tf32)` | 268435456 | 0.015 | 21 | 0.015 | 21 | 0.015 | 21 | 0.015 | 21 | 0.015 | 21 | pass |
| square | 512 | `int8,int8->int32 (int32,default)` | 268435456 | 0.024 | 0 | 0.028 | 0 | 0.010 | 21 | 0.024 | 0 | 0.023 | 0 | pass |
| square | 768 | `bf16,bf16->bf16 (fp32,default)` | 905969664 | 0.015 | 6 | 0.015 | 6 | 0.015 | 6 | 0.015 | 6 | 0.015 | 6 | pass |
| square | 768 | `fp16,fp16->fp16 (fp32,default)` | 905969664 | 0.015 | 6 | 0.015 | 6 | 0.015 | 6 | 0.015 | 6 | 0.015 | 6 | pass |
| square | 768 | `fp32,fp32->fp32 (fp32,default)` | 905969664 | 0.068 | 1 | 0.071 | 1 | 0.069 | 0 | 0.068 | 1 | 0.068 | 1 | pass |
| square | 768 | `fp32,fp32->fp32 (tf32,tf32)` | 905969664 | 0.023 | 21 | 0.023 | 21 | 0.023 | 21 | 0.023 | 21 | 0.023 | 21 | pass |
| square | 768 | `int8,int8->int32 (int32,default)` | 905969664 | 0.043 | 0 | 0.043 | 0 | 0.013 | 21 | 0.043 | 0 | 0.043 | 0 | pass |
| square | 896 | `bf16,bf16->bf16 (fp32,default)` | 1438646272 | 0.017 | 6 | 0.017 | 6 | 0.018 | 6 | 0.017 | 6 | 0.017 | 6 | pass |
| square | 896 | `fp16,fp16->fp16 (fp32,default)` | 1438646272 | 0.021 | 34 | 0.021 | 34 | 0.021 | 34 | 0.021 | 34 | 0.021 | 34 | pass |
| square | 896 | `fp32,fp32->fp32 (fp32,default)` | 1438646272 | 0.103 | 0 | 0.098 | 1 | 0.135 | 0 | 0.103 | 0 | 0.103 | 0 | pass |
| square | 896 | `fp32,fp32->fp32 (tf32,tf32)` | 1438646272 | 0.026 | 21 | 0.026 | 21 | 0.034 | 21 | 0.026 | 21 | 0.026 | 21 | pass |
| square | 896 | `int8,int8->int32 (int32,default)` | 1438646272 | 0.052 | 0 | 0.062 | 0 | 0.017 | 21 | 0.052 | 0 | 0.052 | 0 | pass |
| square | 960 | `bf16,bf16->bf16 (fp32,default)` | 1769472000 | 0.023 | 6 | 0.023 | 6 | 0.022 | 6 | 0.023 | 6 | 0.023 | 6 | pass |
| square | 960 | `fp16,fp16->fp16 (fp32,default)` | 1769472000 | 0.023 | 34 | 0.021 | 34 | 0.023 | 34 | 0.023 | 34 | 0.023 | 34 | pass |
| square | 960 | `fp32,fp32->fp32 (fp32,default)` | 1769472000 | 0.122 | 0 | 0.119 | 20 | 0.127 | 0 | 0.122 | 0 | 0.122 | 0 | pass |
| square | 960 | `fp32,fp32->fp32 (tf32,tf32)` | 1769472000 | 0.035 | 21 | 0.035 | 21 | 0.036 | 21 | 0.035 | 21 | 0.035 | 21 | pass |
| square | 960 | `int8,int8->int32 (int32,default)` | 1769472000 | 0.067 | 0 | 0.067 | 0 | 0.017 | 21 | 0.067 | 0 | 0.067 | 0 | pass |
| square | 992 | `bf16,bf16->bf16 (fp32,default)` | 1952382976 | 0.023 | 6 | 0.023 | 6 | 0.023 | 6 | 0.024 | 6 | 0.023 | 6 | pass |
| square | 992 | `fp16,fp16->fp16 (fp32,default)` | 1952382976 | 0.024 | 6 | 0.023 | 34 | 0.023 | 6 | 0.024 | 6 | 0.024 | 6 | pass |
| square | 992 | `fp32,fp32->fp32 (fp32,default)` | 1952382976 | 0.132 | 0 | 0.137 | 20 | 0.133 | 0 | 0.136 | 0 | 0.131 | 0 | pass |
| square | 992 | `fp32,fp32->fp32 (tf32,tf32)` | 1952382976 | 0.035 | 21 | 0.037 | 21 | 0.036 | 21 | 0.036 | 21 | 0.035 | 21 | pass |
| square | 992 | `int8,int8->int32 (int32,default)` | 1952382976 | 0.067 | 0 | 0.068 | 0 | 0.019 | 21 | 0.067 | 0 | 0.067 | 0 | pass |
| square | 1000 | `bf16,bf16->bf16 (fp32,default)` | 2000000000 | 0.024 | 6 | 0.025 | 6 | 0.023 | 6 | 0.024 | 6 | 0.023 | 6 | pass |
| square | 1000 | `fp16,fp16->fp16 (fp32,default)` | 2000000000 | 0.030 | 6 | 0.028 | 34 | 0.029 | 6 | 0.030 | 6 | 0.029 | 6 | pass |
| square | 1000 | `fp32,fp32->fp32 (fp32,default)` | 2000000000 | 0.132 | 0 | 0.143 | 0 | 0.133 | 0 | 0.138 | 0 | 0.132 | 0 | pass |
| square | 1000 | `fp32,fp32->fp32 (tf32,tf32)` | 2000000000 | 0.037 | 21 | 0.040 | 21 | 0.038 | 21 | 0.039 | 21 | 0.037 | 21 | pass |
| square | 1000 | `int8,int8->int32 (int32,default)` | 2000000000 | 0.069 | 0 | 0.069 | 0 | 0.038 | 23 | 0.069 | 0 | 0.069 | 0 | pass |
| square | 1024 | `bf16,bf16->bf16 (fp32,default)` | 2147483648 | 0.023 | 6 | 0.023 | 6 | 0.023 | 6 | 0.023 | 6 | 0.023 | 6 | pass |
| square | 1024 | `fp16,fp16->fp16 (fp32,default)` | 2147483648 | 0.023 | 34 | 0.023 | 34 | 0.024 | 6 | 0.023 | 34 | 0.023 | 34 | pass |
| square | 1024 | `fp32,fp32->fp32 (fp32,default)` | 2147483648 | 0.135 | 0 | 0.135 | 20 | 0.137 | 0 | 0.135 | 0 | 0.135 | 0 | pass |
| square | 1024 | `fp32,fp32->fp32 (tf32,tf32)` | 2147483648 | 0.036 | 21 | 0.037 | 21 | 0.037 | 21 | 0.036 | 21 | 0.036 | 21 | pass |
| square | 1024 | `int8,int8->int32 (int32,default)` | 2147483648 | 0.069 | 0 | 0.069 | 0 | 0.018 | 21 | 0.069 | 0 | 0.069 | 0 | pass |
| square | 1280 | `bf16,bf16->bf16 (fp32,default)` | 4194304000 | 0.026 | 6 | 0.026 | 6 | 0.026 | 6 | 0.027 | 6 | 0.026 | 6 | pass |
| square | 1280 | `fp16,fp16->fp16 (fp32,default)` | 4194304000 | 0.026 | 6 | 0.029 | 34 | 0.029 | 34 | 0.026 | 6 | 0.026 | 6 | pass |
| square | 1280 | `fp32,fp32->fp32 (fp32,default)` | 4194304000 | 0.256 | 20 | 0.255 | 1 | 0.257 | 0 | 0.255 | 20 | 0.255 | 20 | pass |
| square | 1280 | `fp32,fp32->fp32 (tf32,tf32)` | 4194304000 | 0.044 | 21 | 0.045 | 21 | 0.044 | 21 | 0.044 | 21 | 0.044 | 21 | pass |
| square | 1280 | `int8,int8->int32 (int32,default)` | 4194304000 | 0.100 | 0 | 0.104 | 0 | 0.027 | 21 | 0.100 | 0 | 0.100 | 0 | pass |
| square | 1536 | `bf16,bf16->bf16 (fp32,default)` | 7247757312 | 0.047 | 6 | 0.047 | 6 | 0.048 | 6 | 0.048 | 6 | 0.047 | 6 | pass |
| square | 1536 | `fp16,fp16->fp16 (fp32,default)` | 7247757312 | 0.043 | 34 | 0.043 | 34 | 0.042 | 34 | 0.043 | 34 | 0.043 | 34 | pass |
| square | 1536 | `fp32,fp32->fp32 (fp32,default)` | 7247757312 | 0.428 | 1 | 0.417 | 1 | 0.452 | 0 | 0.425 | 1 | 0.426 | 1 | pass |
| square | 1536 | `fp32,fp32->fp32 (tf32,tf32)` | 7247757312 | 0.081 | 21 | 0.083 | 21 | 0.082 | 21 | 0.082 | 21 | 0.082 | 21 | pass |
| square | 1536 | `int8,int8->int32 (int32,default)` | 7247757312 | 0.154 | 0 | 0.154 | 0 | 0.037 | 21 | 0.154 | 0 | 0.154 | 0 | pass |
| square | 1664 | `bf16,bf16->bf16 (fp32,default)` | 9214885888 | 0.050 | 6 | 0.050 | 6 | 0.051 | 6 | 0.050 | 6 | 0.050 | 6 | pass |
| square | 1664 | `fp16,fp16->fp16 (fp32,default)` | 9214885888 | 0.051 | 21 | 0.051 | 21 | 0.052 | 21 | 0.051 | 21 | 0.051 | 21 | pass |
| square | 1664 | `fp32,fp32->fp32 (fp32,default)` | 9214885888 | 0.542 | 1 | 0.528 | 1 | 0.590 | 0 | 0.542 | 1 | 0.542 | 1 | pass |
| square | 1664 | `fp32,fp32->fp32 (tf32,tf32)` | 9214885888 | 0.096 | 21 | 0.096 | 21 | 0.097 | 21 | 0.096 | 21 | 0.096 | 21 | pass |
| square | 1664 | `int8,int8->int32 (int32,default)` | 9214885888 | 0.200 | 0 | 0.207 | 0 | 0.038 | 21 | 0.200 | 0 | 0.200 | 0 | pass |
| square | 2048 | `bf16,bf16->bf16 (fp32,default)` | 17179869184 | 0.112 | 6 | 0.111 | 6 | 0.113 | 6 | 0.112 | 6 | 0.112 | 6 | pass |
| square | 2048 | `fp16,fp16->fp16 (fp32,default)` | 17179869184 | 0.077 | 34 | 0.082 | 34 | 0.079 | 34 | 0.077 | 34 | 0.077 | 34 | pass |
| square | 2048 | `fp32,fp32->fp32 (fp32,default)` | 17179869184 | 0.978 | 0 | 0.953 | 1 | 0.976 | 0 | 0.978 | 0 | 0.978 | 0 | pass |
| square | 2048 | `fp32,fp32->fp32 (tf32,tf32)` | 17179869184 | 0.179 | 21 | 0.181 | 21 | 0.180 | 21 | 0.179 | 21 | 0.180 | 21 | pass |
| square | 2048 | `int8,int8->int32 (int32,default)` | 17179869184 | 0.300 | 0 | 0.302 | 0 | 0.061 | 21 | 0.299 | 0 | 0.299 | 0 | pass |
| square | 2304 | `bf16,bf16->bf16 (fp32,default)` | 24461180928 | 0.124 | 6 | 0.123 | 6 | 0.126 | 6 | 0.124 | 6 | 0.124 | 6 | pass |
| square | 2304 | `fp16,fp16->fp16 (fp32,default)` | 24461180928 | 0.104 | 21 | 0.113 | 6 | 0.106 | 34 | 0.104 | 21 | 0.104 | 21 | pass |
| square | 2304 | `fp32,fp32->fp32 (fp32,default)` | 24461180928 | 1.288 | 0 | 1.321 | 20 | 1.308 | 0 | 1.288 | 0 | 1.289 | 0 | pass |
| square | 2304 | `fp32,fp32->fp32 (tf32,tf32)` | 24461180928 | 0.216 | 21 | 0.213 | 21 | 0.218 | 21 | 0.216 | 21 | 0.216 | 21 | pass |
| square | 2304 | `int8,int8->int32 (int32,default)` | 24461180928 | 0.335 | 0 | 0.338 | 0 | 0.091 | 21 | 0.335 | 0 | 0.335 | 0 | pass |
| square | 4096 | `bf16,bf16->bf16 (fp32,default)` | 137438953472 | 0.566 | 6 | 0.562 | 6 | 0.569 | 6 | 0.564 | 6 | 0.564 | 6 | pass |
| square | 4096 | `fp16,fp16->fp16 (fp32,default)` | 137438953472 | 0.578 | 6 | 0.573 | 6 | 0.619 | 21 | 0.581 | 6 | 0.582 | 6 | pass |
| square | 4096 | `fp32,fp32->fp32 (fp32,default)` | 137438953472 | 7.238 | 0 | 7.219 | 20 | 7.299 | 0 | 7.236 | 0 | 7.236 | 0 | pass |
| square | 4096 | `fp32,fp32->fp32 (tf32,tf32)` | 137438953472 | 1.163 | 21 | 1.176 | 21 | 1.147 | 21 | 1.159 | 21 | 1.160 | 21 | pass |
| square | 4096 | `int8,int8->int32 (int32,default)` | 137438953472 | 1.924 | 0 | 1.935 | 0 | 0.357 | 21 | 1.923 | 0 | 1.924 | 0 | pass |

## Non-square Suite

| suite | M | N | K | dtype_pair | flop_count | A.T@B(ms) | A.T@B(algo_id) | copy(A.T)@B(ms) | copy(A.T)@B(algo_id) | A@B.T(ms) | A@B.T(algo_id) | A@copy(B.T)(ms) | A@copy(B.T)(algo_id) | verify |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| non_square | 256 | 256 | 992 | `bf16,bf16->bf16 (fp32,default)` | 130023424 | 0.013 | 31 | 0.013 | 31 | 0.013 | 31 | 0.013 | 31 | pass |
| non_square | 256 | 256 | 992 | `fp16,fp16->fp16 (fp32,default)` | 130023424 | 0.013 | 31 | 0.013 | 31 | 0.013 | 31 | 0.013 | 31 | pass |
| non_square | 256 | 256 | 992 | `fp32,fp32->fp32 (fp32,default)` | 130023424 | 0.025 | 1 | 0.028 | 1 | 0.026 | 0 | 0.028 | 1 | pass |
| non_square | 256 | 256 | 992 | `fp32,fp32->fp32 (tf32,tf32)` | 130023424 | 0.017 | 21 | 0.017 | 21 | 0.016 | 21 | 0.017 | 21 | pass |
| non_square | 256 | 256 | 992 | `int8,int8->int32 (int32,default)` | 130023424 | 0.023 | 0 | 0.023 | 0 | 0.011 | 21 | 0.023 | 0 | pass |
| non_square | 256 | 992 | 256 | `bf16,bf16->bf16 (fp32,default)` | 130023424 | 0.011 | 31 | 0.011 | 31 | 0.011 | 31 | 0.011 | 31 | pass |
| non_square | 256 | 992 | 256 | `fp16,fp16->fp16 (fp32,default)` | 130023424 | 0.011 | 31 | 0.011 | 31 | 0.011 | 31 | 0.011 | 31 | pass |
| non_square | 256 | 992 | 256 | `fp32,fp32->fp32 (fp32,default)` | 130023424 | 0.021 | 20 | 0.022 | 1 | 0.023 | 1 | 0.022 | 1 | pass |
| non_square | 256 | 992 | 256 | `fp32,fp32->fp32 (tf32,tf32)` | 130023424 | 0.012 | 21 | 0.012 | 21 | 0.012 | 21 | 0.012 | 21 | pass |
| non_square | 256 | 992 | 256 | `int8,int8->int32 (int32,default)` | 130023424 | 0.020 | 0 | 0.022 | 0 | 0.009 | 21 | 0.023 | 0 | pass |
| non_square | 960 | 320 | 640 | `bf16,bf16->bf16 (fp32,default)` | 393216000 | 0.012 | 31 | 0.012 | 31 | 0.013 | 31 | 0.013 | 31 | pass |
| non_square | 960 | 320 | 640 | `fp16,fp16->fp16 (fp32,default)` | 393216000 | 0.012 | 31 | 0.019 | 21 | 0.015 | 34 | 0.019 | 21 | pass |
| non_square | 960 | 320 | 640 | `fp32,fp32->fp32 (fp32,default)` | 393216000 | 0.038 | 1 | 0.039 | 1 | 0.042 | 1 | 0.039 | 1 | pass |
| non_square | 960 | 320 | 640 | `fp32,fp32->fp32 (tf32,tf32)` | 393216000 | 0.022 | 21 | 0.016 | 21 | 0.016 | 21 | 0.016 | 21 | pass |
| non_square | 960 | 320 | 640 | `int8,int8->int32 (int32,default)` | 393216000 | 0.035 | 0 | 0.033 | 0 | 0.012 | 21 | 0.033 | 0 | pass |
| non_square | 992 | 256 | 256 | `bf16,bf16->bf16 (fp32,default)` | 130023424 | 0.011 | 31 | 0.011 | 31 | 0.010 | 31 | 0.011 | 31 | pass |
| non_square | 992 | 256 | 256 | `fp16,fp16->fp16 (fp32,default)` | 130023424 | 0.011 | 31 | 0.011 | 6 | 0.011 | 6 | 0.011 | 6 | pass |
| non_square | 992 | 256 | 256 | `fp32,fp32->fp32 (fp32,default)` | 130023424 | 0.021 | 1 | 0.022 | 1 | 0.023 | 1 | 0.022 | 1 | pass |
| non_square | 992 | 256 | 256 | `fp32,fp32->fp32 (tf32,tf32)` | 130023424 | 0.012 | 21 | 0.012 | 21 | 0.012 | 21 | 0.012 | 21 | pass |
| non_square | 992 | 256 | 256 | `int8,int8->int32 (int32,default)` | 130023424 | 0.020 | 0 | 0.022 | 0 | 0.009 | 21 | 0.023 | 0 | pass |
| non_square | 1024 | 1024 | 4096 | `bf16,bf16->bf16 (fp32,default)` | 8589934592 | 0.053 | 6 | 0.059 | 6 | 0.059 | 6 | 0.059 | 6 | pass |
| non_square | 1024 | 1024 | 4096 | `fp16,fp16->fp16 (fp32,default)` | 8589934592 | 0.053 | 34 | 0.057 | 34 | 0.059 | 34 | 0.057 | 34 | pass |
| non_square | 1024 | 1024 | 4096 | `fp32,fp32->fp32 (fp32,default)` | 8589934592 | 0.484 | 0 | 0.477 | 0 | 0.481 | 0 | 0.477 | 0 | pass |
| non_square | 1024 | 1024 | 4096 | `fp32,fp32->fp32 (tf32,tf32)` | 8589934592 | 0.112 | 21 | 0.094 | 21 | 0.099 | 21 | 0.095 | 21 | pass |
| non_square | 1024 | 1024 | 4096 | `int8,int8->int32 (int32,default)` | 8589934592 | 0.173 | 0 | 0.172 | 0 | 0.041 | 21 | 0.172 | 0 | pass |
| non_square | 1024 | 4096 | 1024 | `bf16,bf16->bf16 (fp32,default)` | 8589934592 | 0.064 | 6 | 0.064 | 6 | 0.065 | 6 | 0.064 | 6 | pass |
| non_square | 1024 | 4096 | 1024 | `fp16,fp16->fp16 (fp32,default)` | 8589934592 | 0.057 | 6 | 0.051 | 21 | 0.052 | 21 | 0.051 | 21 | pass |
| non_square | 1024 | 4096 | 1024 | `fp32,fp32->fp32 (fp32,default)` | 8589934592 | 0.483 | 1 | 0.492 | 0 | 0.494 | 0 | 0.492 | 0 | pass |
| non_square | 1024 | 4096 | 1024 | `fp32,fp32->fp32 (tf32,tf32)` | 8589934592 | 0.095 | 21 | 0.094 | 21 | 0.098 | 21 | 0.095 | 21 | pass |
| non_square | 1024 | 4096 | 1024 | `int8,int8->int32 (int32,default)` | 8589934592 | 0.157 | 0 | 0.156 | 0 | 0.039 | 21 | 0.156 | 0 | pass |
| non_square | 1536 | 2304 | 1536 | `bf16,bf16->bf16 (fp32,default)` | 10871635968 | NA | NA | NA | NA | 0.049 | 6 | 0.048 | 6 | pass |
| non_square | 1536 | 2304 | 1536 | `fp16,fp16->fp16 (fp32,default)` | 10871635968 | NA | NA | NA | NA | 0.051 | 34 | 0.050 | 21 | pass |
| non_square | 1536 | 2304 | 1536 | `fp32,fp32->fp32 (fp32,default)` | 10871635968 | NA | NA | NA | NA | 0.593 | 0 | 0.588 | 20 | pass |
| non_square | 1536 | 2304 | 1536 | `fp32,fp32->fp32 (tf32,tf32)` | 10871635968 | NA | NA | NA | NA | 0.094 | 21 | 0.093 | 21 | pass |
| non_square | 1536 | 2304 | 1536 | `int8,int8->int32 (int32,default)` | 10871635968 | NA | NA | NA | NA | 0.048 | 21 | 0.155 | 0 | pass |
| non_square | 2048 | 3072 | 2048 | `bf16,bf16->bf16 (fp32,default)` | 25769803776 | NA | NA | NA | NA | 0.115 | 6 | 0.113 | 6 | pass |
| non_square | 2048 | 3072 | 2048 | `fp16,fp16->fp16 (fp32,default)` | 25769803776 | NA | NA | NA | NA | 0.118 | 34 | 0.119 | 34 | pass |
| non_square | 2048 | 3072 | 2048 | `fp32,fp32->fp32 (fp32,default)` | 25769803776 | NA | NA | NA | NA | 1.596 | 0 | 1.498 | 0 | pass |
| non_square | 2048 | 3072 | 2048 | `fp32,fp32->fp32 (tf32,tf32)` | 25769803776 | NA | NA | NA | NA | 0.246 | 21 | 0.237 | 21 | pass |
| non_square | 2048 | 3072 | 2048 | `int8,int8->int32 (int32,default)` | 25769803776 | NA | NA | NA | NA | 0.081 | 21 | 0.394 | 0 | pass |
| non_square | 2304 | 1536 | 1536 | `bf16,bf16->bf16 (fp32,default)` | 10871635968 | 0.048 | 6 | 0.048 | 6 | NA | NA | NA | NA | pass |
| non_square | 2304 | 1536 | 1536 | `fp16,fp16->fp16 (fp32,default)` | 10871635968 | 0.048 | 21 | 0.050 | 21 | NA | NA | NA | NA | pass |
| non_square | 2304 | 1536 | 1536 | `fp32,fp32->fp32 (fp32,default)` | 10871635968 | 0.594 | 20 | 0.579 | 0 | NA | NA | NA | NA | pass |
| non_square | 2304 | 1536 | 1536 | `fp32,fp32->fp32 (tf32,tf32)` | 10871635968 | 0.092 | 21 | 0.091 | 21 | NA | NA | NA | NA | pass |
| non_square | 2304 | 1536 | 1536 | `int8,int8->int32 (int32,default)` | 10871635968 | 0.157 | 0 | 0.155 | 0 | NA | NA | NA | NA | pass |
| non_square | 3072 | 2048 | 2048 | `bf16,bf16->bf16 (fp32,default)` | 25769803776 | 0.111 | 6 | 0.113 | 6 | NA | NA | NA | NA | pass |
| non_square | 3072 | 2048 | 2048 | `fp16,fp16->fp16 (fp32,default)` | 25769803776 | 0.113 | 6 | 0.116 | 6 | NA | NA | NA | NA | pass |
| non_square | 3072 | 2048 | 2048 | `fp32,fp32->fp32 (fp32,default)` | 25769803776 | 1.429 | 1 | 1.470 | 0 | NA | NA | NA | NA | pass |
| non_square | 3072 | 2048 | 2048 | `fp32,fp32->fp32 (tf32,tf32)` | 25769803776 | 0.241 | 21 | 0.228 | 21 | NA | NA | NA | NA | pass |
| non_square | 3072 | 2048 | 2048 | `int8,int8->int32 (int32,default)` | 25769803776 | 0.398 | 0 | 0.394 | 0 | NA | NA | NA | NA | pass |
| non_square | 4096 | 1024 | 1024 | `bf16,bf16->bf16 (fp32,default)` | 8589934592 | 0.064 | 6 | 0.064 | 6 | 0.066 | 6 | 0.064 | 6 | pass |
| non_square | 4096 | 1024 | 1024 | `fp16,fp16->fp16 (fp32,default)` | 8589934592 | 0.049 | 34 | 0.057 | 6 | 0.053 | 34 | 0.057 | 6 | pass |
| non_square | 4096 | 1024 | 1024 | `fp32,fp32->fp32 (fp32,default)` | 8589934592 | 0.487 | 1 | 0.492 | 0 | 0.496 | 0 | 0.492 | 0 | pass |
| non_square | 4096 | 1024 | 1024 | `fp32,fp32->fp32 (tf32,tf32)` | 8589934592 | 0.095 | 21 | 0.093 | 21 | 0.093 | 21 | 0.093 | 21 | pass |
| non_square | 4096 | 1024 | 1024 | `int8,int8->int32 (int32,default)` | 8589934592 | 0.157 | 0 | 0.156 | 0 | 0.039 | 21 | 0.156 | 0 | pass |
| non_square | 8192 | 1024 | 1024 | `bf16,bf16->bf16 (fp32,default)` | 17179869184 | 0.080 | 6 | 0.080 | 6 | 0.095 | 6 | 0.079 | 6 | pass |
| non_square | 8192 | 1024 | 1024 | `fp16,fp16->fp16 (fp32,default)` | 17179869184 | 0.080 | 6 | 0.080 | 6 | 0.081 | 6 | 0.080 | 6 | pass |
| non_square | 8192 | 1024 | 1024 | `fp32,fp32->fp32 (fp32,default)` | 17179869184 | 0.920 | 20 | 0.937 | 0 | 1.000 | 0 | 0.936 | 0 | pass |
| non_square | 8192 | 1024 | 1024 | `fp32,fp32->fp32 (tf32,tf32)` | 17179869184 | 0.161 | 21 | 0.158 | 21 | 0.160 | 21 | 0.159 | 21 | pass |
| non_square | 8192 | 1024 | 1024 | `int8,int8->int32 (int32,default)` | 17179869184 | 0.256 | 0 | 0.253 | 0 | 0.065 | 21 | 0.253 | 0 | pass |

## Column Definitions

### Square Suite

- `suite`: Suite identifier (`square`).
- `N`: Matrix size for square shapes (`M=N=K=N`).
- `dtype_pair`: Normalized dtype description `A,B->C (compute,math_mode)`.
- `flop_count`: Theoretical GEMM FLOPs (`2*N*N*N`).
- `A@B(ms)` etc: Mean GPU time in milliseconds from the NVBench timing run (GEMM-only; transpose materialization is outside timing).
- `A@B(algo_id)` etc: cuBLASLt algorithm ID selected for that case (from `cublasLtMatmulAlgoGetHeuristic`).
- `verify`: `pass` if all cases in the row passed verification; otherwise `fail`.

### Non-square Suite

- `suite`: Suite identifier (`non_square`) summarizing both transpose directions for the same `(M,N,K)` and dtype.
- `M,N,K`: Logical GEMM dimensions for `C[M,N] = A[M,K] @ B[K,N]` (FLOP-matched across non-square cases).
- `dtype_pair`: Normalized dtype description `A,B->C (compute,math_mode)`.
- `flop_count`: Theoretical GEMM FLOPs (`2*M*N*K`) used for row-consistency across compared cases.
- `A.T@B(ms)` / `copy(A.T)@B(ms)`: Times for transpose-A suite (`nonsquare_atb`).
- `A@B.T(ms)` / `A@copy(B.T)(ms)`: Times for transpose-B suite (`nonsquare_abt`).
- `...(algo_id)`: cuBLASLt algorithm ID selected for that record; full per-record config lives in `results.json` under `record.cublaslt.algo`.
- `verify`: `pass` if all present non-square records for the row passed verification; otherwise `fail`.

Notes:
- `NA` means the value is missing (e.g., a case was not run).
- `flop_count` is always `2*M*N*K` even for integer cases; this report intentionally does not compute throughput columns.

## Conclusions

TBD: Populate after collecting results on target GPU(s).

