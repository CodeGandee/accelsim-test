# GEMM Transpose Benchmark Report

- Branch: `main`
- Commit: `7c44854aec96ac066990bb3151978fced52152d9`
- Status: `pass`

## Square Suite

| suite | N | dtype_pair | flop_count | A@B(ms) | A@B(algo_id) | A.T@B(ms) | A.T@B(algo_id) | A@B.T(ms) | A@B.T(algo_id) | copy(A.T)@B(ms) | copy(A.T)@B(algo_id) | A@copy(B.T)(ms) | A@copy(B.T)(algo_id) | verify |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| square | 512 | `bf16,bf16->bf16 (fp32,default)` | 268435456 | 0.012 | 66 | 0.013 | 66 | 0.012 | 66 | 0.012 | 66 | 0.012 | 66 | pass |
| square | 512 | `fp16,fp16->fp16 (fp32,default)` | 268435456 | 0.012 | 66 | 0.012 | 66 | 0.012 | 66 | 0.012 | 66 | 0.012 | 66 | pass |
| square | 512 | `fp32,fp32->fp32 (fp32,default)` | 268435456 | 0.023 | 76 | 0.023 | 76 | 0.023 | 76 | 0.023 | 76 | 0.023 | 76 | pass |
| square | 512 | `fp32,fp32->fp32 (tf32,tf32)` | 268435456 | 0.017 | 73 | 0.017 | 73 | 0.017 | 73 | 0.017 | 73 | 0.017 | 73 | pass |
| square | 512 | `int8,int8->int32 (int32,default)` | 268435456 | 0.018 | 71 | 0.018 | 71 | 0.018 | 71 | 0.018 | 71 | 0.018 | 71 | pass |
| square | 768 | `bf16,bf16->bf16 (fp32,default)` | 905969664 | 0.013 | 66 | 0.013 | 66 | 0.013 | 66 | 0.013 | 66 | 0.013 | 66 | pass |
| square | 768 | `fp16,fp16->fp16 (fp32,default)` | 905969664 | 0.013 | 66 | 0.013 | 66 | 0.013 | 66 | 0.013 | 66 | 0.013 | 66 | pass |
| square | 768 | `fp32,fp32->fp32 (fp32,default)` | 905969664 | 0.031 | 76 | 0.031 | 76 | 0.033 | 76 | 0.032 | 76 | 0.031 | 76 | pass |
| square | 768 | `fp32,fp32->fp32 (tf32,tf32)` | 905969664 | 0.018 | 73 | 0.018 | 73 | 0.019 | 73 | 0.018 | 73 | 0.018 | 73 | pass |
| square | 768 | `int8,int8->int32 (int32,default)` | 905969664 | 0.019 | 71 | 0.019 | 71 | 0.019 | 71 | 0.019 | 71 | 0.019 | 71 | pass |
| square | 896 | `bf16,bf16->bf16 (fp32,default)` | 1438646272 | 0.013 | 66 | 0.013 | 66 | 0.013 | 66 | 0.013 | 66 | 0.013 | 66 | pass |
| square | 896 | `fp16,fp16->fp16 (fp32,default)` | 1438646272 | 0.013 | 66 | 0.013 | 66 | 0.013 | 66 | 0.013 | 66 | 0.013 | 66 | pass |
| square | 896 | `fp32,fp32->fp32 (fp32,default)` | 1438646272 | 0.048 | 76 | 0.047 | 20 | 0.049 | 76 | 0.048 | 76 | 0.048 | 76 | pass |
| square | 896 | `fp32,fp32->fp32 (tf32,tf32)` | 1438646272 | 0.019 | 73 | 0.019 | 73 | 0.019 | 73 | 0.019 | 73 | 0.019 | 73 | pass |
| square | 896 | `int8,int8->int32 (int32,default)` | 1438646272 | 0.019 | 71 | 0.019 | 71 | 0.019 | 71 | 0.019 | 71 | 0.019 | 71 | pass |
| square | 960 | `bf16,bf16->bf16 (fp32,default)` | 1769472000 | 0.014 | 66 | 0.014 | 66 | 0.014 | 66 | 0.014 | 66 | 0.014 | 66 | pass |
| square | 960 | `fp16,fp16->fp16 (fp32,default)` | 1769472000 | 0.014 | 66 | 0.014 | 66 | 0.014 | 66 | 0.014 | 66 | 0.014 | 66 | pass |
| square | 960 | `fp32,fp32->fp32 (fp32,default)` | 1769472000 | 0.050 | 76 | 0.050 | 76 | 0.052 | 76 | 0.051 | 76 | 0.051 | 76 | pass |
| square | 960 | `fp32,fp32->fp32 (tf32,tf32)` | 1769472000 | 0.019 | 73 | 0.019 | 73 | 0.019 | 73 | 0.019 | 73 | 0.019 | 73 | pass |
| square | 960 | `int8,int8->int32 (int32,default)` | 1769472000 | 0.019 | 71 | 0.019 | 71 | 0.019 | 71 | 0.019 | 71 | 0.019 | 71 | pass |
| square | 992 | `bf16,bf16->bf16 (fp32,default)` | 1952382976 | 0.013 | 66 | 0.014 | 66 | 0.014 | 66 | 0.014 | 66 | 0.014 | 66 | pass |
| square | 992 | `fp16,fp16->fp16 (fp32,default)` | 1952382976 | 0.014 | 66 | 0.014 | 66 | 0.014 | 66 | 0.014 | 66 | 0.014 | 66 | pass |
| square | 992 | `fp32,fp32->fp32 (fp32,default)` | 1952382976 | 0.052 | 76 | 0.052 | 76 | 0.054 | 76 | 0.052 | 76 | 0.052 | 76 | pass |
| square | 992 | `fp32,fp32->fp32 (tf32,tf32)` | 1952382976 | 0.019 | 73 | 0.019 | 73 | 0.019 | 73 | 0.019 | 73 | 0.019 | 73 | pass |
| square | 992 | `int8,int8->int32 (int32,default)` | 1952382976 | 0.019 | 71 | 0.019 | 71 | 0.019 | 71 | 0.019 | 71 | 0.019 | 71 | pass |
| square | 1000 | `bf16,bf16->bf16 (fp32,default)` | 2000000000 | 0.015 | 66 | 0.015 | 66 | 0.015 | 66 | 0.015 | 66 | 0.015 | 66 | pass |
| square | 1000 | `fp16,fp16->fp16 (fp32,default)` | 2000000000 | 0.015 | 66 | 0.015 | 66 | 0.015 | 66 | 0.015 | 66 | 0.015 | 66 | pass |
| square | 1000 | `fp32,fp32->fp32 (fp32,default)` | 2000000000 | 0.053 | 76 | 0.053 | 76 | 0.054 | 76 | 0.053 | 76 | 0.053 | 76 | pass |
| square | 1000 | `fp32,fp32->fp32 (tf32,tf32)` | 2000000000 | 0.019 | 73 | 0.019 | 73 | 0.019 | 73 | 0.019 | 73 | 0.019 | 73 | pass |
| square | 1000 | `int8,int8->int32 (int32,default)` | 2000000000 | 0.045 | 64 | 0.047 | 64 | 0.020 | 23 | 0.045 | 64 | 0.045 | 64 | pass |
| square | 1024 | `bf16,bf16->bf16 (fp32,default)` | 2147483648 | 0.014 | 66 | 0.014 | 66 | 0.014 | 66 | 0.014 | 66 | 0.014 | 66 | pass |
| square | 1024 | `fp16,fp16->fp16 (fp32,default)` | 2147483648 | 0.014 | 66 | 0.014 | 66 | 0.014 | 66 | 0.014 | 66 | 0.014 | 66 | pass |
| square | 1024 | `fp32,fp32->fp32 (fp32,default)` | 2147483648 | 0.053 | 76 | 0.053 | 76 | 0.055 | 76 | 0.054 | 76 | 0.054 | 76 | pass |
| square | 1024 | `fp32,fp32->fp32 (tf32,tf32)` | 2147483648 | 0.019 | 73 | 0.019 | 73 | 0.020 | 73 | 0.019 | 73 | 0.019 | 73 | pass |
| square | 1024 | `int8,int8->int32 (int32,default)` | 2147483648 | 0.019 | 71 | 0.019 | 71 | 0.019 | 71 | 0.019 | 71 | 0.019 | 71 | pass |
| square | 1280 | `bf16,bf16->bf16 (fp32,default)` | 4194304000 | 0.015 | 66 | 0.015 | 66 | 0.016 | 66 | 0.015 | 66 | 0.015 | 66 | pass |
| square | 1280 | `fp16,fp16->fp16 (fp32,default)` | 4194304000 | 0.015 | 66 | 0.015 | 66 | 0.016 | 66 | 0.015 | 66 | 0.015 | 66 | pass |
| square | 1280 | `fp32,fp32->fp32 (fp32,default)` | 4194304000 | 0.096 | 76 | 0.094 | 76 | 0.099 | 76 | 0.096 | 76 | 0.096 | 76 | pass |
| square | 1280 | `fp32,fp32->fp32 (tf32,tf32)` | 4194304000 | 0.023 | 73 | 0.023 | 73 | 0.023 | 73 | 0.023 | 73 | 0.023 | 73 | pass |
| square | 1280 | `int8,int8->int32 (int32,default)` | 4194304000 | 0.020 | 71 | 0.020 | 71 | 0.020 | 71 | 0.020 | 71 | 0.020 | 71 | pass |
| square | 1536 | `bf16,bf16->bf16 (fp32,default)` | 7247757312 | 0.017 | 66 | 0.016 | 66 | 0.017 | 66 | 0.017 | 66 | 0.017 | 66 | pass |
| square | 1536 | `fp16,fp16->fp16 (fp32,default)` | 7247757312 | 0.017 | 66 | 0.017 | 66 | 0.017 | 66 | 0.017 | 66 | 0.017 | 66 | pass |
| square | 1536 | `fp32,fp32->fp32 (fp32,default)` | 7247757312 | 0.130 | 76 | 0.130 | 76 | 0.130 | 76 | 0.130 | 76 | 0.130 | 76 | pass |
| square | 1536 | `fp32,fp32->fp32 (tf32,tf32)` | 7247757312 | 0.025 | 73 | 0.025 | 73 | 0.025 | 73 | 0.025 | 73 | 0.026 | 73 | pass |
| square | 1536 | `int8,int8->int32 (int32,default)` | 7247757312 | 0.021 | 71 | 0.021 | 71 | 0.021 | 71 | 0.021 | 71 | 0.021 | 71 | pass |
| square | 1664 | `bf16,bf16->bf16 (fp32,default)` | 9214885888 | 0.020 | 66 | 0.020 | 66 | 0.019 | 66 | 0.020 | 66 | 0.019 | 66 | pass |
| square | 1664 | `fp16,fp16->fp16 (fp32,default)` | 9214885888 | 0.020 | 66 | 0.020 | 66 | 0.019 | 66 | 0.020 | 66 | 0.019 | 66 | pass |
| square | 1664 | `fp32,fp32->fp32 (fp32,default)` | 9214885888 | 0.191 | 76 | 0.184 | 76 | 0.197 | 76 | 0.191 | 76 | 0.191 | 76 | pass |
| square | 1664 | `fp32,fp32->fp32 (tf32,tf32)` | 9214885888 | 0.033 | 73 | 0.033 | 73 | 0.033 | 73 | 0.033 | 73 | 0.033 | 73 | pass |
| square | 1664 | `int8,int8->int32 (int32,default)` | 9214885888 | 0.021 | 71 | 0.021 | 71 | 0.021 | 71 | 0.021 | 71 | 0.021 | 71 | pass |
| square | 2048 | `bf16,bf16->bf16 (fp32,default)` | 17179869184 | 0.024 | 66 | 0.023 | 66 | 0.024 | 66 | 0.023 | 66 | 0.023 | 66 | pass |
| square | 2048 | `fp16,fp16->fp16 (fp32,default)` | 17179869184 | 0.024 | 66 | 0.024 | 66 | 0.024 | 66 | 0.024 | 66 | 0.024 | 66 | pass |
| square | 2048 | `fp32,fp32->fp32 (fp32,default)` | 17179869184 | 0.315 | 76 | 0.303 | 76 | 0.316 | 76 | 0.315 | 76 | 0.315 | 76 | pass |
| square | 2048 | `fp32,fp32->fp32 (tf32,tf32)` | 17179869184 | 0.038 | 73 | 0.038 | 73 | 0.038 | 73 | 0.038 | 73 | 0.038 | 73 | pass |
| square | 2048 | `int8,int8->int32 (int32,default)` | 17179869184 | 0.023 | 71 | 0.023 | 71 | 0.023 | 71 | 0.023 | 71 | 0.023 | 71 | pass |
| square | 2304 | `bf16,bf16->bf16 (fp32,default)` | 24461180928 | 0.027 | 66 | 0.026 | 66 | 0.027 | 66 | 0.027 | 66 | 0.027 | 66 | pass |
| square | 2304 | `fp16,fp16->fp16 (fp32,default)` | 24461180928 | 0.027 | 66 | 0.026 | 66 | 0.027 | 66 | 0.027 | 66 | 0.027 | 66 | pass |
| square | 2304 | `fp32,fp32->fp32 (fp32,default)` | 24461180928 | 0.448 | 76 | 0.430 | 76 | 0.462 | 76 | 0.448 | 76 | 0.447 | 76 | pass |
| square | 2304 | `fp32,fp32->fp32 (tf32,tf32)` | 24461180928 | 0.055 | 73 | 0.054 | 73 | 0.056 | 73 | 0.056 | 73 | 0.056 | 73 | pass |
| square | 2304 | `int8,int8->int32 (int32,default)` | 24461180928 | 0.029 | 71 | 0.028 | 71 | 0.029 | 71 | 0.029 | 71 | 0.029 | 71 | pass |
| square | 4096 | `bf16,bf16->bf16 (fp32,default)` | 137438953472 | 0.096 | 66 | 0.096 | 66 | 0.096 | 66 | 0.096 | 66 | 0.096 | 66 | pass |
| square | 4096 | `fp16,fp16->fp16 (fp32,default)` | 137438953472 | 0.100 | 66 | 0.099 | 66 | 0.099 | 66 | 0.099 | 66 | 0.100 | 66 | pass |
| square | 4096 | `fp32,fp32->fp32 (fp32,default)` | 137438953472 | 2.170 | 76 | 2.317 | 76 | 2.413 | 76 | 2.170 | 76 | 2.170 | 76 | pass |
| square | 4096 | `fp32,fp32->fp32 (tf32,tf32)` | 137438953472 | 0.195 | 73 | 0.187 | 73 | 0.188 | 73 | 0.196 | 73 | 0.196 | 73 | pass |
| square | 4096 | `int8,int8->int32 (int32,default)` | 137438953472 | 0.058 | 71 | 0.058 | 71 | 0.058 | 71 | 0.058 | 71 | 0.058 | 71 | pass |

## Non-square Suite

| suite | M | N | K | dtype_pair | flop_count | A.T@B(ms) | A.T@B(algo_id) | copy(A.T)@B(ms) | copy(A.T)@B(algo_id) | A@B.T(ms) | A@B.T(algo_id) | A@copy(B.T)(ms) | A@copy(B.T)(algo_id) | verify |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| non_square | 256 | 256 | 992 | `bf16,bf16->bf16 (fp32,default)` | 130023424 | 0.011 | 66 | 0.010 | 66 | 0.011 | 66 | 0.011 | 66 | pass |
| non_square | 256 | 256 | 992 | `fp16,fp16->fp16 (fp32,default)` | 130023424 | 0.012 | 66 | 0.011 | 66 | 0.010 | 66 | 0.010 | 66 | pass |
| non_square | 256 | 256 | 992 | `fp32,fp32->fp32 (fp32,default)` | 130023424 | 0.018 | 20 | 0.020 | 20 | 0.023 | 20 | 0.019 | 20 | pass |
| non_square | 256 | 256 | 992 | `fp32,fp32->fp32 (tf32,tf32)` | 130023424 | 0.016 | 73 | 0.016 | 73 | 0.016 | 73 | 0.016 | 73 | pass |
| non_square | 256 | 256 | 992 | `int8,int8->int32 (int32,default)` | 130023424 | 0.018 | 71 | 0.018 | 71 | 0.018 | 71 | 0.018 | 71 | pass |
| non_square | 256 | 992 | 256 | `bf16,bf16->bf16 (fp32,default)` | 130023424 | 0.011 | 66 | 0.010 | 66 | 0.010 | 66 | 0.010 | 66 | pass |
| non_square | 256 | 992 | 256 | `fp16,fp16->fp16 (fp32,default)` | 130023424 | 0.010 | 66 | 0.010 | 66 | 0.010 | 66 | 0.010 | 66 | pass |
| non_square | 256 | 992 | 256 | `fp32,fp32->fp32 (fp32,default)` | 130023424 | 0.016 | 76 | 0.015 | 76 | 0.016 | 76 | 0.016 | 76 | pass |
| non_square | 256 | 992 | 256 | `fp32,fp32->fp32 (tf32,tf32)` | 130023424 | 0.011 | 21 | 0.011 | 21 | 0.011 | 21 | 0.011 | 21 | pass |
| non_square | 256 | 992 | 256 | `int8,int8->int32 (int32,default)` | 130023424 | 0.016 | 71 | 0.016 | 71 | 0.016 | 71 | 0.016 | 71 | pass |
| non_square | 960 | 320 | 640 | `bf16,bf16->bf16 (fp32,default)` | 393216000 | 0.011 | 66 | 0.011 | 66 | 0.011 | 66 | 0.011 | 66 | pass |
| non_square | 960 | 320 | 640 | `fp16,fp16->fp16 (fp32,default)` | 393216000 | 0.010 | 66 | 0.011 | 66 | 0.011 | 66 | 0.011 | 66 | pass |
| non_square | 960 | 320 | 640 | `fp32,fp32->fp32 (fp32,default)` | 393216000 | 0.026 | 76 | 0.026 | 76 | 0.027 | 76 | 0.026 | 76 | pass |
| non_square | 960 | 320 | 640 | `fp32,fp32->fp32 (tf32,tf32)` | 393216000 | 0.016 | 73 | 0.016 | 73 | 0.016 | 73 | 0.016 | 73 | pass |
| non_square | 960 | 320 | 640 | `int8,int8->int32 (int32,default)` | 393216000 | 0.017 | 71 | 0.017 | 71 | 0.017 | 71 | 0.017 | 71 | pass |
| non_square | 992 | 256 | 256 | `bf16,bf16->bf16 (fp32,default)` | 130023424 | 0.010 | 66 | 0.010 | 66 | 0.010 | 66 | 0.010 | 66 | pass |
| non_square | 992 | 256 | 256 | `fp16,fp16->fp16 (fp32,default)` | 130023424 | 0.010 | 66 | 0.010 | 66 | 0.010 | 66 | 0.010 | 66 | pass |
| non_square | 992 | 256 | 256 | `fp32,fp32->fp32 (fp32,default)` | 130023424 | 0.016 | 76 | 0.016 | 76 | 0.016 | 76 | 0.016 | 76 | pass |
| non_square | 992 | 256 | 256 | `fp32,fp32->fp32 (tf32,tf32)` | 130023424 | 0.012 | 21 | 0.012 | 21 | 0.012 | 21 | 0.012 | 21 | pass |
| non_square | 992 | 256 | 256 | `int8,int8->int32 (int32,default)` | 130023424 | 0.016 | 71 | 0.016 | 71 | 0.016 | 71 | 0.016 | 71 | pass |
| non_square | 1024 | 1024 | 4096 | `bf16,bf16->bf16 (fp32,default)` | 8589934592 | 0.018 | 66 | 0.019 | 66 | 0.019 | 66 | 0.019 | 66 | pass |
| non_square | 1024 | 1024 | 4096 | `fp16,fp16->fp16 (fp32,default)` | 8589934592 | 0.018 | 66 | 0.019 | 66 | 0.018 | 66 | 0.019 | 66 | pass |
| non_square | 1024 | 1024 | 4096 | `fp32,fp32->fp32 (fp32,default)` | 8589934592 | 0.170 | 76 | 0.170 | 76 | 0.174 | 76 | 0.170 | 76 | pass |
| non_square | 1024 | 1024 | 4096 | `fp32,fp32->fp32 (tf32,tf32)` | 8589934592 | 0.029 | 73 | 0.030 | 73 | 0.031 | 73 | 0.030 | 73 | pass |
| non_square | 1024 | 1024 | 4096 | `int8,int8->int32 (int32,default)` | 8589934592 | 0.024 | 71 | 0.024 | 71 | 0.024 | 71 | 0.024 | 71 | pass |
| non_square | 1024 | 4096 | 1024 | `bf16,bf16->bf16 (fp32,default)` | 8589934592 | 0.017 | 66 | 0.017 | 66 | 0.017 | 66 | 0.017 | 66 | pass |
| non_square | 1024 | 4096 | 1024 | `fp16,fp16->fp16 (fp32,default)` | 8589934592 | 0.017 | 66 | 0.017 | 66 | 0.018 | 66 | 0.017 | 66 | pass |
| non_square | 1024 | 4096 | 1024 | `fp32,fp32->fp32 (fp32,default)` | 8589934592 | 0.160 | 76 | 0.166 | 76 | 0.165 | 76 | 0.166 | 76 | pass |
| non_square | 1024 | 4096 | 1024 | `fp32,fp32->fp32 (tf32,tf32)` | 8589934592 | 0.027 | 73 | 0.027 | 73 | 0.027 | 73 | 0.027 | 73 | pass |
| non_square | 1024 | 4096 | 1024 | `int8,int8->int32 (int32,default)` | 8589934592 | 0.018 | 71 | 0.018 | 71 | 0.019 | 71 | 0.019 | 71 | pass |
| non_square | 1536 | 2304 | 1536 | `bf16,bf16->bf16 (fp32,default)` | 10871635968 | NA | NA | NA | NA | 0.017 | 66 | 0.017 | 66 | pass |
| non_square | 1536 | 2304 | 1536 | `fp16,fp16->fp16 (fp32,default)` | 10871635968 | NA | NA | NA | NA | 0.018 | 66 | 0.017 | 66 | pass |
| non_square | 1536 | 2304 | 1536 | `fp32,fp32->fp32 (fp32,default)` | 10871635968 | NA | NA | NA | NA | 0.215 | 76 | 0.207 | 76 | pass |
| non_square | 1536 | 2304 | 1536 | `fp32,fp32->fp32 (tf32,tf32)` | 10871635968 | NA | NA | NA | NA | 0.032 | 73 | 0.031 | 73 | pass |
| non_square | 1536 | 2304 | 1536 | `int8,int8->int32 (int32,default)` | 10871635968 | NA | NA | NA | NA | 0.020 | 71 | 0.020 | 71 | pass |
| non_square | 2048 | 3072 | 2048 | `bf16,bf16->bf16 (fp32,default)` | 25769803776 | NA | NA | NA | NA | 0.029 | 66 | 0.028 | 66 | pass |
| non_square | 2048 | 3072 | 2048 | `fp16,fp16->fp16 (fp32,default)` | 25769803776 | NA | NA | NA | NA | 0.028 | 66 | 0.028 | 66 | pass |
| non_square | 2048 | 3072 | 2048 | `fp32,fp32->fp32 (fp32,default)` | 25769803776 | NA | NA | NA | NA | 0.496 | 76 | 0.479 | 76 | pass |
| non_square | 2048 | 3072 | 2048 | `fp32,fp32->fp32 (tf32,tf32)` | 25769803776 | NA | NA | NA | NA | 0.051 | 73 | 0.051 | 73 | pass |
| non_square | 2048 | 3072 | 2048 | `int8,int8->int32 (int32,default)` | 25769803776 | NA | NA | NA | NA | 0.028 | 71 | 0.027 | 71 | pass |
| non_square | 2304 | 1536 | 1536 | `bf16,bf16->bf16 (fp32,default)` | 10871635968 | 0.017 | 66 | 0.017 | 66 | NA | NA | NA | NA | pass |
| non_square | 2304 | 1536 | 1536 | `fp16,fp16->fp16 (fp32,default)` | 10871635968 | 0.017 | 66 | 0.017 | 66 | NA | NA | NA | NA | pass |
| non_square | 2304 | 1536 | 1536 | `fp32,fp32->fp32 (fp32,default)` | 10871635968 | 0.200 | 76 | 0.207 | 76 | NA | NA | NA | NA | pass |
| non_square | 2304 | 1536 | 1536 | `fp32,fp32->fp32 (tf32,tf32)` | 10871635968 | 0.031 | 73 | 0.031 | 73 | NA | NA | NA | NA | pass |
| non_square | 2304 | 1536 | 1536 | `int8,int8->int32 (int32,default)` | 10871635968 | 0.020 | 71 | 0.020 | 71 | NA | NA | NA | NA | pass |
| non_square | 3072 | 2048 | 2048 | `bf16,bf16->bf16 (fp32,default)` | 25769803776 | 0.027 | 66 | 0.028 | 66 | NA | NA | NA | NA | pass |
| non_square | 3072 | 2048 | 2048 | `fp16,fp16->fp16 (fp32,default)` | 25769803776 | 0.028 | 66 | 0.028 | 66 | NA | NA | NA | NA | pass |
| non_square | 3072 | 2048 | 2048 | `fp32,fp32->fp32 (fp32,default)` | 25769803776 | 0.466 | 76 | 0.479 | 76 | NA | NA | NA | NA | pass |
| non_square | 3072 | 2048 | 2048 | `fp32,fp32->fp32 (tf32,tf32)` | 25769803776 | 0.050 | 73 | 0.051 | 73 | NA | NA | NA | NA | pass |
| non_square | 3072 | 2048 | 2048 | `int8,int8->int32 (int32,default)` | 25769803776 | 0.027 | 71 | 0.027 | 71 | NA | NA | NA | NA | pass |
| non_square | 4096 | 1024 | 1024 | `bf16,bf16->bf16 (fp32,default)` | 8589934592 | 0.016 | 66 | 0.017 | 66 | 0.017 | 66 | 0.017 | 66 | pass |
| non_square | 4096 | 1024 | 1024 | `fp16,fp16->fp16 (fp32,default)` | 8589934592 | 0.017 | 66 | 0.017 | 66 | 0.017 | 66 | 0.017 | 66 | pass |
| non_square | 4096 | 1024 | 1024 | `fp32,fp32->fp32 (fp32,default)` | 8589934592 | 0.160 | 76 | 0.166 | 76 | 0.166 | 76 | 0.166 | 76 | pass |
| non_square | 4096 | 1024 | 1024 | `fp32,fp32->fp32 (tf32,tf32)` | 8589934592 | 0.027 | 73 | 0.027 | 73 | 0.028 | 73 | 0.027 | 73 | pass |
| non_square | 4096 | 1024 | 1024 | `int8,int8->int32 (int32,default)` | 8589934592 | 0.019 | 71 | 0.019 | 71 | 0.019 | 71 | 0.019 | 71 | pass |
| non_square | 8192 | 1024 | 1024 | `bf16,bf16->bf16 (fp32,default)` | 17179869184 | 0.024 | 66 | 0.024 | 66 | 0.024 | 66 | 0.024 | 66 | pass |
| non_square | 8192 | 1024 | 1024 | `fp16,fp16->fp16 (fp32,default)` | 17179869184 | 0.024 | 66 | 0.024 | 66 | 0.024 | 66 | 0.025 | 66 | pass |
| non_square | 8192 | 1024 | 1024 | `fp32,fp32->fp32 (fp32,default)` | 17179869184 | 0.329 | 76 | 0.329 | 76 | 0.338 | 76 | 0.329 | 76 | pass |
| non_square | 8192 | 1024 | 1024 | `fp32,fp32->fp32 (tf32,tf32)` | 17179869184 | 0.039 | 73 | 0.039 | 73 | 0.039 | 73 | 0.040 | 73 | pass |
| non_square | 8192 | 1024 | 1024 | `int8,int8->int32 (int32,default)` | 17179869184 | 0.022 | 71 | 0.023 | 71 | 0.023 | 71 | 0.023 | 71 | pass |

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

