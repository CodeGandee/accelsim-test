
Kernel Discovery (nsys)
=======================


This directory contains Nsight Systems kernel-discovery artifacts for a cuBLASLt repro run.
# Command


`pixi run -e cuda13 /usr/local/bin/nsys profile --force-overwrite=true -t cuda,cublas,nvtx -o tmp/algo23_investigation/profiles/n1000_int8_abt_view_algo23/nsys/capture ./cpp/build/Release/repro_algo23_int8_n1000 --variant ABT_view --force-algo 23 --tile-id 18 --stages-id 21 --splitk 1 --nvtx --iters 1 --warmup 0`
# Outputs

- `capture.nsys-rep`: raw capture (nsys-rep)
- `capture.qdrep`: raw capture (qdrep; may be absent on some systems)
- `cuda_gpu_trace.csv`: `cuda_gpu_trace` export (CSV)
- `cuda_gpu_kern_gb_sum.csv`: `cuda_gpu_kern_gb_sum` export (CSV)
- `kernel_list.csv`: compact kernel listing with invocation indices (CSV)
- `invocation.txt`: command + cwd
- `meta.json`: capture metadata
