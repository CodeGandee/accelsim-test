
Kernel Profiling (ncu)
======================


This directory contains Nsight Compute profiling artifacts for a cuBLASLt repro run.
# Command


`pixi run -e cuda13 /home/huangzhe/.pixi/bin/ncu --force-overwrite --log-file tmp/algo23_investigation/profiles/n1000_int8_abt_view_algo23/ncu/ncu.log --set basic --export tmp/algo23_investigation/profiles/n1000_int8_abt_view_algo23/ncu/profile --clock-control base --pipeline-boost-state stable --profile-from-start off ./cpp/build/Release/repro_algo23_int8_n1000 --variant ABT_view --force-algo 23 --tile-id 18 --stages-id 21 --splitk 1 --cuda-profiler-gating --nvtx --iters 1 --warmup 0`
# Outputs

- `profile.ncu-rep`: raw ncu report
- `raw.csv`: exported raw metrics (CSV text)
- `session.csv`: exported session/device info (CSV text)
- `details.csv`: exported section/rule details (CSV text)
- `meta.json`: run metadata
