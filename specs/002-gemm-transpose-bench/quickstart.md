# Quickstart: GEMM Transpose Performance Benchmark

This quickstart describes the intended developer workflow for running the benchmark defined in `/data1/huangzhe/code/accelsim-test/specs/002-gemm-transpose-bench/spec.md` using Pixi (`cuda13`) + NVBench + cuBLASLt.

## Prerequisites

- Linux x86_64 host with an NVIDIA GPU and working NVIDIA driver.
- Pixi installed and usable on `PATH`.
- NVBench source tree available at `/data1/huangzhe/code/accelsim-test/extern/orphan/nvbench` (this is git-ignored; ensure it exists on the machine where you build/run).
- Nsight Compute (`ncu`) available on `PATH` for profiling runs.

## Setup (Pixi)

```bash
cd /data1/huangzhe/code/accelsim-test
pixi install
```

## Build the C++ benchmark (Conan + CMake, inside Pixi `cuda13`)

This is wrapped by a Pixi task:

```bash
cd /data1/huangzhe/code/accelsim-test
pixi run -e cuda13 gemm-transpose-build
```

Notes:
- The build task forces `CUDAHOSTCXX=/usr/bin/g++` to avoid NVCC + libstdc++ header incompatibilities seen with newer Conda toolchains.
- NVBench is expected at `/data1/huangzhe/code/accelsim-test/extern/orphan/nvbench` and the build fails fast if it is missing.

## Run a timing sweep (no profiler)

Primary entrypoint (Python orchestrator):

```bash
cd /data1/huangzhe/code/accelsim-test
pixi run -e cuda13 -- python -m accelsim_test.gemm_transpose_bench timing --out-dir /data1/huangzhe/code/accelsim-test/tmp/gemm_transpose_out --suite all --dtype all
```

Equivalent (via Pixi task wrapper):

```bash
cd /data1/huangzhe/code/accelsim-test
pixi run -e cuda13 gemm-transpose -- timing --out-dir /data1/huangzhe/code/accelsim-test/tmp/gemm_transpose_out --suite all --dtype all
```

Expected outputs:
- `/data1/huangzhe/code/accelsim-test/tmp/gemm_transpose_out/raw/nvbench_timing.json`
- `/data1/huangzhe/code/accelsim-test/tmp/gemm_transpose_out/results.json` (validates against `specs/002-gemm-transpose-bench/contracts/results.schema.json`)

## Run profiling (Nsight Compute, per configuration/case)

```bash
cd /data1/huangzhe/code/accelsim-test
pixi run -e cuda13 -- python -m accelsim_test.gemm_transpose_bench profile --out-dir /data1/huangzhe/code/accelsim-test/tmp/gemm_transpose_out
```

Expected outputs:
- `/data1/huangzhe/code/accelsim-test/tmp/gemm_transpose_out/profiles/.../*.ncu-rep`
- Updated `/data1/huangzhe/code/accelsim-test/tmp/gemm_transpose_out/results.json` with per-record artifact references

## Generate the stakeholder report

```bash
cd /data1/huangzhe/code/accelsim-test
pixi run -- python -m accelsim_test.gemm_transpose_bench report --out-dir /data1/huangzhe/code/accelsim-test/tmp/gemm_transpose_out
```

Expected output:
- `/data1/huangzhe/code/accelsim-test/tmp/gemm_transpose_out/report.md` containing square-suite and non-square-suite tables with required ratios and `flop_count` consistency (FR-010 / FR-010a).
