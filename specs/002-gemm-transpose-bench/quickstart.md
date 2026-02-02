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

Planned: this will be wrapped by a Pixi task, but the underlying flow is:

```bash
cd /data1/huangzhe/code/accelsim-test
pixi run -e cuda13 -- bash -lc 'cd cpp && conan profile detect --force && conan install . -b missing && cmake --preset conan-release && cmake --build --preset conan-release -j'
```

## Run a timing sweep (no profiler)

Planned primary entrypoint (Python orchestrator):

```bash
cd /data1/huangzhe/code/accelsim-test
pixi run -e cuda13 -- python -m accelsim_test.gemm_transpose_bench timing --out-dir /data1/huangzhe/code/accelsim-test/tmp/gemm_transpose_out --suite all --dtype all
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
