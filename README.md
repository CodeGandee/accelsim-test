# Accel-Sim Test Project

## Overview

This project, `accelsim-test`, is a dedicated workspace for evaluating the **[Accel-Sim Framework](https://accel-sim.github.io/)** as a potential component for a larger **LLM Inference Simulation Framework**.

## Objective

The primary goal is to determine if Accel-Sim can accurately and efficiently estimate theoretical inference performance of Large Language Models (LLMs) across different hardware settings, specifically:

*   **Low-Level Estimation:** Leveraging Accel-Sim's detailed architecture modeling to predict performance metrics at the kernel and instruction level.
*   **Cross-Architecture Simulation:** validating performance on existing GPUs and projecting performance for unseen or future GPU architectures (e.g., hypothetical next-gen configurations).
*   **Integration Feasibility:** Assessing the effort required to integrate Accel-Sim's trace-driven or execution-driven modes into a high-level LLM performance modeling pipeline.

## Scope

This repository will contain:

*   Configuration files for simulating specific GPU architectures (e.g., NVIDIA A100, H100, and hypothetical specs).
*   Test kernels and micro-benchmarks relevant to LLM inference (e.g., GEMM, Attention mechanisms).
*   Scripts to drive Accel-Sim simulations and parse output metrics.
*   Documentation of findings regarding accuracy, simulation speed, and ease of use.

## Project Layout

*   `src/accelsim_test/`: main Python package (src-layout).
*   `extern/tracked/accel-sim-framework/`: Accel-Sim submodule (reproducible).
*   `scripts/accelsim/`: helper scripts for building/running Accel-Sim.
*   `cpp/`: Conan-managed C++ scratch project for profiling tooling (see below).
*   `tmp/`: scratch space (includes `tmp/build-check/` for CUDA build verification).
*   `context/`: project knowledge base (hints/issues/notes).

## Development Environments (Pixi)

This repo uses Pixi (`pyproject.toml`, `pixi.lock`) and defines multiple environments:

*   `default`: Python tooling (lint/typecheck/etc).
*   `accelsim`: dependencies to build/run Accel-Sim (includes CUDA toolkit pinned for that flow).
*   `cuda13`: a project-local CUDA build environment (CUDA 13.0 + cuDNN) for compiling/running `.cu` code without relying on system CUDA.

Common commands:

```bash
pixi install
pixi install -e accelsim
pixi install -e cuda13
```

Repo tasks (examples):

```bash
pixi run accelsim-build
pixi run accelsim-smoke
pixi run accelsim-short-tests
pixi run pytest
```

CUDA build sanity check (uses the `cuda13` env):

```bash
pixi run -e cuda13 cuda13-build-check
```

## GEMM Transpose Benchmark (NVBench + cuBLASLt)

This repo includes an NVBench-based CUDA GEMM transpose microbenchmark with a Python orchestrator (timing → export → report).

- Quickstart: `specs/002-gemm-transpose-bench/quickstart.md`
- Build (CUDA 13 env): `pixi run -e cuda13 gemm-transpose-build`
- Run (timing): `pixi run -e cuda13 gemm-transpose -- timing --out-dir tmp/gemm_transpose_out --suite all --dtype all`
- Generate report: `pixi run -- python -m accelsim_test.gemm_transpose_bench report --out-dir tmp/gemm_transpose_out`

## C++ Profiling Scratch (`cpp/`)

The `cpp/` directory is a small Conan 2 + CMake project used for experimenting with profiling-related C++ code and dependency integration (e.g., CUTLASS via Conan).

Typical local dev loop:

```bash
cd cpp
conan profile detect --force   # once per machine
conan install . -b missing
cmake --preset conan-release
cmake --build --preset conan-release -j
./build/Release/accelsim_profiling
```

## References

*   [Accel-Sim Official Website](https://accel-sim.github.io/)
*   [Accel-Sim GitHub Repository](https://github.com/accel-sim/accel-sim-framework)
