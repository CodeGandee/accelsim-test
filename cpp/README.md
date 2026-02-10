# C++ Subproject (`cpp/`)

This folder contains the C++/CUDA side of the repository. It is built as a Conan 2 + CMake (Ninja) project and is typically orchestrated by Python tooling at the repo root (Pixi tasks and `python -m ...` entry points).

## What Is In Here

Key files and folders:

- `cpp/conanfile.py`: Conan 2 recipe that provides third-party dependencies (Cutlass, Eigen, nlohmann_json, etc.) and generates CMake toolchain + presets.
- `cpp/CMakeLists.txt`: CMake project that builds the CUDA/C++ executables.
- `cpp/src/`: Source code for executables and reusable helpers.
- `cpp/build/`: Default Conan/CMake build output (generated).
- `cpp/compile_commands.json`: Symlink to the active build's compile database (generated).

NVBench dependency:

- NVBench is expected as a local source tree at `extern/orphan/nvbench/` and is added via `add_subdirectory(...)`.

## Build (Manual)

From repo root, using the Pixi environment that provides CUDA + Conan + CMake:

```bash
pixi install -e cuda128

export CUDAHOSTCXX=/usr/bin/g++
cd cpp
conan profile detect --force
conan install . -b missing
cmake --preset conan-release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build --preset conan-release -j
```

Notes:

- `CMAKE_CUDA_ARCHITECTURES=native` builds for the locally-visible GPU (recommended for portability across A100/B200). For reproducible binaries, set an explicit architecture (for example `80`).
- If `extern/orphan/nvbench/` is missing, CMake configuration will fail (this repo treats it as a local-only dependency).

## Build (Via Pixi Task)

The root `pyproject.toml` defines a convenience task:

```bash
pixi run -e cuda128 gemm-transpose-build
```

## Executables

These are the main CMake targets currently built from `cpp/src/`:

- `gemm_transpose_bench`
  - NVBench-based benchmark runner for the transpose-vs-copy GEMM experiment (cuBLASLt).
  - Typically invoked via Python: `pixi run -e cuda128 gemm-transpose ...` (see `src/accelsim_test/gemm_transpose_bench/`).
- `cublaslt_algo_caps_dump`
  - Dumps `cublasLtMatmulAlgoCapGetAttribute()` capability fields for a given `(algo_id, types)` set, and writes JSON.
  - Used to generate `cublaslt_algo_caps.json` alongside sweep results.
- `repro_algo23_int8_n1000`
  - Minimal repro for the square `N=1000` int8 case where `ABT_view` selects algo 23 and is faster.
  - Also attempts to force algo 23 for other transpose modes and reports `NA` if the override is rejected by `cublasLtMatmulAlgoCheck`.
- `cublaslt_usable_algo_sweep`
  - Case-level usable-algo sweep engine for row-major int8 `AB` vs `ABT_view`.
  - Enumerates `algo_id`s via cuBLASLt discovery APIs, checks usability via `cublasLtMatmulAlgoCheck`, and times usable candidates.
  - Typically invoked via Python: `pixi run python scripts/cublaslt_usable_algo_sweep.py`.
- `accelsim_profiling`
  - A separate CUDA/C++ application target used for general profiling/experiments (not the transpose sweep runner).

## Running The N=1000 Algo-23 Repro

```bash
pixi run -e cuda128 bash -lc ./cpp/build/Release/repro_algo23_int8_n1000
```

Optional controls:

- `ACCELSIM_REPRO_ITERS`: timed iterations (default `2000`)
- `ACCELSIM_REPRO_WARMUP`: warmup iterations (default `200`)
- `./repro_algo23_int8_n1000 <device_id>`: pick a CUDA device index
