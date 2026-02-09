cmake_minimum_required(VERSION 3.30.4)

# CUDA 13's nvcc does not reliably support the conda-forge GCC 14 toolchain yet.
# Prefer a system GCC (typically 13.x on Ubuntu 24.04) as the nvcc host compiler.
#
# NOTE: We override cached conda/pixi compilers because they can make NVBench
# (and CUDA stdlib headers) fail to compile under CUDA 13.
if (NOT DEFINED CMAKE_CUDA_HOST_COMPILER OR CMAKE_CUDA_HOST_COMPILER MATCHES "\\.pixi/|conda|x86_64-conda-linux-gnu")
  set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++" CACHE FILEPATH "Host compiler used by nvcc" FORCE)
endif()

project(accelsim_profiling CXX CUDA)

# Default to building for the locally-available GPU to keep the benchmark
# runnable across machines (e.g., A100 vs B200). Override explicitly via:
#   cmake -DCMAKE_CUDA_ARCHITECTURES=80 ...
# if you need reproducible arch-specific binaries.
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(NvidiaCutlass CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(SqliteOrm CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(rapidcsv CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
find_package(Catch2 3 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# NVTX (nvToolsExt) is used for deterministic profiling range markers in repro tools.
# Some FindCUDAToolkit versions do not expose a CUDA::nvToolsExt imported target, so
# discover the library path explicitly.
find_library(NVTOOLSEXT_LIBRARY
    NAMES nvToolsExt
    HINTS ${CUDAToolkit_LIBRARY_DIR} ${CUDAToolkit_LIBRARY_ROOT}
    PATH_SUFFIXES lib64 lib
)

# NVBench (source lives in extern/orphan/nvbench; expected to be available on the machine)
set(NVBENCH_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../extern/orphan/nvbench")
if (NOT EXISTS "${NVBENCH_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "NVBench not found at ${NVBENCH_SOURCE_DIR}. Expected extern/orphan/nvbench to exist.")
endif()

add_subdirectory("${NVBENCH_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/_deps/nvbench-build" EXCLUDE_FROM_ALL)

add_executable(accelsim_profiling src/accelsim_profiling.cpp src/main.cu)

target_link_libraries(accelsim_profiling PRIVATE 
    nvidia::cutlass::cutlass 
    benchmark::benchmark 
    Eigen3::Eigen
    sqlite_orm::sqlite_orm
    nlohmann_json::nlohmann_json
    rapidcsv::rapidcsv
    yaml-cpp::yaml-cpp
)

add_executable(gemm_transpose_bench
    src/gemm_transpose_bench.cu
    src/cublaslt_gemm.cu
)
target_link_libraries(gemm_transpose_bench PRIVATE
    nvbench::main
    CUDA::cublasLt
    CUDA::cudart
    nlohmann_json::nlohmann_json
    yaml-cpp::yaml-cpp
)

add_executable(cublaslt_algo_caps_dump
    src/cublaslt_algo_caps_dump.cpp
)
target_link_libraries(cublaslt_algo_caps_dump PRIVATE
    CUDA::cublasLt
    CUDA::cudart
    nlohmann_json::nlohmann_json
)

add_executable(repro_algo23_int8_n1000
    src/repro_algo23_int8_n1000.cu
    src/cublaslt_gemm.cu
)
target_link_libraries(repro_algo23_int8_n1000 PRIVATE
    CUDA::cublasLt
    CUDA::cudart
)
if (NVTOOLSEXT_LIBRARY)
  target_link_libraries(repro_algo23_int8_n1000 PRIVATE ${NVTOOLSEXT_LIBRARY})
endif()

install(TARGETS accelsim_profiling DESTINATION "."
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        )

install(TARGETS gemm_transpose_bench DESTINATION "."
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        )
