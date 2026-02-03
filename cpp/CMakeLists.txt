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

set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(NvidiaCutlass CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(SqliteOrm CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(rapidcsv CONFIG REQUIRED)
find_package(Catch2 3 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

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
)

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
