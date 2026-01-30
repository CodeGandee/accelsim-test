cmake_minimum_required(VERSION 3.15)
project(accelsim_profiling CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 80)

find_package(NvidiaCutlass CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)

add_executable(accelsim_profiling src/accelsim_profiling.cpp src/main.cu)

target_link_libraries(accelsim_profiling PRIVATE nvidia::cutlass::cutlass benchmark::benchmark)

install(TARGETS accelsim_profiling DESTINATION "."
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        )
