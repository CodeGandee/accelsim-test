cmake_minimum_required(VERSION 3.15)
project(accelsim_profiling CXX)



add_executable(accelsim_profiling src/accelsim_profiling.cpp src/main.cpp)



install(TARGETS accelsim_profiling DESTINATION "."
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        )
