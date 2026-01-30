#pragma once

#include <vector>
#include <string>


#ifdef _WIN32
  #define ACCELSIM_PROFILING_EXPORT __declspec(dllexport)
#else
  #define ACCELSIM_PROFILING_EXPORT
#endif

ACCELSIM_PROFILING_EXPORT void accelsim_profiling();
ACCELSIM_PROFILING_EXPORT void accelsim_profiling_print_vector(const std::vector<std::string> &strings);
