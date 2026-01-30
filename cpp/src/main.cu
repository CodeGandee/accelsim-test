#include "accelsim_profiling.h"
#include <vector>
#include <string>
#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>

int main() {
    accelsim_profiling();

    std::vector<std::string> vec;
    vec.push_back("test_package");

    accelsim_profiling_print_vector(vec);
    
    std::cout << "Cutlass imported successfully." << std::endl;
    // Basic check of Cutlass types
    std::cout << "Cutlass Half type size: " << sizeof(cutlass::half_t) << std::endl;

    return 0;
}