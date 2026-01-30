#include "accelsim_profiling.h"
#include <vector>
#include <string>

int main() {
    accelsim_profiling();

    std::vector<std::string> vec;
    vec.push_back("test_package");

    accelsim_profiling_print_vector(vec);
}
