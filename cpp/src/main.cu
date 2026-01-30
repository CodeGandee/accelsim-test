#include "accelsim_profiling.h"
#include <vector>
#include <string>
#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <Eigen/Dense>
#include <rapidcsv.h>
#include <sqlite_orm/sqlite_orm.h>
#include <nlohmann/json.hpp>

int main() {
    accelsim_profiling();

    std::vector<std::string> vec;
    vec.push_back("test_package");

    accelsim_profiling_print_vector(vec);
    
    std::cout << "--- Dependency Check ---" << std::endl;
    std::cout << "[Cutlass] Half type size: " << sizeof(cutlass::half_t) << std::endl;
    std::cout << "[Eigen] Version: " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << std::endl;
    
    // RapidCSV check
    try {
        rapidcsv::Document doc("", rapidcsv::LabelParams(), rapidcsv::SeparatorParams('\t'));
        std::cout << "[RapidCSV] Document object created successfully." << std::endl;
    } catch (...) {
        std::cout << "[RapidCSV] Failed to create document (expected for empty string, but library linked)." << std::endl;
    }

    // JSON check
    nlohmann::json j;
    j["library"] = "nlohmann_json";
    std::cout << "[JSON] " << j.dump() << std::endl;

    // SQLite ORM check
    std::cout << "[SQLite ORM] Library included successfully." << std::endl;

    return 0;
}