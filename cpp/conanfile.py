from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout, CMakeDeps


class accelsim_profilingRecipe(ConanFile):
    name = "accelsim_profiling"
    version = "0.1"
    package_type = "application"

    # Optional metadata
    license = "<Put the package license here>"
    author = "<Put your name here> <And your email here>"
    url = "<Package recipe repository url here, for issues about the package>"
    description = "<Description of accelsim_profiling package here>"
    topics = ("<Put some tag here>", "<here>", "<and here>")

    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"

    # Sources are located in the same place as this recipe, copy them to the recipe
    exports_sources = "CMakeLists.txt", "src/*"

    def layout(self):
        cmake_layout(self)

    def requirements(self):
        self.requires("cutlass/3.5.1")
        self.requires("benchmark/1.9.4")
        self.requires("eigen/3.4.0")
        self.requires("sqlite_orm/1.8.2")
        self.requires("nlohmann_json/3.11.3")
        self.requires("rapidcsv/8.84")

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
        # Prefer Ninja for faster incremental builds (provided by the pixi env in this repo).
        tc.generator = "Ninja"
        # Force C++17 or higher
        tc.variables["CMAKE_CXX_STANDARD"] = "17"
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    

    
