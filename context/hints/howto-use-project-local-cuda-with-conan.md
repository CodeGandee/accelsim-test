# How to Use a Project-Local CUDA Toolkit with Conan (Without Touching System CUDA)

## Goal

Build a C++/CUDA project with a CUDA Toolkit (nvcc + headers/libs) that is installed in project space (for example via Pixi/Conda or a local CUDA install prefix), while still using Conan for C/C++ dependency management.

## Key Constraints (Reality Check)

- The NVIDIA driver is system-level; Conan (and most user-space tools) cannot swap the kernel driver per-project. Your project-local CUDA Toolkit must be compatible with the installed driver at runtime.
- Conan can point your build system at a specific CUDA compiler/toolkit, but it typically does not “install the CUDA Toolkit” for you from ConanCenter; treat the toolkit as an external toolchain and configure Conan/CMake to use it.

## Step 1: Install an Alternate CUDA Toolkit in Project Space

Examples (choose one approach):

- Pixi/Conda environment that provides `nvcc` and CUDA headers/libs, then build inside that environment so `$PATH` resolves to the desired `nvcc`.
- A locally installed CUDA Toolkit prefix (for example `/opt/cuda-12.4` or `<repo>/.toolchains/cuda-12.4`) that contains `bin/nvcc`, `include/`, and `lib64/`.

Verify your intended toolkit:

```bash
which nvcc
nvcc --version
```

## Step 2: Tell Conan Which CUDA Compiler to Use

Conan 2 supports selecting compilers via the `tools.build:compiler_executables` conf, including a `cuda` entry.

Option A: Put it in a Conan profile (recommended for repeatability)

`~/.conan2/profiles/cuda-local`:

```ini
[conf]
tools.build:compiler_executables={"cuda":"/path/to/your/cuda/bin/nvcc"}
```

Then:

```bash
conan install . -pr cuda-local -b missing
```

Option B: Pass it on the command line (quick experiments)

```bash
conan install . -c tools.build:compiler_executables='{"cuda":"/path/to/your/cuda/bin/nvcc"}' -b missing
```

## Step 3: Make CMake Resolve the Same Toolkit

If your CMake project uses the CUDA language or `FindCUDAToolkit`, you can force the toolkit choice using one (or more) of:

```bash
cmake -S . -B build \
  -DCMAKE_CUDA_COMPILER=/path/to/your/cuda/bin/nvcc \
  -DCUDAToolkit_ROOT=/path/to/your/cuda
```

Or via environment variables:

```bash
export CUDACXX=/path/to/your/cuda/bin/nvcc
export CUDAToolkit_ROOT=/path/to/your/cuda
```

In many cases, if the intended `nvcc` is first in `$PATH`, CMake will find it without extra flags, but pinning `CMAKE_CUDA_COMPILER`/`CUDAToolkit_ROOT` is more deterministic.

## Step 4: Verify You Are Not Using System CUDA by Accident

After configuring, check the build directory for the selected compiler/toolkit:

```bash
rg -n "CMAKE_CUDA_COMPILER|CUDAToolkit_ROOT" build/**/CMakeCache.txt
```

And confirm at runtime which CUDA libraries are being loaded (if applicable) with `ldd`/`LD_DEBUG=libs` (Linux).

## Sources

- Conan configuration reference (`tools.build:compiler_executables` supports `cuda` key): https://docs.conan.io/2/reference/config_files/global_conf.html
- Conan profiles reference (setting `[conf]` in profiles): https://docs.conan.io/2/reference/config_files/profiles.html
- CMake `FindCUDAToolkit` search order (`CMAKE_CUDA_COMPILER`, `CUDACXX`, `CUDAToolkit_ROOT`, PATH): https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
