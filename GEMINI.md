# Project Context: accelsim-test

## Overview
**accelsim-test** is a dedicated research and development workspace for evaluating the **Accel-Sim Framework** in the context of Large Language Model (LLM) inference simulation.

The primary objective is to validate if Accel-Sim's detailed architecture modeling can accurately estimate performance metrics for LLM workloads on current and future GPU architectures.

### Key Technologies
*   **Orchestration Language:** Python 3.12 (Managed by [Pixi](https://prefix.dev/))
*   **Performance Language:** C++17/20 (Managed by Conan/CMake)
*   **Core Dependencies:** `scipy`, `omegaconf`, `attrs`, `imageio`, `mdutils`
*   **Core Submodule:** `accel-sim-framework` (Forked at `extern/tracked/accel-sim-framework`, tracking branch `hz-dev`)

## Directory Structure
*   `src/accelsim_test/`: Main Python package source code (Orchestrator).
*   `cpp/`: High-performance C++ subproject (Simulation Engines, Kernels).
*   `extern/tracked/`: Contains tracked external submodules, specifically the forked `accel-sim-framework`.
*   `tests/`: Directory for unit, integration, and manual tests.
    *   `tests/*/cpp/`: Dedicated C++ tests (Catch2, standalone mains).
*   `docs/`: Project documentation.
*   `scripts/`: Utility scripts for simulations and analysis.
*   `context/`: Project-specific context, plans, and logs.
*   `magic-context/`: Shared submodule containing general guidelines, specialized agent skills, and reference documentation.

## Building and Running

This project uses **Pixi** for global environment and workflow management, with **Conan/CMake** handling the C++ subsystem.

### Environment Setup
To set up the environment and install dependencies (Python & C++ tools):
```bash
pixi install
```

### Running Code
To run Python scripts within the managed environment:
```bash
pixi run python <script.py>
```
Or, to spawn a shell within the environment:
```bash
pixi shell
```

### C++ Subproject Build
The `cpp/` directory is a standard CMake project managed by Conan. It should ideally be built via Pixi tasks (defined in `pyproject.toml`) to ensure the correct environment.
```bash
# Example (check pyproject.toml for actual task names)
pixi run build-cpp
```

### Managing Dependencies
*   **Add PyPI dependency:** `pixi add --pypi <package_name>`
*   **Add Conda dependency:** `pixi add <package_name>`
*   **Add C++ dependency:** Edit `cpp/conanfile.py` and run `conan install .` (inside `cpp/`).

### Submodules
The project relies on Git submodules. Ensure they are initialized:
```bash
git submodule update --init --recursive
```
*   `accel-sim-framework` is located in `extern/tracked/` and tracks the `hz-dev` branch of the `imsight-forks` fork.

## Development Conventions

*   **Code Style (Python):** Adhere to standard Python practices. `ruff` and `mypy` are included in the environment.
*   **Code Style (C++):** Adhere to modern C++17/20 standards. `clang-tidy` is used for static analysis. Strict `m_` prefix for members.
*   **Architecture:** Python acts as the orchestrator; C++ acts as the high-performance engine.
*   **Configuration:** Use `pyproject.toml` for project configuration and dependency definition.
*   **Version Control:**
    *   Main branch: `main`
    *   Submodules: HTTPS is preferred for cloning/pulling, SSH for pushing (if you have rights).
*   **Testing:** 
    *   Python tests: `tests/unit/`, `tests/manual/`
    *   C++ tests: `tests/unit/cpp/`, `tests/manual/cpp/`

## Current State
The project has been initialized with the basic directory structure and dependencies. The `accel-sim-framework` submodule is linked and ready for modification on the `hz-dev` branch. The main package `accelsim_test` is currently a placeholder awaiting implementation of simulation drivers and analysis tools. The `cpp/` subproject structure is established for performance-critical components.
