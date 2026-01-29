# Project Context: accelsim-test

## Overview
**accelsim-test** is a dedicated research and development workspace for evaluating the **Accel-Sim Framework** in the context of Large Language Model (LLM) inference simulation.

The primary objective is to validate if Accel-Sim's detailed architecture modeling can accurately estimate performance metrics for LLM workloads on current and future GPU architectures.

### Key Technologies
*   **Language:** Python 3.12
*   **Package Manager:** [Pixi](https://prefix.dev/) (conda-forge channel)
*   **Core Dependencies:** `scipy`, `omegaconf`, `attrs`, `imageio`, `mdutils`
*   **Core Submodule:** `accel-sim-framework` (Forked at `extern/tracked/accel-sim-framework`, tracking branch `hz-dev`)

## Directory Structure
*   `src/accelsim_test/`: Main Python package source code.
*   `extern/tracked/`: Contains tracked external submodules, specifically the forked `accel-sim-framework`.
*   `tests/`: Directory for unit, integration, and manual tests.
*   `docs/`: Project documentation.
*   `scripts/`: Utility scripts for simulations and analysis.
*   `context/`: Project-specific context, plans, and logs.
*   `magic-context/`: Shared submodule containing general guidelines, specialized agent skills, and reference documentation.

## Building and Running

This project uses **Pixi** for environment management.

### Environment Setup
To set up the environment and install dependencies:
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

### Managing Dependencies
*   **Add PyPI dependency:** `pixi add --pypi <package_name>`
*   **Add Conda dependency:** `pixi add <package_name>`

### Submodules
The project relies on Git submodules. Ensure they are initialized:
```bash
git submodule update --init --recursive
```
*   `accel-sim-framework` is located in `extern/tracked/` and tracks the `hz-dev` branch of the `imsight-forks` fork.

## Development Conventions

*   **Code Style:** Adhere to standard Python practices. `ruff` and `mypy` are included in the environment for linting and type checking.
*   **Configuration:** Use `pyproject.toml` for project configuration and dependency definition.
*   **Version Control:**
    *   Main branch: `main`
    *   Submodules: HTTPS is preferred for cloning/pulling, SSH for pushing (if you have rights).
*   **Testing:** Place tests in the `tests/` directory.

## Current State
The project has been initialized with the basic directory structure and dependencies. The `accel-sim-framework` submodule is linked and ready for modification on the `hz-dev` branch. The main package `accelsim_test` is currently a placeholder awaiting implementation of simulation drivers and analysis tools.
