# Repository Guidelines

## Project Model (Python + C++)

This is a mixed Python/C++ project:

- **Python is the orchestrator**: environment management (Pixi), repo automation/CLIs, and “glue” code that coordinates one or more C++ builds/runs (executables, modules, functions).
- **C++ is a subproject** under `cpp/`: built and dependency-managed with **Conan** + **CMake** (often invoked from Python tooling).

## Project Structure & Module Organization

- `src/accelsim_test/`: main Python package (src-layout).
- `cpp/`: C++ subproject (Conan 2 + CMake) for performance-critical code and experiments.
- `tests/`: test scaffolding (`unit/`, `integration/`, `manual/`).
- `scripts/`: repository helper scripts / CLIs (add new entry points here).
- `docs/`: Markdown documentation (MkDocs Material is listed as a dependency, but no site config is committed yet).
- `extern/`: third-party code.
  - `extern/tracked/`: git submodules pinned in `.gitmodules` (reproducible).
  - `extern/orphan/`: local-only clones (git-ignored).
    - `extern/orphan/nvbench/`: NVBench source tree (primary CUDA benchmarking library for this repo).
- `context/`: project knowledge base (design notes, plans, issues, logs).
- `tmp/`: scratch space for experiments; do not rely on it for production inputs.

## Build, Test, and Development Commands

This repo uses `pixi` at the workspace root for a reproducible Python environment (`pyproject.toml`, `pixi.lock`).

- `pixi install`: create/update the environment from the lockfile.
- `pixi shell`: activate the environment in your shell.
- `pixi run ruff check .`: lint the repository (Ruff is a dependency).
- `pixi run mypy src`: type-check package code (Mypy is a dependency).

The C++ subproject uses Conan + CMake (run from `cpp/`):

- `cd cpp`
- `conan profile detect --force` (once per machine)
- `conan install . -b missing`
- `cmake --preset conan-release`
- `cmake --build --preset conan-release -j`

Benchmarking:
- Prefer **NVBench** for CUDA benchmarking (primary measurement harness in this repo).
- Be rigorous: use NVBench warmup + statistical stopping criteria (`--min-time`, `--max-noise`) and avoid ad-hoc timing loops.
- NVBench source is expected at `extern/orphan/nvbench/` (use it directly when integrating benchmarks).

Profiling tools:
- `nsys`: NVIDIA Nsight Systems (available on the host system).
- `ncu`: NVIDIA Nsight Compute (installed via `pixi global` and expected to be on `PATH`; if not, add Pixi’s bin dir on the fly, e.g. `export PATH=\"$HOME/.pixi/bin:$PATH\"`).

Submodules (required for external dependencies like Accel-Sim):
- `git submodule update --init --recursive`

## Coding Style & Naming Conventions

- Python: 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes.
- Python: prefer type hints for public functions and data structures; keep “orchestration” logic in Python (calling C++ via subprocess, Python bindings, or other adapters).
- C++: keep code and build logic inside `cpp/` (dependencies in `cpp/conanfile.py`, build via CMake presets); follow existing style in that subproject.
- Keep library code in `src/` and avoid importing from `extern/` (tracked code is vendor/reference).

## Testing Guidelines

Tests should live under `tests/` and mirror package structure where possible.
Use `test_*.py` filenames and keep `unit/` tests fast and deterministic. If you introduce a new test runner (e.g., `pytest`), add it to dependencies and document the command in this file.

## Commit & Pull Request Guidelines

Commit history mostly follows Conventional Commits (e.g., `feat: ...`). Prefer:
`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:` with a short, imperative summary.

PRs should include:
- What changed and why (link issues/tasks if applicable).
- How to reproduce/verify (commands + expected output).
- Notes when updating submodules (new commit SHA/branch and reason).

## Active Technologies
- Python 3.12 (Pixi) + CUDA C++17 (NVCC via Pixi `cuda13`) + Pixi tasks (workflow), Ruff + Mypy (Python QA), `attrs`/Hydra (Python config), Conan 2 + CMake/Ninja (C++ build), cuBLASLt (GEMM), NVBench (timing; `extern/orphan/nvbench`), Nsight Compute `ncu` (profiling) (002-gemm-transpose-bench)
- Files (structured JSON/CSV export + Markdown report + per-case `*.ncu-rep` artifacts) (002-gemm-transpose-bench)
- Python 3.12 (Pixi) + CUDA C++ (nvcc; PTX-mode simulation target) + Pixi tasks (workflow), Accel-Sim submodule (`extern/tracked/accel-sim-framework`), bash runner glue, CUDA compiler (`nvcc`) via Pixi when available (fallback to system `nvcc`) (003-accelsim-dummy-ptx-sim)
- Files under `tmp/<run_id>/` (binary, PTX, `gpgpusim.config` copy, logs, metadata) (003-accelsim-dummy-ptx-sim)

## Recent Changes
- 002-gemm-transpose-bench: Added Python 3.12 (Pixi) + CUDA C++17 (NVCC via Pixi `cuda13`) + Pixi tasks (workflow), Ruff + Mypy (Python QA), `attrs`/Hydra (Python config), Conan 2 + CMake/Ninja (C++ build), cuBLASLt (GEMM), NVBench (timing; `extern/orphan/nvbench`), Nsight Compute `ncu` (profiling)
