<!-- 
Sync Impact Report:
- Version Change: 1.1.0 (Hybrid Integration) -> 1.2.0 (Architectural Hierarchy)
- Modified Principles: 
  - Added "System Architecture" as Principle I to define Python's role as orchestrator.
  - Renumbered subsequent principles.
  - Clarified "Build Authority": Pixi (Python) is the global workflow manager; Conan (C++) is the subsystem build manager.
- Templates Status: 
  - .specify/templates/plan-template.md: ✅ Compatible.
  - .specify/templates/spec-template.md: ✅ Compatible.
  - .specify/templates/tasks-template.md: ✅ Compatible.
-->

# accelsim-test Constitution

## Core Principles

### I. System Architecture
- **Orchestration First**: Python is the primary entry point and orchestrator. It manages workflows, test runners, simulation drivers, and analysis tools.
- **High-Performance Subsystem**: C++ (`cpp/`) is a distinct subproject dedicated to high-performance compute kernels and simulation engines. It is invoked, wrapped, or managed by the Python layer, not the other way around.
- **Directory Structure**: Python resides in the repository root (`src/`, `tests/`); C++ resides strictly in `cpp/`.

### II. C++ Language Standards (Subproject)
- **Modern Idioms**: Prioritize C++17/20 (RAII, smart pointers, move semantics).
- **Naming Conventions**: `PascalCase` for types, `lower_snake_case` for functions/variables.
- **Member Variables**: STRICT `m_` prefix for data members.
- **Safety**: `const` correctness is non-negotiable. Public APIs must be safe (`[[nodiscard]]`, `std::optional`, no raw owning pointers).
- **Headers**: Use `#pragma once` and avoid `using namespace`.

### III. Python Language Standards (Orchestrator)
- **Type Safety**: All code must be type-annotated and pass `mypy`. Linting via `ruff` is mandatory.
- **Functional Classes**: 
  - Follow strict OO design with `m_` prefix for member variables.
  - Use factory methods (`cls.from_x`) instead of complex constructors.
  - Expose read-only properties; use explicit `set_x()` methods for mutation.
- **Data Models**: 
  - Use `attrs` (default) or `pydantic` (web schemas).
  - Use framework-native field naming (NO `m_` prefix).
  - Keep logic-free; define fields declaratively.
- **Style**: Prefer absolute imports.

### IV. Build & Environment Authority
- **Global Workflow (Python/Pixi)**: **Pixi** is the master environment and workflow manager. All build tasks, test runs, and scripts are defined as Pixi tasks.
- **Subsystem Build (C++/Conan)**: C++ dependency management (Conan) and compilation (CMake) are encapsulated processes. They are triggered by the global Python/Pixi workflow, ensuring a unified developer experience.
  - System Python is PROHIBITED.

### V. Three-Tier Testing Strategy
Testing is mandatory for all features.
1. **Manual Tests** (Feature Validation):
   - **Python (Primary)**: Interactive scripts in `tests/manual/` focusing on visualization/inspection. Orchestrates full scenarios.
   - **C++ (Kernel Level)**: Standalone `main()` executables in `tests/manual/cpp/` for low-level validation.
2. **Unit Tests** (Component Isolation):
   - **Python**: `pytest` in `tests/unit/`. Mock external resources.
   - **C++**: Catch2 framework in `tests/unit/cpp/`.
3. **Integration Tests** (System Flows):
   - End-to-end scenarios in `tests/integration/` where Python drives C++ components to validate interoperability and correctness.

### VI. Documentation & Performance
- **Documentation**: 
  - **Python**: NumPy-style docstrings (Orchestrator APIs).
  - **C++**: Doxygen-style comments (Engine APIs).
  - **General**: No hard line breaks in Markdown prose.
- **Performance**: Critical paths (usually C++) must be profiled (perf, Nsight) and benchmarked. Python profiling focuses on orchestration overhead.

## Development Workflow

### Quality Gates & Review
- **Pre-commit**: All code must compile/interpret without warnings.
- **Static Analysis**: `clang-tidy` (C++) and `ruff` (Python) checks must pass.
- **Type Check**: `mypy` must pass for all Python code.
- **Test Pass**: All Unit tests must pass before review.
- **Review Focus**:
  - **Architecture**: Ensure logic resides in the correct layer (Performance -> C++, Orchestration -> Python).
  - **Safety**: Resource Management (C++) and Type Safety (Python).

## Governance

This Constitution supersedes all other practices. Amendments require documentation, approval, and a migration plan.

**Compliance**: All Pull Requests and Code Reviews must explicitly verify compliance with these principles. Deviations must be justified by complexity or technical impossibility, not convenience.

**Version**: 1.2.0 | **Ratified**: 2026-02-02 | **Last Amended**: 2026-02-02