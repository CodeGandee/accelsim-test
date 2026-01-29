# Repository Guidelines

## Project Structure & Module Organization

- `src/accelsim_test/`: main Python package (src-layout).
- `tests/`: test scaffolding (`unit/`, `integration/`, `manual/`).
- `scripts/`: repository helper scripts / CLIs (add new entry points here).
- `docs/`: Markdown documentation (MkDocs Material is listed as a dependency, but no site config is committed yet).
- `extern/`: third-party code.
  - `extern/tracked/`: git submodules pinned in `.gitmodules` (reproducible).
  - `extern/orphan/`: local-only clones (git-ignored).
- `context/`: project knowledge base (design notes, plans, issues, logs).
- `tmp/`: scratch space for experiments; do not rely on it for production inputs.

## Build, Test, and Development Commands

This repo uses `pixi` for a reproducible Python environment (`pyproject.toml`, `pixi.lock`).

- `pixi install`: create/update the environment from the lockfile.
- `pixi shell`: activate the environment in your shell.
- `pixi run ruff check .`: lint the repository (Ruff is a dependency).
- `pixi run mypy src`: type-check package code (Mypy is a dependency).

Submodules (required for external dependencies like Accel-Sim):
- `git submodule update --init --recursive`

## Coding Style & Naming Conventions

- Python: 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes.
- Prefer type hints for public functions and data structures.
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

