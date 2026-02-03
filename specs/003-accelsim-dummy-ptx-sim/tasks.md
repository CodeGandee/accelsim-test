# Tasks: Accel-Sim Dummy CUDA PTX Simulation

**Input**: Design documents from `/data1/huangzhe/code/accelsim-test/specs/003-accelsim-dummy-ptx-sim/`  
**Prerequisites**: `/data1/huangzhe/code/accelsim-test/specs/003-accelsim-dummy-ptx-sim/plan.md`, `/data1/huangzhe/code/accelsim-test/specs/003-accelsim-dummy-ptx-sim/spec.md`, `/data1/huangzhe/code/accelsim-test/specs/003-accelsim-dummy-ptx-sim/research.md`, `/data1/huangzhe/code/accelsim-test/specs/003-accelsim-dummy-ptx-sim/data-model.md`, `/data1/huangzhe/code/accelsim-test/specs/003-accelsim-dummy-ptx-sim/contracts/`

**Tests**: Included (pytest unit tests for Python orchestration utilities only; no end-to-end simulator run in unit tests) per `plan.md` + `research.md` (Decision 8).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: User story mapping: US1..US5 (from `spec.md`)
- Include exact file paths in descriptions

## Implementation Guides

- `context/tasks/working/003-accelsim-dummy-ptx-sim/impl-guide-ph1-setup.md`
- `context/tasks/working/003-accelsim-dummy-ptx-sim/impl-guide-ph2-foundational.md`
- `context/tasks/working/003-accelsim-dummy-ptx-sim/impl-guide-ph3-us1-end-to-end.md`
- `context/tasks/working/003-accelsim-dummy-ptx-sim/impl-guide-ph4-us2-ptx-provenance.md`
- `context/tasks/working/003-accelsim-dummy-ptx-sim/impl-guide-ph5-us3-correctness.md`
- `context/tasks/working/003-accelsim-dummy-ptx-sim/impl-guide-ph6-us4-config-preset.md`
- `context/tasks/working/003-accelsim-dummy-ptx-sim/impl-guide-ph7-us5-prereqs-failfast.md`
- `context/tasks/working/003-accelsim-dummy-ptx-sim/impl-guide-ph8-polish.md`
- `context/tasks/working/003-accelsim-dummy-ptx-sim/impl-integrate-phases.md`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 [P] Create Python package skeleton in `/data1/huangzhe/code/accelsim-test/src/accelsim_test/accelsim_dummy_ptx_sim/` (`__init__.py`, `__main__.py`, `model.py`, `paths.py`, `artifacts.py`, `toolchain.py`, `prereqs.py`, `workflow.py`)
- [X] T002 [P] Create minimal CUDA program source directory and stub in `/data1/huangzhe/code/accelsim-test/cpp/accelsim_dummy_ptx_sim/matmul.cu`
- [X] T003 Add Pixi task wrapper `accelsim-dummy-ptx-sim` in `/data1/huangzhe/code/accelsim-test/pyproject.toml` that runs `python -m accelsim_test.accelsim_dummy_ptx_sim run` in the `accelsim` environment

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T004 [P] Implement core entities from `/data1/huangzhe/code/accelsim-test/specs/003-accelsim-dummy-ptx-sim/data-model.md` in `/data1/huangzhe/code/accelsim-test/src/accelsim_test/accelsim_dummy_ptx_sim/model.py` (`SimulationRun`, `RunArtifacts`, `PrerequisiteCheck`) with `to_dict()` helpers for `metadata.json`
- [X] T005 [P] Implement repo root detection + path builders in `/data1/huangzhe/code/accelsim-test/src/accelsim_test/accelsim_dummy_ptx_sim/paths.py` (run dir under `/data1/huangzhe/code/accelsim-test/tmp/accelsim_dummy_ptx_sim/<run_id>/`, filesystem-safe `run_id` sanitizer, SM80_A100 config source path constant from `research.md` Decision 2)
- [X] T006 [P] Implement artifact directory layout + metadata writer in `/data1/huangzhe/code/accelsim-test/src/accelsim_test/accelsim_dummy_ptx_sim/artifacts.py` (create `bin/`, `ptx/`, `run/`, write `/data1/huangzhe/code/accelsim-test/tmp/accelsim_dummy_ptx_sim/<run_id>/metadata.json`)
- [X] T007 [P] Implement CLI parsing for `run` command in `/data1/huangzhe/code/accelsim-test/src/accelsim_test/accelsim_dummy_ptx_sim/__main__.py` with options from `/data1/huangzhe/code/accelsim-test/specs/003-accelsim-dummy-ptx-sim/contracts/cli.md` (`--run-id`, `--compiler`, `--config-preset`)
- [X] T008 Define the typed workflow API and wire CLI → workflow in `/data1/huangzhe/code/accelsim-test/src/accelsim_test/accelsim_dummy_ptx_sim/workflow.py` (single `run(...) -> int` entrypoint that always writes `metadata.json`)

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Run a minimal PTX simulation end-to-end (Priority: P1) 🎯 MVP

**Goal**: A developer can run a minimal CUDA program under Accel-Sim PTX-mode simulation and get a per-run artifact directory containing the executable, PTX, config copy, and simulator log.

**Independent Test**: Run `pixi run -e accelsim python -m accelsim_test.accelsim_dummy_ptx_sim run --run-id 2026-02-03T00-00-00Z` and verify `/data1/huangzhe/code/accelsim-test/tmp/accelsim_dummy_ptx_sim/2026-02-03T00-00-00Z/` contains `bin/`, `ptx/`, `run/`, `metadata.json`, and `run/matmul.sim.log` includes an Accel-Sim banner line.

### Implementation (US1)

- [X] T009 [P] [US1] Implement a minimal CUDA program that runs quickly and prints a completion indicator (e.g., `DONE`) in `/data1/huangzhe/code/accelsim-test/cpp/accelsim_dummy_ptx_sim/matmul.cu`
- [X] T010 [P] [US1] Implement `nvcc` selection + compile command builder in `/data1/huangzhe/code/accelsim-test/src/accelsim_test/accelsim_dummy_ptx_sim/toolchain.py` (`--compiler auto|pixi|system`, emit executable + standalone PTX, target `compute_80` per `research.md` Decision 3)
- [X] T011 [US1] Implement end-to-end workflow in `/data1/huangzhe/code/accelsim-test/src/accelsim_test/accelsim_dummy_ptx_sim/workflow.py`: prerequisite stubs (no fail-fast yet), compile into `bin/matmul` + `ptx/matmul.ptx`, copy SM80_A100 `gpgpusim.config` into `run/`, run via `bash -lc 'export GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN=; source extern/tracked/accel-sim-framework/gpu-simulator/setup_environment.sh; cd .../run; ../bin/matmul'` and capture stdout/stderr into `run/matmul.sim.log`
- [X] T012 [US1] Validate and update `/data1/huangzhe/code/accelsim-test/specs/003-accelsim-dummy-ptx-sim/quickstart.md` so its commands and expected artifact paths match the implemented CLI/workflow
- [X] T013 [P] [US1] Add unit tests for run-id sanitization and artifact layout in `/data1/huangzhe/code/accelsim-test/tests/unit/test_accelsim_dummy_ptx_sim_paths.py`

**Checkpoint**: User Story 1 runs end-to-end and produces the required artifact directory

---

## Phase 4: User Story 2 - Inspect the exact PTX used by the simulator (Priority: P2)

**Goal**: PTX is preserved per-run and can be tied unambiguously to the executed program (no accidental overwrites).

**Independent Test**: Run twice with different `--run-id` values and verify (1) each run has its own `ptx/matmul.ptx`, (2) `metadata.json` contains the PTX path + hash, and (3) reusing the same `--run-id` fails rather than overwriting artifacts.

### Implementation (US2)

- [X] T014 [P] [US2] Add run-dir collision prevention (error if artifacts dir exists) and PTX/source hashing utilities in `/data1/huangzhe/code/accelsim-test/src/accelsim_test/accelsim_dummy_ptx_sim/artifacts.py` (record `ptx_sha256` in `metadata.json`)
- [X] T015 [US2] Copy the compiled CUDA source into run artifacts (e.g., `src/matmul.cu`) and record its path/hash in `/data1/huangzhe/code/accelsim-test/src/accelsim_test/accelsim_dummy_ptx_sim/workflow.py` to make PTX provenance auditable per run
- [X] T016 [P] [US2] Add unit tests for hashing + collision behavior in `/data1/huangzhe/code/accelsim-test/tests/unit/test_accelsim_dummy_ptx_sim_artifacts.py`

---

## Phase 5: User Story 3 - Validate numerical correctness at small sizes (Priority: P3)

**Goal**: The program self-checks numerical correctness against a CPU reference (small sizes) and prints an unambiguous `PASS`/`FAIL`; the workflow records status and distinguishes correctness failures from runtime/simulator failures.

**Independent Test**: Run once and observe deterministic `PASS` in `run/matmul.sim.log` and `metadata.json`. Force a mismatch (temporary code edit) and verify the run is marked `fail` with `failure_reason=correctness_mismatch`.

### Implementation (US3)

- [X] T017 [P] [US3] Implement CPU reference + deterministic inputs + `PASS`/`FAIL` printing (non-zero exit on `FAIL`) in `/data1/huangzhe/code/accelsim-test/cpp/accelsim_dummy_ptx_sim/matmul.cu`
- [X] T018 [US3] Parse `PASS`/`FAIL` from `run/matmul.sim.log`, set `SimulationRun.status` + `failure_reason`, and enforce exit code `0` only on `PASS` in `/data1/huangzhe/code/accelsim-test/src/accelsim_test/accelsim_dummy_ptx_sim/workflow.py`
- [X] T019 [P] [US3] Add unit tests for log parsing and status mapping in `/data1/huangzhe/code/accelsim-test/tests/unit/test_accelsim_dummy_ptx_sim_status.py`

---

## Phase 6: User Story 4 - Switch between supported simulator configurations (Priority: P4)

**Goal**: The workflow supports selecting the supported simulator config preset (Ampere/A100-like) and records which preset was used.

**Independent Test**: Run with `--config-preset sm80_a100` and verify `metadata.json` records the preset and `run/gpgpusim.config` is copied from the preset source path.

### Implementation (US4)

- [X] T020 [P] [US4] Implement config preset mapping/validation (support `sm80_a100` only for this feature) in `/data1/huangzhe/code/accelsim-test/src/accelsim_test/accelsim_dummy_ptx_sim/paths.py`
- [X] T021 [P] [US4] Wire `--config-preset` option through CLI parsing in `/data1/huangzhe/code/accelsim-test/src/accelsim_test/accelsim_dummy_ptx_sim/__main__.py`
- [X] T022 [US4] Update config-copy + metadata recording to use the selected preset in `/data1/huangzhe/code/accelsim-test/src/accelsim_test/accelsim_dummy_ptx_sim/workflow.py`
- [X] T023 [P] [US4] Add unit tests for preset mapping/validation in `/data1/huangzhe/code/accelsim-test/tests/unit/test_accelsim_dummy_ptx_sim_presets.py`

---

## Phase 7: User Story 5 - Fail fast with actionable errors (Priority: P5)

**Goal**: The workflow detects missing prerequisites early and fails with actionable error messages that tell the developer what to do next.

**Independent Test**: With a missing prerequisite (e.g., uninitialized submodule or missing simulator build), running the CLI fails fast with a clear message and non-zero exit code, and still writes `metadata.json` with `status=fail`.

### Implementation (US5)

- [X] T024 [P] [US5] Implement prerequisite checks in `/data1/huangzhe/code/accelsim-test/src/accelsim_test/accelsim_dummy_ptx_sim/prereqs.py` (`submodule_initialized`, `simulator_built`, `nvcc_available`, `config_preset_exists`, `tmp_writable`) returning `list[PrerequisiteCheck]` with actionable `details` strings (commands to run)
- [X] T025 [US5] Integrate prerequisite checks into `/data1/huangzhe/code/accelsim-test/src/accelsim_test/accelsim_dummy_ptx_sim/workflow.py` so failures short-circuit before compile/run, print a concise checklist of missing items, and record `failure_reason` in `metadata.json`
- [X] T026 [P] [US5] Add unit tests for prerequisite checks in `/data1/huangzhe/code/accelsim-test/tests/unit/test_accelsim_dummy_ptx_sim_prereqs.py`
- [X] T027 [US5] Add a troubleshooting section (common failures + exact fix commands) to `/data1/huangzhe/code/accelsim-test/specs/003-accelsim-dummy-ptx-sim/quickstart.md`

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [X] T028 [P] Add a manual smoke-run helper (skips if prerequisites are missing) in `/data1/huangzhe/code/accelsim-test/tests/manual/run_accelsim_dummy_ptx_sim_smoke.py`
- [X] T029 [P] Update repository docs to link the new quickstart in `/data1/huangzhe/code/accelsim-test/README.md`
- [X] T030 Validate the full workflow against `/data1/huangzhe/code/accelsim-test/specs/003-accelsim-dummy-ptx-sim/contracts/cli.md` and update `/data1/huangzhe/code/accelsim-test/specs/003-accelsim-dummy-ptx-sim/quickstart.md` if any paths/flags differ

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: Depend on Foundational completion
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependency Graph

```text
Phase 2 (Foundation)
  └─ US1 (P1: end-to-end run)
       ├─ US2 (P2: PTX provenance + no-overwrite)
       ├─ US3 (P3: correctness PASS/FAIL)
       ├─ US4 (P4: config preset switch)
       └─ US5 (P5: fail-fast prerequisites)
```

---

## Parallel Execution Examples (Per User Story)

### Parallel Example: User Story 1

```text
Can run in parallel:
- T009 [US1] .../cpp/accelsim_dummy_ptx_sim/matmul.cu
- T010 [US1] .../src/accelsim_test/accelsim_dummy_ptx_sim/toolchain.py
```

### Parallel Example: User Story 2

```text
Can run in parallel:
- T014 [US2] .../src/accelsim_test/accelsim_dummy_ptx_sim/artifacts.py
- T016 [US2] .../tests/unit/test_accelsim_dummy_ptx_sim_artifacts.py
```

### Parallel Example: User Story 3

```text
Can run in parallel:
- T017 [US3] .../cpp/accelsim_dummy_ptx_sim/matmul.cu
- T019 [US3] .../tests/unit/test_accelsim_dummy_ptx_sim_status.py
```

### Parallel Example: User Story 4

```text
Can run in parallel:
- T020 [US4] .../src/accelsim_test/accelsim_dummy_ptx_sim/paths.py
- T023 [US4] .../tests/unit/test_accelsim_dummy_ptx_sim_presets.py
```

### Parallel Example: User Story 5

```text
Can run in parallel:
- T024 [US5] .../src/accelsim_test/accelsim_dummy_ptx_sim/prereqs.py
- T026 [US5] .../tests/unit/test_accelsim_dummy_ptx_sim_prereqs.py
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Run the independent test for US1

### Incremental Delivery (Recommended)

1. Setup + Foundational → foundation ready
2. US1 → end-to-end simulation artifacts exist
3. US2 → PTX provenance + no-overwrite guarantees
4. US3 → correctness PASS/FAIL gating
5. US4 → preset selection + recording
6. US5 → fail-fast prerequisite UX
7. Polish → docs + manual smoke script

---

## Notes

- `[P]` tasks should not touch the same file(s)
- All run artifacts must be written under `/data1/huangzhe/code/accelsim-test/tmp/accelsim_dummy_ptx_sim/<run_id>/` only (see `spec.md` FR-009)
- Avoid trace-driven flows (PTX-mode only; see `spec.md` FR-006)
