# Phase Integration Guide: Accel-Sim Dummy CUDA PTX Simulation

**Feature**: `003-accelsim-dummy-ptx-sim` | **Phases**: 8

## Overview

This feature is a file-based workflow orchestrated by Python (Pixi) that compiles a minimal CUDA program to embedded PTX and then runs it under Accel-Sim’s PTX-mode simulation. Each run produces a self-contained artifact directory under `tmp/accelsim_dummy_ptx_sim/<run_id>/` that captures the executable, emitted PTX, simulator config copy, simulator output, and a machine-readable `metadata.json`.

Implementation is staged so each phase adds a thin slice: scaffolding → core models/paths → end-to-end run → provenance → correctness gating → preset selection → fail-fast prerequisites → polish/docs.

## Phase Flow

**MUST HAVE: End-to-End Sequence Diagram**

```mermaid
sequenceDiagram
    participant Dev as Developer
    participant CLI as __main__<br/>(Python)
    participant WF as workflow.run<br/>(Python)
    participant PR as prereqs.check_all<br/>(Python)
    participant TC as toolchain.compile<br/>(Python)
    participant SIM as Accel-Sim env<br/>setup_environment
    participant FS as tmp/RUN_ID<br/>artifacts

    Dev->>CLI: run --run-id <id>
    CLI->>WF: run(...)
    WF->>PR: check_all(repo_root)
    PR-->>WF: prerequisite list

    alt prerequisites missing
        WF->>FS: write metadata.json<br/>(status=fail)
        WF-->>CLI: exit != 0
        CLI-->>Dev: actionable error
    else prerequisites ok
        WF->>FS: mkdir bin/ ptx/ run/
        WF->>TC: compile matmul.cu<br/>emit exe + ptx
        TC-->>WF: compile command
        WF->>FS: copy gpgpusim.config
        WF->>SIM: execute bin/matmul<br/>from run/
        SIM-->>FS: write matmul.sim.log
        WF->>WF: parse PASS/FAIL
        WF->>FS: write metadata.json
        WF-->>CLI: exit 0 on PASS
        CLI-->>Dev: artifact dir path
    end
```

## Artifact Flow Between Phases

```mermaid
graph TD
    subgraph P1["Phase 1: Setup"]
        P1A[T001: Python pkg skeleton]
        P1B[T002: CUDA source matmul.cu]
        P1C[T003: Pixi task wrapper]
    end

    subgraph P2["Phase 2: Foundational"]
        P2M[T004: model.py]
        P2P[T005: paths.py]
        P2A[T006: artifacts.py]
        P2C[T007: __main__.py]
        P2W[T008: workflow.py]
    end

    subgraph P3["Phase 3: US1 End-to-End"]
        P3S[T009: matmul.cu logic]
        P3T[T010: toolchain.py]
        P3W[T011: workflow simulate]
        P3Q[T012: quickstart update]
        P3U[T013: unit tests paths]
    end

    subgraph RUN["Run Artifacts: tmp/RUN_ID/"]
        R1[bin/matmul]
        R2[ptx/matmul.ptx]
        R3[run/gpgpusim.config]
        R4[run/matmul.sim.log]
        R5[metadata.json]
    end

    P1A --> P2C;
    P2M --> P2W;
    P2P --> P2W;
    P2A --> P2W;
    P2C --> P2W;

    P3T --> R1;
    P3T --> R2;
    P3W --> R3;
    P3W --> R4;
    P3W --> R5;
```

## System Architecture

```mermaid
classDiagram
    class SimulationRun {
        +run_id: str
        +status: str
        +failure_reason: str
        +to_dict() dict
    }

    class RunArtifacts {
        +exe_path: Path
        +ptx_path: Path
        +config_path: Path
        +stdout_log_path: Path
        +metadata_path: Path
        +to_dict() dict
    }

    class Workflow {
        +run(run_id,compiler,config_preset) int
    }

    class Toolchain {
        +compile_cuda_program(...) str
    }

    class Paths {
        +find_repo_root() Path
        +run_artifacts_dir(...) Path
        +preset_config_source(...) Path
    }

    class Prereqs {
        +check_all(repo_root) list
    }

    class Artifacts {
        +ensure_new_run_dir(dir)
        +create_artifact_dirs(dir) dict
        +write_metadata(path,payload)
    }

    Workflow --> SimulationRun: builds
    Workflow --> RunArtifacts: writes
    Workflow --> Toolchain: compiles
    Workflow --> Paths: resolves paths
    Workflow --> Prereqs: checks
    Workflow --> Artifacts: creates/writes
```

## Use Cases

```mermaid
graph LR
    Dev((Developer))

    UC1[Build simulator]
    UC2[Run dummy PTX sim]
    UC3[Inspect PTX artifact]
    UC4[Validate PASS/FAIL]
    UC5[Troubleshoot prereqs]

    Dev --> UC1;
    Dev --> UC2;
    Dev --> UC3;
    Dev --> UC4;
    Dev --> UC5;

    UC1 -.->|prerequisite| UC2;
    UC2 -.->|produces| UC3;
    UC2 -.->|produces| UC4;
    UC5 -.->|unblocks| UC2;
```

## Activity Flow

```mermaid
stateDiagram-v2
    [*] --> PrereqCheck

    PrereqCheck --> PrereqFail: missing prereq
    PrereqFail --> [*]: write metadata fail

    PrereqCheck --> Compile
    Compile --> ConfigCopy
    ConfigCopy --> Simulate
    Simulate --> ParseResult
    ParseResult --> WriteMetadata
    WriteMetadata --> [*]
```

## Inter-Phase Dependencies

### Phase 1 → Phase 2

**Artifacts**:

- No runtime artifacts; this is code scaffolding only.

**Code Dependencies**:

- Phase 2 assumes the module layout created in Phase 1:
  - `src/accelsim_test/accelsim_dummy_ptx_sim/{model,paths,artifacts,workflow,__main__}.py`

### Phase 2 → Phase 3

**Artifacts**:

- The run directory layout (Phase 2) is the contract Phase 3 populates:
  - `<workspace>/tmp/accelsim_dummy_ptx_sim/<run_id>/{bin,ptx,run}/`
  - `<workspace>/tmp/accelsim_dummy_ptx_sim/<run_id>/metadata.json`

**Code Dependencies**:

```python
from accelsim_test.accelsim_dummy_ptx_sim import artifacts, paths, toolchain
from accelsim_test.accelsim_dummy_ptx_sim.model import SimulationRun
```

### Phase 3 → Phase 4/5/6/7

**Artifacts**:

- `ptx/matmul.ptx` becomes hashable/provenanced (Phase 4).
- `run/matmul.sim.log` becomes a correctness signal (Phase 5).
- `run/gpgpusim.config` becomes preset-selected (Phase 6).
- `metadata.json` becomes richer with prerequisite check results (Phase 7).

## Integration Testing

```bash
# Unit-test the orchestration helpers (no simulator):
pixi run pytest -q tests/unit/test_accelsim_dummy_ptx_sim_*.py

# Manual end-to-end run (requires simulator prerequisites):
pixi run -e accelsim python -m accelsim_test.accelsim_dummy_ptx_sim run --run-id 2026-02-03T00-00-00Z
```

## Critical Integration Points

1. **Compute capability alignment**
   - Compile with `compute_80` (Decision 3) and use SM80_A100 preset (Decision 2).
   - Mismatch here is a common “kernel load” failure mode.

2. **Working directory + config placement**
   - The simulator expects `gpgpusim.config` in the working directory where the app is run.
   - Always `cd` into `<run_dir>/run/` before executing `../bin/matmul`.

3. **No writes outside `tmp/`**
   - All artifacts must stay under `tmp/accelsim_dummy_ptx_sim/<run_id>/` (FR-009).

4. **`metadata.json` is written on failure**
   - Even if prerequisites fail, record `status=fail` and actionable `failure_reason`.

5. **Run directory collisions**
   - Refuse to overwrite existing `<run_id>` directories to preserve PTX provenance (US2).

## References

- Individual phase guides: `context/tasks/working/003-accelsim-dummy-ptx-sim/impl-guide-ph*.md`
- Spec: `specs/003-accelsim-dummy-ptx-sim/spec.md`
- Tasks breakdown: `specs/003-accelsim-dummy-ptx-sim/tasks.md`
- Data model: `specs/003-accelsim-dummy-ptx-sim/data-model.md`
- Contracts: `specs/003-accelsim-dummy-ptx-sim/contracts/`
