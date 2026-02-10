# cublaslt-kernel-discovery Specification

## Purpose
TBD - created by archiving change algo23-investigation. Update Purpose after archive.
## Requirements
### Requirement: Capture kernel discovery artifacts
The system SHALL capture and persist GPU kernel discovery artifacts for a given cuBLASLt matmul reproduction program run.

#### Scenario: Persisted discovery bundle
- **WHEN** the user runs kernel discovery for a target repro program and a selected variant of the matmul (e.g., `ABT_view`)
- **THEN** the system MUST write a discovery bundle under the chosen output directory containing:
  - a Nsight Systems capture file (e.g., `.qdrep`) sufficient to inspect kernel launches in the UI
  - a machine-readable kernel listing (e.g., CSV/JSON) that includes kernel name and per-launch index
  - an invocation log describing the exact command line, working directory, and environment used to capture the run

### Requirement: Include launch configuration in the kernel listing
The system SHALL include kernel launch configuration details in the discovery output so kernels can be matched and filtered robustly across tool name formatting changes.

#### Scenario: Export kernel grid/block dimensions
- **WHEN** the system exports the machine-readable kernel listing from the discovery capture
- **THEN** the listing MUST include grid and block dimensions for each kernel (or a grouped summary key that includes them), in addition to kernel name and invocation index.

### Requirement: Support narrowing to relevant kernels
The system SHALL support narrowing discovery output to the kernel launches relevant to the matmul of interest so that downstream profiling can target the correct kernel invocation(s).

#### Scenario: Select kernels by filter
- **WHEN** the user provides a kernel-name regex and/or an NVTX range filter
- **THEN** the system MUST record the filter parameters and MUST produce an output kernel listing that enables unambiguous selection of the kernel invocation(s) to profile (e.g., by kernel name + invocation number).

### Requirement: Standard report-directory layout
The system SHALL store discovery artifacts in a predictable directory layout rooted at a user-specified output directory to enable analysis and stakeholder-facing markdown to reference the artifacts without regenerating the narrative.

#### Scenario: Store under profiles directory
- **WHEN** discovery artifacts are written for a user-provided output directory `<out_dir>`
- **THEN** the system MUST store them under a subdirectory rooted at `<out_dir>/profiles/` and MUST avoid writing into the stakeholder report directly.

### Requirement: Support arbitrary output locations (including tmp for testing)
The system SHALL support writing discovery artifacts to an arbitrary, user-specified output directory path.

#### Scenario: Write to tmp during testing
- **WHEN** the user specifies an output directory under `tmp/<subdir>/`
- **THEN** the system MUST write the discovery bundle under that directory and MUST NOT assume the output directory is under `reports/`.

