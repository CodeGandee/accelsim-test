## ADDED Requirements

### Requirement: Produce ncu profiling artifacts
The system SHALL collect Nsight Compute profiling artifacts for a specified kernel invocation (or set of invocations) associated with a cuBLASLt matmul reproduction run.

#### Scenario: Export ncu report
- **WHEN** the user requests profiling for a selected kernel name (or NVTX range) and an invocation selection (e.g., first matching launch)
- **THEN** the system MUST export an ncu report artifact (e.g., `.ncu-rep`) into the chosen output directory and MUST record the exact `ncu` command line used.

### Requirement: Support deterministic profiling scoping
The system SHALL support deterministic scoping of profiling to the intended matmul region to avoid collecting unrelated kernel launches.

#### Scenario: Gate profiling by range or profiler start/stop
- **WHEN** the user requests profiling scoped to a specific matmul variant
- **THEN** the system MUST support at least one of the following scoping mechanisms and record which mechanism was used:
  - NVTX range inclusion/exclusion
  - `cudaProfilerStart/Stop`-based gating (with `ncu` configured to respect start/stop)

### Requirement: Support side-by-side comparisons
The system SHALL support generating comparable profiling bundles for two or more variants of the same matmul (e.g., `ABT_view algo=23` vs `ABT_view algo=64`) to allow attributing performance differences to kernel/runtime behavior.

#### Scenario: Profile fast and baseline paths
- **WHEN** the user requests profiling for a fast path and a baseline path for the same `(M,N,K,dtypes)` problem
- **THEN** the system MUST store the profiling outputs in distinct, clearly labeled subdirectories and MUST include enough metadata to distinguish the compared variants (transpose flags, algo_id, tile_id, stages_id, splitK).

### Requirement: Baseline comparison uses a forced algorithm configuration
The system SHALL support using a forced baseline algorithm configuration for comparisons to isolate kernel-path differences.

#### Scenario: Baseline uses forced algo 64
- **WHEN** the user requests a comparison between `ABT_view` fast path (`algo_id=23`) and a baseline path
- **THEN** the system MUST support forcing `ABT_view` to use `algo_id=64` for the baseline bundle (when supported by `cublasLtMatmulAlgoCheck`) and MUST record the forced algorithm configuration in the profiling metadata.

### Requirement: Standard report-directory layout
The system SHALL store ncu artifacts in a predictable directory layout rooted at a user-specified output directory to enable analysis and stakeholder-facing markdown to reference the artifacts without regenerating the narrative.

#### Scenario: Store under profiles directory
- **WHEN** ncu artifacts are written for a user-provided output directory `<out_dir>`
- **THEN** the system MUST store them under a subdirectory rooted at `<out_dir>/profiles/` and MUST avoid writing into the stakeholder report directly.

### Requirement: Support arbitrary output locations (including tmp for testing)
The system SHALL support writing profiling artifacts to an arbitrary, user-specified output directory path.

#### Scenario: Write to tmp during testing
- **WHEN** the user specifies an output directory under `tmp/<subdir>/`
- **THEN** the system MUST write the profiling bundles under that directory and MUST NOT assume the output directory is under `reports/`.
