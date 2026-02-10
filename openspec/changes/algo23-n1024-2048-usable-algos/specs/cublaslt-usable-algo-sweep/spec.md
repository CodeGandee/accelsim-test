## ADDED Requirements

### Requirement: Run the focused shape/variant matrix
The system SHALL run a focused int8 experiment for square GEMM with row-major matrix layouts that includes only the `AB` and `ABT_view` variants.

#### Scenario: Execute required matrix
- **WHEN** the user runs the usable-algo sweep experiment with default scope
- **THEN** the system MUST execute both `AB` and `ABT_view` for each of `N=1000`, `N=1024`, and `N=2048` with `M=N=K` and row-major A/B/C layouts.

### Requirement: Enumerate candidate cuBLASLt algorithms per case
The system SHALL enumerate candidate cuBLASLt algorithm configurations for each `(shape, variant)` case before final timing comparison.

#### Scenario: Produce per-case candidate list
- **WHEN** the system prepares a case `(N, variant)`
- **THEN** it MUST generate a candidate list of algorithm configurations from cuBLASLt discovery APIs and record each candidate with at least `algo_id`, tile/stages/splitK-related fields when available.

### Requirement: Classify candidate usability with AlgoCheck
The system SHALL evaluate candidate compatibility using `cublasLtMatmulAlgoCheck` under a fixed workspace policy and classify each candidate as usable or non-usable.

#### Scenario: Candidate accepted or rejected
- **WHEN** a candidate algorithm configuration is tested for a specific `(N, variant)`
- **THEN** the system MUST run `cublasLtMatmulAlgoCheck` for that case and record whether it is usable.
- **AND** for non-usable candidates, the system MUST record failure status information sufficient to distinguish unsupported vs other check failures.

### Requirement: Benchmark all usable candidates
The system SHALL benchmark every candidate classified as usable for each `(N, variant)` case under consistent timing settings.

#### Scenario: Timing run for usable candidates
- **WHEN** candidate usability classification is complete for a case
- **THEN** the system MUST run timed measurements for all usable candidates using a consistent warmup/iteration configuration and record per-candidate timing statistics.

### Requirement: Report heuristic-selected algorithm alongside candidate results
The system SHALL record the heuristic-selected algorithm for each case and present it together with candidate usability and timing outcomes.

#### Scenario: Compare heuristic pick against full usable set
- **WHEN** case-level results are finalized
- **THEN** the system MUST include the heuristic-selected configuration in the output and indicate whether it is the fastest measured usable candidate.

### Requirement: Emit reproducible raw artifacts and summary report
The system SHALL emit machine-readable raw artifacts and a concise markdown summary sufficient to explain why `algo_id=23` appears or does not appear across `N=1000/1024/2048`.

#### Scenario: Write outputs under experiment directory
- **WHEN** the experiment finishes
- **THEN** the system MUST write raw per-case candidate records (including usability status and timing) to CSV/JSON files under the selected output directory.
- **AND** the system MUST write a markdown summary that includes AB-vs-ABT comparison tables for each `N` and an explicit statement of `algo_id=23` availability and rank/performance where applicable.
