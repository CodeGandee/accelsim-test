# Feature Specification: GEMM Transpose Performance Benchmark

**Feature Branch**: `[002-gemm-transpose-bench]`  
**Created**: 2026-02-02  
**Status**: Draft  
**Input**: User description: "revise the current spec according to the updated context/tasks/req-cuda-gemm-test.md , and create 5 user stories"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run square-suite benchmark (Priority: P1)

As a performance engineer, I want to benchmark matrix multiplication when one operand is transposed (as a view vs as a materialized copy) for square matrices, so I can quantify how transpose handling changes runtime while holding math cost constant.

**Why this priority**: This is the strictest apples-to-apples comparison: same inputs, same output shape, same theoretical FLOP count.

**Independent Test**: Run the square suite for a single size and numeric type pair and verify the results export contains all square-suite cases with timings and derived ratios.

**Acceptance Scenarios**:

1. **Given** square matrices `A[N,N]` and `B[N,N]`, **When** I run the square suite, **Then** results are produced for all required cases: `A@B`, `Aᵀ@B`, `A@Bᵀ`, `copy(Aᵀ)@B`, `A@copy(Bᵀ)`.
2. **Given** square-suite results, **When** I review the comparison fields, **Then** the report includes slowdowns relative to `A@B` and overhead factors comparing materialized-transpose vs transpose-view.

---

### User Story 2 - Run non-square-suite benchmark (Priority: P2)

As a performance engineer, I want to benchmark transpose-A and transpose-B workloads for non-square matrices with matched theoretical FLOP counts, so I can study transpose effects across aspect ratios without invalid shape combinations.

**Why this priority**: Aspect ratio is a major driver of performance; the benchmark must cover non-square shapes in a way that preserves fair compute-cost comparison.

**Independent Test**: Run the non-square suite for a single `(M,N,K)` and numeric type pair and verify that transpose-view and materialized-transpose results exist for both transpose directions.

**Acceptance Scenarios**:

1. **Given** matrices shaped to make transpose-A multiplication valid, **When** I run transpose-A cases, **Then** results are produced for `Aᵀ@B` and `copy(Aᵀ)@B`.
2. **Given** matrices shaped to make transpose-B multiplication valid, **When** I run transpose-B cases, **Then** results are produced for `A@Bᵀ` and `A@copy(Bᵀ)`.

---

### User Story 3 - Validate numerical correctness (Priority: P3)

As a performance engineer, I want each reported configuration to be sanity-checked against an independent reference computation, so I can trust that performance comparisons are not polluted by incorrect outputs.

**Why this priority**: Performance numbers without correctness are not actionable; correctness must be verified for every configuration and case.

**Independent Test**: Run a small configuration and confirm the export marks verification as pass and records an error summary; introduce an intentional mismatch and confirm it is flagged as fail.

**Acceptance Scenarios**:

1. **Given** a benchmark run completes, **When** verification is enabled, **Then** every configuration/case is labeled pass/fail and includes an error summary appropriate to the numeric types used.
2. **Given** verification fails for any configuration, **When** the run finishes, **Then** the exported results clearly identify the failing configurations and the overall run indicates failure.

---

### User Story 4 - Export structured results for analysis (Priority: P4)

As an analyst, I want benchmark results exported in a structured, machine-readable format with complete metadata, so I can reproduce analyses and build dashboards without manual parsing.

**Why this priority**: Structured export is required for automated analysis, regression tracking, and reproducibility.

**Independent Test**: Run a small sweep and confirm every output record contains configuration parameters, timing fields, `flop_count`, derived throughput fields, and verification status.

**Acceptance Scenarios**:

1. **Given** a benchmark sweep completes, **When** I load the exported dataset, **Then** every record includes the configuration (suite, shape, numeric type pair, case), the measured timing, and environment metadata.
2. **Given** a single comparison row in the final report, **When** I check the computed FLOP count, **Then** all cases on that row share the same `flop_count` (otherwise they are split into separate rows).

---

### User Story 5 - Generate stakeholder report with comparison tables (Priority: P5)

As a stakeholder, I want a concise report that summarizes the impact of transpose handling on performance across shapes and numeric types, so I can understand conclusions without reading raw data.

**Why this priority**: The goal is to communicate findings and guide decisions; a clear report is the primary deliverable for non-technical audiences.

**Independent Test**: Generate the report from an exported dataset and verify it contains the required tables, ratios, and a short conclusion section.

**Acceptance Scenarios**:

1. **Given** an exported results dataset, **When** I generate the report, **Then** it includes one square-suite comparison table and one non-square-suite comparison table with per-row `flop_count`, per-case timings, and ratio columns.
2. **Given** the report tables, **When** I compare transpose-view vs materialized-transpose rows, **Then** the report highlights where materialization overhead is significant and where transpose-view changes low-level execution behavior and overall performance.

---

### Edge Cases

- Requested shapes are invalid for the selected suite/case.
- Requested shapes exceed available device memory.
- Verification is too slow for large cases and must use a reduced verification mode while still providing a per-configuration sanity check.
- Results export is incomplete/corrupted or missing required fields.
- Profiling artifacts cannot be produced for one or more configurations (tool unavailable or run aborted).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST support two suites of comparisons: a square suite and a non-square suite, each with clearly defined valid shapes and cases.
- **FR-002**: Square suite MUST compare the five cases for `A[N,N]` and `B[N,N]`: `A@B`, `Aᵀ@B`, `A@Bᵀ`, `copy(Aᵀ)@B`, `A@copy(Bᵀ)`.
- **FR-003**: Non-square suite MUST compare these cases:
  - transpose-A: `Aᵀ@B` and `copy(Aᵀ)@B` for inputs shaped so the multiplication is valid.
  - transpose-B: `A@Bᵀ` and `A@copy(Bᵀ)` for inputs shaped so the multiplication is valid.
- **FR-004**: The benchmark MUST support sweeps over relevant matrix sizes and aspect ratios, including both small (cache-friendly) and large (cache-stressing) configurations.
- **FR-005**: The benchmark MUST support the required numeric type pairs, including 8-bit integer and 16-bit/32-bit floating point combinations, and MUST record the effective compute/output types used for each run.
- **FR-006**: Each case MUST be executed enough times (with warmup) to produce a stable average timing suitable for comparison.
- **FR-007**: Timing MUST reflect accelerator execution time for the matrix multiplication and MUST exclude data transfer time (moving inputs/outputs to/from accelerator memory).
- **FR-008**: For each configuration/case, the system MUST compute and export a theoretical operation count (`flop_count`, measured in floating-point operations) and it MUST be consistent across all cases compared within a single report row.
- **FR-009**: The system MUST export results in a structured machine-readable format containing, at minimum: suite, shape parameters, numeric type pair, case identifier, average timing, `flop_count`, derived throughput, verification status, and environment metadata.
- **FR-010**: The system MUST generate a stakeholder report containing:
  - a square-suite comparison table with per-case timings and ratios relative to `A@B` plus materialization-overhead ratios,
  - a non-square-suite comparison table with per-case timings and materialization-overhead ratios,
  - and a short conclusions section explaining observed trends.
- **FR-011**: The system MUST perform correctness verification for every configuration/case against an independent reference computation and record pass/fail plus an error summary.
- **FR-012**: The system MUST support a profiling-enabled run mode that captures low-level execution behavior per configuration/case and links the artifacts to exported results for analysis.

### Assumptions & Dependencies

- A single accelerator-capable machine is available to run the benchmark.
- The benchmark has access to a stable execution environment so that repeated runs can be compared fairly.
- An independent reference implementation is available for correctness checking.
- Some performance variability is expected on shared systems; the benchmark mitigates this via warmup and repeated measurement.

### Key Entities *(include if feature involves data)*

- **Suite**: Whether a run belongs to square or non-square comparisons.
- **Configuration**: Shape parameters (N or M/N/K), numeric type pair, and case identifier.
- **Result Record**: Timing, `flop_count`, throughput, ratios (as applicable), and verification status for a configuration/case.
- **Run Metadata**: Hardware/software environment details needed to interpret and reproduce results.
- **Report**: Stakeholder-facing summary tables and conclusions derived from result records.
- **Profiling Artifact**: Per-configuration/case trace/report data used to explain observed performance differences.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A default benchmark run produces a complete structured export containing 100% of required fields for 100% of executed configurations/cases.
- **SC-002**: The stakeholder report includes both required tables and enforces per-row `flop_count` consistency for all ratio comparisons.
- **SC-003**: For a fixed set of parameters on the same machine, repeated runs produce sufficiently stable comparisons to support conclusions (e.g., within 10% for large configurations).
- **SC-004**: Correctness verification is performed for every configuration/case and failures are clearly surfaced in both exports and report.
- **SC-005**: Profiling artifacts are produced and attributable for each configuration/case in the profiling-enabled run mode, enabling kernel-level interpretation of observed differences.
