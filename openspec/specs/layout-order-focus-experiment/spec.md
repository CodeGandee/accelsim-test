# layout-order-focus-experiment Specification

## Purpose
TBD - created by archiving change layout-order-n1000-focus-experiment. Update Purpose after archive.
## Requirements
### Requirement: Run layout-order focus matrix
The system SHALL provide a focused experiment that runs the following GEMM transpose-view cases for `M=N=K=1000` (int8 inputs, int32 accumulation/output):

- `AB`: `trans_a=N`, `trans_b=N`
- `ATB_view`: `trans_a=T`, `trans_b=N`
- `ABT_view`: `trans_a=N`, `trans_b=T`

The experiment SHALL execute a full 4×3 matrix by varying the cuBLASLt matrix-layout order of A and B independently:

- A order:
  - Row-major: `CUBLASLT_ORDER_ROW`
  - Column-major: `CUBLASLT_ORDER_COL`
- B order:
  - Row-major: `CUBLASLT_ORDER_ROW`
  - Column-major: `CUBLASLT_ORDER_COL`

#### Scenario: Execute a selected (A order, B order) run
- **WHEN** the user selects an A order and a B order (e.g., `order_a=row`, `order_b=col`)
- **THEN** the system runs `AB`, `ATB_view`, and `ABT_view` using those matrix-layout orders and records the selected algorithm/kernel evidence.
- **AND** the output artifacts MUST record both the A and B layout orders used for the run.

#### Scenario: Execute mixed-order runs
- **WHEN** the user selects mixed A/B layout orders (i.e., `order_a != order_b`)
- **THEN** the system runs `AB`, `ATB_view`, and `ABT_view` using those mixed matrix-layout orders and records the selected algorithm/kernel evidence.

### Requirement: Limited output-order variation (C order)
The system SHALL support varying the output matrix layout order (C/D layouts) and SHALL include a limited output-order sensitivity check for the baseline `order_a=row, order_b=row` case.

#### Scenario: Sweep output order for row/row
- **WHEN** the user runs the experiment matrix
- **THEN** the system MUST run the `order_a=row, order_b=row` cases for both:
  - `order_c=row` and
  - `order_c=col`
- **AND** the output artifacts MUST record the output order used for each run.

### Requirement: Optional math-equivalence mode
The system SHALL support an option to generate inputs such that the three transpose-view cases produce the same mathematical result, to avoid “different GEMM” concerns during performance comparison.

#### Scenario: Symmetric-input mode
- **WHEN** the user enables `symmetric_inputs=true`
- **THEN** the system generates symmetric `A` and symmetric `B` (i.e., `A=Aᵀ` and `B=Bᵀ`) and may verify that outputs across `AB`, `ATB_view`, and `ABT_view` match within exact int32 equality (or an explicitly documented tolerance if needed).

### Requirement: Configurable output directory and reproducible artifacts
The system SHALL write all outputs under a user-specified directory (e.g., `tmp/<subdir>`), and SHALL keep per-case artifacts in deterministic subdirectories so that results can be compared and referenced from reports.

#### Scenario: Write outputs to a chosen directory
- **WHEN** the user passes `--out-dir <dir>`
- **THEN** the system writes all result artifacts under `<dir>` and does not assume any fixed location under `reports/`.

### Requirement: Kernel-level evidence capture (optional)
The system SHALL support optional kernel-evidence capture for each case, including:

- the selected cuBLASLt algorithm/config (at minimum `algo_id`, plus tile/stages/splitK where available),
- the executed kernel name(s) for the timed GEMM region (e.g., from Nsight Systems GPU trace), and
- optional Nsight Compute profiling output for the timed GEMM kernel.

#### Scenario: Capture kernel discovery with Nsight Systems
- **WHEN** the user enables `nsys=true`
- **THEN** the system runs Nsight Systems kernel discovery and records a kernel list identifying the timed GEMM kernel for each case.

#### Scenario: Capture kernel profiling with Nsight Compute
- **WHEN** the user enables `ncu=true`
- **THEN** the system runs Nsight Compute on the timed GEMM kernel and records a report (`.ncu-rep`) and exported CSVs suitable for post-hoc analysis.

