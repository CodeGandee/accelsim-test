## ADDED Requirements

### Requirement: Run layout-order focus matrix
The system SHALL provide a focused experiment that runs the following GEMM transpose-view cases for `M=N=K=1000` (int8 inputs, int32 accumulation/output):

- `AB`: `trans_a=N`, `trans_b=N`
- `ATB_view`: `trans_a=T`, `trans_b=N`
- `ABT_view`: `trans_a=N`, `trans_b=T`

The experiment SHALL execute the full 2×3 matrix by varying the cuBLASLt matrix-layout order:

- Row-major: `CUBLASLT_ORDER_ROW`
- Column-major: `CUBLASLT_ORDER_COL`

#### Scenario: Execute row-major runs
- **WHEN** the user selects `order=row`
- **THEN** the system runs `AB`, `ATB_view`, and `ABT_view` using row-major matrix layouts and records the selected algorithm/kernel evidence.

#### Scenario: Execute column-major runs
- **WHEN** the user selects `order=col`
- **THEN** the system runs `AB`, `ATB_view`, and `ABT_view` using column-major matrix layouts and records the selected algorithm/kernel evidence.

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
