# Data Model: GEMM Transpose Performance Benchmark

This document defines the entities and invariants used by the benchmark export and reporting pipeline for `/data1/huangzhe/code/accelsim-test/specs/002-gemm-transpose-bench/spec.md`.

## Entities

### 1) `Run`

Represents one end-to-end benchmark execution (timing sweep and optionally profiling sweep).

Fields:
- `run_id` (string, required): Unique identifier (e.g., timestamp + git short SHA).
- `started_at` / `finished_at` (RFC3339 string, required).
- `status` (enum, required): `pass` | `fail`.
- `failure_reason` (string, optional): Present when `status=fail` (e.g., correctness failure, missing profiling artifact).
- `git` (object, required): `{ "branch": string, "commit": string, "dirty": bool }`.
- `environment` (object, required): See `RunMetadata` below.
- `settings` (object, required): Global NVBench stopping parameters and benchmark defaults used for the run.
- `artifacts_dir` (string, required): Base directory for artifacts (prefer path relative to the export file; allow absolute in local runs).

State transitions:
- `started -> completed(pass|fail)`.

### 2) `BenchmarkCase`

Identifies which matmul variant was executed.

Enum values:
- Square suite: `AB`, `ATB_view`, `ABT_view`, `ATB_copyA`, `ABT_copyB`
- Non-square suite: `ATB_view`, `ATB_copyA`, `ABT_view`, `ABT_copyB`

Validation rules:
- Square suite rows must include all five cases (FR-002).
- Non-square suite rows must include the two cases for the selected direction (FR-003).

### 3) `Suite`

Identifies the comparison family.

Fields:
- `suite` (enum, required): `square` | `nonsquare_atb` | `nonsquare_abt`.

Validation rules:
- `square` implies `M=N=K` and includes cases `AB`, `ATB_*`, `ABT_*`.
- `nonsquare_atb` uses inputs shaped to make transpose-A valid; only `ATB_*` cases are valid.
- `nonsquare_abt` uses inputs shaped to make transpose-B valid; only `ABT_*` cases are valid.

### 4) `Configuration`

Represents a single parameter point for execution (suite + shape + dtype + math mode), independent of case.

Fields:
- `suite` (enum, required): See `Suite`.
- `shape` (object, required): `{ "m": int, "n": int, "k": int }` for the GEMM `C[m,n] = A[m,k] @ B[k,n]` in logical math terms.
- `dtype` (object, required): `{ "a": string, "b": string, "c": string, "compute": string, "math_mode": string }`.
- `seed` (int, required): Input data RNG seed for reproducibility.
- `nvbench_axes` (object, required): The concrete NVBench axis values used to generate/identify the configuration.

Validation rules:
- `m`, `n`, `k` must be positive.
- Dtype combinations must match the supported set defined in `/data1/huangzhe/code/accelsim-test/context/tasks/req-cuda-gemm-test.md`.

### 5) `ResultRecord`

Represents the measured output for a single configuration/case.

Fields:
- `configuration` (object, required): Embedded `Configuration` (or a `config_id` reference).
- `case` (enum, required): `BenchmarkCase`.
- `timing` (object, required): `{ "gpu_time_ms": number, "cpu_time_ms": number|null, "measurement": "cold"|"batch", "samples": int|null, "nvbench_raw": object|null }`.
- `flop_count` (int, required): Always `2*m*n*k` (FR-008 / FR-008b).
- `ratios` (object, optional): Derived ratio fields populated during normalization (e.g., `ratio_to_ab`, `ratio_copy_over_view`).
- `verification` (object, required): See `VerificationResult`.
- `profiling` (object, optional): See `ProfilingArtifactRef` (present when profiling mode enabled).

Validation rules:
- `flop_count` is constant across all cases compared within one report row (FR-008).
- For integer cases, throughput-per-second metrics must be omitted in final stakeholder report (FR-008b); raw timing is still recorded.

### 6) `VerificationResult`

Represents per-record correctness status.

Fields:
- `status` (enum, required): `pass` | `fail`.
- `mode` (enum, required): `sampled` | `full`.
- `max_abs_error` (number, optional): For float outputs.
- `max_rel_error` (number, optional): For float outputs.
- `num_samples` (int, optional): For sampled mode.
- `details` (string, optional): Human-readable summary (e.g., first failing index, thresholds).

Validation rules:
- Every `ResultRecord` must include a verification result (FR-011).
- Any `fail` must cause overall run `status=fail` (FR-011a), while still exporting all records.

### 7) `ProfilingArtifactRef`

Links profiling outputs to a specific configuration/case.

Fields:
- `ncu_rep` (string, required): Path to `*.ncu-rep`.
- `ncu_summary` (string, optional): Path to extracted metrics (CSV/text).
- `command` (string, required): The exact `ncu ... -- <bench> ...` command used (for reproducibility).

Validation rules:
- In profiling-enabled mode, every executed configuration/case must include artifact references (FR-012a).

### 8) `ReportRow`

Represents the stakeholder-facing aggregation for one configuration (square suite row has 5 cases; non-square row has 2 cases).

Fields:
- `configuration` (object, required): `Configuration`.
- `cases` (object, required): Map from `BenchmarkCase` to `{ "gpu_time_ms": number, "verification": "pass"|"fail" }`.
- `flop_count` (int, required).
- `ratios` (object, required): Includes ratios required by FR-010 (e.g., per-case slowdown vs `AB` and copy-over-view ratios).

Validation rules:
- A report row must not compare cases with differing `flop_count` (FR-008 / FR-010).
