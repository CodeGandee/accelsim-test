# Research: GEMM Transpose Performance Benchmark

This document records key technical decisions for implementing `/data1/huangzhe/code/accelsim-test/specs/002-gemm-transpose-bench/spec.md` and resolves any open questions from planning by committing to specific, actionable choices.

## Decision 1: Python orchestrates; C++ measures

Decision: Implement the performance-critical benchmark in C++/CUDA under `/data1/huangzhe/code/accelsim-test/cpp/` and use Python under `/data1/huangzhe/code/accelsim-test/src/accelsim_test/` to orchestrate sweeps, profiling runs, normalization/export, and report generation.

Rationale: This matches the repository architecture (Python orchestrator, C++ high-performance subsystem) and keeps timing behavior stable and close to the CUDA/cuBLASLt call sites.

Alternatives considered: A pure-Python benchmark (e.g., via PyTorch/CuPy) was rejected because the requirements mandate cuBLASLt directly and NVBench as the timing harness, and because Python introduces additional variability in measurement and profiling attribution.

## Decision 2: NVBench is the only timing harness (rigor requirement)

Decision: Use NVBench for all kernel timing, warmup, and statistical stopping criteria; do not implement custom timing loops with `cudaEventRecord` or wall-clock timing. NVBench source is used directly from `/data1/huangzhe/code/accelsim-test/extern/orphan/nvbench`.

Rationale: NVBench provides a robust and configurable methodology (warmup, convergence/noise thresholds, batch/cold measurements, JSON/CSV output, and `--profile` mode) and is explicitly required by `/data1/huangzhe/code/accelsim-test/context/tasks/req-cuda-gemm-test.md`.

Alternatives considered: Manual CUDA event timing was rejected (non-compliant). Google Benchmark was rejected because it is not GPU-first and does not provide NVBench’s CUDA-specific controls and outputs.

## Decision 3: GEMM backend is cuBLASLt only

Decision: All GEMM cases are executed via cuBLASLt (no custom kernels, no CUTLASS GEMM implementation for timing).

Rationale: This is explicitly in-scope and required; it isolates the comparison to cuBLASLt algorithm selection and operand-layout handling (view vs copy).

Alternatives considered: CUTLASS kernels were rejected for timing because they change the kernel selection space; they may still be used only as a non-timed reference path if needed for correctness debugging, but the primary correctness approach avoids depending on a second GEMM implementation.

## Decision 4: Represent “transpose view” without data movement; represent “copy(transpose)” with explicit transpose outside timing

Decision: Implement view-vs-copy comparisons as follows:

- View cases (`Aᵀ@B`, `A@Bᵀ`): avoid any transpose kernel and represent the transposed operand through cuBLASLt layout/operation descriptors so the underlying buffer is reused.
- Copy cases (`copy(Aᵀ)@B`, `A@copy(Bᵀ)`): explicitly materialize the transposed operand into a new contiguous buffer outside the NVBench timed region, then time only the GEMM call.

Rationale: This matches FR-008a and allows a clean separation between “data movement cost” (transpose materialization) and GEMM execution cost. Keeping transpose materialization outside the timed region ensures `flop_count` consistency across compared cases.

Alternatives considered: Timing “copy + GEMM” together was rejected because it violates the specification’s requirement to time GEMM only for copy cases.

## Decision 5: Parameterization and sweeps use NVBench axes and CLI overrides

Decision: Encode the sweep space in NVBench axes (suite, case, M/N/K, dtype pair, math mode) and rely on NVBench CLI to (a) run full sweeps and (b) run a single configuration/case for profiling by overriding each axis to a single value.

Rationale: NVBench’s axis system and `--axis` CLI make it straightforward to run full timing sweeps and then deterministic per-case profiling invocations without building separate binaries for each case.

Alternatives considered: A custom Python sweep driver that repeatedly calls into the C++ benchmark API was rejected because it would bypass NVBench’s core measurement controls and complicate profiling/attribution.

## Decision 6: Profiling uses per-case `ncu` runs with NVBench `--profile`

Decision: For profiling-enabled runs, invoke Nsight Compute (`ncu`) once per configuration/case and run the NVBench executable with `--profile` plus axis overrides so each `*.ncu-rep` artifact maps 1:1 to a single result record.

Rationale: This satisfies FR-012/FR-012a and the requirement that profiling artifacts are attributable per configuration/case, while minimizing profiling overhead and benchmark-run variability.

Alternatives considered: Profiling a full sweep in one `ncu` session was rejected because it makes attribution ambiguous and mixes runs.

## Decision 7: Correctness verification uses sampled dot-product checks (with full checks only for small shapes)

Decision: Perform correctness verification for every configuration/case by sampling a fixed number of output elements and computing their dot-products on CPU from the original host-side inputs; compare against the corresponding device outputs extracted to a small host buffer. For small shapes (configurable threshold), optionally compute a full CPU reference to tighten coverage.

Rationale: A full CPU GEMM reference is infeasible for large matrices (e.g., thousands), but sampled verification is independent, fast, and still detects common correctness failures (wrong transpose mapping, wrong leading dimensions, dtype/compute-type mistakes, etc.). This satisfies FR-011 while keeping the benchmark runtime practical.

Alternatives considered: Using another GPU GEMM implementation as “reference” was rejected as insufficiently independent and as adding a second performance-critical dependency.

## Decision 8: Structured export is normalized by Python (NVBench JSON is treated as an input, not the final contract)

Decision: Use NVBench `--json` output as the raw measurement log, then normalize it into a project-owned schema (`contracts/results.schema.json`) that adds required metadata (suite/case semantics, `flop_count`, ratios, verification status, profiling artifact references, environment info) and enforces per-row `flop_count` consistency rules for reporting.

Rationale: NVBench’s JSON format is stable for measurement capture but does not directly satisfy the full export contract required by FR-009/FR-010 (e.g., suite/case semantics, per-row ratio logic, verification and profiling linkage).

Alternatives considered: Using NVBench JSON “as-is” was rejected because the project needs stable, feature-specific fields and derived ratios that are not guaranteed to exist in the upstream schema.

## Decision 9: Reproducibility baseline is Pixi `cuda13`; environment metadata is always captured

Decision: All benchmark execution commands run under the Pixi environment `cuda13`, and every export includes CUDA toolkit/runtime, driver version, GPU model, and key NVBench stopping parameters (`min_time`, `max_noise`, `min_samples`, `stopping_criterion`).

Rationale: Reproducibility depends on controlling toolchain versions and capturing enough metadata to interpret differences across runs and machines.

Alternatives considered: Allowing “any local CUDA install” was rejected because it makes results difficult to compare and violates the repo’s build authority principle (Pixi as global workflow manager).
