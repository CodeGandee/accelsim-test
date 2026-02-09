## Why

On B200 (CUDA 13.0), the `N=1000` int8 square case shows a large performance discontinuity: `ABT_view` selects cuBLASLt `algo_id=23` and runs ~2–3x faster than `AB` / `ATB_view` (which select `algo_id=64`). We need to understand the exact kernel path and runtime behavior behind `algo_id=23` to explain the result to stakeholders and to make informed decisions about transpose-as-view vs transpose-as-copy strategies.

## What Changes

- Add a reproducible way to identify the **exact GPU kernel(s)** used by cuBLASLt for `algo_id=23` in the `ABT_view` case (including launch parameters and any auxiliary kernels, if present).
- Add a reproducible way to collect **Nsight Compute (ncu)** profiling artifacts for the `algo_id=23` fast path and a comparable baseline (e.g., `algo_id=64`), suitable for side-by-side analysis.
- Standardize where program-generated profiling artifacts are stored under a user-specified output directory so stakeholder-facing markdown can reference them without being auto-generated (in testing, outputs will go to `tmp/<subdir>/`).

## Capabilities

### New Capabilities

- `cublaslt-kernel-discovery`: Given a concrete cuBLASLt matmul repro (including transpose flags), capture kernel names and launch configuration details (e.g., via Nsight Systems / CUDA trace) and save them as artifacts under a user-specified output directory.
- `cublaslt-ncu-profiling`: Given a kernel name (or NVTX range) from kernel discovery, collect ncu reports for the relevant kernel invocation(s) and export the profiling bundle (e.g., `.ncu-rep` plus CSV/text summaries) into the output directory.

### Modified Capabilities

<!-- None -->

## Impact

- Code: likely touches `cpp/` repro programs (to isolate variants and optionally add NVTX ranges) and adds repository scripts/CLIs for repeatable profiling runs.
- Tooling: requires Pixi env `cuda13` and host tooling `ncu` / `nsys` availability on PATH.
- Reporting: profiling artifacts will be stored under a user-specified output directory (e.g., `tmp/<subdir>/profiles/...`) and can be manually copied into a report directory for stakeholder-facing writeups.
