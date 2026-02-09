## Context

We have a reproducible performance discontinuity on B200 (CUDA 13.0) for the square int8 case `N=1000`:

- `AB` / `ATB_view` select cuBLASLt `algo_id=64` and are slower.
- `ABT_view` selects cuBLASLt `algo_id=23` (tile=18, stages=21) and is ~2–3x faster.
- Forcing `algo_id=23` into `AB` / `ATB_view` fails `cublasLtMatmulAlgoCheck`, suggesting the fast path is only eligible under the `trans_b=T` configuration and associated layout/stride constraints.

The report directory `reports/transpose_matmul/gemm_transpose_full_sweep_20260209_025629/` already contains:

- Sweep results (`results.json`, `report.md`, `all_timings.md`)
- A standalone repro output (`repro_algo23_int8_n1000.txt`)
- A cuBLASLt caps dump (`cublaslt_algo_caps.json`)

The missing piece is attribution: which GPU kernel(s) actually execute for `algo_id=23`, what the launch configuration is, and what runtime behavior (TC utilization, memory stalls, occupancy, etc.) explains the speedup.

Key constraints / preferences:

- Stakeholder reports are not meant to be program-generated. Profiling artifacts should be stored in the report directory and referenced from stakeholder markdown.
- Profiling data collection should be standalone: tools should write to a user-specified output directory (in testing: `tmp/<subdir>/`), and any copying into `reports/...` should be an explicit, manual step.
- The repo already relies on `pixi` environments (notably `cuda13`), and profiling tools (`nsys`, `ncu`) are expected to be available on PATH.

## Goals / Non-Goals

**Goals:**

- Make kernel identification and profiling for the `algo_id=23` fast path **repeatable**.
- Ensure we can unambiguously target the correct kernel invocation(s) for profiling (avoid “profile the wrong launch”).
- Store all program-generated profiling artifacts in a consistent output-rooted layout (e.g., under `<out_dir>/profiles/`) with enough metadata to reproduce the exact run.
- Enable a side-by-side comparison between a “fast” path (`ABT_view` + `algo_id=23`) and a baseline path (e.g., `ABT_view` + forced `algo_id=64`, or `AB` + `algo_id=64`).

**Non-Goals:**

- Building a general-purpose cuBLASLt benchmarking framework beyond what is needed for this investigation.
- Guaranteeing that `algo_id` maps to a stable, public kernel symbol across CUDA versions (cuBLASLt is free to change internals).
- Automatically generating a narrative stakeholder report from the profiling data.

## Decisions

### Decision: Use a two-stage workflow (kernel discovery → ncu profiling)

**Choice:** Treat kernel discovery (finding exact kernel name(s) and the right invocation) as a first-class step, then run ncu targeted to that kernel invocation.

**Rationale:** cuBLASLt `algo_id` is not a kernel name; the only reliable way to learn the executed kernel(s) is to trace launches. Separating discovery from profiling reduces iteration time and avoids collecting huge ncu reports while still unsure what to profile.

**Alternatives considered:**

- Profile directly with broad `ncu` filters (risk: capturing unrelated kernels; expensive).
- Rely on cuBLASLt logging (not consistently available; may omit kernel-level details).

### Decision: Prefer NVTX ranges for unambiguous filtering (if needed)

**Choice:** Modify the repro binary to optionally emit NVTX ranges around each variant’s GEMM loop (e.g., `AB`, `ATB_view`, `ABT_view`, and forced-algo variants), and run `ncu` with `--nvtx --nvtx-include ...` when possible.

**Rationale:** Kernel names can be long, unstable across versions, and/or renamed by tooling. NVTX-based selection makes it easier to profile exactly one logical region even if cuBLASLt launches helper kernels (reformat/epilogue/etc.).

**Alternatives considered:**

- Filtering purely by kernel name regex (`ncu -k regex:...`): workable but more fragile, especially if multiple similar kernels match.

### Decision: Add “single-variant” execution for profiling

**Choice:** Extend the repro to run exactly one selected variant per invocation (e.g., `--variant ABT_view --force-algo 23`) and to reduce iteration count for profiling runs.

**Rationale:** Profilers work best when the binary’s control flow is simple: one matmul variant, minimal warmup, minimal unrelated kernel launches. This reduces the need for `--launch-skip` gymnastics and avoids profiling the wrong iteration.

**Alternatives considered:**

- Keep current multi-variant run and use invocation indices: workable but brittle when warmup/iters change.

### Decision: Standardize profiling artifact layout under the output directory

**Choice:** Write profiling outputs under `<out_dir>/profiles/<case_id>/...` where `<case_id>` encodes the problem and variant, e.g.:

- `profiles/n1000_int8_abt_view_algo23/`
- `profiles/n1000_int8_abt_view_algo64/`

Each directory should include:

- `README.md` (brief “what this is”, and exact commands used)
- raw profiler outputs (`.qdrep`, `.ncu-rep`)
- exported summaries (CSV/text) for quick diffing without opening GUIs
- a small `meta.json` capturing GPU/driver/toolkit versions and relevant knobs (variant, algo_id, tile/stages, iters/warmup)

**Rationale:** This keeps stakeholder markdown manual while making it easy to reference artifacts and reproduce captures.

## Risks / Trade-offs

- **[Kernel-name instability across CUDA versions]** → Use NVTX ranges for targeting; store raw `.qdrep`/`.ncu-rep` artifacts so analysis remains grounded in the captured run.
- **[Profiling overhead changes kernel behavior]** → Keep profiling to minimal iterations; use consistent clock/control settings where possible; compare relative metrics (e.g., achieved occupancy, tensor utilization) rather than absolute time only.
- **[Multiple kernels per logical matmul]** (e.g., internal pack/reformat) → Prefer NVTX filtering and record complete kernel listings from discovery; profile both the main GEMM kernel and any significant auxiliary kernels if they differ between fast/baseline paths.
- **[Tool availability / permissions]** → Detect `nsys`/`ncu` availability and fail with actionable instructions; keep profiling scripts optional and non-blocking for environments without the tools.
