# Contract: CLI Interfaces (Benchmark + Orchestrator)

This feature defines two command-line interfaces: a C++ NVBench executable (timing/profiling primitive) and a Python orchestrator (workflow glue, export normalization, reporting). The Python CLI is the primary user entry point.

## 1) C++ benchmark executable (NVBench)

Location (planned): built from `/data1/huangzhe/code/accelsim-test/cpp/` as an executable such as `gemm_transpose_bench` that embeds NVBench benchmarks.

Invocation shape:
- `gemm_transpose_bench [NVBench options...]`

Required NVBench options used by this project:
- `--json <path|stdout>`: write NVBench JSON output for timing runs.
- `--profile`: run each benchmark configuration once (used under `ncu` for profiling).
- `--benchmark <name>` / `--axis <spec>`: used by the orchestrator to run a single configuration/case by overriding each axis to a singleton.

NVBench axes (contracted names and types):
- `suite` (string): `square` | `nonsquare_atb` | `nonsquare_abt`
- `case` (string): `AB` | `ATB_view` | `ABT_view` | `ATB_copyA` | `ABT_copyB` (subset depends on suite)
- `shape` (string): logical GEMM dimensions encoded as `MxNxK` for `C[M,N] = A[M,K] @ B[K,N]` (used to avoid correlated `m/n/k` cartesian products in sweeps)
- `dtype` (string): canonical dtype pair key (e.g., `fp16_fp16_fp16`, `bf16_bf16_bf16`, `fp32_fp32_fp32`, `int8_int8_int32`) as defined by the Python config layer
- `math_mode` (string): e.g., `default`, `tf32` (only meaningful for fp32 where supported)

Behavioral constraints:
- Timed region must measure GEMM only; transpose materialization for `*_copy*` cases must be outside NVBench timing (FR-008a).
- H2D/D2H transfers and handle initialization must not occur inside the timed region (FR-007).

## 2) Python orchestrator CLI (primary)

Location (planned): `/data1/huangzhe/code/accelsim-test/src/accelsim_test/gemm_transpose_bench/`

Primary commands (contract):

### `timing`

Runs a timing sweep (no profiler) and emits a normalized results export.

Inputs:
- `--out-dir <path>` (required): output directory for raw NVBench JSON, normalized results, and generated report.
- `--suite <square|nonsquare_atb|nonsquare_abt|all>` (optional, default `all`)
- `--dtype <key|all>` (optional, default `all`)
- `--shape-set <name|all>` (optional): named shape sets per `/data1/huangzhe/code/accelsim-test/context/tasks/req-cuda-gemm-test.md`
- `--nvbench-args "<string>"` (optional): pass-through NVBench args (e.g., `--min-time 1 --max-noise 0.5 --devices 0`)

Outputs:
- `raw/nvbench_timing.json`: NVBench JSON output.
- `results.json`: normalized export (must validate against `contracts/results.schema.json`).

Exit code:
- `0` only if all correctness checks pass.
- Non-zero if any configuration/case fails verification (FR-011a) or if required outputs are missing.

### `profile`

Runs per-configuration/case Nsight Compute profiling (`ncu`) and links artifacts back into the normalized results export.

Inputs:
- `--out-dir <path>` (required): must match the timing run output directory.
- `--ncu-args "<string>"` (optional): pass-through Nsight Compute options (e.g., metrics set, kernel name filters).

Outputs:
- `profiles/<config_id>/<case>/profile.ncu-rep` (path pattern may vary but must be stable and attributable).
- `results.json`: updated to include per-record profiling artifact references (FR-012a).

Exit code:
- Non-zero if any expected profiling artifact cannot be produced (unless explicitly configured to allow partial profiling, which is not required by the spec).

### `report`

Generates the stakeholder-facing markdown report from `results.json`.

Inputs:
- `--out-dir <path>` (required)

Outputs:
- `report.md`: includes square-suite and non-square-suite tables with all executed configurations/cases by default (FR-010a) and required ratios (FR-010).

Exit code:
- Non-zero if `results.json` is invalid or missing required fields.
