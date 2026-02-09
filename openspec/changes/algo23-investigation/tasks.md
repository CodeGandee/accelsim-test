## 1. Repro Instrumentation (C++)

- [x] 1.1 Add CLI flags to `repro_algo23_int8_n1000` to run a single selected variant (e.g., `AB`, `ATB_view`, `ABT_view`) with configurable warmup/iters.
- [x] 1.2 Add an option to force a specific `(algo_id, tile_id, stages_id, splitK, etc.)` for the selected variant (and print the `AlgoCheck` failure reason when rejected).
- [x] 1.3 Add optional NVTX ranges around the selected variant’s timed GEMM region to enable unambiguous profiler filtering.

## 2. Kernel Discovery (nsys)

- [x] 2.1 Add a Python CLI under `scripts/` to run Nsight Systems kernel discovery for a specified repro command via `pixi run -e cuda13 ...`.
- [x] 2.2 Implement export of a machine-readable kernel listing from the `nsys` capture (kernel name + invocation index at minimum), suitable for selecting the correct kernel invocation for ncu.
- [x] 2.3 Write discovery artifacts into an output-rooted layout under `<out_dir>/profiles/...` (in testing: `tmp/<subdir>/profiles/...`) including `meta.json`, the raw `.qdrep`, the kernel listing, and a `README.md` generated with `mdutils` describing the exact commands used.

## 3. Nsight Compute Profiling (ncu)

- [x] 3.1 Add a Python CLI under `scripts/` to run `ncu` targeting either an NVTX range or a kernel-name regex and exporting a `.ncu-rep` into the output-rooted `profiles/` layout.
- [x] 3.2 Export lightweight summaries (CSV/text) from the `.ncu-rep` for quick comparison without opening the UI, and record the exact `ncu` invocation in `meta.json` / `README.md`.
- [x] 3.3 Add a convenience mode to collect comparable bundles for the fast path (`ABT_view` + `algo_id=23`) and a baseline path (e.g., `ABT_view` + `algo_id=64`) with clearly labeled output directories.

## 4. Validation & Reporting

- [x] 4.1 Run the workflow end-to-end for the B200 `N=1000` int8 case and store artifacts under `tmp/<subdir>/profiles/` (and optionally copy into a report directory as an explicit follow-up step).
- [x] 4.2 Update the stakeholder report manually to reference the profiling artifacts and to answer “what kernel is `algo_id=23` using, and why is it faster?” with evidence-backed claims.
- [x] 4.3 Add minimal tests (pytest) for path/layout generation and CLI argument parsing for the new profiling scripts.
