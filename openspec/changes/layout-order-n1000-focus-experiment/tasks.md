## 1. C++ repro support (order + symmetric inputs)

- [ ] 1.1 Add `--order {row|col}` support to the N=1000 int8 repro (set `CUBLASLT_MATRIX_LAYOUT_ORDER` for A/B/C layouts).
- [ ] 1.2 Add `--symmetric-inputs` option to generate symmetric A and symmetric B for “same-math” comparisons across `AB`/`ATB_view`/`ABT_view`.
- [ ] 1.3 (Optional) Add `--force-algo <id>` passthrough for controlled comparisons under both layout orders.
- [ ] 1.4 Print a machine-parsable summary per run (selected `algo_id` + key config fields) suitable for a Python orchestrator to consume.

## 2. Python orchestration (2×3 matrix, output to chosen dir)

- [ ] 2.1 Add a script to run the 6-case experiment matrix (row/col × AB/ATB_view/ABT_view) and write all artifacts under `--out-dir`.
- [ ] 2.2 Write a concise Markdown summary (via `mdutils`) and a JSON/CSV index of results (case → order/transpose/algo/timing).

## 3. Optional profiling hooks (kernel evidence)

- [ ] 3.1 Integrate optional Nsight Systems kernel discovery per case (reuse `scripts/cublaslt_kernel_discovery.py`).
- [ ] 3.2 Integrate optional Nsight Compute profiling per case (reuse `scripts/cublaslt_ncu_profile.py`) and store `.ncu-rep` + CSV exports under `<out-dir>/profiles/`.

## 4. Verification + docs

- [ ] 4.1 Add a small unit test for CLI validation / output layout (no GPU required).
- [ ] 4.2 Document how to run the experiment in `reports/transpose_matmul/...` (or under `context/`) and how to interpret “winner flips” across layout orders.
