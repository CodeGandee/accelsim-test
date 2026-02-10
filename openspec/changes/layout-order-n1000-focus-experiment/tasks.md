## 1. C++ repro support (per-matrix order + symmetric inputs)

- [x] 1.1 Add per-matrix layout-order flags (e.g., `--order-a {row|col}`, `--order-b {row|col}`, `--order-c {row|col}`) to the N=1000 int8 repro (set `CUBLASLT_MATRIX_LAYOUT_ORDER` for each layout and allow mixed row/col cases).
- [x] 1.2 Add `--symmetric-inputs` option to generate symmetric A and symmetric B for “same-math” comparisons across `AB`/`ATB_view`/`ABT_view`.
- [x] 1.3 (Optional) Add `--force-algo <id>` passthrough for controlled comparisons under both layout orders.
- [x] 1.4 Print a machine-parsable summary per run (selected `algo_id` + key config fields) suitable for a Python orchestrator to consume.

## 2. Python orchestration (4×3 matrix + limited output-order sweep, output to chosen dir)

- [x] 2.1 Add a script to run the 12-case experiment matrix (A/B order pairs × AB/ATB_view/ABT_view), plus a limited output-order sweep for `order_a=row, order_b=row` (`order_c=row` and `order_c=col`), and write all artifacts under `--out-dir`.
- [x] 2.2 Write a concise Markdown summary (via `mdutils`) and a JSON/CSV index of results (case → order_a/order_b/order_c/transpose/algo/timing).

## 3. Optional profiling hooks (kernel evidence)

- [x] 3.1 Integrate optional Nsight Systems kernel discovery per case (reuse `scripts/cublaslt_kernel_discovery.py`).
- [x] 3.2 Integrate optional Nsight Compute profiling per case (reuse `scripts/cublaslt_ncu_profile.py`) and store `.ncu-rep` + CSV exports under `<out-dir>/profiles/`.

## 4. Verification + docs

- [x] 4.1 Add a small unit test for CLI validation / output layout (no GPU required).
- [x] 4.2 Document how to run the experiment in `reports/transpose_matmul/...` (or under `context/`) and how to interpret “winner flips” across A/B layout-order combinations and the limited `order_c` sensitivity check.
