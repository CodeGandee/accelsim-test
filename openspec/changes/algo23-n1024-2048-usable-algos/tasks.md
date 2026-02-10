## 1. Experiment CLI and Case Matrix

- [ ] 1.1 Add/extend experiment entrypoint to run only `AB` and `ABT_view` for int8 row-major square cases.
- [ ] 1.2 Add fixed case matrix for `N=1000`, `N=1024`, and `N=2048` with configurable warmup/iters/workspace policy.
- [ ] 1.3 Record environment and invocation metadata in run artifacts for reproducibility.

## 2. Candidate Enumeration and Usability Classification

- [ ] 2.1 Implement per-case candidate enumeration from cuBLASLt discovery APIs and normalize candidate fields (`algo_id`, tile, stages, splitK, and related config).
- [ ] 2.2 Implement per-candidate `cublasLtMatmulAlgoCheck` evaluation under fixed workspace and record usable/non-usable with status details.
- [ ] 2.3 Capture heuristic-selected algorithm per case in the same output schema for side-by-side comparison.

## 3. Candidate Timing and Result Export

- [ ] 3.1 Benchmark all usable candidates per case with consistent timing configuration and collect per-candidate stats.
- [ ] 3.2 Export raw candidate results (including usability + timing) to JSON and CSV under a dedicated report directory.
- [ ] 3.3 Generate markdown tables comparing `AB` vs `ABT_view` for each `N`, including `algo_id=23` availability and ranking/performance notes.

## 4. Validation and Integration

- [ ] 4.1 Add or update tests for case-matrix generation and result/schema integrity where applicable.
- [ ] 4.2 Run targeted validation commands and confirm artifacts are generated in expected paths.
- [ ] 4.3 Integrate experiment summary into the stakeholder report section discussing `algo_id=23` behavior across sizes.
