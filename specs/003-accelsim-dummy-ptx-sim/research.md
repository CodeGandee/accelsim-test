# Research: Accel-Sim Dummy CUDA PTX Simulation

This document records key technical decisions for implementing `/data1/huangzhe/code/accelsim-test/specs/003-accelsim-dummy-ptx-sim/spec.md` and resolves planning questions by committing to specific, actionable choices.

## Decision 1: PTX-mode only (no trace-driven/SASS)

Decision: Implement only PTX-mode simulation for this feature; do not add trace-driven (SASS/NVBit) flows.

Rationale: This keeps the feature a fast “sanity-check” workflow and aligns with the clarified scope in the spec (`FR-006`).

Alternatives considered: Adding trace-driven simulation was rejected because it introduces additional prerequisites (trace capture tooling, trace bundles) and shifts the feature away from “dummy PTX simulation” toward a larger validation pipeline.

## Decision 2: Use the SM80_A100 simulator preset

Decision: Use the Ampere/A100-like preset config shipped in the submodule at:
- `/data1/huangzhe/code/accelsim-test/extern/tracked/accel-sim-framework/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM80_A100/gpgpusim.config`

Rationale: The spec requires A100-like preset support and explicitly does not require multiple presets. This config sets compute capability 8.0 and `-gpgpu_ptx_force_max_capability 80`.

Alternatives considered: Supporting multiple presets (e.g., V100 and A100) was rejected by clarification; supporting an arbitrary user-supplied config path was deferred to future scope.

## Decision 3: Compile embedded PTX targeting compute_80

Decision: Compile the dummy CUDA program with embedded PTX compatible with the simulator preset (compute capability 8.0), i.e. use `-gencode arch=compute_80,code=compute_80` and also emit a standalone `.ptx` for inspection.

Rationale: GPGPU-Sim’s run instructions require aligning the program’s embedded PTX capability with the config’s forced capability; otherwise kernels may fail to load or be mis-modeled. The preset uses capability 80, so `compute_80` is the correct target.

Alternatives considered: Compiling for `sm_80` SASS was rejected because this feature is PTX-mode (the simulator extracts and simulates PTX); compiling for another capability would mismatch the chosen preset.

## Decision 4: Run by executing the CUDA program under the simulator environment

Decision: Run the dummy program by:
1) building Accel-Sim once (Pixi task), then
2) sourcing the simulator environment (`setup_environment.sh`) and executing the CUDA program from a working directory that contains the copied `gpgpusim.config`.

Rationale: This is the simplest and most direct workflow described in upstream GPGPU-Sim documentation (“copy config to app working dir; source setup_environment; run the executable”) and avoids modifying or depending on Accel-Sim’s job launcher YAMLs for this MVP.

Alternatives considered: Using Accel-Sim’s `util/job_launching/run_simulations.py` was rejected for MVP because it typically requires editing/creating launcher YAML definitions inside the submodule, which is undesirable for a minimal repo-local sanity check.

## Decision 5: Prefer Pixi-provided `nvcc`, fall back to system `nvcc`

Decision: The workflow selects the CUDA compiler as:
1) preferred: `pixi run -e accelsim nvcc ...`
2) fallback: `nvcc ...` from the host environment

Rationale: Pixi is the repository’s workflow authority; using the Pixi-provided toolchain improves reproducibility. A system fallback reduces friction on machines where the Pixi CUDA compiler is unavailable or configured differently.

Alternatives considered: “Pixi-only” was rejected to avoid blocking developer machines; “system-only” was rejected because it undermines reproducibility and conflicts with the repo’s build authority principle.

## Decision 6: Deterministic correctness check with CPU reference + `PASS`/`FAIL`

Decision: The minimal program computes a small matmul on GPU, computes a CPU reference for the same inputs, and prints an unambiguous `PASS` or `FAIL` based on a fixed tolerance and deterministic inputs.

Rationale: A completion-only signal is insufficient; correctness makes the workflow trustworthy and helps separate simulator/toolchain errors from logic errors.

Alternatives considered: A checksum-only output was rejected because it is harder to interpret and does not encode a clear pass/fail outcome.

## Decision 7: Artifacts stored only under `tmp/<run_id>/`

Decision: Store all run artifacts under a per-run directory:
- `/data1/huangzhe/code/accelsim-test/tmp/accelsim_dummy_ptx_sim/<run_id>/`

Rationale: This matches the clarified requirement (`FR-009`) and keeps the workflow self-contained without requiring write access elsewhere.

Alternatives considered: Copying curated outputs to `reports/` was rejected for this feature; it can be a future enhancement.

## Decision 8: Testing strategy is lightweight by default

Decision: Provide:
- unit tests for Python orchestration utilities (path construction, command rendering, artifact discovery),
- a manual smoke path for end-to-end simulation (may be slow / environment-dependent).

Rationale: End-to-end simulation is heavy and depends on external toolchains; unit tests still provide guardrails for regressions in the orchestrator.

Alternatives considered: A mandatory integration test that runs the simulator in CI was rejected due to runtime and environment constraints.
