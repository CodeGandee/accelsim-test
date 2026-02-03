# Accel-Sim Dummy PTX Simulation — Stakeholder Report

- Feature: `003-accelsim-dummy-ptx-sim`
- Goal: provide a minimal “sanity-check” workflow to compile a tiny CUDA program and run it under Accel-Sim **PTX-mode** simulation.
- Primary user outcome: a developer can run one command and get a per-run artifact directory with logs + a clear `PASS`/`FAIL`.

## What was delivered

### 1) Reproducible CLI workflow (Python orchestrator)

Command (primary):

```bash
pixi run -e accelsim python -m accelsim_test.accelsim_dummy_ptx_sim run --run-id <RUN_ID>
```

Wrapper task (equivalent):

```bash
pixi run -e accelsim accelsim-dummy-ptx-sim -- --run-id <RUN_ID>
```

Key behavior:
- **PTX-mode only** (trace-driven / SASS modes are out of scope).
- Uses the **SM80_A100** simulator preset (`--config-preset sm80_a100`).
- Prefers Pixi `nvcc`, with system `nvcc` fallback (`--compiler auto|pixi|system`).
- Fails fast with actionable prerequisite guidance (still writes metadata on failure).

### 2) Minimal CUDA workload with correctness signal

The CUDA program runs a tiny matmul and prints:
- `PASS` on match vs CPU reference
- `FAIL` on mismatch (non-zero exit code)

### 3) Per-run artifacts (evidence trail)

All outputs are written under:

```text
tmp/accelsim_dummy_ptx_sim/<run_id>/
```

Contents:

```text
bin/matmul                 # compiled executable
ptx/matmul.ptx             # standalone PTX (for inspection)
run/gpgpusim.config        # copied preset config used for the run
run/matmul.sim.log         # captured simulator stdout/stderr
src/matmul.cu              # copied source for provenance
metadata.json              # machine-readable summary (commands, git info, hashes, status)
```

## How to validate (independent)

1) Build Accel-Sim once:

```bash
pixi install -e accelsim
pixi run -e accelsim accelsim-build
pixi run -e accelsim accelsim-smoke
```

2) Run one simulation:

```bash
pixi run -e accelsim python -m accelsim_test.accelsim_dummy_ptx_sim run --run-id 2026-02-03T00-00-00Z
```

3) Confirm expected signals:
- `tmp/accelsim_dummy_ptx_sim/2026-02-03T00-00-00Z/run/matmul.sim.log` contains an Accel-Sim banner line and `PASS`
- `tmp/accelsim_dummy_ptx_sim/2026-02-03T00-00-00Z/metadata.json` records `status=pass`

## Status and limitations

- Unit tests cover the Python orchestration utilities; end-to-end simulation is intentionally **not** run in unit tests (environment/time dependent).
- Only one simulator config preset is supported (`sm80_a100`), by design.

## References

- Quickstart: `specs/003-accelsim-dummy-ptx-sim/quickstart.md`
- CLI contract: `specs/003-accelsim-dummy-ptx-sim/contracts/cli.md`
- Task checklist: `specs/003-accelsim-dummy-ptx-sim/tasks.md`
