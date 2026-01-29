# How to Compare Accel-Sim Results with a Real GPU Run

## HEADER
- **Purpose**: Provide a repeatable procedure to collect “ground truth” GPU measurements and compare them against Accel-Sim output.
- **Status**: Active
- **Date**: 2026-01-29
- **Dependencies**: `extern/tracked/accel-sim-framework`, a working GPU + `nvidia-smi`, and a runnable workload (e.g., `tmp/accelsim-matmul/`)
- **Target**: Developers (including future maintainers) and AI assistants

## Key idea

`nvidia-smi` is good for device identity and coarse utilization/power/clock logging, but it does not directly give accurate per-kernel latency. For meaningful comparisons, measure time inside the workload (CUDA events) and optionally collect profiler counters; then align the simulator mode/config to the same GPU and workload.

## 1) Capture real GPU “baseline” info

Run before each experiment and save logs under your run folder (example: `tmp/accelsim-matmul/run/`):

```bash
nvidia-smi --query-gpu=index,name,uuid,driver_version,cuda_version,pstate,power.limit --format=csv | tee tmp/accelsim-matmul/run/nvidia-smi.info.csv
```

Optional: log dynamic telemetry during the run (separate terminal):

```bash
nvidia-smi dmon -s pucvmt -d 1 | tee tmp/accelsim-matmul/run/nvidia-smi.dmon.log
```

## 2) Measure real runtime correctly (recommended)

Add CUDA event timing in your `.cu` program (around the kernel, and optionally end-to-end including memcpy) and print a compact line like:
- `kernel_ms=<...> end_to_end_ms=<...> M=<...> N=<...> K=<...>`

Run multiple iterations (warmup + N measured) and report median/min to reduce noise.

## 3) Reduce run-to-run variance (optional, but helps)

- Avoid contention (exclusive node/GPU if possible).
- Prefer fixed problem sizes and pinned inputs.
- If you have permissions, consider stabilizing clocks/power (persistence mode / application clocks / power cap) and record what you changed. If you cannot change clocks, at least log `pstate`, `clocks.sm`, `clocks.mem`, and `power.draw` via `nvidia-smi` during the run.

## 4) Align Accel-Sim configuration and mode

- Pick a config matching your GPU generation (e.g., Volta V100 ~ `SM7_QV100`, Ampere A100 ~ `SM80_A100`): see `extern/tracked/accel-sim-framework/gpu-simulator/gpgpu-sim/configs/tested-cfgs/`.
- Use the right mode for the question:
  - **SASS trace-driven mode** is preferred for accuracy comparisons (simulate the same dynamic instruction stream captured on hardware).
  - **PTX mode** is useful for quick sanity/trends but may differ from hardware due to JIT/ISA/scheduling effects.

## 5) Use Nsight Compute (ncu) to validate “reasonableness” (recommended)

If `ncu` is installed (for example via `pixi global` as `nsight-compute`), use it to profile the real GPU run and compare the “shape” of performance against Accel-Sim.

Quick checks:

```bash
command -v ncu && ncu --version
```

Run `ncu` and save a report (example workload path):

```bash
ncu --target-processes all --kernel-name-base demangled -o tmp/accelsim-matmul/run/ncu ./tmp/accelsim-matmul/bin/matmul
```

Collect useful sections (high signal for bottleneck triage):

```bash
ncu --target-processes all --kernel-name-base demangled --set speed-of-light --section LaunchStats --section Occupancy --section MemoryWorkloadAnalysis --section InstructionStats -o tmp/accelsim-matmul/run/ncu ./tmp/accelsim-matmul/bin/matmul
```

If there are many kernels, narrow the capture:
- Use `--kernel-name <regex_or_exact>` and/or `--kernel-id`, and/or `--launch-skip/--launch-count`.

What to compare vs Accel-Sim (aim for consistency, not exact equality):
- Occupancy/parallelism: `ncu` achieved occupancy / active warps trends vs what your chosen `gpgpusim.config` implies.
- Bottleneck class: if `ncu` indicates memory-bound vs compute-bound, Accel-Sim should show corresponding stall/IPC and memory pressure patterns.
- Bandwidth: `ncu` achieved DRAM throughput vs Accel-Sim’s reported memory traffic/bandwidth (order-of-magnitude and scaling across input sizes).
- Instruction mix: `ncu` instruction breakdown vs Accel-Sim’s instruction mix trends (especially for simple, non-library kernels).

## 5) Compare like-for-like outputs

At minimum compare:
- Real: `kernel_ms` from CUDA events (per kernel, per input size).
- Sim: Accel-Sim/GPGPU-Sim output stats (cycles, IPC, SIM_TIME) collected from the run logs (or `util/job_launching/get_stats.py` when using the launcher).

Recommended practice:
- Sweep several sizes (e.g., multiple `(M,N,K)`), then compare trends and ratios (ordering and scaling) rather than a single absolute point.
