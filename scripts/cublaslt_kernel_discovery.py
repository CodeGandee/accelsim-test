"""
CLI: Nsight Systems kernel discovery for cuBLASLt repro programs.

This script wraps a reproduction command with `nsys profile`, exports a CUDA GPU
trace table, and generates a compact kernel listing that includes invocation
indices and grid/block dimensions.

Run via Pixi (recommended):
    pixi run python scripts/cublaslt_kernel_discovery.py --out-dir tmp/foo --case-id n1000_int8_abt23 -- -- ./cpp/build/Release/repro_algo23_int8_n1000 --variant ABT_view --force-algo 23 --nvtx --iters 1 --warmup 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from accelsim_test.profiling.cublaslt_profiling import run_nsys_kernel_discovery


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run nsys kernel discovery and export kernel listing.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory root (artifacts under out_dir/profiles/...).")
    parser.add_argument("--case-id", type=str, required=True, help="Case identifier (used under out_dir/profiles/<case_id>/).")
    parser.add_argument("--pixi-env", type=str, default="cuda13", help="Pixi environment to run the repro under (default: cuda13).")
    parser.add_argument("--no-nvtx-prefix", action="store_true", help="Do not prefix kernel names with NVTX range names in exported tables.")
    parser.add_argument("repro_cmd", nargs=argparse.REMAINDER, help="Repro command (argv list). Use `--` before the command.")
    return parser.parse_args()


def main() -> int:
    """Entry point."""
    args = _parse_args()
    repro_cmd = [c for c in args.repro_cmd if c != "--"]
    if not repro_cmd:
        raise SystemExit("Missing repro command. Provide it after `--`.")

    run_nsys_kernel_discovery(
        out_dir=args.out_dir,
        case_id=args.case_id,
        pixi_env=args.pixi_env,
        repro_cmd=repro_cmd,
        nvtx_prefix_names=not args.no_nvtx_prefix,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
