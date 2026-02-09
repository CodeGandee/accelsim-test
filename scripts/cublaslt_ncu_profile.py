"""
CLI: Nsight Compute profiling for cuBLASLt repro programs.

This script profiles a reproduction command with `ncu`, exports the `.ncu-rep`
report, and writes CSV summaries for quick comparison.

Run via Pixi (recommended):
    pixi run python scripts/cublaslt_ncu_profile.py --out-dir tmp/foo --case-id n1000_int8_abt23 --scope profiler -- -- ./cpp/build/Release/repro_algo23_int8_n1000 --variant ABT_view --force-algo 23 --cuda-profiler-gating --iters 1 --warmup 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from accelsim_test.profiling.cublaslt_profiling import run_ncu_profile


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run ncu profiling and export CSV summaries.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory root (artifacts under out_dir/profiles/...).")
    parser.add_argument("--case-id", type=str, default=None, help="Case identifier (used under out_dir/profiles/<case_id>/).")
    parser.add_argument("--case-prefix", type=str, default=None, help="Case prefix used by --compare-abt23-vs-abt64 (writes <prefix>_algo23 and <prefix>_algo64).")
    parser.add_argument("--pixi-env", type=str, default="cuda13", help="Pixi environment to run the repro under (default: cuda13).")
    parser.add_argument("--set", dest="set_name", type=str, default="full", help="Nsight Compute section set (default: full).")
    parser.add_argument("--scope", type=str, default="profiler", choices=["profiler", "nvtx", "kernel"], help="Profiling scope mode.")
    parser.add_argument("--nvtx-include", type=str, default=None, help="NVTX include filter (required for --scope nvtx).")
    parser.add_argument("--kernel-regex", type=str, default=None, help="Kernel name regex (required for --scope kernel).")
    parser.add_argument("--launch-count", type=int, default=1, help="Number of matching launches to profile (scope=kernel).")
    parser.add_argument("--launch-skip", type=int, default=0, help="Matching launches to skip before profiling (scope=kernel).")
    parser.add_argument("--compare-abt23-vs-abt64", action="store_true", help="Profile ABT_view forced algo 23 vs forced algo 64 into two case directories.")
    parser.add_argument("--iters", type=int, default=1, help="Repro timed iterations (compare mode only; default: 1).")
    parser.add_argument("--warmup", type=int, default=0, help="Repro warmup iterations (compare mode only; default: 0).")
    parser.add_argument("repro_cmd", nargs=argparse.REMAINDER, help="Repro command (argv list). Use `--` before the command.")
    return parser.parse_args()


def main() -> int:
    """Entry point."""
    args = _parse_args()
    repro_cmd = [c for c in args.repro_cmd if c != "--"]
    if not repro_cmd:
        raise SystemExit("Missing repro command. Provide it after `--`.")

    if args.compare_abt23_vs_abt64:
        if args.case_prefix is None:
            raise SystemExit("--case-prefix is required when --compare-abt23-vs-abt64 is set.")
        if args.scope != "profiler":
            raise SystemExit("Compare mode currently requires --scope profiler (uses cudaProfilerStart/Stop gating).")

        base_cmd = repro_cmd
        forbidden = {"--variant", "--force-algo", "--tile-id", "--stages-id", "--splitk", "--iters", "--warmup"}
        if any(a in forbidden for a in base_cmd):
            raise SystemExit(
                "Compare mode expects a base repro command without variant/force/iters/warmup flags; "
                "the script will supply those flags."
            )

        def _profile_one(*, algo_id: int, tile_id: int, stages_id: int) -> None:
            case_id = f"{args.case_prefix}_algo{algo_id}"
            cmd = [
                *base_cmd,
                "--variant",
                "ABT_view",
                "--force-algo",
                str(algo_id),
                "--tile-id",
                str(tile_id),
                "--stages-id",
                str(stages_id),
                "--splitk",
                "1",
                "--cuda-profiler-gating",
                "--nvtx",
                "--iters",
                str(max(1, args.iters)),
                "--warmup",
                str(max(0, args.warmup)),
            ]
            run_ncu_profile(
                out_dir=args.out_dir,
                case_id=case_id,
                pixi_env=args.pixi_env,
                repro_cmd=cmd,
                set_name=args.set_name,
                scope=args.scope,
                nvtx_include=args.nvtx_include,
                kernel_regex=args.kernel_regex,
                launch_count=max(1, args.launch_count),
                launch_skip=max(0, args.launch_skip),
            )

        _profile_one(algo_id=23, tile_id=18, stages_id=21)
        _profile_one(algo_id=64, tile_id=20, stages_id=8)
    else:
        if args.case_id is None:
            raise SystemExit("--case-id is required unless --compare-abt23-vs-abt64 is set.")
        run_ncu_profile(
            out_dir=args.out_dir,
            case_id=args.case_id,
            pixi_env=args.pixi_env,
            repro_cmd=repro_cmd,
            set_name=args.set_name,
            scope=args.scope,
            nvtx_include=args.nvtx_include,
            kernel_regex=args.kernel_regex,
            launch_count=max(1, args.launch_count),
            launch_skip=max(0, args.launch_skip),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
