from __future__ import annotations

import argparse
from pathlib import Path

from .algo_map import algo_map_run
from .profiling import profile_run
from .report import report_run
from .runner import timing_run
from .sweep import sweep_run


def _abs_path(p: str) -> Path:
    return Path(p).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="accelsim_test.gemm_transpose_bench",
        description="GEMM transpose benchmark orchestrator (NVBench + cuBLASLt).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    timing = sub.add_parser("timing", help="Run NVBench timing sweep (no profiler).")
    timing.add_argument("--out-dir", type=_abs_path, required=True)
    timing.add_argument("--suite", default="all", choices=["square", "nonsquare_atb", "nonsquare_abt", "all"])
    timing.add_argument("--dtype", default="all", help="Dtype key (or 'all').")
    timing.add_argument("--shape-set", default="all", help="Named shape set (or 'all').")
    timing.add_argument("--algo-map", type=_abs_path, default=None, help="Path to cuBLASLt algo-map JSON (pins algorithms).")
    timing.add_argument(
        "--nvbench-args",
        default="",
        help="Extra NVBench CLI args (e.g. \"--min-time 1 --max-noise 0.5 --devices 0\").",
    )

    profile = sub.add_parser("profile", help="Run Nsight Compute profiling per configuration/case.")
    profile.add_argument("--out-dir", type=_abs_path, required=True)
    profile.add_argument("--ncu-args", default="", help="Extra Nsight Compute args.")
    profile.add_argument("--nvbench-args", default="--profile", help="NVBench args for profiling runs.")

    report = sub.add_parser("report", help="Generate Markdown reports from results.json (no benchmark run).")
    report.add_argument("--out-dir", type=_abs_path, required=True)

    sweep = sub.add_parser("sweep", help="Run full timing sweep (experiments only).")
    sweep.add_argument("--out-dir", type=_abs_path, required=True)
    sweep.add_argument("--shape-set", default="full_sweep_required", help="Named shape set (default: full_sweep_required).")
    sweep.add_argument("--dtype", default="all", help="Dtype key (or 'all').")
    sweep.add_argument("--algo-map", type=_abs_path, default=None, help="Path to cuBLASLt algo-map JSON (pins algorithms).")
    sweep.add_argument(
        "--nvbench-args",
        default="--stopping-criterion stdrel --min-time 0.5 --max-noise 0.3 --min-samples 20 --devices 0",
        help="NVBench args for timing runs.",
    )

    algo = sub.add_parser("algo-map", help="Extract cuBLASLt algorithm configs from an existing results.json.")
    algo.add_argument("--results", type=_abs_path, required=True)
    algo.add_argument("--out", type=_abs_path, required=True, help="Output JSON path for algo-map.")
    algo.add_argument("--suite", default="all", choices=["square", "nonsquare_atb", "nonsquare_abt", "all"])

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)

    if ns.cmd == "timing":
        return timing_run(
            out_dir=ns.out_dir,
            suite=ns.suite,
            dtype=ns.dtype,
            shape_set=ns.shape_set,
            algo_map=ns.algo_map,
            nvbench_args=ns.nvbench_args,
        )
    if ns.cmd == "profile":
        return profile_run(out_dir=ns.out_dir, ncu_args=ns.ncu_args, nvbench_args=ns.nvbench_args)
    if ns.cmd == "report":
        return report_run(out_dir=ns.out_dir)
    if ns.cmd == "sweep":
        return sweep_run(out_dir=ns.out_dir, shape_set=ns.shape_set, dtype=ns.dtype, algo_map=ns.algo_map, nvbench_args=ns.nvbench_args)
    if ns.cmd == "algo-map":
        return algo_map_run(results_path=ns.results, out_path=ns.out, suite=ns.suite)

    raise AssertionError(f"Unhandled cmd: {ns.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
