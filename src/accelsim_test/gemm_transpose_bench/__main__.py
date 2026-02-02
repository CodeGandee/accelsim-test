from __future__ import annotations

import argparse
from pathlib import Path

from .profiling import profile_run
from .report import report_run
from .runner import timing_run


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
    timing.add_argument(
        "--nvbench-args",
        default="",
        help="Extra NVBench CLI args (e.g. \"--min-time 1 --max-noise 0.5 --devices 0\").",
    )

    profile = sub.add_parser("profile", help="Run Nsight Compute profiling per configuration/case.")
    profile.add_argument("--out-dir", type=_abs_path, required=True)
    profile.add_argument("--ncu-args", default="", help="Extra Nsight Compute args.")
    profile.add_argument("--nvbench-args", default="--profile", help="NVBench args for profiling runs.")

    report = sub.add_parser("report", help="Generate stakeholder report from results.json.")
    report.add_argument("--out-dir", type=_abs_path, required=True)

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
            nvbench_args=ns.nvbench_args,
        )
    if ns.cmd == "profile":
        return profile_run(out_dir=ns.out_dir, ncu_args=ns.ncu_args, nvbench_args=ns.nvbench_args)
    if ns.cmd == "report":
        return report_run(out_dir=ns.out_dir)

    raise AssertionError(f"Unhandled cmd: {ns.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
