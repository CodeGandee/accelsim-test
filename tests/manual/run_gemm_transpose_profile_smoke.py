from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from accelsim_test.gemm_transpose_bench.profiling import profile_run
from accelsim_test.gemm_transpose_bench.runner import timing_run


def _abs_path(p: str) -> Path:
    return Path(p).expanduser().resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manual smoke: run one gemm_transpose_bench config under ncu.")
    parser.add_argument("--out-dir", type=_abs_path, required=True)
    parser.add_argument("--ncu-args", default="", help="Extra Nsight Compute args.")
    ns = parser.parse_args(argv)

    if shutil.which("ncu") is None:
        print("ncu not found on PATH; skipping profiling smoke.")
        return 0

    ns.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = ns.out_dir / "results.json"
    if not results_path.exists():
        rc = timing_run(
            out_dir=ns.out_dir,
            suite="square",
            dtype="fp16_fp16_fp16",
            shape_set="smoke_square",
            nvbench_args="--min-time 0.01 --min-samples 1 --max-noise 100",
        )
        if rc != 0:
            return rc

    return profile_run(out_dir=ns.out_dir, ncu_args=ns.ncu_args, nvbench_args="--profile")


if __name__ == "__main__":
    raise SystemExit(main())
