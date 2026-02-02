from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from accelsim_test.gemm_transpose_bench.export import validate_results_schema
from accelsim_test.gemm_transpose_bench.runner import find_benchmark_executable, timing_run


def _has_cuda_gpu() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.STDOUT)
    except Exception:
        return False
    return bool(out.decode(errors="replace").strip())


@pytest.mark.integration
def test_nonsquare_smoke(tmp_path: Path) -> None:
    if os.environ.get("PIXI_ENVIRONMENT_NAME") != "cuda13":
        pytest.skip("requires pixi cuda13 environment")
    if not _has_cuda_gpu():
        pytest.skip("requires CUDA GPU")
    try:
        find_benchmark_executable()
    except FileNotFoundError:
        pytest.skip("requires built gemm_transpose_bench executable")

    out_dir = tmp_path / "gemm_transpose_out"
    rc = timing_run(
        out_dir=out_dir,
        suite="nonsquare_atb",
        dtype="fp16_fp16_fp16",
        shape_set="smoke_nonsquare",
        nvbench_args="--min-time 0.01 --min-samples 1 --max-noise 100",
    )
    assert rc == 0

    results_path = out_dir / "results.json"
    results = json.loads(results_path.read_text())
    validate_results_schema(results)

    atb_records = [r for r in results.get("records", []) if r.get("suite") == "nonsquare_atb"]
    assert {r["case"] for r in atb_records} == {"ATB_view", "ATB_copyA"}
