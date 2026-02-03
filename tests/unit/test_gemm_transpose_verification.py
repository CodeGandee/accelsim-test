from __future__ import annotations

from pathlib import Path

import pytest

from accelsim_test.gemm_transpose_bench.export import normalize_nvbench_results


def _make_nvbench_state(*, verify_pass: int) -> dict:
    return {
        "is_skipped": False,
        "device": 0,
        "axis_values": [
            {"name": "suite", "type": "string", "value": "square"},
            {"name": "case", "type": "string", "value": "AB"},
            {"name": "dtype", "type": "string", "value": "fp16_fp16_fp16"},
            {"name": "math_mode", "type": "string", "value": "default"},
            {"name": "shape", "type": "string", "value": "512x512x512"},
        ],
        "summaries": [
            {"tag": "nv/cold/time/gpu/mean", "data": [{"name": "value", "type": "float64", "value": "0.001"}]},
            {"tag": "accelsim/verification/pass", "data": [{"name": "value", "type": "int64", "value": str(verify_pass)}]},
            {"tag": "accelsim/verification/mode", "data": [{"name": "value", "type": "string", "value": "sampled"}]},
            {"tag": "accelsim/verification/max_abs_error", "data": [{"name": "value", "type": "float64", "value": "0.0"}]},
            {"tag": "accelsim/verification/max_rel_error", "data": [{"name": "value", "type": "float64", "value": "0.0"}]},
        ],
    }


@pytest.mark.parametrize("verify_pass,expected_run_status", [(1, "pass"), (0, "fail")])
def test_export_marks_run_fail_on_verification_failure(tmp_path: Path, verify_pass: int, expected_run_status: str) -> None:
    nvbench = {
        "meta": {"argv": [], "version": {"nvbench": {"string": "0.0.0"}}},
        "devices": [{"id": 0, "name": "Test GPU", "sm_version": 80}],
        "benchmarks": [{"name": "gemm_transpose_bench", "states": [_make_nvbench_state(verify_pass=verify_pass)]}],
    }

    results = normalize_nvbench_results(
        nvbench,
        git_branch="main",
        git_commit="deadbeef",
        git_dirty=False,
        pixi_env="cuda13",
        nvbench_source_path="/abs/path/to/nvbench",
        artifacts_dir=tmp_path,
        nvbench_settings={"stopping_criterion": "stdrel", "min_time_s": 0.1, "max_noise_pct": 5.0, "min_samples": 1},
    )

    assert results["run"]["status"] == expected_run_status
    assert results["records"][0]["verification"]["status"] == ("pass" if verify_pass == 1 else "fail")

