from __future__ import annotations

from accelsim_test.gemm_transpose_bench.report import generate_report


def test_generate_report_contains_required_tables() -> None:
    results = {
        "schema_version": "0.1.0",
        "run": {"git": {"branch": "main", "commit": "deadbeef"}, "status": "pass"},
        "records": [
            {
                "suite": "square",
                "case": "AB",
                "shape": {"m": 512, "n": 512, "k": 512},
                "dtype": {"a": "fp16", "b": "fp16", "c": "fp16", "compute": "fp32", "math_mode": "default"},
                "timing": {"measurement": "cold", "gpu_time_ms": 1.0},
                "flop_count": 2 * 512 * 512 * 512,
                "ratios": {},
                "verification": {"status": "pass", "mode": "sampled", "max_abs_error": 0.0, "max_rel_error": 0.0},
                "profiling": None,
            },
            {
                "suite": "square",
                "case": "ATB_view",
                "shape": {"m": 512, "n": 512, "k": 512},
                "dtype": {"a": "fp16", "b": "fp16", "c": "fp16", "compute": "fp32", "math_mode": "default"},
                "timing": {"measurement": "cold", "gpu_time_ms": 2.0},
                "flop_count": 2 * 512 * 512 * 512,
                "ratios": {},
                "verification": {"status": "pass", "mode": "sampled", "max_abs_error": 0.0, "max_rel_error": 0.0},
                "profiling": None,
            },
            {
                "suite": "nonsquare_atb",
                "case": "ATB_view",
                "shape": {"m": 992, "n": 256, "k": 256},
                "dtype": {"a": "fp16", "b": "fp16", "c": "fp16", "compute": "fp32", "math_mode": "default"},
                "timing": {"measurement": "cold", "gpu_time_ms": 3.0},
                "flop_count": 2 * 992 * 256 * 256,
                "ratios": {},
                "verification": {"status": "pass", "mode": "sampled", "max_abs_error": 0.0, "max_rel_error": 0.0},
                "profiling": None,
            },
            {
                "suite": "nonsquare_atb",
                "case": "ATB_copyA",
                "shape": {"m": 992, "n": 256, "k": 256},
                "dtype": {"a": "fp16", "b": "fp16", "c": "fp16", "compute": "fp32", "math_mode": "default"},
                "timing": {"measurement": "cold", "gpu_time_ms": 6.0},
                "flop_count": 2 * 992 * 256 * 256,
                "ratios": {},
                "verification": {"status": "pass", "mode": "sampled", "max_abs_error": 0.0, "max_rel_error": 0.0},
                "profiling": None,
            },
        ],
    }

    report_md = generate_report(results)
    assert "## Square Suite" in report_md
    assert "timed_ms_AB" in report_md
    assert "slow_ATB_view_vs_AB" in report_md
    assert "## Non-square Suite" in report_md
    assert "over_ATB_copyA_vs_view" in report_md

