from __future__ import annotations

from accelsim_test.gemm_transpose_bench.report import compute_ratios_in_place


def test_compute_ratios_square_suite() -> None:
    results = {
        "schema_version": "0.1.0",
        "run": {},
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
                "suite": "square",
                "case": "ATB_copyA",
                "shape": {"m": 512, "n": 512, "k": 512},
                "dtype": {"a": "fp16", "b": "fp16", "c": "fp16", "compute": "fp32", "math_mode": "default"},
                "timing": {"measurement": "cold", "gpu_time_ms": 3.0},
                "flop_count": 2 * 512 * 512 * 512,
                "ratios": {},
                "verification": {"status": "pass", "mode": "sampled", "max_abs_error": 0.0, "max_rel_error": 0.0},
                "profiling": None,
            },
        ],
    }

    compute_ratios_in_place(results)
    by_case = {r["case"]: r for r in results["records"]}
    assert by_case["AB"]["ratios"]["ratio_to_ab"] == 1.0
    assert by_case["ATB_view"]["ratios"]["ratio_to_ab"] == 2.0
    assert by_case["ATB_copyA"]["ratios"]["ratio_to_ab"] == 3.0
    assert by_case["ATB_copyA"]["ratios"]["ratio_copy_over_view"] == 1.5


def test_compute_ratios_non_square_copy_over_view_only() -> None:
    results = {
        "schema_version": "0.1.0",
        "run": {},
        "records": [
            {
                "suite": "nonsquare_atb",
                "case": "ATB_view",
                "shape": {"m": 992, "n": 256, "k": 256},
                "dtype": {"a": "fp16", "b": "fp16", "c": "fp16", "compute": "fp32", "math_mode": "default"},
                "timing": {"measurement": "cold", "gpu_time_ms": 4.0},
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
                "timing": {"measurement": "cold", "gpu_time_ms": 10.0},
                "flop_count": 2 * 992 * 256 * 256,
                "ratios": {},
                "verification": {"status": "pass", "mode": "sampled", "max_abs_error": 0.0, "max_rel_error": 0.0},
                "profiling": None,
            },
        ],
    }

    compute_ratios_in_place(results)
    by_case = {r["case"]: r for r in results["records"]}
    assert "ratio_to_ab" not in by_case["ATB_view"]["ratios"]
    assert by_case["ATB_copyA"]["ratios"]["ratio_copy_over_view"] == 2.5

