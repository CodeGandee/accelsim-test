from __future__ import annotations

from pathlib import Path

from accelsim_test.gemm_transpose_bench.export import validate_results_schema


def test_results_schema_minimal_payload_validates() -> None:
    results = {
        "schema_version": "0.1.0",
        "run": {
            "run_id": "deadbeef",
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:00:01Z",
            "status": "pass",
            "failure_reason": "",
            "git": {"branch": "main", "commit": "deadbeef", "dirty": False},
            "environment": {
                "platform": {"os": "linux", "arch": "x86_64"},
                "gpu": {"device_name": "Test GPU"},
                "cuda": {"toolkit_version": "13.0"},
                "pixi_env": "cuda13",
                "nvbench": {"source_path": "/abs/path/to/nvbench", "version": "0.0.0"},
            },
            "settings": {"nvbench": {"stopping_criterion": "stdrel", "min_time_s": 0.1, "max_noise_pct": 5.0, "min_samples": 1}},
            "artifacts_dir": "/tmp/out",
        },
        "records": [
            {
                "suite": "square",
                "case": "AB",
                "shape": {"m": 512, "n": 512, "k": 512},
                "dtype": {"a": "fp16", "b": "fp16", "c": "fp16", "compute": "fp32", "math_mode": "default"},
                "timing": {"measurement": "cold", "gpu_time_ms": 1.0, "cpu_time_ms": 1.0, "samples": 1, "nvbench_raw": None},
                "flop_count": 2 * 512 * 512 * 512,
                "ratios": {},
                "verification": {"status": "pass", "mode": "sampled", "max_abs_error": 0.0, "max_rel_error": 0.0},
                "profiling": None,
            }
        ],
    }

    validate_results_schema(results, schema_path=None)


def test_results_schema_file_exists() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    schema_path = repo_root / "specs" / "002-gemm-transpose-bench" / "contracts" / "results.schema.json"
    assert schema_path.exists()

