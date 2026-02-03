from __future__ import annotations

import json
from pathlib import Path

from accelsim_test.gemm_transpose_bench.report import report_run


def test_report_run_writes_reports_without_rerunning_bench(tmp_path: Path) -> None:
    results = {
        "schema_version": "0.1.0",
        "run": {"git": {"branch": "main", "commit": "deadbeef", "dirty": False}, "status": "pass"},
        "records": [],
    }
    (tmp_path / "results.json").write_text(json.dumps(results))

    rc = report_run(out_dir=tmp_path)
    assert rc == 0

    report_md = (tmp_path / "report.md").read_text()
    assert "# GEMM Transpose Benchmark Report" in report_md

    all_timings_md = (tmp_path / "all_timings.md").read_text()
    assert "# GEMM Transpose Benchmark Timings (All Records)" in all_timings_md

    stakeholder = (tmp_path / "stakeholder_report.md").read_text()
    assert "# GEMM Transpose Sweep — Stakeholder Report" in stakeholder


def test_report_run_does_not_overwrite_existing_stakeholder_report(tmp_path: Path) -> None:
    results = {
        "schema_version": "0.1.0",
        "run": {"git": {"branch": "main", "commit": "deadbeef", "dirty": False}, "status": "pass"},
        "records": [],
    }
    (tmp_path / "results.json").write_text(json.dumps(results))
    (tmp_path / "stakeholder_report.md").write_text("KEEP\n")

    rc = report_run(out_dir=tmp_path)
    assert rc == 0
    assert (tmp_path / "stakeholder_report.md").read_text() == "KEEP\n"
