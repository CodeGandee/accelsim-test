from __future__ import annotations

from pathlib import Path

from accelsim_test.accelsim_dummy_ptx_sim.workflow import parse_pass_fail


def test_parse_pass_fail_pass(tmp_path: Path) -> None:
    p = tmp_path / "ok.log"
    p.write_text("Accel-Sim [build foo]\nPASS\n")
    assert parse_pass_fail(p) == ("pass", None)


def test_parse_pass_fail_fail(tmp_path: Path) -> None:
    p = tmp_path / "bad.log"
    p.write_text("FAIL\nMismatch...\n")
    assert parse_pass_fail(p) == ("fail", "correctness_mismatch")


def test_parse_pass_fail_missing_marker(tmp_path: Path) -> None:
    p = tmp_path / "missing.log"
    p.write_text("no markers here\n")
    assert parse_pass_fail(p) == ("fail", "missing_pass_fail_marker")


def test_parse_pass_fail_missing_banner(tmp_path: Path) -> None:
    p = tmp_path / "no_banner.log"
    p.write_text("PASS\n")
    assert parse_pass_fail(p) == ("fail", "missing_simulator_banner")
