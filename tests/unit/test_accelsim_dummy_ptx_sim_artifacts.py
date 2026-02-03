from __future__ import annotations

from pathlib import Path

import pytest

from accelsim_test.accelsim_dummy_ptx_sim import artifacts


def test_ensure_new_run_dir_refuses_overwrite(tmp_path: Path) -> None:
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    with pytest.raises(FileExistsError):
        artifacts.ensure_new_run_dir(run_dir)


def test_sha256_file_stable(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_text("abc")
    assert artifacts.sha256_file(p) == artifacts.sha256_file(p)

