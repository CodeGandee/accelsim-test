from __future__ import annotations

from pathlib import Path

import pytest

from accelsim_test.profiling.cublaslt_profiling import profiles_case_dir, validate_case_id


def test_validate_case_id_accepts_common_ids() -> None:
    validate_case_id("n1000_int8_abt_view_algo23")
    validate_case_id("a")
    validate_case_id("A-1._b")


def test_validate_case_id_rejects_bad_ids() -> None:
    with pytest.raises(ValueError):
        validate_case_id("")
    with pytest.raises(ValueError):
        validate_case_id("../escape")
    with pytest.raises(ValueError):
        validate_case_id("has space")


def test_profiles_case_dir_builds_expected_path(tmp_path: Path) -> None:
    out = profiles_case_dir(tmp_path, "case-1")
    assert out == tmp_path / "profiles" / "case-1"

