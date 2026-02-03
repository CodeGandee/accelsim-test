from __future__ import annotations

from pathlib import Path

import pytest

from accelsim_test.accelsim_dummy_ptx_sim import artifacts, paths


def test_sanitize_run_id_filesystem_safe() -> None:
    assert paths.sanitize_run_id("hello world") == "hello-world"
    assert paths.sanitize_run_id("  2026-02-03T00:00:00Z  ") == "2026-02-03T00-00-00Z"


def test_sanitize_run_id_rejects_empty() -> None:
    with pytest.raises(ValueError):
        paths.sanitize_run_id("   ")


def test_run_artifacts_dir_under_tmp() -> None:
    repo = Path("/tmp/fake-repo")
    p = paths.run_artifacts_dir(repo_root=repo, run_id="r1")
    assert str(p).endswith("/tmp/fake-repo/tmp/accelsim_dummy_ptx_sim/r1")


def test_run_artifacts_paths_layout() -> None:
    root = Path("/tmp/fake-repo/tmp/accelsim_dummy_ptx_sim/r1")
    a = artifacts.run_artifacts_paths(artifacts_dir=root)
    assert str(a.exe_path).endswith("/bin/matmul")
    assert str(a.ptx_path).endswith("/ptx/matmul.ptx")
    assert str(a.config_path).endswith("/run/gpgpusim.config")
    assert str(a.stdout_log_path).endswith("/run/matmul.sim.log")
    assert str(a.metadata_path).endswith("/metadata.json")

