from __future__ import annotations

import os
from pathlib import Path

import pytest

from accelsim_test.accelsim_dummy_ptx_sim import prereqs


def test_check_tmp_writable_passes(tmp_path: Path) -> None:
    c = prereqs.check_tmp_writable(tmp_path)
    assert c.status == "pass"


def test_check_submodule_initialized_fails_without_marker(tmp_path: Path) -> None:
    c = prereqs.check_submodule_initialized(tmp_path)
    assert c.status == "fail"


def test_check_submodule_initialized_passes_with_env_script(tmp_path: Path) -> None:
    p = tmp_path / "extern" / "tracked" / "accel-sim-framework" / "gpu-simulator"
    p.mkdir(parents=True)
    (p / "setup_environment.sh").write_text("#!/usr/bin/env bash\n")
    c = prereqs.check_submodule_initialized(tmp_path)
    assert c.status == "pass"


def test_check_simulator_built_passes_with_exe(tmp_path: Path) -> None:
    exe = tmp_path / "extern" / "tracked" / "accel-sim-framework" / "gpu-simulator" / "bin" / "release" / "accel-sim.out"
    exe.parent.mkdir(parents=True)
    exe.write_text("stub")
    c = prereqs.check_simulator_built(tmp_path)
    assert c.status == "pass"


def test_check_config_preset_exists_passes_with_config(tmp_path: Path) -> None:
    cfg = (
        tmp_path
        / "extern"
        / "tracked"
        / "accel-sim-framework"
        / "gpu-simulator"
        / "gpgpu-sim"
        / "configs"
        / "tested-cfgs"
        / "SM80_A100"
        / "gpgpusim.config"
    )
    cfg.parent.mkdir(parents=True)
    cfg.write_text("stub")
    c = prereqs.check_config_preset_exists(repo_root=tmp_path, preset="sm80_a100")
    assert c.status == "pass"


def test_check_nvcc_available_system_respects_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    nvcc = bin_dir / "nvcc"
    nvcc.write_text("#!/usr/bin/env bash\necho nvcc\n")
    nvcc.chmod(0o755)

    monkeypatch.setenv("PATH", os.fspath(bin_dir))
    c = prereqs.check_nvcc_available(mode="system")
    assert c.status == "pass"

    monkeypatch.setenv("PATH", os.fspath(tmp_path))
    c2 = prereqs.check_nvcc_available(mode="system")
    assert c2.status == "fail"

