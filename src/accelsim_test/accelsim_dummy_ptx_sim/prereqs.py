from __future__ import annotations

import os
import shutil
from pathlib import Path

from . import paths, toolchain
from .model import PrerequisiteCheck


def check_submodule_initialized(repo_root: Path) -> PrerequisiteCheck:
    accel_sim_root = repo_root / "extern" / "tracked" / "accel-sim-framework"
    git_marker = accel_sim_root / ".git"
    env_script = accel_sim_root / "gpu-simulator" / "setup_environment.sh"
    if git_marker.exists() or env_script.exists():
        return PrerequisiteCheck(check_name="submodule_initialized", status="pass")
    return PrerequisiteCheck(
        check_name="submodule_initialized",
        status="fail",
        details="Run: git submodule update --init --recursive",
    )


def check_simulator_built(repo_root: Path) -> PrerequisiteCheck:
    exe = repo_root / "extern" / "tracked" / "accel-sim-framework" / "gpu-simulator" / "bin" / "release" / "accel-sim.out"
    if exe.exists():
        return PrerequisiteCheck(check_name="simulator_built", status="pass")
    return PrerequisiteCheck(
        check_name="simulator_built",
        status="fail",
        details="Run: pixi run -e accelsim accelsim-build",
    )


def check_nvcc_available(*, mode: toolchain.CompilerMode) -> PrerequisiteCheck:
    """Best-effort check for nvcc availability.

    This remains lightweight (no compile) to keep fail-fast UX responsive.
    """
    if mode == "system":
        if shutil.which("nvcc") is not None:
            return PrerequisiteCheck(check_name="nvcc_available", status="pass")
        return PrerequisiteCheck(
            check_name="nvcc_available",
            status="fail",
            details="Install CUDA Toolkit and ensure `nvcc` is on PATH (e.g., `which nvcc`).",
        )

    # pixi/auto: accept either nvcc on PATH (already in env) or pixi available to run nvcc via `pixi run`.
    if shutil.which("nvcc") is not None or shutil.which("pixi") is not None:
        return PrerequisiteCheck(check_name="nvcc_available", status="pass")
    return PrerequisiteCheck(
        check_name="nvcc_available",
        status="fail",
        details="Run: pixi install -e accelsim  (or install system CUDA and ensure `nvcc` is on PATH)",
    )


def check_config_preset_exists(*, repo_root: Path, preset: str) -> PrerequisiteCheck:
    try:
        preset_cli = paths.normalize_config_preset(preset)
        _, cfg = paths.preset_config_source(repo_root=repo_root, preset=preset_cli)
    except Exception as e:
        return PrerequisiteCheck(check_name="config_preset_exists", status="fail", details=str(e))

    if cfg.exists():
        return PrerequisiteCheck(check_name="config_preset_exists", status="pass")
    return PrerequisiteCheck(
        check_name="config_preset_exists",
        status="fail",
        details=f"Missing config: {cfg} (did you init submodules?)",
    )


def check_tmp_writable(repo_root: Path) -> PrerequisiteCheck:
    tmp = repo_root / "tmp"
    try:
        tmp.mkdir(exist_ok=True)
        test = tmp / f".write_test_{os.getpid()}"
        test.write_text("ok")
        test.unlink()
        return PrerequisiteCheck(check_name="tmp_writable", status="pass")
    except Exception as e:
        return PrerequisiteCheck(check_name="tmp_writable", status="fail", details=str(e))


def check_all(*, repo_root: Path, compiler: toolchain.CompilerMode, config_preset: str) -> list[PrerequisiteCheck]:
    return [
        check_submodule_initialized(repo_root),
        check_simulator_built(repo_root),
        check_nvcc_available(mode=compiler),
        check_config_preset_exists(repo_root=repo_root, preset=config_preset),
        check_tmp_writable(repo_root),
    ]
