from __future__ import annotations

import re
from pathlib import Path
from typing import Literal


def find_repo_root() -> Path:
    """Return the repository root directory.

    This feature relies on a repo-local Accel-Sim submodule path, so we locate the root
    by searching upwards for `pyproject.toml`.
    """
    start = Path(__file__).resolve()
    for parent in start.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError(f"Failed to locate repo root from: {start}")


def sanitize_run_id(run_id: str) -> str:
    """Make run_id filesystem-safe and non-empty."""
    s = run_id.strip()
    if not s:
        raise ValueError("run_id must be non-empty")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "-", s)
    s = s.strip("-")
    if not s:
        raise ValueError("run_id must contain at least one alphanumeric character after sanitization")
    return s


def run_artifacts_dir(*, repo_root: Path, run_id: str) -> Path:
    return (repo_root / "tmp" / "accelsim_dummy_ptx_sim" / sanitize_run_id(run_id)).resolve()


def sm80_a100_config_source(*, repo_root: Path) -> Path:
    """Source config path from `research.md` Decision 2."""
    return (
        repo_root
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


ConfigPresetCli = Literal["sm80_a100"]
ConfigPresetName = Literal["SM80_A100"]


def normalize_config_preset(preset: str) -> ConfigPresetCli:
    if preset != "sm80_a100":
        raise ValueError(f"Unsupported config preset: {preset}")
    return "sm80_a100"


def preset_config_source(*, repo_root: Path, preset: ConfigPresetCli) -> tuple[ConfigPresetName, Path]:
    if preset == "sm80_a100":
        return "SM80_A100", sm80_a100_config_source(repo_root=repo_root)
    raise AssertionError("unreachable")
