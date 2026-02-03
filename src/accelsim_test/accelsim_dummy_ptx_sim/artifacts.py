from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .model import RunArtifacts

DEFAULT_PROGRAM_NAME = "matmul"


def ensure_new_run_dir(artifacts_dir: Path) -> None:
    """Raise FileExistsError if artifacts_dir already exists (prevents overwrites)."""
    if artifacts_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run dir: {artifacts_dir}")


def ensure_empty_run_dir(artifacts_dir: Path) -> None:
    """Backward-compatible alias for `ensure_new_run_dir`."""
    ensure_new_run_dir(artifacts_dir)


def create_artifact_dirs(artifacts_dir: Path) -> dict[str, Path]:
    """Create bin/ ptx/ run/ directories and return their paths."""
    artifacts_dir.mkdir(parents=True, exist_ok=False)
    bin_dir = artifacts_dir / "bin"
    ptx_dir = artifacts_dir / "ptx"
    run_dir = artifacts_dir / "run"
    bin_dir.mkdir()
    ptx_dir.mkdir()
    run_dir.mkdir()
    return {"bin": bin_dir, "ptx": ptx_dir, "run": run_dir}


def run_artifacts_paths(*, artifacts_dir: Path, program_name: str = DEFAULT_PROGRAM_NAME) -> RunArtifacts:
    artifacts_dir = artifacts_dir.resolve()
    return RunArtifacts(
        exe_path=artifacts_dir / "bin" / program_name,
        ptx_path=artifacts_dir / "ptx" / f"{program_name}.ptx",
        config_path=artifacts_dir / "run" / "gpgpusim.config",
        stdout_log_path=artifacts_dir / "run" / f"{program_name}.sim.log",
        metadata_path=artifacts_dir / "metadata.json",
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_metadata(metadata_path: Path, payload: dict[str, Any]) -> None:
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
