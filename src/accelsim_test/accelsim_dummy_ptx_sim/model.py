from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import attrs

RunStatus = Literal["pass", "fail"]
CompilerSource = Literal["pixi", "system"]
ConfigPreset = Literal["SM80_A100"]


@attrs.define(frozen=True, slots=True)
class PrerequisiteCheck:
    check_name: str
    status: RunStatus
    details: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"check_name": self.check_name, "status": self.status, "details": self.details}


@attrs.define(frozen=True, slots=True)
class RunArtifacts:
    exe_path: Path
    ptx_path: Path
    config_path: Path
    stdout_log_path: Path
    metadata_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "exe_path": str(self.exe_path),
            "ptx_path": str(self.ptx_path),
            "config_path": str(self.config_path),
            "stdout_log_path": str(self.stdout_log_path),
            "metadata_path": str(self.metadata_path),
        }


@attrs.define(frozen=True, slots=True)
class SimulationRun:
    run_id: str
    started_at: str
    finished_at: str | None
    status: RunStatus
    failure_reason: str | None
    mode: Literal["ptx"]
    config_preset: ConfigPreset
    compiler_source: CompilerSource
    artifacts_dir: Path
    git: dict[str, Any]
    commands: dict[str, str]
    prerequisites: list[PrerequisiteCheck] = attrs.field(factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "mode": self.mode,
            "config_preset": self.config_preset,
            "compiler_source": self.compiler_source,
            "artifacts_dir": str(self.artifacts_dir),
            "git": self.git,
            "commands": self.commands,
            "prerequisites": [c.to_dict() for c in self.prerequisites],
        }
