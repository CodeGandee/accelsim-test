from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import attrs

from . import artifacts, paths, prereqs, toolchain
from .model import CompilerSource, PrerequisiteCheck, RunArtifacts, RunStatus, SimulationRun


def _now_rfc3339() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _git_info(repo_root: Path) -> dict[str, Any]:
    try:
        branch = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root).decode().strip())
        return {"branch": branch, "commit": commit, "dirty": dirty}
    except Exception:
        return {"branch": "unknown", "commit": "unknown", "dirty": False}


def _compiler_source_from_mode(mode: str) -> CompilerSource:
    if mode == "system":
        return "system"
    return "pixi"


def _render_run_simulation_command(*, repo_root: Path, run_dir: Path, rel_exe: str) -> tuple[list[str], str]:
    env_script = (
        repo_root / "extern" / "tracked" / "accel-sim-framework" / "gpu-simulator" / "setup_environment.sh"
    )
    script = " ; ".join(
        [
            "export GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN=",
            f"source {shlex.quote(str(env_script))}",
            f"cd {shlex.quote(str(run_dir))}",
            shlex.quote(rel_exe),
        ]
    )
    argv = ["bash", "-lc", script]
    return argv, shlex.join(argv)


def parse_pass_fail(log_path: Path) -> tuple[RunStatus, str | None]:
    """Return (status, failure_reason) based on PASS/FAIL markers in the log."""
    try:
        lines = [ln.strip() for ln in log_path.read_text(errors="replace").splitlines()]
    except Exception:
        return "fail", "missing_log"

    has_banner = any("Accel-Sim [build" in ln for ln in lines)
    has_pass = any(ln == "PASS" for ln in lines)
    has_fail = any(ln == "FAIL" for ln in lines)

    if has_pass and not has_fail:
        if not has_banner:
            return "fail", "missing_simulator_banner"
        return "pass", None
    if has_fail:
        return "fail", "correctness_mismatch"
    if any("CUDA error" in ln for ln in lines):
        return "fail", "runtime_error"
    return "fail", "missing_pass_fail_marker"


def format_prereq_failures(checks: list[PrerequisiteCheck]) -> str:
    lines: list[str] = ["Missing prerequisites:"]
    for c in checks:
        if c.status != "fail":
            continue
        hint = f" - {c.details}" if c.details else ""
        lines.append(f"- {c.check_name}{hint}")
    return "\n".join(lines)


def run(*, run_id: str | None, compiler: toolchain.CompilerMode, config_preset: str) -> int:
    """Run the dummy PTX simulation workflow.

    This function must always attempt to write `metadata.json` in the per-run artifacts dir.
    """
    repo_root = paths.find_repo_root()
    chosen_run_id = run_id or _default_run_id()
    artifacts_dir = paths.run_artifacts_dir(repo_root=repo_root, run_id=chosen_run_id)

    try:
        artifacts.ensure_new_run_dir(artifacts_dir)
        artifacts.create_artifact_dirs(artifacts_dir)
    except FileExistsError as e:
        print(str(e), file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Failed to create run dir: {e}", file=sys.stderr)
        return 2

    run_artifacts: RunArtifacts = artifacts.run_artifacts_paths(artifacts_dir=artifacts_dir)
    started_at = _now_rfc3339()
    run_meta = SimulationRun(
        run_id=paths.sanitize_run_id(chosen_run_id),
        started_at=started_at,
        finished_at=None,
        status="fail",
        failure_reason=None,
        mode="ptx",
        config_preset="SM80_A100",
        compiler_source=_compiler_source_from_mode(compiler),
        artifacts_dir=artifacts_dir,
        git=_git_info(repo_root),
        commands={
            "build_simulator": "pixi run -e accelsim accelsim-build",
            "compile_app": "",
            "run_simulation": "",
        },
        prerequisites=[],
    )

    ptx_sha256: str | None = None
    source_info: dict[str, Any] | None = None
    config_info: dict[str, Any] | None = None
    exit_code = 1
    try:
        src = repo_root / "cpp" / "accelsim_dummy_ptx_sim" / "matmul.cu"

        checks = prereqs.check_all(repo_root=repo_root, compiler=compiler, config_preset=config_preset)
        run_meta = attrs.evolve(run_meta, prerequisites=checks)
        if any(c.status == "fail" for c in checks):
            print(format_prereq_failures(checks), file=sys.stderr)
            run_meta = attrs.evolve(run_meta, status="fail", failure_reason="missing_prerequisites")
            exit_code = 2
            return exit_code

        preset_cli = paths.normalize_config_preset(config_preset)
        preset_name, config_src = paths.preset_config_source(repo_root=repo_root, preset=preset_cli)
        shutil.copy2(config_src, run_artifacts.config_path)
        config_info = {
            "preset_cli": preset_cli,
            "preset": preset_name,
            "source_path": str(config_src.resolve()),
        }
        run_meta = attrs.evolve(run_meta, config_preset=preset_name)

        compiler_mode = cast(toolchain.CompilerMode, compiler)
        compiler_source, compile_cmd = toolchain.compile_cuda_program(
            mode=compiler_mode,
            src=src,
            exe_out=run_artifacts.exe_path,
            ptx_out=run_artifacts.ptx_path,
        )
        ptx_sha256 = artifacts.sha256_file(run_artifacts.ptx_path)

        run_meta = attrs.evolve(
            run_meta,
            compiler_source=compiler_source,
            commands={
                **run_meta.commands,
                "compile_app": compile_cmd,
            },
        )

        copied_src_dir = artifacts_dir / "src"
        copied_src_dir.mkdir(parents=True, exist_ok=True)
        copied_src = copied_src_dir / src.name
        shutil.copy2(src, copied_src)
        source_info = {
            "original_path": str(src.resolve()),
            "copied_path": str(copied_src.resolve()),
            "sha256": artifacts.sha256_file(copied_src),
        }

        run_dir = run_artifacts.config_path.parent
        run_argv, run_cmd = _render_run_simulation_command(repo_root=repo_root, run_dir=run_dir, rel_exe="../bin/matmul")
        with run_artifacts.stdout_log_path.open("wb") as f:
            proc = subprocess.run(run_argv, cwd=repo_root, stdout=f, stderr=subprocess.STDOUT, check=False)

        status, failure_reason = parse_pass_fail(run_artifacts.stdout_log_path)
        run_meta = attrs.evolve(
            run_meta,
            status=status,
            failure_reason=failure_reason,
            commands={**run_meta.commands, "run_simulation": run_cmd},
        )
        exit_code = 0 if status == "pass" else (proc.returncode or 1)
    except Exception as e:
        run_meta = attrs.evolve(run_meta, status="fail", failure_reason=str(e))
        exit_code = 1
    finally:
        finished_at = _now_rfc3339()
        run_meta = attrs.evolve(run_meta, finished_at=finished_at)
        payload: dict[str, Any] = {"simulation_run": run_meta.to_dict(), "artifacts": run_artifacts.to_dict()}
        if ptx_sha256 is not None:
            payload["ptx_sha256"] = ptx_sha256
        if source_info is not None:
            payload["source"] = source_info
        if config_info is not None:
            payload["config"] = config_info
        try:
            artifacts.write_metadata(run_artifacts.metadata_path, payload)
        except Exception as e:
            print(f"Failed to write metadata: {e}", file=sys.stderr)
            return 2

    return exit_code
