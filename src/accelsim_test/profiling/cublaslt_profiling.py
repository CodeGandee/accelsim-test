"""
cuBLASLt profiling orchestration helpers.

This module provides small, composable functions used by repository CLI scripts
under `scripts/` to:

- Run Nsight Systems (`nsys`) captures for kernel discovery
- Export kernel-trace tables and generate a machine-readable kernel listing
- Run Nsight Compute (`ncu`) to capture `.ncu-rep` reports with deterministic scoping
- Export lightweight CSV summaries from `.ncu-rep` reports

All outputs are written under a user-specified output directory in the layout:

`<out_dir>/profiles/<case_id>/...`

The stakeholder report is intentionally not generated here; it should reference
the produced artifacts.
"""

from __future__ import annotations

import csv
import json
import platform
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mdutils.mdutils import MdUtils  # type: ignore[import-untyped]


def _utc_now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _run_capture(cmd: list[str], *, cwd: Path | None = None) -> str | None:
    """Run a command and capture combined stdout/stderr as text."""
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, cwd=cwd)
    except Exception:
        return None
    return out.decode(errors="replace").strip()


def _run_checked(cmd: list[str], *, cwd: Path | None = None) -> None:
    """Run a command and raise if it fails."""
    subprocess.run(cmd, cwd=cwd, check=True)


def _which(name: str) -> str | None:
    """Return the absolute path of a command found on PATH."""
    return shutil.which(name)


def _tool_version(cmd: list[str]) -> str | None:
    """Return the first line of a tool version command, if available."""
    out = _run_capture(cmd)
    if not out:
        return None
    return out.splitlines()[0].strip()


def _git_state(repo_root: Path) -> dict[str, Any] | None:
    """Return a best-effort git state snapshot (branch/commit/dirty)."""
    git = _which("git")
    if git is None:
        return None
    commit = _run_capture([git, "rev-parse", "HEAD"], cwd=repo_root)
    branch = _run_capture([git, "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    dirty = _run_capture([git, "status", "--porcelain=v1"], cwd=repo_root)
    if not commit or not branch:
        return None
    return {"commit": commit, "branch": branch, "dirty": bool(dirty)}


_CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_case_id(case_id: str) -> None:
    """
    Validate a case identifier used in on-disk artifact layout.

    Parameters
    ----------
    case_id:
        Short identifier used under `<out_dir>/profiles/<case_id>/`.
        Allowed characters are `[A-Za-z0-9._-]` and it must start with an
        alphanumeric character.
    """
    if not _CASE_ID_RE.fullmatch(case_id):
        raise ValueError(
            f"Invalid case_id '{case_id}'. Expected /^[A-Za-z0-9][A-Za-z0-9._-]{{0,127}}$/."
        )


def profiles_case_dir(out_dir: Path, case_id: str) -> Path:
    """
    Return the directory used to store artifacts for a profiling case.
    """
    validate_case_id(case_id)
    return out_dir / "profiles" / case_id


def _write_json(path: Path, obj: Any) -> None:
    """Write JSON with stable formatting (indent + sorted keys)."""
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _ensure_dir(path: Path) -> None:
    """Create a directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def _repo_root() -> Path:
    """Return the repository root directory (workspace root)."""
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class NsysDiscoveryResult:
    case_dir: Path
    nsys_rep: Path
    qdrep: Path | None
    cuda_gpu_trace_csv: Path
    kernel_list_csv: Path


def _parse_int_field(row: dict[str, str], key: str) -> int | None:
    """Parse an integer field from a CSV row, returning None on missing/invalid."""
    v = row.get(key, "").strip()
    if not v:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def _build_kernel_list_from_cuda_gpu_trace(trace_csv: Path, *, out_csv: Path) -> None:
    """
    Build a compact kernel listing from an `nsys stats --report cuda_gpu_trace` CSV.

    The `cuda_gpu_trace` report includes kernels and memory operations. This function
    keeps only kernel launches (identified by having numeric grid and block dims).
    It assigns both a global index and a per-kernel-name index.
    """
    with trace_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    per_name_counts: dict[str, int] = {}
    out_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        grd_x = _parse_int_field(row, "GrdX")
        blk_x = _parse_int_field(row, "BlkX")
        if grd_x is None or blk_x is None:
            continue
        if grd_x <= 0 or blk_x <= 0:
            continue

        name = (row.get("Name") or "").strip()
        name_idx = per_name_counts.get(name, 0)
        per_name_counts[name] = name_idx + 1

        start_key = "Start (ns)" if "Start (ns)" in row else "Start"
        dur_key = "Duration (ns)" if "Duration (ns)" in row else "Duration"
        stc_key = "StcSMem (MB)" if "StcSMem (MB)" in row else "StcSMem"
        dym_key = "DymSMem (MB)" if "DymSMem (MB)" in row else "DymSMem"

        out_rows.append(
            {
                "global_index": len(out_rows),
                "trace_row_index": idx,
                "name_index": name_idx,
                "start_ns": row.get(start_key, "").strip(),
                "duration_ns": row.get(dur_key, "").strip(),
                "corr_id": row.get("CorrId", "").strip(),
                "grid": f"{row.get('GrdX','').strip()}x{row.get('GrdY','').strip()}x{row.get('GrdZ','').strip()}",
                "block": f"{row.get('BlkX','').strip()}x{row.get('BlkY','').strip()}x{row.get('BlkZ','').strip()}",
                "reg_per_thread": row.get("Reg/Trd", "").strip(),
                "static_smem_mb": row.get(stc_key, "").strip(),
                "dynamic_smem_mb": row.get(dym_key, "").strip(),
                "device": row.get("Device", "").strip(),
                "ctx": row.get("Ctx", "").strip(),
                "stream": row.get("Strm", "").strip(),
                "name": name,
            }
        )

    _ensure_dir(out_csv.parent)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else ["name"])
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)


def run_nsys_kernel_discovery(
    *,
    out_dir: Path,
    case_id: str,
    pixi_env: str,
    repro_cmd: list[str],
    nvtx_prefix_names: bool = True,
) -> NsysDiscoveryResult:
    """
    Run Nsight Systems kernel discovery and export a machine-readable kernel listing.

    Parameters
    ----------
    out_dir:
        Output root directory. Artifacts are written under `<out_dir>/profiles/<case_id>/nsys/`.
    case_id:
        Case identifier used for the output directory name.
    pixi_env:
        Pixi environment name used to run the reproduction program (e.g., `cuda13`).
    repro_cmd:
        Reproduction command to run (argv list). This module will execute it via:
        `pixi run -e <pixi_env> <repro_cmd...>`.
    nvtx_prefix_names:
        If true, export `nsys stats` tables with NVTX range names prefixed into kernel names.
        This requires the repro program to emit NVTX ranges.
    """
    case_dir = profiles_case_dir(out_dir, case_id)
    nsys_dir = case_dir / "nsys"
    _ensure_dir(nsys_dir)

    nsys = _which("nsys")
    if nsys is None:
        raise RuntimeError("nsys not found on PATH")

    capture_prefix = nsys_dir / "capture"

    prof_cmd = [
        "pixi",
        "run",
        "-e",
        pixi_env,
        nsys,
        "profile",
        "--force-overwrite=true",
        "-t",
        "cuda,cublas,nvtx",
        "-o",
        str(capture_prefix),
        *repro_cmd,
    ]
    _run_checked(prof_cmd, cwd=_repo_root())

    nsys_rep = capture_prefix.with_suffix(".nsys-rep")
    qdrep = capture_prefix.with_suffix(".qdrep")
    if not nsys_rep.exists():
        raise RuntimeError(f"Expected nsys output not found: {nsys_rep}")
    qdrep_path = qdrep if qdrep.exists() else None

    # Clean old derived outputs to avoid ambiguous "created file" detection.
    for p in nsys_dir.glob("*.csv"):
        p.unlink()
    for p in nsys_dir.glob("*.sqlite"):
        p.unlink()

    def _stats_to_csv(report: str, *, dest_name: str) -> Path:
        before = {p.name for p in nsys_dir.glob("*.csv")}
        _run_checked(
            [
                nsys,
                "stats",
                "--force-export=true",
                "--report",
                report,
                "--format",
                "csv",
                "--output",
                ".",
                str(nsys_rep.resolve()),
            ],
            cwd=nsys_dir,
        )
        after = {p.name for p in nsys_dir.glob("*.csv")}
        created = sorted(after - before)
        if not created:
            raise RuntimeError(f"nsys stats produced no CSV output for report '{report}'")
        if len(created) == 1:
            produced = nsys_dir / created[0]
        else:
            needle = report.split(":")[0]
            matches = [n for n in created if needle in n]
            if len(matches) != 1:
                raise RuntimeError(f"Ambiguous nsys stats outputs for '{report}': {created}")
            produced = nsys_dir / matches[0]

        dest = nsys_dir / dest_name
        if dest.exists():
            dest.unlink()
        produced.rename(dest)
        return dest

    trace_report = "cuda_gpu_trace"
    kern_gb_sum_report = "cuda_gpu_kern_gb_sum"
    if nvtx_prefix_names:
        trace_report = f"{trace_report}:nvtx-name"
        kern_gb_sum_report = f"{kern_gb_sum_report}:nvtx-name"
    trace_csv = _stats_to_csv(trace_report, dest_name="cuda_gpu_trace.csv")
    kern_gb_sum_csv = _stats_to_csv(kern_gb_sum_report, dest_name="cuda_gpu_kern_gb_sum.csv")

    kern_list = nsys_dir / "kernel_list.csv"
    _build_kernel_list_from_cuda_gpu_trace(trace_csv, out_csv=kern_list)

    invocation_txt = nsys_dir / "invocation.txt"
    invocation_txt.write_text(f"cwd: {_repo_root()}\ncommand: {shlex.join(prof_cmd)}\n")

    meta = {
        "tool": "nsys",
        "timestamp_utc": _utc_now_iso(),
        "case_id": case_id,
        "out_dir": str(out_dir),
        "pixi_env": pixi_env,
        "command": prof_cmd,
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "tool_versions": {"nsys": _tool_version([nsys, "--version"]), "pixi": _tool_version(["pixi", "--version"])},
        "git": _git_state(_repo_root()),
        "outputs": {
            "nsys_rep": str(nsys_rep),
            "qdrep": str(qdrep_path) if qdrep_path is not None else None,
            "cuda_gpu_trace_csv": str(trace_csv),
            "cuda_gpu_kern_gb_sum_csv": str(kern_gb_sum_csv),
            "kernel_list_csv": str(kern_list),
            "invocation_txt": str(invocation_txt),
        },
    }
    _write_json(nsys_dir / "meta.json", meta)

    md = MdUtils(file_name=str(nsys_dir / "README"), title="Kernel Discovery (nsys)")
    md.new_paragraph("This directory contains Nsight Systems kernel-discovery artifacts for a cuBLASLt repro run.")
    md.new_header(level=1, title="Command")
    md.new_paragraph(f"`{shlex.join(prof_cmd)}`")
    md.new_header(level=1, title="Outputs")
    md.new_list(
        [
            f"`{nsys_rep.name}`: raw capture (nsys-rep)",
            "`capture.qdrep`: raw capture (qdrep; may be absent on some systems)",
            f"`{trace_csv.name}`: `cuda_gpu_trace` export (CSV)",
            f"`{kern_gb_sum_csv.name}`: `cuda_gpu_kern_gb_sum` export (CSV)",
            f"`{kern_list.name}`: compact kernel listing with invocation indices (CSV)",
            "`invocation.txt`: command + cwd",
            "`meta.json`: capture metadata",
        ]
    )
    md.create_md_file()

    return NsysDiscoveryResult(
        case_dir=case_dir,
        nsys_rep=nsys_rep,
        qdrep=qdrep_path,
        cuda_gpu_trace_csv=trace_csv,
        kernel_list_csv=kern_list,
    )


@dataclass(frozen=True, slots=True)
class NcuProfileResult:
    case_dir: Path
    report: Path
    raw_csv: Path
    session_csv: Path
    details_csv: Path


def run_ncu_profile(
    *,
    out_dir: Path,
    case_id: str,
    pixi_env: str,
    repro_cmd: list[str],
    set_name: str = "full",
    scope: str = "profiler",
    nvtx_include: str | None = None,
    kernel_regex: str | None = None,
    launch_count: int = 1,
    launch_skip: int = 0,
) -> NcuProfileResult:
    """
    Run Nsight Compute (`ncu`) and export CSV summaries.

    Parameters
    ----------
    out_dir:
        Output root directory. Artifacts are written under `<out_dir>/profiles/<case_id>/ncu/`.
    case_id:
        Case identifier used for the output directory name.
    pixi_env:
        Pixi environment name used to run the reproduction program (e.g., `cuda13`).
    repro_cmd:
        Reproduction command to run (argv list). This module will execute it via:
        `pixi run -e <pixi_env> <repro_cmd...>`.
    set_name:
        Nsight Compute section set identifier (e.g., `basic`, `full`).
    scope:
        Profiling scoping mode: `profiler`, `nvtx`, or `kernel`.
    nvtx_include:
        NVTX include filter (required when `scope == "nvtx"`).
    kernel_regex:
        Kernel name regex (required when `scope == "kernel"`).
    launch_count:
        Number of matching kernel launches to profile (used for `kernel` scoping).
    launch_skip:
        Number of matching kernel launches to skip before profiling (used for `kernel` scoping).
    """
    case_dir = profiles_case_dir(out_dir, case_id)
    ncu_dir = case_dir / "ncu"
    _ensure_dir(ncu_dir)

    ncu = _which("ncu")
    if ncu is None:
        raise RuntimeError("ncu not found on PATH")

    export_base = ncu_dir / "profile"
    rep_path = export_base.with_suffix(".ncu-rep")

    log_file = ncu_dir / "ncu.log"
    cmd: list[str] = [
        "pixi",
        "run",
        "-e",
        pixi_env,
        ncu,
        "--force-overwrite",
        "--log-file",
        str(log_file),
        "--set",
        set_name,
        "--export",
        str(export_base),
        "--clock-control",
        "base",
        "--pipeline-boost-state",
        "stable",
    ]
    if scope == "profiler":
        cmd += ["--profile-from-start", "off"]
    elif scope == "nvtx":
        if not nvtx_include:
            raise ValueError("nvtx_include is required when scope='nvtx'")
        cmd += ["--nvtx", "--nvtx-include", nvtx_include]
    elif scope == "kernel":
        if not kernel_regex:
            raise ValueError("kernel_regex is required when scope='kernel'")
        cmd += ["-k", f"regex:{kernel_regex}", "-c", str(launch_count), "-s", str(launch_skip)]
    else:
        raise ValueError("scope must be one of: profiler, nvtx, kernel")

    cmd += [*repro_cmd]
    _run_checked(cmd, cwd=_repo_root())

    if not rep_path.exists():
        raise RuntimeError(f"Expected ncu output not found: {rep_path}")

    # Export lightweight summaries from the report.
    raw_csv = ncu_dir / "raw.csv"
    session_csv = ncu_dir / "session.csv"
    details_csv = ncu_dir / "details.csv"
    raw_csv.write_text(_run_capture([ncu, "--import", str(rep_path), "--page", "raw", "--csv"]) or "")
    session_csv.write_text(_run_capture([ncu, "--import", str(rep_path), "--page", "session", "--csv"]) or "")
    details_csv.write_text(_run_capture([ncu, "--import", str(rep_path), "--page", "details", "--csv"]) or "")

    meta = {
        "tool": "ncu",
        "timestamp_utc": _utc_now_iso(),
        "case_id": case_id,
        "out_dir": str(out_dir),
        "pixi_env": pixi_env,
        "scope": {"mode": scope, "nvtx_include": nvtx_include, "kernel_regex": kernel_regex, "launch_count": launch_count, "launch_skip": launch_skip},
        "command": cmd,
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "tool_versions": {"ncu": _tool_version([ncu, "--version"]), "pixi": _tool_version(["pixi", "--version"])},
        "git": _git_state(_repo_root()),
        "outputs": {
            "ncu_rep": str(rep_path),
            "raw_csv": str(raw_csv),
            "session_csv": str(session_csv),
            "details_csv": str(details_csv),
            "ncu_log": str(log_file),
        },
    }
    _write_json(ncu_dir / "meta.json", meta)

    md = MdUtils(file_name=str(ncu_dir / "README"), title="Kernel Profiling (ncu)")
    md.new_paragraph("This directory contains Nsight Compute profiling artifacts for a cuBLASLt repro run.")
    md.new_header(level=1, title="Command")
    md.new_paragraph(f"`{shlex.join(cmd)}`")
    md.new_header(level=1, title="Outputs")
    md.new_list(
        [
            f"`{rep_path.name}`: raw ncu report",
            f"`{raw_csv.name}`: exported raw metrics (CSV text)",
            f"`{session_csv.name}`: exported session/device info (CSV text)",
            f"`{details_csv.name}`: exported section/rule details (CSV text)",
            "`meta.json`: run metadata",
        ]
    )
    md.create_md_file()

    return NcuProfileResult(
        case_dir=case_dir,
        report=rep_path,
        raw_csv=raw_csv,
        session_csv=session_csv,
        details_csv=details_csv,
    )
