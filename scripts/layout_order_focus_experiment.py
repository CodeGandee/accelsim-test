"""
CLI: Layout-order focus experiment for N=1000 int8 cuBLASLt GEMM.

This script runs the view-only transpose variants (`AB`, `ATB_view`, `ABT_view`) across
independent A/B storage-order combinations (row/col). It optionally runs a limited
output-order (C) sensitivity check for the baseline A=row, B=row case.

Artifacts are written under a user-specified output directory:
- `cases/<case_id>/stdout.txt` (and `stderr.txt`)
- `results.json` + `results.csv`
- `report.md` (concise summary via mdutils)
- optional profiling bundles under `profiles/<case_id>/...` (nsys/ncu)

Example:
    pixi run python scripts/layout_order_focus_experiment.py \
      --out-dir tmp/layout_order_n1000 \
      --repro-bin ./cpp/build/Release/repro_algo23_int8_n1000 \
      --pixi-env cuda13 \
      --iters 2000 --warmup 200
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mdutils.mdutils import MdUtils  # type: ignore[import-untyped]

from accelsim_test.profiling.cublaslt_profiling import (
    profiles_case_dir,
    run_ncu_profile,
    run_nsys_kernel_discovery,
    validate_case_id,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_capture(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_summary_lines(output: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    prefix = "ACCELSIM_GEMM_RUN "
    for line in output.splitlines():
        if not line.startswith(prefix):
            continue
        payload = line[len(prefix) :].strip()
        out.append(json.loads(payload))
    return out


def _best_effort_git_state(repo_root: Path) -> dict[str, Any] | None:
    cp = _run_capture(["git", "rev-parse", "HEAD"], cwd=repo_root)
    if cp.returncode != 0:
        return None
    commit = cp.stdout.strip()
    branch_cp = _run_capture(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    branch = branch_cp.stdout.strip() if branch_cp.returncode == 0 else None
    dirty_cp = _run_capture(["git", "status", "--porcelain=v1"], cwd=repo_root)
    dirty = bool(dirty_cp.stdout.strip()) if dirty_cp.returncode == 0 else None
    return {"commit": commit, "branch": branch, "dirty": dirty}


def _main_kernel_from_kernel_list_csv(kernel_list_csv: Path) -> dict[str, Any] | None:
    if not kernel_list_csv.exists():
        return None
    with kernel_list_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None

    def _duration_ns(row: dict[str, str]) -> int:
        raw = (row.get("duration_ns") or "").strip()
        try:
            return int(raw)
        except ValueError:
            return 0

    best = max(rows, key=_duration_ns)
    return {
        "name": (best.get("name") or "").strip(),
        "grid": (best.get("grid") or "").strip(),
        "block": (best.get("block") or "").strip(),
        "duration_ns": (best.get("duration_ns") or "").strip(),
        "global_index": (best.get("global_index") or "").strip(),
        "name_index": (best.get("name_index") or "").strip(),
    }


@dataclass(frozen=True, slots=True)
class CaseSpec:
    order_a: str
    order_b: str
    order_c: str
    variant: str


def generate_case_specs() -> list[CaseSpec]:
    orders = ["row", "col"]
    variants = ["AB", "ATB_view", "ABT_view"]

    cases: list[CaseSpec] = []
    # Full A/B order matrix (keep C=row).
    for order_a in orders:
        for order_b in orders:
            for variant in variants:
                cases.append(CaseSpec(order_a=order_a, order_b=order_b, order_c="row", variant=variant))

    # Limited output-order sensitivity: only add C=col for baseline A=row, B=row.
    for variant in variants:
        cases.append(CaseSpec(order_a="row", order_b="row", order_c="col", variant=variant))

    return cases


def make_case_id(*, n: int, dtype: str, case: CaseSpec, symmetric_inputs: bool) -> str:
    variant = case.variant.lower()
    case_id = f"n{n}_{dtype}_a_{case.order_a}_b_{case.order_b}_c_{case.order_c}_{variant}"
    if symmetric_inputs:
        case_id = f"{case_id}_sym"
    validate_case_id(case_id)
    return case_id


def build_repro_cmd(
    *,
    repro_bin: Path,
    case: CaseSpec,
    iters: int,
    warmup: int,
    device: int | None,
    symmetric_inputs: bool,
    nvtx: bool,
    cuda_profiler_gating: bool,
    summary_json: bool,
) -> list[str]:
    cmd = [
        str(repro_bin),
        "--variant",
        case.variant,
        "--iters",
        str(iters),
        "--warmup",
        str(warmup),
        "--order-a",
        case.order_a,
        "--order-b",
        case.order_b,
        "--order-c",
        case.order_c,
    ]
    if device is not None:
        cmd += ["--device", str(device)]
    if symmetric_inputs:
        cmd += ["--symmetric-inputs"]
    if nvtx:
        cmd += ["--nvtx"]
    if cuda_profiler_gating:
        cmd += ["--cuda-profiler-gating"]
    if summary_json:
        cmd += ["--summary-json"]
    return cmd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the N=1000 layout-order focus experiment matrix.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory root.")
    parser.add_argument("--repro-bin", type=Path, required=True, help="Path to repro binary (repro_algo23_int8_n1000).")
    parser.add_argument("--pixi-env", type=str, default="cuda13", help="Pixi environment for running the repro (default: cuda13).")
    parser.add_argument("--device", type=int, default=None, help="CUDA device index to pass through to the repro.")
    parser.add_argument("--iters", type=int, default=2000, help="Timed iterations for measurement runs (default: 2000).")
    parser.add_argument("--warmup", type=int, default=200, help="Warmup iterations for measurement runs (default: 200).")
    parser.add_argument("--symmetric-inputs", action="store_true", help="Use symmetric A and B for same-math comparisons.")
    parser.add_argument("--nsys", action="store_true", help="Capture Nsight Systems kernel discovery per case.")
    parser.add_argument("--ncu", action="store_true", help="Capture Nsight Compute report per case (scope=profiler gating).")
    parser.add_argument("--profile-iters", type=int, default=1, help="Timed iterations for profiling runs (default: 1).")
    parser.add_argument("--profile-warmup", type=int, default=0, help="Warmup iterations for profiling runs (default: 0).")
    parser.add_argument("--ncu-set", dest="ncu_set", type=str, default="basic", help="Nsight Compute section set (default: basic).")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = _repo_root()

    if not args.repro_bin.exists():
        raise SystemExit(f"Repro binary not found: {args.repro_bin}")

    out_dir: Path = args.out_dir
    _ensure_dir(out_dir)
    _ensure_dir(out_dir / "cases")

    cases = generate_case_specs()

    run_meta = {
        "tool": "layout_order_focus_experiment",
        "timestamp_utc": _utc_now_iso(),
        "out_dir": str(out_dir),
        "repro_bin": str(args.repro_bin),
        "pixi_env": args.pixi_env,
        "device": args.device,
        "iters": args.iters,
        "warmup": args.warmup,
        "symmetric_inputs": bool(args.symmetric_inputs),
        "nsys": bool(args.nsys),
        "ncu": bool(args.ncu),
        "profile_iters": args.profile_iters,
        "profile_warmup": args.profile_warmup,
        "ncu_set": args.ncu_set,
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "git": _best_effort_git_state(repo_root),
    }
    _write_json(out_dir / "meta.json", run_meta)

    results: list[dict[str, Any]] = []

    for case in cases:
        case_id = make_case_id(n=1000, dtype="int8", case=case, symmetric_inputs=bool(args.symmetric_inputs))
        case_dir = out_dir / "cases" / case_id
        _ensure_dir(case_dir)

        repro_cmd = build_repro_cmd(
            repro_bin=args.repro_bin,
            case=case,
            iters=max(1, int(args.iters)),
            warmup=max(0, int(args.warmup)),
            device=args.device,
            symmetric_inputs=bool(args.symmetric_inputs),
            nvtx=False,
            cuda_profiler_gating=False,
            summary_json=True,
        )
        full_cmd = ["pixi", "run", "-e", args.pixi_env, *repro_cmd]
        cp = _run_capture(full_cmd, cwd=repo_root)
        (case_dir / "stdout.txt").write_text(cp.stdout)
        (case_dir / "stderr.txt").write_text(cp.stderr)
        (case_dir / "invocation.txt").write_text(f"cwd: {repo_root}\ncommand: {shlex.join(full_cmd)}\n")

        summary_lines = _parse_summary_lines(cp.stdout + "\n" + cp.stderr)
        summary = summary_lines[-1] if summary_lines else None

        algo = {
            "algo_id": summary.get("algo_id") if summary else None,
            "tile_id": summary.get("tile_id") if summary else None,
            "stages_id": summary.get("stages_id") if summary else None,
            "splitk_num": summary.get("splitk_num") if summary else None,
        }
        avg_ms = summary.get("avg_ms") if summary else None
        time_us = (float(avg_ms) * 1000.0) if isinstance(avg_ms, (int, float)) else None

        result: dict[str, Any] = {
            "case_id": case_id,
            "order_a": case.order_a,
            "order_b": case.order_b,
            "order_c": case.order_c,
            "variant": case.variant,
            "symmetric_inputs": bool(args.symmetric_inputs),
            "iters": int(args.iters),
            "warmup": int(args.warmup),
            "ok": bool(summary.get("ok")) if summary else False,
            "avg_ms": avg_ms,
            "time_us": time_us,
            **algo,
            "returncode": int(cp.returncode),
        }
        if summary and "reason" in summary:
            result["reason"] = summary["reason"]
        results.append(result)

        # Optional profiling hooks (run with minimal iters/warmup; always enable NVTX).
        repro_cmd_profile = build_repro_cmd(
            repro_bin=args.repro_bin,
            case=case,
            iters=max(1, int(args.profile_iters)),
            warmup=max(0, int(args.profile_warmup)),
            device=args.device,
            symmetric_inputs=bool(args.symmetric_inputs),
            nvtx=True,
            cuda_profiler_gating=bool(args.ncu),
            summary_json=False,
        )

        if args.nsys:
            run_nsys_kernel_discovery(
                out_dir=out_dir,
                case_id=case_id,
                pixi_env=args.pixi_env,
                repro_cmd=repro_cmd_profile,
                nvtx_prefix_names=True,
            )
            kern = _main_kernel_from_kernel_list_csv(profiles_case_dir(out_dir, case_id) / "nsys" / "kernel_list.csv")
            if kern is not None:
                result["nsys_main_kernel"] = kern

        if args.ncu:
            run_ncu_profile(
                out_dir=out_dir,
                case_id=case_id,
                pixi_env=args.pixi_env,
                repro_cmd=repro_cmd_profile,
                set_name=args.ncu_set,
                scope="profiler",
            )

    _write_json(out_dir / "results.json", {"meta": run_meta, "results": results})

    # CSV index for quick grepping/diffing.
    csv_path = out_dir / "results.csv"
    fieldnames = [
        "case_id",
        "order_a",
        "order_b",
        "order_c",
        "variant",
        "symmetric_inputs",
        "iters",
        "warmup",
        "ok",
        "avg_ms",
        "time_us",
        "algo_id",
        "tile_id",
        "stages_id",
        "splitk_num",
        "returncode",
        "reason",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    # Minimal markdown report (two tables).
    md = MdUtils(file_name=str(out_dir / "report"), title="Layout-order focus experiment (N=1000 int8)")
    md.new_header(level=1, title="Run Metadata")
    md.new_paragraph(f"`{_utc_now_iso()}`")
    md.new_paragraph(f"Output: `{out_dir}`")
    md.new_paragraph(f"Repro: `{args.repro_bin}` (pixi env: `{args.pixi_env}`)")

    md.new_header(level=1, title="Results (A/B order matrix; order_c=row)")
    matrix_rows = [
        r
        for r in results
        if r["order_c"] == "row"
        and (r["order_a"], r["order_b"]) in {("row", "row"), ("row", "col"), ("col", "row"), ("col", "col")}
    ]
    matrix_rows.sort(key=lambda r: (r["order_a"], r["order_b"], r["variant"]))

    table_lines = [
        "| order_a | order_b | variant | time (us) | algo_id | tile | stages |",
        "|--------:|--------:|---------|----------:|--------:|-----:|-------:|",
    ]
    for r in matrix_rows:
        table_lines.append(
            "| "
            + " | ".join(
                [
                    str(r["order_a"]),
                    str(r["order_b"]),
                    str(r["variant"]),
                    "" if r["time_us"] is None else f"{float(r['time_us']):.2f}",
                    "" if r["algo_id"] is None else str(r["algo_id"]),
                    "" if r["tile_id"] is None else str(r["tile_id"]),
                    "" if r["stages_id"] is None else str(r["stages_id"]),
                ]
            )
            + " |"
        )
    md.new_paragraph("\n".join(table_lines))

    md.new_header(level=1, title="Output order check (baseline order_a=row, order_b=row)")
    outcheck_rows = [r for r in results if r["order_a"] == "row" and r["order_b"] == "row" and r["variant"] in {"AB", "ATB_view", "ABT_view"}]
    outcheck_rows.sort(key=lambda r: (r["order_c"], r["variant"]))
    table2_lines = [
        "| order_c | variant | time (us) | algo_id | tile | stages |",
        "|--------:|---------|----------:|--------:|-----:|-------:|",
    ]
    for r in outcheck_rows:
        table2_lines.append(
            "| "
            + " | ".join(
                [
                    str(r["order_c"]),
                    str(r["variant"]),
                    "" if r["time_us"] is None else f"{float(r['time_us']):.2f}",
                    "" if r["algo_id"] is None else str(r["algo_id"]),
                    "" if r["tile_id"] is None else str(r["tile_id"]),
                    "" if r["stages_id"] is None else str(r["stages_id"]),
                ]
            )
            + " |"
        )
    md.new_paragraph("\n".join(table2_lines))

    md.create_md_file()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

