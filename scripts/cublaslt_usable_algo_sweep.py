"""
CLI: cuBLASLt usable-algo sweep for int8 square GEMM (row-major).

This experiment focuses on the AB vs ABT_view question for the known int8
transpose-matmul behavior. It:

- Runs a fixed case matrix: N in {1000, 1024, 2048} × variant in {AB, ABT_view}
- Enumerates cuBLASLt `algo_id`s per case, selects one best config per `algo_id`,
  classifies usability via `cublasLtMatmulAlgoCheck`, and times all usable candidates.
- Writes reproducible artifacts under the chosen output directory:
  - meta.json
  - results.json / results.csv (candidate-level)
  - merged_table.csv (NA-filled table keyed by algo_id)
  - report.md (concise stakeholder-facing summary)

Example:
    pixi run python scripts/cublaslt_usable_algo_sweep.py --out-dir tmp/usable_algo_sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mdutils.mdutils import MdUtils  # type: ignore[import-untyped]

from accelsim_test.profiling.cublaslt_profiling import validate_case_id


SUMMARY_PREFIX = "ACCELSIM_USABLE_ALGO_CASE "


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


def _parse_summary_lines(output: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.startswith(SUMMARY_PREFIX):
            continue
        payload = line[len(SUMMARY_PREFIX) :].strip()
        out.append(json.loads(payload))
    return out


def find_sweep_executable() -> Path:
    env = os.environ.get("ACCELSIM_TEST_CUBLASLT_USABLE_ALGO_SWEEP_EXE")
    if env:
        p = Path(env).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"ACCELSIM_TEST_CUBLASLT_USABLE_ALGO_SWEEP_EXE points to missing file: {p}")
        return p

    root = _repo_root()
    candidates = [
        root / "cpp" / "build" / "Release" / "cublaslt_usable_algo_sweep",
        root / "cpp" / "build" / "Debug" / "cublaslt_usable_algo_sweep",
        root / "cpp" / "build" / "cublaslt_usable_algo_sweep",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not find cublaslt_usable_algo_sweep executable. Build it first or set ACCELSIM_TEST_CUBLASLT_USABLE_ALGO_SWEEP_EXE."
    )


@dataclass(frozen=True, slots=True)
class CaseSpec:
    n: int
    variant: str  # "AB" or "ABT_view"


def generate_case_specs() -> list[CaseSpec]:
    ns = [1000, 1024, 2048]
    variants = ["AB", "ABT_view"]
    return [CaseSpec(n=n, variant=v) for n in ns for v in variants]


def make_case_id(*, case: CaseSpec) -> str:
    variant = case.variant.lower()
    case_id = f"n{case.n}_int8_a_row_b_row_c_row_{variant}"
    validate_case_id(case_id)
    return case_id


def build_sweep_cmd(
    *,
    sweep_bin: Path,
    case: CaseSpec,
    iters: int,
    warmup: int,
    device: int | None,
    max_workspace_bytes: int,
    max_algo_ids: int | None,
) -> list[str]:
    cmd = [
        str(sweep_bin),
        "--n",
        str(case.n),
        "--variant",
        case.variant,
        "--iters",
        str(iters),
        "--warmup",
        str(warmup),
        "--max-workspace-bytes",
        str(max_workspace_bytes),
        "--summary-json",
    ]
    if device is not None:
        cmd += ["--device", str(device)]
    if max_algo_ids is not None:
        cmd += ["--max-algo-ids", str(max_algo_ids)]
    return cmd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a cuBLASLt usable-algo sweep for int8 AB vs ABT_view.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory root.")
    parser.add_argument("--sweep-bin", type=Path, default=None, help="Path to cublaslt_usable_algo_sweep executable.")
    parser.add_argument("--pixi-env", type=str, default="cuda13", help="Pixi env for running the sweep binary (default: cuda13).")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index (default: 0).")
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations per candidate (default: 50).")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations per candidate (default: 10).")
    parser.add_argument(
        "--max-workspace-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help="Fixed workspace policy applied during AlgoCheck (default: 64MiB).",
    )
    parser.add_argument(
        "--max-algo-ids",
        type=int,
        default=None,
        help="Optional cap on evaluated algo IDs per case (for runtime control).",
    )
    return parser.parse_args()


def _merged_time_table(
    *, candidates: list[dict[str, Any]], ns: list[int], variants: list[str]
) -> list[dict[str, Any]]:
    """
    Build a merged algo-id keyed table with NA-filled cells per (N, variant).

    The `candidates` list is expected to include per-candidate timing fields:
    - n: int
    - variant: str
    - algo_id: int
    - usable: bool
    - time_us: float | None
    """
    # key: (algo_id) -> row dict
    rows: dict[int, dict[str, Any]] = {}

    for rec in candidates:
        algo_id = rec.get("algo_id")
        n = rec.get("n")
        variant = rec.get("variant")
        if not isinstance(algo_id, int) or not isinstance(n, int) or not isinstance(variant, str):
            continue
        row = rows.setdefault(algo_id, {"algo_id": algo_id})
        col = f"n{n}_{variant.lower()}_time_us"
        if bool(rec.get("usable")):
            time_us = rec.get("time_us")
            if isinstance(time_us, (int, float)):
                row[col] = float(time_us)

    # Ensure all columns exist for all rows (fill with NA).
    for row in rows.values():
        for n in ns:
            for v in variants:
                col = f"n{n}_{v.lower()}_time_us"
                row.setdefault(col, "NA")

    # Stable ordering by algo_id.
    return [rows[k] for k in sorted(rows)]


def main() -> int:
    args = _parse_args()
    repo_root = _repo_root()

    sweep_bin = args.sweep_bin.expanduser().resolve() if args.sweep_bin is not None else find_sweep_executable()
    if not sweep_bin.exists():
        raise SystemExit(f"Sweep binary not found: {sweep_bin}")

    out_dir: Path = args.out_dir
    _ensure_dir(out_dir)
    _ensure_dir(out_dir / "cases")

    cases = generate_case_specs()

    run_meta: dict[str, Any] = {
        "tool": "cublaslt_usable_algo_sweep",
        "timestamp_utc": _utc_now_iso(),
        "out_dir": str(out_dir),
        "sweep_bin": str(sweep_bin),
        "pixi_env": args.pixi_env,
        "device": int(args.device) if args.device is not None else None,
        "iters": int(args.iters),
        "warmup": int(args.warmup),
        "max_workspace_bytes": int(args.max_workspace_bytes),
        "max_algo_ids": int(args.max_algo_ids) if args.max_algo_ids is not None else None,
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "git": _best_effort_git_state(repo_root),
        "case_matrix": [{"n": c.n, "variant": c.variant} for c in cases],
    }
    _write_json(out_dir / "meta.json", run_meta)

    case_results: list[dict[str, Any]] = []
    flat_candidates: list[dict[str, Any]] = []

    for case in cases:
        case_id = make_case_id(case=case)
        case_dir = out_dir / "cases" / case_id
        _ensure_dir(case_dir)

        sweep_cmd = build_sweep_cmd(
            sweep_bin=sweep_bin,
            case=case,
            iters=max(1, int(args.iters)),
            warmup=max(0, int(args.warmup)),
            device=int(args.device) if args.device is not None else None,
            max_workspace_bytes=max(0, int(args.max_workspace_bytes)),
            max_algo_ids=int(args.max_algo_ids) if args.max_algo_ids is not None else None,
        )
        full_cmd = ["pixi", "run", "-e", args.pixi_env, *sweep_cmd]
        cp = _run_capture(full_cmd, cwd=repo_root)
        (case_dir / "stdout.txt").write_text(cp.stdout)
        (case_dir / "stderr.txt").write_text(cp.stderr)
        (case_dir / "invocation.txt").write_text(f"cwd: {repo_root}\ncommand: {shlex.join(full_cmd)}\n")

        summary_lines = _parse_summary_lines(cp.stdout + "\n" + cp.stderr)
        summary = summary_lines[-1] if summary_lines else None
        if summary is None:
            case_results.append(
                {
                    "case_id": case_id,
                    "n": case.n,
                    "variant": case.variant,
                    "ok": False,
                    "returncode": int(cp.returncode),
                    "reason": "missing_summary_json",
                }
            )
            continue

        # Persist the raw case payload for auditability.
        _write_json(case_dir / "case.json", summary)

        summary_case: dict[str, Any] = {
            "case_id": case_id,
            "n": case.n,
            "variant": case.variant,
            "returncode": int(cp.returncode),
            **summary,
        }
        case_results.append(summary_case)

        # Flatten candidates for results.csv / merged_table.csv.
        for cand in summary.get("candidates", []) or []:
            if not isinstance(cand, dict):
                continue
            cfg = cand.get("config")
            cfg_map = cfg if isinstance(cfg, dict) else {}
            flat = {
                "case_id": case_id,
                "n": case.n,
                "variant": case.variant,
                "iters": int(args.iters),
                "warmup": int(args.warmup),
                **cfg_map,
                **{k: v for k, v in cand.items() if k != "config"},
            }
            flat_candidates.append(flat)

    # Derived per-case summary: fastest candidate and heuristic-vs-candidates.
    by_case: dict[str, list[dict[str, Any]]] = {}
    for rec in flat_candidates:
        cid = rec.get("case_id")
        if isinstance(cid, str):
            by_case.setdefault(cid, []).append(rec)

    def _as_time_us(v: Any) -> float | None:
        if isinstance(v, (int, float)):
            return float(v)
        return None

    for c in case_results:
        cid = c.get("case_id")
        if not isinstance(cid, str):
            continue
        cands = by_case.get(cid, [])
        usable = [r for r in cands if bool(r.get("usable")) and _as_time_us(r.get("time_us")) is not None]
        usable.sort(key=lambda r: float(r["time_us"]))  # type: ignore[arg-type]

        fastest = usable[0] if usable else None
        c["candidate_count"] = len(cands)
        c["usable_candidate_count"] = len(usable)
        if fastest is not None:
            c["fastest_candidate"] = {"algo_id": fastest.get("algo_id"), "time_us": fastest.get("time_us")}
        else:
            c["fastest_candidate"] = None

        heur = c.get("heuristic") if isinstance(c.get("heuristic"), dict) else {}
        heur_cfg = heur.get("config") if isinstance(heur.get("config"), dict) else {}
        heur_algo_id = heur_cfg.get("algo_id")
        heur_time_us = _as_time_us(heur.get("time_us")) if bool(heur.get("timing_ok")) else None
        c["heuristic_algo_id"] = heur_algo_id if isinstance(heur_algo_id, int) else None
        c["heuristic_time_us"] = heur_time_us

        best_candidate_time_us = _as_time_us(fastest.get("time_us")) if fastest is not None else None
        if heur_time_us is None or best_candidate_time_us is None:
            c["heuristic_is_fastest_vs_candidates"] = None
        else:
            # True if heuristic is at least as fast as the best per-algo-id candidate (within a tiny epsilon).
            c["heuristic_is_fastest_vs_candidates"] = bool(heur_time_us <= best_candidate_time_us + 1e-6)

    results_obj = {"meta": run_meta, "cases": case_results, "candidates": flat_candidates}
    _write_json(out_dir / "results.json", results_obj)

    # Candidate-level CSV export.
    csv_path = out_dir / "results.csv"
    fieldnames = [
        "case_id",
        "n",
        "variant",
        "algo_id",
        "usable",
        "time_us",
        "required_workspace_bytes",
        "check_status",
        "check_status_code",
        "tile_id",
        "stages_id",
        "splitk_num",
        "reduction_scheme",
        "cta_swizzling",
        "custom_option",
        "inner_shape_id",
        "cluster_shape_id",
        "waves_count",
        "notes",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in flat_candidates:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    # Merged NA-filled table keyed by algo_id (time only).
    ns = [1000, 1024, 2048]
    variants = ["AB", "ABT_view"]
    merged_rows = _merged_time_table(candidates=flat_candidates, ns=ns, variants=variants)
    merged_path = out_dir / "merged_table.csv"
    merged_fields = ["algo_id"] + [f"n{n}_{v.lower()}_time_us" for n in ns for v in variants]
    with merged_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fields)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow({k: row.get(k, "NA") for k in merged_fields})

    # Markdown report: keep it concise and point to raw artifacts for full details.
    md = MdUtils(file_name=str(out_dir / "report"), title="cuBLASLt usable-algo sweep (int8 square, row-major)")
    md.new_header(level=1, title="Run Metadata")
    md.new_paragraph(f"`{run_meta['timestamp_utc']}`")
    md.new_paragraph(f"Output: `{out_dir}`")
    md.new_paragraph(f"Sweep bin: `{sweep_bin}` (pixi env: `{args.pixi_env}`)")
    md.new_paragraph(f"Workspace policy: `max_workspace_bytes={int(args.max_workspace_bytes)}`")
    md.new_paragraph(f"Timing policy: `warmup={int(args.warmup)}`, `iters={int(args.iters)}`")

    # By-N comparison (top-K).
    md.new_header(level=1, title="Results (AB vs ABT_view)")
    top_k = 30

    # Executive summary: highlight algo_id=23 and fastest-per-case.
    md.new_header(level=2, title="Executive Summary")

    def _case_summary(n: int, variant: str) -> dict[str, Any] | None:
        for c in case_results:
            if c.get("n") == n and c.get("variant") == variant:
                return c if isinstance(c, dict) else None
        return None

    def _fmt_time(v: Any) -> str:
        if v is None or v == "NA":
            return "NA"
        try:
            return f"{float(v):.2f}us"
        except Exception:
            return "NA"

    summary_lines: list[str] = []
    def _usable_times(*, n: int, variant: str) -> list[tuple[int, float]]:
        rows = [
            r
            for r in flat_candidates
            if r.get("n") == n
            and r.get("variant") == variant
            and bool(r.get("usable"))
            and isinstance(r.get("algo_id"), int)
            and isinstance(r.get("time_us"), (int, float))
        ]
        return sorted([(int(r["algo_id"]), float(r["time_us"])) for r in rows], key=lambda t: t[1])

    def _rank_of(*, n: int, variant: str, algo_id: int) -> tuple[int, int] | None:
        times = _usable_times(n=n, variant=variant)
        for i, (aid, _t) in enumerate(times):
            if aid == algo_id:
                return (i + 1, len(times))
        return None

    for n in ns:
        ab = _case_summary(n, "AB") or {}
        abt = _case_summary(n, "ABT_view") or {}
        fastest_ab = (ab.get("fastest_candidate") or {}) if isinstance(ab.get("fastest_candidate"), dict) else {}
        fastest_abt = (abt.get("fastest_candidate") or {}) if isinstance(abt.get("fastest_candidate"), dict) else {}
        best_ab_t = _as_time_us(fastest_ab.get("time_us"))
        best_abt_t = _as_time_us(fastest_abt.get("time_us"))
        speedup = (best_ab_t / best_abt_t) if (best_ab_t is not None and best_abt_t is not None and best_abt_t != 0) else None

        speedup_msg = ""
        if speedup is not None:
            # speedup = AB / ABT_view
            if abs(speedup - 1.0) <= 0.01:
                speedup_msg = " ⇒ ≈ equal (best-of-case)."
            elif speedup > 1.0:
                speedup_msg = f" ⇒ ABT_view is ~{speedup:.2f}× faster (best-of-case)."
            else:
                speedup_msg = f" ⇒ AB is ~{(1.0 / speedup):.2f}× faster (best-of-case)."

        summary_lines.append(
            f"- N={n}: best AB algo_id={fastest_ab.get('algo_id','NA')} ({_fmt_time(best_ab_t)}), "
            f"best ABT_view algo_id={fastest_abt.get('algo_id','NA')} ({_fmt_time(best_abt_t)})"
            + speedup_msg
        )

        row23 = next((r for r in merged_rows if r.get("algo_id") == 23), None)
        if row23 is None:
            continue
        ab23 = row23.get(f"n{n}_ab_time_us")
        abt23 = row23.get(f"n{n}_abt_view_time_us")
        rank_abt = _rank_of(n=n, variant="ABT_view", algo_id=23)
        rank_ab = _rank_of(n=n, variant="AB", algo_id=23)
        rank_s = (
            "NA"
            if rank_abt is None
            else f"{rank_abt[0]}/{rank_abt[1]}"
        )
        summary_lines.append(
            f"  - algo_id=23: AB={_fmt_time(ab23)} (rank {('NA' if rank_ab is None else f'{rank_ab[0]}/{rank_ab[1]}')}), "
            f"ABT_view={_fmt_time(abt23)} (rank {rank_s})"
        )
    md.new_paragraph("\n".join(summary_lines))

    def _time_key(v: Any) -> float:
        try:
            return float(v)
        except Exception:
            return float("inf")

    for n in ns:
        md.new_header(level=2, title=f"N={n}")
        n_rows = [
            r
            for r in merged_rows
            if r.get(f"n{n}_ab_time_us") != "NA" or r.get(f"n{n}_abt_view_time_us") != "NA"
        ]

        def _best_time(row: dict[str, Any]) -> float:
            return min(_time_key(row.get(f"n{n}_ab_time_us")), _time_key(row.get(f"n{n}_abt_view_time_us")))

        n_rows.sort(key=_best_time)
        # Always include algo_id=23 row if present.
        algo23 = next((r for r in merged_rows if r.get("algo_id") == 23), None)
        display = list(n_rows[:top_k])
        if algo23 is not None and algo23 not in display:
            display.append(algo23)

        table_lines = [
            "| algo_id | AB time_us | ABT_view time_us | ABT/AB |",
            "|------:|----------:|---------------:|------:|",
        ]
        for row in display:
            ab = row.get(f"n{n}_ab_time_us")
            abt = row.get(f"n{n}_abt_view_time_us")
            ab_f = _time_key(ab)
            abt_f = _time_key(abt)
            ratio = (abt_f / ab_f) if (ab_f != float("inf") and ab_f != 0 and abt_f != float("inf")) else None

            def _fmt(x: Any) -> str:
                if x == "NA" or x is None:
                    return "NA"
                try:
                    return f"{float(x):.2f}"
                except Exception:
                    return "NA"

            ratio_s = "NA" if ratio is None else f"{ratio:.3f}"
            table_lines.append(
                "| "
                + " | ".join([str(row.get("algo_id", "")), _fmt(ab), _fmt(abt), ratio_s])
                + " |"
            )
        md.new_paragraph("\n".join(table_lines))
        md.new_paragraph(f"Full merged table: `{merged_path.name}` (all algo_id rows).")

    md.create_md_file()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
