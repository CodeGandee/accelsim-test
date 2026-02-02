from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_results(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _format_float(v: float | None) -> str:
    if v is None:
        return "NA"
    return f"{v:.3f}"


def _format_int(v: int | None) -> str:
    if v is None:
        return "NA"
    return str(v)


def _dtype_key(dtype: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(dtype.get("a", "unknown")),
        str(dtype.get("b", "unknown")),
        str(dtype.get("c", "unknown")),
        str(dtype.get("compute", "unknown")),
        str(dtype.get("math_mode", "unknown")),
    )


def _dtype_label(dtype: dict[str, Any]) -> str:
    a, b, c, compute, math_mode = _dtype_key(dtype)
    return f"{a},{b}->{c} ({compute},{math_mode})"


def _group_key(rec: dict[str, Any]) -> tuple[str, int, int, int, tuple[str, str, str, str, str]]:
    s = rec["shape"]
    return (rec["suite"], s["m"], s["n"], s["k"], _dtype_key(rec["dtype"]))


def _ensure_flop_consistency(records: list[dict[str, Any]]) -> None:
    flops = {r["flop_count"] for r in records}
    if len(flops) != 1:
        raise ValueError("flop_count mismatch within report row")


def _safe_ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return num / den


def compute_ratios_in_place(results: dict[str, Any]) -> None:
    """Compute per-record ratio fields required by reporting/export.

    - Square suite: ratio_to_ab (all cases), ratio_copy_over_view (copy cases).
    - Non-square suites: ratio_copy_over_view (copy cases only).
    """
    by_cfg: dict[tuple[str, int, int, int, tuple[str, str, str, str, str]], list[dict[str, Any]]] = {}
    for r in results.get("records", []):
        by_cfg.setdefault(_group_key(r), []).append(r)

    for (_suite, _m, _n, _k, _dtype), recs in by_cfg.items():
        _ensure_flop_consistency(recs)
        by_case = {r["case"]: r for r in recs}

        def time_ms(case_name: str) -> float | None:
            rec = by_case.get(case_name)
            if rec is None:
                return None
            return rec.get("timing", {}).get("gpu_time_ms")

        ab_t = time_ms("AB")
        atb_v_t = time_ms("ATB_view")
        abt_v_t = time_ms("ABT_view")

        for r in recs:
            t = r.get("timing", {}).get("gpu_time_ms")
            existing_ratios = r.get("ratios")
            ratios: dict[str, Any] = existing_ratios if isinstance(existing_ratios, dict) else {}

            if ab_t is not None and t is not None:
                ratios["ratio_to_ab"] = float(t / ab_t) if ab_t != 0 else 0.0

            if r["case"] == "ATB_copyA":
                v = _safe_ratio(t, atb_v_t)
                if v is not None:
                    ratios["ratio_copy_over_view"] = float(v)
            if r["case"] == "ABT_copyB":
                v = _safe_ratio(t, abt_v_t)
                if v is not None:
                    ratios["ratio_copy_over_view"] = float(v)

            r["ratios"] = ratios


def generate_report(results: dict[str, Any]) -> str:
    records = list(results.get("records", []))

    lines: list[str] = []
    lines.append("# GEMM Transpose Benchmark Report")
    lines.append("")
    run = results.get("run", {})
    lines.append(f"- Branch: `{run.get('git', {}).get('branch', '')}`")
    lines.append(f"- Commit: `{run.get('git', {}).get('commit', '')}`")
    lines.append(f"- Status: `{run.get('status', '')}`")
    lines.append("")

    # Table A: Square suite summary (all executed rows by default).
    lines.append("## Square Suite")
    lines.append("")
    header = [
        "suite",
        "N",
        "dtype_pair",
        "flop_count",
        "A@B(ms)",
        "A@B(algo_id)",
        "A.T@B(ms)",
        "A.T@B(algo_id)",
        "A@B.T(ms)",
        "A@B.T(algo_id)",
        "copy(A.T)@B(ms)",
        "copy(A.T)@B(algo_id)",
        "A@copy(B.T)(ms)",
        "A@copy(B.T)(algo_id)",
        "verify",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    square_groups: dict[tuple[int, tuple[str, str, str, str, str]], list[dict[str, Any]]] = {}
    for r in records:
        if r.get("suite") != "square":
            continue
        s = r.get("shape", {})
        m = int(s.get("m"))
        n = int(s.get("n"))
        k = int(s.get("k"))
        if not (m == n == k):
            continue
        square_groups.setdefault((m, _dtype_key(r["dtype"])), []).append(r)

    for (n, _dtype_tuple), recs in sorted(square_groups.items()):
        _ensure_flop_consistency(recs)
        by_case_name = {r["case"]: r for r in recs}

        def time_ms_case(case_name: str) -> float | None:
            rec = by_case_name.get(case_name)
            if rec is None:
                return None
            return rec.get("timing", {}).get("gpu_time_ms")

        def algo_id_case(case_name: str) -> int | None:
            rec = by_case_name.get(case_name)
            if rec is None:
                return None
            algo = rec.get("cublaslt", {}).get("algo", {})
            v = algo.get("id")
            return int(v) if isinstance(v, int) else None

        ab = time_ms_case("AB")
        atb_view = time_ms_case("ATB_view")
        abt_view = time_ms_case("ABT_view")
        atb_copy = time_ms_case("ATB_copyA")
        abt_copy = time_ms_case("ABT_copyB")

        verify = "fail" if any(r["verification"]["status"] == "fail" for r in recs) else "pass"
        flop_count = recs[0]["flop_count"]
        dtype_label = _dtype_label(recs[0]["dtype"])

        lines.append(
            "| "
            + " | ".join(
                [
                    "square",
                    str(n),
                    f"`{dtype_label}`",
                    str(flop_count),
                    _format_float(ab),
                    _format_int(algo_id_case("AB")),
                    _format_float(atb_view),
                    _format_int(algo_id_case("ATB_view")),
                    _format_float(abt_view),
                    _format_int(algo_id_case("ABT_view")),
                    _format_float(atb_copy),
                    _format_int(algo_id_case("ATB_copyA")),
                    _format_float(abt_copy),
                    _format_int(algo_id_case("ABT_copyB")),
                    verify,
                ]
            )
            + " |"
        )

    lines.append("")
    # Table B: Non-square suite summary (FLOP-matched; transpose-A and transpose-B).
    lines.append("## Non-square Suite")
    lines.append("")
    header2 = [
        "suite",
        "M",
        "N",
        "K",
        "dtype_pair",
        "flop_count",
        "A.T@B(ms)",
        "A.T@B(algo_id)",
        "copy(A.T)@B(ms)",
        "copy(A.T)@B(algo_id)",
        "A@B.T(ms)",
        "A@B.T(algo_id)",
        "A@copy(B.T)(ms)",
        "A@copy(B.T)(algo_id)",
        "verify",
    ]
    lines.append("| " + " | ".join(header2) + " |")
    lines.append("|" + "|".join(["---"] * len(header2)) + "|")

    nonsquare_groups: dict[tuple[int, int, int, tuple[str, str, str, str, str]], list[dict[str, Any]]] = {}
    for r in records:
        if r.get("suite") not in {"nonsquare_atb", "nonsquare_abt"}:
            continue
        s = r.get("shape", {})
        key = (int(s.get("m")), int(s.get("n")), int(s.get("k")), _dtype_key(r["dtype"]))
        nonsquare_groups.setdefault(key, []).append(r)

    for (m, n, k, _dtype_tuple), recs in sorted(nonsquare_groups.items()):
        by_suite_case: dict[tuple[str, str], dict[str, Any]] = {(r["suite"], r["case"]): r for r in recs}
        flop_count = 2 * m * n * k
        if any(r["flop_count"] != flop_count for r in recs):
            raise ValueError("flop_count mismatch within non-square report row")

        def time_ms_suite_case(suite: str, case: str) -> float | None:
            rec = by_suite_case.get((suite, case))
            if rec is None:
                return None
            return rec.get("timing", {}).get("gpu_time_ms")

        def algo_id_suite_case(suite: str, case: str) -> int | None:
            rec = by_suite_case.get((suite, case))
            if rec is None:
                return None
            algo = rec.get("cublaslt", {}).get("algo", {})
            v = algo.get("id")
            return int(v) if isinstance(v, int) else None

        atb_view = time_ms_suite_case("nonsquare_atb", "ATB_view")
        atb_copy = time_ms_suite_case("nonsquare_atb", "ATB_copyA")
        abt_view = time_ms_suite_case("nonsquare_abt", "ABT_view")
        abt_copy = time_ms_suite_case("nonsquare_abt", "ABT_copyB")

        verify = "fail" if any(r["verification"]["status"] == "fail" for r in recs) else "pass"
        dtype_label = _dtype_label(recs[0]["dtype"])

        lines.append(
            "| "
            + " | ".join(
                [
                    "non_square",
                    str(m),
                    str(n),
                    str(k),
                    f"`{dtype_label}`",
                    str(flop_count),
                    _format_float(atb_view),
                    _format_int(algo_id_suite_case("nonsquare_atb", "ATB_view")),
                    _format_float(atb_copy),
                    _format_int(algo_id_suite_case("nonsquare_atb", "ATB_copyA")),
                    _format_float(abt_view),
                    _format_int(algo_id_suite_case("nonsquare_abt", "ABT_view")),
                    _format_float(abt_copy),
                    _format_int(algo_id_suite_case("nonsquare_abt", "ABT_copyB")),
                    verify,
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Column Definitions")
    lines.append("")
    lines.append("### Square Suite")
    lines.append("")
    lines.append("- `suite`: Suite identifier (`square`).")
    lines.append("- `N`: Matrix size for square shapes (`M=N=K=N`).")
    lines.append("- `dtype_pair`: Normalized dtype description `A,B->C (compute,math_mode)`.")
    lines.append("- `flop_count`: Theoretical GEMM FLOPs (`2*N*N*N`).")
    lines.append("- `A@B(ms)` etc: Mean GPU time in milliseconds from the NVBench timing run (GEMM-only; transpose materialization is outside timing).")
    lines.append("- `A@B(algo_id)` etc: cuBLASLt algorithm ID used for that case (heuristic-selected or pinned via algo-map).")
    lines.append("- `verify`: `pass` if all cases in the row passed verification; otherwise `fail`.")
    lines.append("")
    lines.append("### Non-square Suite")
    lines.append("")
    lines.append("- `suite`: Suite identifier (`non_square`) summarizing both transpose directions for the same `(M,N,K)` and dtype.")
    lines.append("- `M,N,K`: Logical GEMM dimensions for `C[M,N] = A[M,K] @ B[K,N]` (FLOP-matched across non-square cases).")
    lines.append("- `dtype_pair`: Normalized dtype description `A,B->C (compute,math_mode)`.")
    lines.append("- `flop_count`: Theoretical GEMM FLOPs (`2*M*N*K`) used for row-consistency across compared cases.")
    lines.append("- `A.T@B(ms)` / `copy(A.T)@B(ms)`: Times for transpose-A suite (`nonsquare_atb`).")
    lines.append("- `A@B.T(ms)` / `A@copy(B.T)(ms)`: Times for transpose-B suite (`nonsquare_abt`).")
    lines.append("- `...(algo_id)`: cuBLASLt algorithm ID used for that record; full per-record config lives in `results.json` under `record.cublaslt.algo`.")
    lines.append("- `verify`: `pass` if all present non-square records for the row passed verification; otherwise `fail`.")
    lines.append("")
    lines.append("Notes:")
    lines.append("- `NA` means the value is missing (e.g., a case was not run).")
    lines.append("- `flop_count` is always `2*M*N*K` even for integer cases; this report intentionally does not compute throughput columns.")
    lines.append("")
    lines.append("## Conclusions")
    lines.append("")
    lines.append("TBD: Populate after collecting results on target GPU(s).")
    lines.append("")

    return "\n".join(lines)


def generate_all_timings(results: dict[str, Any]) -> str:
    records = list(results.get("records", []))

    def _algo_field(rec: dict[str, Any], key: str) -> int | None:
        algo = rec.get("cublaslt", {}).get("algo", {})
        v = algo.get(key)
        return int(v) if isinstance(v, int) else None

    def _time_ms(rec: dict[str, Any]) -> float | None:
        return rec.get("timing", {}).get("gpu_time_ms")

    def _samples(rec: dict[str, Any]) -> int | None:
        v = rec.get("timing", {}).get("samples")
        return int(v) if isinstance(v, int) else None

    lines: list[str] = []
    lines.append("# GEMM Transpose Benchmark Timings (All Records)")
    lines.append("")
    run = results.get("run", {})
    lines.append(f"- Branch: `{run.get('git', {}).get('branch', '')}`")
    lines.append(f"- Commit: `{run.get('git', {}).get('commit', '')}`")
    lines.append(f"- Status: `{run.get('status', '')}`")
    lines.append("")

    header = [
        "suite",
        "case",
        "M",
        "N",
        "K",
        "dtype_pair",
        "time(ms)",
        "samples",
        "verify",
        "algo_id",
        "tile_id",
        "splitk_num",
        "stages_id",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    def _sort_key(rec: dict[str, Any]) -> tuple:
        s = rec.get("shape", {})
        return (
            str(rec.get("suite", "")),
            str(rec.get("case", "")),
            int(s.get("m", 0)),
            int(s.get("n", 0)),
            int(s.get("k", 0)),
            _dtype_key(rec.get("dtype", {})),
        )

    for r in sorted(records, key=_sort_key):
        s = r.get("shape", {})
        m = int(s.get("m", 0))
        n = int(s.get("n", 0))
        k = int(s.get("k", 0))
        dtype_label = _dtype_label(r.get("dtype", {}))
        verify = str(r.get("verification", {}).get("status", "unknown"))

        lines.append(
            "| "
            + " | ".join(
                [
                    str(r.get("suite", "")),
                    str(r.get("case", "")),
                    str(m),
                    str(n),
                    str(k),
                    f"`{dtype_label}`",
                    _format_float(_time_ms(r)),
                    _format_int(_samples(r)),
                    verify,
                    _format_int(_algo_field(r, "id")),
                    _format_int(_algo_field(r, "tile_id")),
                    _format_int(_algo_field(r, "splitk_num")),
                    _format_int(_algo_field(r, "stages_id")),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("Notes:")
    lines.append("- `time(ms)` is the NVBench cold GPU mean time for the record.")
    lines.append("- `samples` is `nv/cold/sample_size` for the record.")
    lines.append("- `algo_id` and related fields come from `record.cublaslt.algo` in `results.json`.")
    lines.append("")
    return "\n".join(lines)


def write_stakeholder_report_template(*, out_dir: Path, results: dict[str, Any]) -> None:
    path = out_dir / "stakeholder_report.md"
    if path.exists():
        return

    run = results.get("run", {})
    git = run.get("git", {})
    env = run.get("environment", {})
    gpu = env.get("gpu", {})
    cuda = env.get("cuda", {})
    nvbench = env.get("nvbench", {})
    nvbench_settings = run.get("settings", {}).get("nvbench", {})

    path.write_text(
        "\n".join(
            [
                "# GEMM Transpose Sweep — Stakeholder Report",
                "",
                "- Run ID: `TBD`",
                f"- Git: `{git.get('branch', '')}` @ `{git.get('commit', '')}` (dirty: `{git.get('dirty', '')}`)",
                f"- GPU: `{gpu.get('device_name', '')}` ({gpu.get('sm', '')}), driver `{gpu.get('driver_version', '')}`",
                f"- CUDA toolkit: `{cuda.get('toolkit_version', '')}`",
                f"- Pixi env: `{env.get('pixi_env', '')}`",
                f"- NVBench: `{nvbench.get('version', '')}` (source: `{nvbench.get('source_path', '')}`)",
                f"- NVBench settings: `{nvbench_settings}`",
                "",
                "## Executive Summary",
                "",
                "- TBD",
                "",
                "## Key Results (curated)",
                "",
                "- TBD (select representative rows from `all_timings.md`)",
                "",
                "## Analysis",
                "",
                "- TBD (algorithm selection boundaries, stability, etc.)",
                "",
                "## Correctness & Verification (critical path)",
                "",
                "- TBD (include code references for timed region, cuBLASLt call, view vs copy semantics, and verification approach)",
                "",
                "## Reproduction",
                "",
                "- Build: `pixi run -e cuda13 gemm-transpose-build`",
                "- Sweep: `pixi run -e cuda13 gemm-transpose sweep --out-dir <abs_out_dir>`",
                "- Reporting: `pixi run -e cuda13 gemm-transpose report --out-dir <abs_out_dir>`",
                "",
                "## Appendix",
                "",
                "- Generated summary: `report.md`",
                "- Full timing table: `all_timings.md`",
                "- Normalized export: `results.json`",
                "- Raw NVBench JSON: `raw/`",
                "",
            ]
        )
        + "\n"
    )


def report_run(*, out_dir: Path) -> int:
    results_path = out_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.json at {results_path}")

    results = _load_results(results_path)
    report_md = generate_report(results)
    (out_dir / "report.md").write_text(report_md + "\n")
    all_md = generate_all_timings(results)
    (out_dir / "all_timings.md").write_text(all_md + "\n")
    write_stakeholder_report_template(out_dir=out_dir, results=results)
    return 0 if results.get("run", {}).get("status") == "pass" else 1


def all_timings_run(*, out_dir: Path) -> int:
    results_path = out_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.json at {results_path}")

    results = _load_results(results_path)
    all_md = generate_all_timings(results)
    (out_dir / "all_timings.md").write_text(all_md + "\n")
    return 0 if results.get("run", {}).get("status") == "pass" else 1
