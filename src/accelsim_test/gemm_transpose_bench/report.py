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
    compute_ratios_in_place(results)

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
        "timed_ms_AB",
        "timed_ms_ATB_view",
        "timed_ms_ABT_view",
        "timed_ms_ATB_copyA",
        "timed_ms_ABT_copyB",
        "slow_ATB_view_vs_AB",
        "slow_ABT_view_vs_AB",
        "over_ATB_copyA_vs_view",
        "over_ABT_copyB_vs_view",
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
                    _format_float(atb_view),
                    _format_float(abt_view),
                    _format_float(atb_copy),
                    _format_float(abt_copy),
                    _format_float(_safe_ratio(atb_view, ab)),
                    _format_float(_safe_ratio(abt_view, ab)),
                    _format_float(_safe_ratio(atb_copy, atb_view)),
                    _format_float(_safe_ratio(abt_copy, abt_view)),
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
        "timed_ms_ATB_view",
        "timed_ms_ATB_copyA",
        "over_ATB_copyA_vs_view",
        "timed_ms_ABT_view",
        "timed_ms_ABT_copyB",
        "over_ABT_copyB_vs_view",
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
                    _format_float(atb_copy),
                    _format_float(_safe_ratio(atb_copy, atb_view)),
                    _format_float(abt_view),
                    _format_float(abt_copy),
                    _format_float(_safe_ratio(abt_copy, abt_view)),
                    verify,
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Conclusions")
    lines.append("")
    lines.append("TBD: Populate after collecting results on target GPU(s).")
    lines.append("")

    return "\n".join(lines)


def report_run(*, out_dir: Path) -> int:
    results_path = out_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.json at {results_path}")

    results = _load_results(results_path)
    report_md = generate_report(results)
    (out_dir / "report.md").write_text(report_md + "\n")
    return 0
