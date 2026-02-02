from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_results(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _line_number(path: Path, needle: str) -> int | None:
    try:
        for i, line in enumerate(path.read_text().splitlines(), start=1):
            if needle in line:
                return i
    except OSError:
        return None
    return None


def _fmt_code_ref(path: Path, lineno: int | None) -> str:
    try:
        rel = path.resolve().relative_to(_find_repo_root().resolve())
    except Exception:
        rel = path
    if lineno is None:
        return f"`{rel}`"
    return f"`{rel}:{lineno}`"


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
        "algo_id_AB",
        "timed_ms_ATB_view",
        "algo_id_ATB_view",
        "timed_ms_ABT_view",
        "algo_id_ABT_view",
        "timed_ms_ATB_copyA",
        "algo_id_ATB_copyA",
        "timed_ms_ABT_copyB",
        "algo_id_ABT_copyB",
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
        "algo_id_ATB_view",
        "timed_ms_ATB_copyA",
        "algo_id_ATB_copyA",
        "over_ATB_copyA_vs_view",
        "timed_ms_ABT_view",
        "algo_id_ABT_view",
        "timed_ms_ABT_copyB",
        "algo_id_ABT_copyB",
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
                    _format_float(_safe_ratio(atb_copy, atb_view)),
                    _format_float(abt_view),
                    _format_int(algo_id_suite_case("nonsquare_abt", "ABT_view")),
                    _format_float(abt_copy),
                    _format_int(algo_id_suite_case("nonsquare_abt", "ABT_copyB")),
                    _format_float(_safe_ratio(abt_copy, abt_view)),
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
    lines.append("- `timed_ms_*`: Mean GPU time in milliseconds from the NVBench timing run (GEMM-only; transpose materialization is outside timing).")
    lines.append("- `algo_id_*`: cuBLASLt algorithm ID selected for that case (from `cublasLtMatmulAlgoGetHeuristic`).")
    lines.append("- `slow_*_vs_AB`: Slowdown vs baseline `AB`, computed as `timed_ms_case / timed_ms_AB`.")
    lines.append("- `over_*_vs_view`: Materialization overhead factor, computed as `timed_ms_copy / timed_ms_view`.")
    lines.append("- `verify`: `pass` if all cases in the row passed verification; otherwise `fail`.")
    lines.append("")
    lines.append("### Non-square Suite")
    lines.append("")
    lines.append("- `suite`: Suite identifier (`non_square`) summarizing both transpose directions for the same `(M,N,K)` and dtype.")
    lines.append("- `M,N,K`: Logical GEMM dimensions for `C[M,N] = A[M,K] @ B[K,N]` (FLOP-matched across non-square cases).")
    lines.append("- `dtype_pair`: Normalized dtype description `A,B->C (compute,math_mode)`.")
    lines.append("- `flop_count`: Theoretical GEMM FLOPs (`2*M*N*K`) used for row-consistency across compared cases.")
    lines.append("- `timed_ms_ATB_*`: Times for the transpose-A suite (`nonsquare_atb`): `ATB_view` and `ATB_copyA`.")
    lines.append("- `timed_ms_ABT_*`: Times for the transpose-B suite (`nonsquare_abt`): `ABT_view` and `ABT_copyB`.")
    lines.append("- `algo_id_*`: cuBLASLt algorithm ID selected for the corresponding suite/case; full per-record config lives in `results.json` under `record.cublaslt.algo`.")
    lines.append("- `over_*_vs_view`: Materialization overhead factor, computed as `timed_ms_copy / timed_ms_view` for the corresponding direction.")
    lines.append("- `verify`: `pass` if all present non-square records for the row passed verification; otherwise `fail`.")
    lines.append("")
    lines.append("Notes:")
    lines.append("- `NA` means the value is missing (e.g., a case was not run).")
    lines.append("- `flop_count` is always `2*M*N*K` even for integer cases; this report intentionally does not compute throughput columns.")
    lines.append("")

    lines.append("## Kernel Dispatch (cuBLASLt)")
    lines.append("")
    lines.append("This benchmark invokes GEMM via **cuBLASLt**; the concrete GPU kernel name is selected by cuBLASLt heuristics.")
    lines.append("")
    repo_root = _find_repo_root()
    gemm_bench_path = repo_root / "cpp" / "src" / "gemm_transpose_bench.cu"
    lt_impl_path = repo_root / "cpp" / "src" / "cublaslt_gemm.cu"

    timed_ln = _line_number(gemm_bench_path, "state.exec(nvbench::exec_tag::sync")
    cublaslt_ln = _line_number(lt_impl_path, "cublasLtMatmul(")
    heur_ln = _line_number(lt_impl_path, "cublasLtMatmulAlgoGetHeuristic")

    lines.append("- Timed region (NVBench) calls `plan.Run(...)` inside `state.exec(...)`: " + _fmt_code_ref(gemm_bench_path, timed_ln))
    lines.append("- cuBLASLt launch call is `cublasLtMatmul(...)` inside `CublasLtGemmPlan::Run(...)`: " + _fmt_code_ref(lt_impl_path, cublaslt_ln))
    lines.append("- Algo selection uses `cublasLtMatmulAlgoGetHeuristic(...)`: " + _fmt_code_ref(lt_impl_path, heur_ln))
    lines.append("")
    lines.append("Transpose form (what we mean by `*_view` vs `*_copy*`):")
    lines.append("- `*_view`: sets cuBLASLt transpose flags (`CUBLAS_OP_T`) and calls GEMM directly (no explicit transpose buffer).")
    lines.append("- `*_copy*`: explicitly transposes on-device outside the timed region, then calls GEMM with `op(N),op(N)` on the materialized buffer.")
    lines.append("")
    lines.append("Key call sites (representative):")
    lines.append("")
    lines.append("```cpp")
    lines.append("// Timed region (GEMM-only):")
    lines.append("state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {")
    lines.append("  plan.Run(launch.get_stream(), a_used, b_used, c_dev, &alpha, &beta);")
    lines.append("});")
    lines.append("```")
    lines.append("")
    lines.append("```cpp")
    lines.append("// cuBLASLt dispatch (launches the GEMM kernel):")
    lines.append("cublasLtMatmul(m_handle, m_matmul_desc, alpha, a, m_a_layout, b, m_b_layout, beta, c, m_c_layout, c, m_c_layout,")
    lines.append("              &m_algo, m_workspace, m_workspace_bytes, stream);")
    lines.append("```")
    lines.append("")
    lines.append("If you ran `profile`, each record's `profiling.command` and `*.ncu-rep` capture the concrete kernel name (e.g., `ampere_fp16_...`).")
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
