from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import DTYPES, iter_dtypes, iter_shapes
from .export import validate_results_schema, write_results
from .runner import timing_run


def _expected_cases(suite: str) -> list[str]:
    if suite == "square":
        return ["AB", "ATB_view", "ABT_view", "ATB_copyA", "ABT_copyB"]
    if suite == "nonsquare_atb":
        return ["ATB_view", "ATB_copyA"]
    if suite == "nonsquare_abt":
        return ["ABT_view", "ABT_copyB"]
    raise ValueError(f"Unknown suite: {suite!r}")


def _record_key(rec: dict[str, Any]) -> tuple:
    s = rec.get("shape", {}) or {}
    d = rec.get("dtype", {}) or {}
    return (
        rec.get("suite"),
        rec.get("case"),
        int(s.get("m", 0)),
        int(s.get("n", 0)),
        int(s.get("k", 0)),
        str(d.get("a", "")),
        str(d.get("b", "")),
        str(d.get("c", "")),
        str(d.get("compute", "")),
        str(d.get("math_mode", "")),
    )


def _expected_record_keys(*, suite: str, shape_set: str, dtype: str) -> set[tuple]:
    cases = _expected_cases(suite)
    shapes = list(iter_shapes(suite, shape_set))

    if dtype == "all":
        dtype_cfgs = list(DTYPES.values())
    else:
        dtype_cfgs = list(iter_dtypes(dtype))

    expected: set[tuple] = set()
    for sh in shapes:
        for cfg in dtype_cfgs:
            for case in cases:
                expected.add(
                    (
                        suite,
                        case,
                        sh.m,
                        sh.n,
                        sh.k,
                        cfg.a,
                        cfg.b,
                        cfg.c,
                        cfg.compute,
                        cfg.math_mode,
                    )
                )
    return expected


def sweep_run(*, out_dir: Path, shape_set: str, dtype: str, algo_map: Path | None, nvbench_args: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    for suite in ("square", "nonsquare_atb", "nonsquare_abt"):
        timing_run(out_dir=out_dir, suite=suite, dtype=dtype, shape_set=shape_set, algo_map=algo_map, nvbench_args=nvbench_args)

    results_path = out_dir / "results.json"
    results = json.loads(results_path.read_text())

    expected: set[tuple] = set()
    for suite in ("square", "nonsquare_atb", "nonsquare_abt"):
        expected |= _expected_record_keys(suite=suite, shape_set=shape_set, dtype=dtype)

    actual = {_record_key(r) for r in results.get("records", []) or []}
    missing = sorted(expected - actual)
    if missing:
        results.setdefault("run", {})["status"] = "fail"
        results["run"]["failure_reason"] = f"missing {len(missing)} expected record(s)"
        # Keep the message short; details can be computed from expected/actual.
        validate_results_schema(results)
        write_results(results_path, results)

    return 0 if results.get("run", {}).get("status") == "pass" else 1
