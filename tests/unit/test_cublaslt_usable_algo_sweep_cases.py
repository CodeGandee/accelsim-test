from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_generate_case_specs_has_expected_matrix() -> None:
    mod = _load_module(
        "cublaslt_usable_algo_sweep",
        Path(__file__).resolve().parents[2] / "scripts" / "cublaslt_usable_algo_sweep.py",
    )

    cases = mod.generate_case_specs()
    assert len(cases) == 6

    ns = sorted({c.n for c in cases})
    variants = sorted({c.variant for c in cases})
    assert ns == [1000, 1024, 2048]
    assert variants == ["AB", "ABT_view"]


def test_case_ids_are_unique() -> None:
    mod = _load_module(
        "cublaslt_usable_algo_sweep",
        Path(__file__).resolve().parents[2] / "scripts" / "cublaslt_usable_algo_sweep.py",
    )

    cases = mod.generate_case_specs()
    ids = [mod.make_case_id(case=c) for c in cases]  # type: ignore[attr-defined]
    assert len(ids) == len(set(ids))


def test_merged_time_table_fills_na_for_missing_or_nonusable() -> None:
    mod = _load_module(
        "cublaslt_usable_algo_sweep",
        Path(__file__).resolve().parents[2] / "scripts" / "cublaslt_usable_algo_sweep.py",
    )

    candidates = [
        {"algo_id": 23, "n": 1000, "variant": "AB", "usable": False, "time_us": None},
        {"algo_id": 23, "n": 1000, "variant": "ABT_view", "usable": True, "time_us": 10.0},
        {"algo_id": 64, "n": 1000, "variant": "AB", "usable": True, "time_us": 30.0},
    ]
    rows = mod._merged_time_table(candidates=candidates, ns=[1000, 1024], variants=["AB", "ABT_view"])  # type: ignore[attr-defined]

    by_id = {r["algo_id"]: r for r in rows}
    assert by_id[23]["n1000_ab_time_us"] == "NA"
    assert by_id[23]["n1000_abt_view_time_us"] == 10.0
    assert by_id[23]["n1024_ab_time_us"] == "NA"
    assert by_id[64]["n1000_ab_time_us"] == 30.0

