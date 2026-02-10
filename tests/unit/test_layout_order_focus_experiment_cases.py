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


def test_generate_case_specs_has_expected_size_and_sweep() -> None:
    mod = _load_module(
        "layout_order_focus_experiment",
        Path(__file__).resolve().parents[2] / "scripts" / "layout_order_focus_experiment.py",
    )

    cases = mod.generate_case_specs()
    assert len(cases) == 15

    sweep = [c for c in cases if c.order_a == "row" and c.order_b == "row" and c.order_c == "col"]
    assert sorted([c.variant for c in sweep]) == ["AB", "ABT_view", "ATB_view"]


def test_case_ids_are_unique() -> None:
    mod = _load_module(
        "layout_order_focus_experiment",
        Path(__file__).resolve().parents[2] / "scripts" / "layout_order_focus_experiment.py",
    )

    cases = mod.generate_case_specs()
    ids = [
        mod.make_case_id(n=1000, dtype="int8", case=c, symmetric_inputs=False)  # type: ignore[attr-defined]
        for c in cases
    ]
    assert len(ids) == len(set(ids))
