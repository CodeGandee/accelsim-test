from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import Shape, dtype_key_from_dtype_obj


ALGO_MAP_SCHEMA = "accelsim_test.cublaslt_algo_map/v1"

ALGO_FIELDS: set[str] = {
    "id",
    "tile_id",
    "splitk_num",
    "reduction_scheme",
    "cta_swizzling",
    "custom_option",
    "stages_id",
    "inner_shape_id",
    "cluster_shape_id",
    "required_workspace_bytes",
    "waves_count",
}


def algo_map_key(*, suite: str, case: str, dtype_key: str, shape: Shape) -> str:
    return f"{suite}|{case}|{dtype_key}|{shape.to_axis_value()}"


def build_algo_map(results: dict[str, Any], *, suites: Iterable[str] | None = None) -> dict[str, Any]:
    suite_filter = set(suites) if suites is not None else None

    algos: dict[str, dict[str, int]] = {}
    for rec in results.get("records", []) or []:
        suite = str(rec.get("suite", ""))
        if suite_filter is not None and suite not in suite_filter:
            continue
        case = str(rec.get("case", ""))
        shape_obj = rec.get("shape", {}) or {}
        dtype_obj = rec.get("dtype", {}) or {}

        dtype_key = dtype_key_from_dtype_obj(dtype_obj)
        if dtype_key == "unknown":
            raise ValueError(f"Could not map dtype object to dtype key: {dtype_obj!r}")

        shape = Shape(m=int(shape_obj.get("m", 0)), n=int(shape_obj.get("n", 0)), k=int(shape_obj.get("k", 0)))
        key = algo_map_key(suite=suite, case=case, dtype_key=dtype_key, shape=shape)

        cublaslt = rec.get("cublaslt")
        if not isinstance(cublaslt, dict):
            cublaslt = {}
        algo = cublaslt.get("algo")
        if not isinstance(algo, dict):
            raise ValueError(f"Missing cublaslt.algo for record key={key}")

        extracted: dict[str, int] = {}
        for k in ALGO_FIELDS:
            v = algo.get(k)
            if isinstance(v, bool):
                continue
            if isinstance(v, int):
                extracted[k] = v
            elif isinstance(v, float) and v.is_integer():
                extracted[k] = int(v)

        if "id" not in extracted:
            raise ValueError(f"Missing algo.id for record key={key}")

        algos[key] = extracted

    now = datetime.now(timezone.utc).isoformat()
    run = results.get("run", {}) if isinstance(results.get("run", {}), dict) else {}
    return {
        "schema": ALGO_MAP_SCHEMA,
        "generated_at": now,
        "source": {
            "run_id": str(run.get("run_id", "")),
            "git": run.get("git", {}),
            "artifacts_dir": str(run.get("artifacts_dir", "")),
        },
        "algos": dict(sorted(algos.items())),
    }


def algo_map_run(*, results_path: Path, out_path: Path, suite: str) -> int:
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.json at {results_path}")

    results = json.loads(results_path.read_text())
    suites = None if suite == "all" else [suite]
    algo_map = build_algo_map(results, suites=suites)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(algo_map, indent=2, sort_keys=True) + "\n")
    return 0
