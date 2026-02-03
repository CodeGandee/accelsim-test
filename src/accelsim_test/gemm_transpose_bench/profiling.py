from __future__ import annotations

import json
import shutil
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .config import dtype_key_from_dtype_obj
from .runner import build_nvbench_single_axis_overrides, find_benchmark_executable


def _load_results(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_results(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def profile_run(*, out_dir: Path, ncu_args: str, nvbench_args: str) -> int:
    results_path = out_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.json at {results_path}")

    ncu_path = shutil.which("ncu")
    if ncu_path is None:
        raise FileNotFoundError("ncu not found on PATH (expected via `pixi global` or system install).")

    bench_exe = find_benchmark_executable()
    results = _load_results(results_path)

    profiles_dir = out_dir / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []

    nvbench_args_list = shlex.split(nvbench_args)
    if not any(a in {"--devices", "--device", "-d"} for a in nvbench_args_list):
        nvbench_args_list = ["--devices", "0", *nvbench_args_list]

    for rec in results.get("records", []):
        suite = rec["suite"]
        case = rec["case"]
        shape = rec["shape"]
        dtype = rec["dtype"]
        shape_axis = f"{shape['m']}x{shape['n']}x{shape['k']}"
        dtype_key = dtype_key_from_dtype_obj({k: str(v) for k, v in dtype.items()})
        if dtype_key == "unknown":
            raise ValueError(f"Cannot infer dtype key for profiling from record dtype: {dtype}")

        # Store artifacts in an attributable directory.
        rec_dir = profiles_dir / f"{suite}" / f"{dtype_key}" / f"{shape_axis}" / f"{case}"
        rec_dir.mkdir(parents=True, exist_ok=True)
        rep_path = rec_dir / "profile.ncu-rep"

        axis_overrides = build_nvbench_single_axis_overrides(
            suite=suite,
            case=case,
            shape=shape_axis,
            dtype=dtype_key,
            math_mode=str(dtype.get("math_mode", "default")),
        )

        cmd = [
            ncu_path,
            "--export",
            str(rep_path),
            *shlex.split(ncu_args),
            "--",
            str(bench_exe),
            *nvbench_args_list,
            "--benchmark",
            "gemm_transpose_bench",
            *axis_overrides,
        ]

        proc = subprocess.run(cmd, check=False)

        ncu_rep = str(rep_path.relative_to(out_dir)) if rep_path.is_relative_to(out_dir) else str(rep_path)
        rec["profiling"] = {"ncu_rep": ncu_rep, "command": " ".join(cmd)}

        if proc.returncode != 0:
            failures.append(f"{suite}/{dtype_key}/{shape_axis}/{case}: ncu exit {proc.returncode}")
        if not rep_path.exists():
            failures.append(f"{suite}/{dtype_key}/{shape_axis}/{case}: missing {rep_path}")

    _write_results(results_path, results)
    return 0 if not failures else 1
