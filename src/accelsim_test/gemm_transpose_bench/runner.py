from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

from .config import CASES, SUITES, iter_dtypes, iter_shapes
from .export import normalize_nvbench_results, parse_nvbench_json, validate_results_schema, write_results


def find_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def find_benchmark_executable() -> Path:
    # Allow explicit override (useful on CI and non-standard build dirs).
    env = os.environ.get("ACCELSIM_TEST_GEMM_TRANSPOSE_BENCH_EXE")
    if env:
        p = Path(env).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"ACCELSIM_TEST_GEMM_TRANSPOSE_BENCH_EXE points to missing file: {p}")
        return p

    root = find_repo_root()
    candidates = [
        root / "cpp" / "build" / "Release" / "gemm_transpose_bench",
        root / "cpp" / "build" / "Debug" / "gemm_transpose_bench",
        root / "cpp" / "build" / "gemm_transpose_bench",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not find benchmark executable. Build it first or set ACCELSIM_TEST_GEMM_TRANSPOSE_BENCH_EXE."
    )


def nvbench_axis_args(axis: dict[str, str]) -> list[str]:
    args: list[str] = []
    for k, v in axis.items():
        args += ["--axis", f"{k}={v}"]
    return args


def _axis_list(values: list[str]) -> str:
    # NVBench CLI syntax for selecting multiple values from a string axis.
    return f"[{','.join(values)}]"


def build_nvbench_single_axis_overrides(
    *, suite: str, case: str, dtype: str, shape: str, math_mode: str | None = None
) -> list[str]:
    axis = {"suite": suite, "case": case, "dtype": dtype, "shape": shape}
    if math_mode is not None:
        axis["math_mode"] = math_mode
    return nvbench_axis_args(axis)


def build_nvbench_sweep_axis_overrides(
    *, suites: list[str], cases: list[str], dtypes: list[str], shapes: list[str] | None = None
) -> list[str]:
    axis_overrides: list[str] = []
    axis_overrides += ["--axis", f"suite={_axis_list(suites)}"]
    axis_overrides += ["--axis", f"case={_axis_list(cases)}"]
    axis_overrides += ["--axis", f"dtype={_axis_list(dtypes)}"]
    if shapes:
        axis_overrides += ["--axis", f"shape={_axis_list(shapes)}"]
    return axis_overrides


def _git_info(repo_root: Path) -> tuple[str, str, bool]:
    def _run(cmd: list[str]) -> str:
        out = subprocess.check_output(cmd, cwd=repo_root)
        return out.decode().strip()

    try:
        branch = _run(["git", "branch", "--show-current"])
        commit = _run(["git", "rev-parse", "HEAD"])
        dirty = bool(_run(["git", "status", "--porcelain=v1"]))
        return branch, commit, dirty
    except Exception:
        return "unknown", "unknown", False


def _record_key(rec: dict) -> tuple:
    s = rec.get("shape", {})
    d = rec.get("dtype", {})
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


def _merge_results(existing: dict, new: dict) -> dict:
    merged = dict(existing)
    merged_run = dict(existing.get("run", {}))

    # Keep the original started_at if present; update finished_at to the newest.
    new_run = new.get("run", {})
    if isinstance(new_run, dict):
        if "finished_at" in new_run:
            merged_run["finished_at"] = new_run["finished_at"]
        if "artifacts_dir" in new_run:
            merged_run["artifacts_dir"] = new_run["artifacts_dir"]
        # Merge failure state: any failure makes the run fail.
        if new_run.get("status") == "fail":
            merged_run["status"] = "fail"
            merged_run["failure_reason"] = new_run.get("failure_reason", "")

    merged["run"] = merged_run

    by_key: dict[tuple, dict] = {}
    for r in existing.get("records", []) or []:
        by_key[_record_key(r)] = r
    for r in new.get("records", []) or []:
        by_key[_record_key(r)] = r
    merged["records"] = list(by_key.values())

    validate_results_schema(merged)
    return merged


def timing_run(*, out_dir: Path, suite: str, dtype: str, shape_set: str, nvbench_args: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)

    bench_exe = find_benchmark_executable()
    nvbench_json_path = out_dir / "raw" / ("nvbench_timing.json" if suite == "all" else f"nvbench_timing_{suite}.json")

    suites = list(SUITES) if suite == "all" else [suite]
    dtype_keys = [d.key for d in iter_dtypes(dtype)]
    cases = list(CASES)

    # Shape axis is a string to avoid correlated m/n/k cartesian products.
    shape_values: list[str] = []
    for s in suites:
        for sh in iter_shapes(s, shape_set):
            shape_values.append(sh.to_axis_value())
    shape_values = sorted(set(shape_values))

    axis_overrides = build_nvbench_sweep_axis_overrides(suites=suites, cases=cases, dtypes=dtype_keys, shapes=shape_values)

    nvbench_args_list = shlex.split(nvbench_args)
    if not any(a in {"--devices", "--device", "-d"} for a in nvbench_args_list):
        # Default to a single device to keep exports unambiguous.
        nvbench_args_list = ["--devices", "0", *nvbench_args_list]

    cmd = [
        str(bench_exe),
        *nvbench_args_list,
        "--benchmark",
        "gemm_transpose_bench",
        "--json",
        str(nvbench_json_path),
        *axis_overrides,
    ]
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Benchmark failed with exit code {proc.returncode}: {' '.join(cmd)}")

    nvbench = parse_nvbench_json(nvbench_json_path)
    repo_root = find_repo_root()
    branch, commit, dirty = _git_info(repo_root)
    results = normalize_nvbench_results(
        nvbench,
        git_branch=branch,
        git_commit=commit,
        git_dirty=dirty,
        pixi_env=os.environ.get("PIXI_ENVIRONMENT_NAME", "unknown"),
        nvbench_source_path=str((repo_root / "extern" / "orphan" / "nvbench").resolve()),
        artifacts_dir=out_dir,
        nvbench_settings={"stopping_criterion": "stdrel", "min_time_s": 0.5, "max_noise_pct": 0.5, "min_samples": 10},
    )
    results_path = out_dir / "results.json"
    if results_path.exists():
        existing = json.loads(results_path.read_text())
        results = _merge_results(existing, results)
    write_results(results_path, results)

    # Fail overall run if any verification failed.
    return 0 if results["run"]["status"] == "pass" else 1
