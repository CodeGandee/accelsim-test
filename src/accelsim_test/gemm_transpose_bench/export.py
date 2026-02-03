from __future__ import annotations

import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .config import DTYPES, Shape


@dataclass(frozen=True, slots=True)
class NvbenchSummary:
    tag: str
    data: dict[str, Any]


def _named_values_to_dict(items: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for it in items:
        name = it.get("name")
        typ = it.get("type")
        val = it.get("value")
        if name is None:
            continue
        if typ in {"int64", "float64"} and isinstance(val, str):
            # NVBench writes these as strings to avoid precision loss.
            out[name] = float(val) if typ == "float64" else int(val)
        else:
            out[name] = val
    return out


def _summaries_to_map(summaries: list[dict[str, Any]]) -> dict[str, NvbenchSummary]:
    out: dict[str, NvbenchSummary] = {}
    for s in summaries:
        tag = s.get("tag")
        if not isinstance(tag, str):
            continue
        data_items = s.get("data") or []
        data = _named_values_to_dict(data_items) if isinstance(data_items, list) else {}
        # Inline convenience fields:
        for k in ("name", "description", "hint", "hide"):
            if k in s:
                data[k] = s[k]
        out[tag] = NvbenchSummary(tag=tag, data=data)
    return out


def _axis_values_to_map(axis_values: list[dict[str, Any]]) -> dict[str, Any]:
    # NVBench encodes axis_values as a list of {name,type,value} objects.
    out: dict[str, Any] = {}
    for av in axis_values:
        name = av.get("name")
        typ = av.get("type")
        val = av.get("value")
        if not isinstance(name, str):
            continue
        if typ == "int64":
            out[name] = int(val) if isinstance(val, int) else int(str(val))
        elif typ == "float64":
            out[name] = float(val) if isinstance(val, float) else float(str(val))
        else:
            out[name] = val
    return out


def parse_nvbench_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_results_schema_path() -> Path:
    return _find_repo_root() / "specs" / "002-gemm-transpose-bench" / "contracts" / "results.schema.json"


def validate_results_schema(results: dict[str, Any], *, schema_path: Path | None = None) -> None:
    schema_path = _default_results_schema_path() if schema_path is None else schema_path
    schema = json.loads(schema_path.read_text())
    Draft202012Validator(schema).validate(results)


def _extract_time_seconds(summaries: dict[str, NvbenchSummary], tag: str) -> float | None:
    summ = summaries.get(tag)
    if summ is None:
        return None
    value = summ.data.get("value")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _run_capture(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except Exception:
        return None
    return out.decode(errors="replace").strip()


def _detect_cuda_toolkit_version() -> str | None:
    out = _run_capture(["nvcc", "--version"])
    if not out:
        return None
    m = re.search(r"release\\s+(\\d+\\.\\d+)", out)
    if m:
        return m.group(1)
    m = re.search(r"V(\\d+\\.\\d+\\.\\d+)", out)
    if m:
        return m.group(1)
    return None


def _detect_nvidia_driver_version() -> str | None:
    out = _run_capture(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if not out:
        return None
    return out.splitlines()[0].strip()


def normalize_nvbench_results(
    nvbench: dict[str, Any],
    *,
    git_branch: str,
    git_commit: str,
    git_dirty: bool,
    pixi_env: str,
    nvbench_source_path: str,
    artifacts_dir: Path,
    nvbench_settings: dict[str, Any],
    cublaslt_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []

    devices_by_id: dict[int, dict[str, Any]] = {}
    for dev in nvbench.get("devices", []) or []:
        if not isinstance(dev, dict):
            continue
        dev_id = dev.get("id")
        if isinstance(dev_id, int):
            devices_by_id[dev_id] = dev

    first_device_id: int | None = None

    for bench in nvbench.get("benchmarks", []):
        for st in bench.get("states", []):
            if st.get("is_skipped"):
                continue
            if first_device_id is None and isinstance(st.get("device"), int):
                first_device_id = st["device"]

            axis_map = _axis_values_to_map(st.get("axis_values", []))
            suite = str(axis_map.get("suite"))
            case = str(axis_map.get("case"))
            dtype_key = str(axis_map.get("dtype"))
            shape = Shape.from_axis_value(str(axis_map.get("shape")))
            math_mode = str(axis_map.get("math_mode", "default"))

            dtype_cfg = DTYPES.get(dtype_key)
            if dtype_cfg is None:
                # Allow forward-compat: unknown dtype keys can still be recorded.
                dtype_obj = {"a": "unknown", "b": "unknown", "c": "unknown", "compute": "unknown", "math_mode": math_mode}
            else:
                dtype_obj = {
                    "a": dtype_cfg.a,
                    "b": dtype_cfg.b,
                    "c": dtype_cfg.c,
                    "compute": dtype_cfg.compute,
                    "math_mode": dtype_cfg.math_mode,
                }

            summaries = _summaries_to_map(st.get("summaries", []))
            gpu_s = _extract_time_seconds(summaries, "nv/cold/time/gpu/mean")
            cpu_s = _extract_time_seconds(summaries, "nv/cold/time/cpu/mean")
            if gpu_s is None:
                raise ValueError("Missing required NVBench summary: nv/cold/time/gpu/mean")
            samples = None
            cold_samples = summaries.get("nv/cold/sample_size")
            if cold_samples is not None:
                v = cold_samples.data.get("value")
                if isinstance(v, int):
                    samples = v
                elif isinstance(v, str) and v.isdigit():
                    samples = int(v)

            # Verification summaries are produced by the benchmark itself.
            v_pass = summaries.get("accelsim/verification/pass")
            v_mode = summaries.get("accelsim/verification/mode")
            v_abs = summaries.get("accelsim/verification/max_abs_error")
            v_rel = summaries.get("accelsim/verification/max_rel_error")
            v_details = summaries.get("accelsim/verification/details")
            algo_summ = summaries.get("accelsim/cublaslt/algo")

            verification_status = "pass"
            if v_pass is not None:
                vp = v_pass.data.get("value")
                if isinstance(vp, int) and vp == 0:
                    verification_status = "fail"
                if isinstance(vp, str) and vp.strip() == "0":
                    verification_status = "fail"

            verification_mode = "sampled"
            if v_mode is not None and isinstance(v_mode.data.get("value"), str):
                verification_mode = v_mode.data["value"]

            def _float_from_summary(s: NvbenchSummary | None) -> float | None:
                if s is None:
                    return None
                v = s.data.get("value")
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, str):
                    try:
                        return float(v)
                    except ValueError:
                        return None
                return None

            details = None
            if v_details is not None:
                dv = v_details.data.get("value")
                if isinstance(dv, str):
                    details = dv

            algo_obj: dict[str, int] | None = None
            if algo_summ is not None:
                allowed = {
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
                extracted: dict[str, int] = {}
                for k, v in algo_summ.data.items():
                    if k not in allowed:
                        continue
                    if isinstance(v, bool):
                        continue
                    if isinstance(v, int):
                        extracted[k] = v
                    elif isinstance(v, float) and v.is_integer():
                        extracted[k] = int(v)
                if "id" in extracted:
                    algo_obj = extracted

            record = {
                "suite": suite,
                "case": case,
                "shape": {"m": shape.m, "n": shape.n, "k": shape.k},
                "dtype": dtype_obj,
                "timing": {
                    "measurement": "cold",
                    "gpu_time_ms": gpu_s * 1e3,
                    "cpu_time_ms": None if cpu_s is None else cpu_s * 1e3,
                    "samples": samples,
                    "nvbench_raw": None,
                },
                "flop_count": shape.flop_count,
                **({} if algo_obj is None else {"cublaslt": {"algo": algo_obj}}),
                "ratios": {},
                "verification": {
                    "status": verification_status,
                    "mode": verification_mode,
                    "max_abs_error": _float_from_summary(v_abs) or 0.0,
                    "max_rel_error": _float_from_summary(v_rel) or 0.0,
                    **({} if details is None else {"details": details}),
                },
                "profiling": None,
            }
            records.append(record)

    now = datetime.now(timezone.utc).isoformat()
    device_name = "unknown"
    device_sm = None
    if first_device_id is not None and first_device_id in devices_by_id:
        dev = devices_by_id[first_device_id]
        name = dev.get("name")
        if isinstance(name, str):
            device_name = name
        sm_version = dev.get("sm_version")
        if isinstance(sm_version, int):
            device_sm = f"sm_{sm_version}"

    driver_version = _detect_nvidia_driver_version()
    cuda_toolkit_version = _detect_cuda_toolkit_version()

    settings_obj: dict[str, Any] = {"nvbench": nvbench_settings}
    if cublaslt_settings is not None:
        settings_obj["cublaslt"] = cublaslt_settings

    run_obj = {
        "run_id": f"{git_commit}",
        "started_at": now,
        "finished_at": now,
        "status": "pass",
        "failure_reason": "",
        "git": {"branch": git_branch, "commit": git_commit, "dirty": git_dirty},
        "environment": {
            "platform": {"os": platform.system().lower(), "arch": platform.machine().lower()},
            "gpu": {"device_name": device_name, **({} if device_sm is None else {"sm": device_sm}), **({} if driver_version is None else {"driver_version": driver_version})},
            "cuda": {"toolkit_version": cuda_toolkit_version or "unknown"},
            "pixi_env": pixi_env,
            "nvbench": {"source_path": nvbench_source_path, "version": str(nvbench.get("meta", {}).get("version", {}).get("nvbench", {}).get("string", ""))},
        },
        "settings": settings_obj,
        "artifacts_dir": str(artifacts_dir),
    }

    # Mark run fail if any record verification fails.
    failures = [r for r in records if r["verification"]["status"] == "fail"]
    if failures:
        run_obj["status"] = "fail"
        run_obj["failure_reason"] = f"{len(failures)} record(s) failed verification"

    out = {"schema_version": "0.1.0", "run": run_obj, "records": records}
    validate_results_schema(out)
    return out


def write_results(path: Path, results: dict[str, Any]) -> None:
    path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
