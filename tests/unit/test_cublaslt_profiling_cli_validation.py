from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kernel_discovery_requires_repro_command(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module(
        "cublaslt_kernel_discovery",
        Path(__file__).resolve().parents[2] / "scripts" / "cublaslt_kernel_discovery.py",
    )

    monkeypatch.setattr(sys, "argv", ["cublaslt_kernel_discovery.py", "--out-dir", "tmp/x", "--case-id", "case1", "--"])
    with pytest.raises(SystemExit):
        mod.main()


def test_ncu_profile_requires_case_id_when_not_comparing(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module(
        "cublaslt_ncu_profile",
        Path(__file__).resolve().parents[2] / "scripts" / "cublaslt_ncu_profile.py",
    )

    monkeypatch.setattr(sys, "argv", ["cublaslt_ncu_profile.py", "--out-dir", "tmp/x", "--", "./repro"])
    with pytest.raises(SystemExit):
        mod.main()


def test_ncu_compare_requires_case_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module(
        "cublaslt_ncu_profile",
        Path(__file__).resolve().parents[2] / "scripts" / "cublaslt_ncu_profile.py",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cublaslt_ncu_profile.py",
            "--out-dir",
            "tmp/x",
            "--compare-abt23-vs-abt64",
            "--",
            "./repro",
        ],
    )
    with pytest.raises(SystemExit):
        mod.main()


def test_ncu_compare_rejects_forbidden_base_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module(
        "cublaslt_ncu_profile",
        Path(__file__).resolve().parents[2] / "scripts" / "cublaslt_ncu_profile.py",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cublaslt_ncu_profile.py",
            "--out-dir",
            "tmp/x",
            "--case-prefix",
            "case",
            "--compare-abt23-vs-abt64",
            "--",
            "./repro",
            "--variant",
            "ABT_view",
        ],
    )
    with pytest.raises(SystemExit):
        mod.main()
