from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Literal

from .model import CompilerSource

CompilerMode = Literal["auto", "pixi", "system"]


def resolve_nvcc(mode: CompilerMode) -> tuple[CompilerSource, list[str]]:
    """Return (compiler_source, nvcc_prefix_argv).

    - compiler_source is recorded in metadata: "pixi" or "system"
    - nvcc_prefix_argv is the command prefix used to invoke nvcc
      (e.g., ["pixi","run","-e","accelsim","nvcc"] or ["nvcc"]).
    """
    if mode == "system":
        if shutil.which("nvcc") is None:
            raise FileNotFoundError("nvcc not found on PATH (system compiler requested)")
        return "system", ["nvcc"]

    if mode == "pixi":
        if shutil.which("pixi") is None:
            raise FileNotFoundError("pixi not found on PATH (pixi compiler requested)")
        return "pixi", ["pixi", "run", "-e", "accelsim", "nvcc"]

    # auto: prefer pixi, fall back to system
    if shutil.which("pixi") is not None:
        try:
            subprocess.check_output(["pixi", "run", "-e", "accelsim", "nvcc", "--version"], stderr=subprocess.DEVNULL)
            return "pixi", ["pixi", "run", "-e", "accelsim", "nvcc"]
        except Exception:
            pass

    if shutil.which("nvcc") is not None:
        return "system", ["nvcc"]

    raise FileNotFoundError("nvcc not found (checked pixi and system)")


def _build_nvcc_exe_argv(*, nvcc_prefix: list[str], src: Path, exe_out: Path) -> list[str]:
    return [
        *nvcc_prefix,
        str(src),
        "-O2",
        "-std=c++17",
        "-lineinfo",
        "-gencode",
        "arch=compute_80,code=compute_80",
        "-o",
        str(exe_out),
    ]


def _build_nvcc_ptx_argv(*, nvcc_prefix: list[str], src: Path, ptx_out: Path) -> list[str]:
    return [
        *nvcc_prefix,
        str(src),
        "-O2",
        "-std=c++17",
        "-lineinfo",
        "-gencode",
        "arch=compute_80,code=compute_80",
        "--ptx",
        "-o",
        str(ptx_out),
    ]


def compile_cuda_program(*, mode: CompilerMode, src: Path, exe_out: Path, ptx_out: Path) -> tuple[CompilerSource, str]:
    """Compile and return (compiler_source, rendered compile command string for metadata)."""
    compiler_source, nvcc_prefix = resolve_nvcc(mode)
    exe_argv = _build_nvcc_exe_argv(nvcc_prefix=nvcc_prefix, src=src, exe_out=exe_out)
    ptx_argv = _build_nvcc_ptx_argv(nvcc_prefix=nvcc_prefix, src=src, ptx_out=ptx_out)

    subprocess.check_call(exe_argv)
    subprocess.check_call(ptx_argv)

    rendered = "\n".join([shlex.join(exe_argv), shlex.join(ptx_argv)])
    return compiler_source, rendered
