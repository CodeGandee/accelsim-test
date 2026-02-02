"""CUDA GEMM transpose benchmark (Python orchestrator layer).

This package orchestrates timing/profiling runs of the C++/CUDA NVBench benchmark
binary, normalizes NVBench JSON output into a stable project-owned schema, and
generates stakeholder-facing reports.
"""

from __future__ import annotations
