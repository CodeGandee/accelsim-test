from __future__ import annotations

from collections.abc import Iterable

import attrs


@attrs.define(frozen=True, slots=True)
class Shape:
    m: int
    n: int
    k: int

    @property
    def flop_count(self) -> int:
        return 2 * self.m * self.n * self.k

    def to_axis_value(self) -> str:
        return f"{self.m}x{self.n}x{self.k}"

    @staticmethod
    def from_axis_value(v: str) -> "Shape":
        parts = v.split("x")
        if len(parts) != 3:
            raise ValueError(f"Invalid shape axis value: {v!r}")
        m_s, n_s, k_s = parts
        return Shape(m=int(m_s), n=int(n_s), k=int(k_s))


@attrs.define(frozen=True, slots=True)
class DtypeConfig:
    key: str
    a: str
    b: str
    c: str
    compute: str
    math_mode: str


SUITES: tuple[str, ...] = ("square", "nonsquare_atb", "nonsquare_abt")
CASES: tuple[str, ...] = ("AB", "ATB_view", "ABT_view", "ATB_copyA", "ABT_copyB")


DTYPES: dict[str, DtypeConfig] = {
    "fp16_fp16_fp16": DtypeConfig(key="fp16_fp16_fp16", a="fp16", b="fp16", c="fp16", compute="fp32", math_mode="default"),
    "bf16_bf16_bf16": DtypeConfig(key="bf16_bf16_bf16", a="bf16", b="bf16", c="bf16", compute="fp32", math_mode="default"),
    "fp32_fp32_fp32": DtypeConfig(key="fp32_fp32_fp32", a="fp32", b="fp32", c="fp32", compute="fp32", math_mode="default"),
    "fp32_fp32_fp32_tf32": DtypeConfig(
        key="fp32_fp32_fp32_tf32", a="fp32", b="fp32", c="fp32", compute="tf32", math_mode="tf32"
    ),
    "int8_int8_int32": DtypeConfig(key="int8_int8_int32", a="int8", b="int8", c="int32", compute="int32", math_mode="default"),
}


# Shape sets derived from /context/tasks/req-cuda-gemm-test.md.
# The orchestrator can select subsets via CLI, but the defaults must cover
# a representative set (small/safe + cache-stressing).
SHAPE_SETS: dict[str, dict[str, list[Shape]]] = {
    # Square suite: small/safe control set (dims <= 1000)
    "square_safe": {
        "square": [Shape(n, n, n) for n in (512, 768, 896, 960, 992, 1000)],
        "nonsquare_atb": [],
        "nonsquare_abt": [],
    },
    # Square-ish baseline coverage (benchmark matrix minimum).
    "square_ish": {
        "square": [Shape(n, n, n) for n in (512, 1024, 2048, 4096)],
        "nonsquare_atb": [],
        "nonsquare_abt": [],
    },
    # Cache-resident set (fits in L2 even if a transpose is materialized) for fp16/bf16.
    "cache_resident_fp16_bf16": {
        "square": [Shape(n, n, n) for n in (1024, 1536, 2048)],
        "nonsquare_atb": [Shape(4096, 1024, 1024), Shape(1024, 4096, 1024), Shape(1024, 1024, 4096)],
        "nonsquare_abt": [Shape(4096, 1024, 1024), Shape(1024, 4096, 1024), Shape(1024, 1024, 4096)],
    },
    # Cache-resident set (fits in L2) for fp32.
    "cache_resident_fp32": {
        "square": [Shape(n, n, n) for n in (768, 1024, 1280, 1536)],
        "nonsquare_atb": [],
        "nonsquare_abt": [],
    },
    # Cache-spill boundaries (examples; full sweep is selectable via config extension)
    "cache_spill_fp16_bf16": {
        "square": [Shape(2304, 2304, 2304)],
        "nonsquare_atb": [Shape(3072, 2048, 2048), Shape(8192, 1024, 1024)],
        "nonsquare_abt": [Shape(2048, 3072, 2048), Shape(8192, 1024, 1024)],
    },
    "cache_spill_fp32": {
        "square": [Shape(1664, 1664, 1664)],
        "nonsquare_atb": [Shape(2304, 1536, 1536)],
        "nonsquare_abt": [Shape(1536, 2304, 1536)],
    },
    # Non-square aspect ratio probes (dims <= 1000)
    "nonsquare_safe": {
        "square": [],
        "nonsquare_atb": [Shape(992, 256, 256), Shape(256, 992, 256), Shape(256, 256, 992), Shape(960, 320, 640)],
        "nonsquare_abt": [Shape(992, 256, 256), Shape(256, 992, 256), Shape(256, 256, 992), Shape(960, 320, 640)],
    },
    # Minimal sets intended for fast smoke runs (CI/local sanity).
    "smoke_square": {
        "square": [Shape(512, 512, 512)],
        "nonsquare_atb": [],
        "nonsquare_abt": [],
    },
    "smoke_nonsquare": {
        "square": [],
        "nonsquare_atb": [Shape(992, 256, 256)],
        "nonsquare_abt": [Shape(256, 992, 256)],
    },
    # Full sweep required by context/tasks/req-cuda-gemm-test.md (union of baseline, cache-resident,
    # cache-spill, and safety sets). Intended for a single end-to-end run.
    "full_sweep_required": {
        "square": [
            Shape(512, 512, 512),
            Shape(768, 768, 768),
            Shape(896, 896, 896),
            Shape(960, 960, 960),
            Shape(992, 992, 992),
            Shape(1000, 1000, 1000),
            Shape(1024, 1024, 1024),
            Shape(1280, 1280, 1280),
            Shape(1536, 1536, 1536),
            Shape(1664, 1664, 1664),
            Shape(2048, 2048, 2048),
            Shape(2304, 2304, 2304),
            Shape(4096, 4096, 4096),
        ],
        "nonsquare_atb": [
            Shape(992, 256, 256),
            Shape(256, 992, 256),
            Shape(256, 256, 992),
            Shape(960, 320, 640),
            Shape(4096, 1024, 1024),
            Shape(1024, 4096, 1024),
            Shape(1024, 1024, 4096),
            Shape(3072, 2048, 2048),
            Shape(8192, 1024, 1024),
            Shape(2304, 1536, 1536),
        ],
        "nonsquare_abt": [
            Shape(992, 256, 256),
            Shape(256, 992, 256),
            Shape(256, 256, 992),
            Shape(960, 320, 640),
            Shape(4096, 1024, 1024),
            Shape(1024, 4096, 1024),
            Shape(1024, 1024, 4096),
            Shape(2048, 3072, 2048),
            Shape(8192, 1024, 1024),
            Shape(1536, 2304, 1536),
        ],
    },
}


def iter_shapes(suite: str, shape_set: str) -> Iterable[Shape]:
    if shape_set == "all":
        for named in SHAPE_SETS.values():
            for s in named.get(suite, []):
                yield s
        return

    if shape_set not in SHAPE_SETS:
        raise KeyError(f"Unknown shape_set={shape_set!r}. Known: {sorted(SHAPE_SETS)}")
    yield from SHAPE_SETS[shape_set].get(suite, [])


def iter_dtypes(dtype: str) -> Iterable[DtypeConfig]:
    if dtype == "all":
        return DTYPES.values()
    if dtype not in DTYPES:
        raise KeyError(f"Unknown dtype={dtype!r}. Known: {sorted(DTYPES)}")
    return (DTYPES[dtype],)


def dtype_key_from_dtype_obj(dtype_obj: dict[str, str]) -> str:
    """Best-effort reverse mapping from normalized dtype fields to a canonical dtype key."""
    a = dtype_obj.get("a", "")
    b = dtype_obj.get("b", "")
    c = dtype_obj.get("c", "")
    compute = dtype_obj.get("compute", "")
    math_mode = dtype_obj.get("math_mode", "")

    for key, cfg in DTYPES.items():
        if (
            cfg.a == a
            and cfg.b == b
            and cfg.c == c
            and cfg.compute == compute
            and cfg.math_mode == math_mode
        ):
            return key
    return "unknown"
