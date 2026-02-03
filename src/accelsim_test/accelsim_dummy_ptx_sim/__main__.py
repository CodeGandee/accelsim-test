from __future__ import annotations

import argparse
from typing import cast

from . import toolchain
from . import workflow

def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the dummy PTX simulation workflow."""
    parser = argparse.ArgumentParser(
        prog="accelsim_test.accelsim_dummy_ptx_sim",
        description="Compile and run a dummy CUDA program under Accel-Sim (PTX mode).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run compile + simulate workflow.")
    run.add_argument("--run-id", default=None, help="Filesystem-safe run id (default: timestamp).")
    run.add_argument("--compiler", default="auto", choices=["auto", "pixi", "system"])
    run.add_argument("--config-preset", default="sm80_a100", choices=["sm80_a100"])

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns process exit code."""
    parser = build_parser()
    ns = parser.parse_args(argv)

    if ns.cmd == "run":
        return workflow.run(run_id=ns.run_id, compiler=cast(toolchain.CompilerMode, ns.compiler), config_preset=ns.config_preset)

    raise AssertionError(f"Unhandled cmd: {ns.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
