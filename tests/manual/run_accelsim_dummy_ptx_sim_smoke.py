from __future__ import annotations

import subprocess
from datetime import datetime, timezone

from accelsim_test.accelsim_dummy_ptx_sim import paths, prereqs, workflow


def _default_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"smoke-{ts}"


def main() -> int:
    repo_root = paths.find_repo_root()
    run_id = _default_run_id()

    checks = prereqs.check_all(repo_root=repo_root, compiler="auto", config_preset="sm80_a100")
    if any(c.status == "fail" for c in checks):
        print("Skipping: missing prerequisites")
        print(workflow.format_prereq_failures(checks))
        return 0

    cmd = [
        "pixi",
        "run",
        "-e",
        "accelsim",
        "python",
        "-m",
        "accelsim_test.accelsim_dummy_ptx_sim",
        "run",
        "--run-id",
        run_id,
        "--compiler",
        "auto",
        "--config-preset",
        "sm80_a100",
    ]
    rc = subprocess.call(cmd, cwd=repo_root)
    print(f"Artifacts: {paths.run_artifacts_dir(repo_root=repo_root, run_id=run_id)}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
