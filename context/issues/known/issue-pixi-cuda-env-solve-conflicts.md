# Issue: Pixi CUDA Environment Solve Conflicts (nvidia vs conda-forge, cuDNN, strict channel priority)

## Date

2026-01-30

## Summary

Creating a project-local CUDA build environment with Pixi can fail to solve when mixing `conda-forge` and `nvidia` channels, especially for CUDA 13.x + cuDNN, due to strict channel priority and similarly named packages with incompatible dependency models.

## Symptoms

- `pixi install -e cuda13` fails with solver errors mentioning one or more of:
  - `cuda-toolkit 13.0.*` excluded because candidate not in requested channel / strict channel priority.
  - `cuda-version 13.0` excluded because not using option from `https://conda.anaconda.org/nvidia/`.
  - `cudnn` pulling in `cudatoolkit` (legacy metapackage) and only allowing CUDA <= 11.x, conflicting with `cuda-toolkit 13.x`.
- `pixi install` unexpectedly fails while solving a different environment (e.g., default/accelsim) even if you only intend to build/install a new CUDA environment, because environments share a solve-group by default.
- Confusing CLI discovery: `pixi env ...` / `pixi environment ...` subcommands do not exist in Pixi 0.62.2; environment management is under `pixi workspace environment ...`.

## Root Causes

1) Strict channel priority + unpinned CUDA packages

If workspace channels are ordered with `conda-forge` before `nvidia`, Pixi prefers `conda-forge` packages and may exclude lower-priority `nvidia` packages needed to satisfy CUDA 13.x dependency chains (for example `cuda-version`), even if you explicitly request some `nvidia::cuda-*` packages.

2) Two different cuDNN packaging ecosystems

There are different `cudnn` packages across channels that depend on different CUDA metapackages:
- The `nvidia` channel cuDNN packages are aligned with `cuda-version >=13,<14` (CUDA 13.x ecosystem).
- Other `cudnn` packages can depend on the legacy `cudatoolkit` metapackage (often CUDA <= 11.x).

If `cudnn` is not pinned to the `nvidia` channel, the solver may pick an incompatible `cudnn` and then fail when combined with `cuda-toolkit=13.0.*`.

3) Shared solve-group coupling

If `cuda13` shares the same solve-group as `default` and `accelsim`, solving `cuda13` can be constrained by `accelsim` (e.g., `cuda-toolkit=12.8.*`) and vice versa. This can cause errors that appear unrelated to the environment you are installing.

## Reproduction (Representative)

- Workspace channels: `["conda-forge", "nvidia"]`
- Create `cuda13` feature with:
  - `cuda-toolkit=13.0.*`
  - `cuda-nvcc=13.0.*`
  - `cudnn` (unpinned)
- Run:
  - `pixi install -e cuda13`

## Mitigations / Workarounds

1) Prefer `nvidia` channel first in workspace channels

In `pyproject.toml`:

```toml
[tool.pixi.workspace]
channels = ["nvidia", "conda-forge"]
```

2) Pin CUDA ecosystem packages to the intended channel

For CUDA 13.x envs:

```toml
[tool.pixi.feature.cuda13.dependencies]
cuda-toolkit = { version = "13.0.*", channel = "nvidia" }
cuda-nvcc    = { version = "13.0.*", channel = "nvidia" }
cudnn        = { channel = "nvidia" }
```

If another feature/environment needs CUDA from `conda-forge` (example: `accelsim` using CUDA 12.8), also pin it to avoid being pulled from `nvidia` after reordering channels:

```toml
[tool.pixi.feature.accelsim.dependencies]
cuda-toolkit = { version = "12.8.*", channel = "conda-forge" }
```

3) Separate solve-groups for different CUDA stacks

Keep independent CUDA stacks from constraining each other:

```toml
[tool.pixi.environments]
cuda13 = { features = ["cuda13"], solve-group = "cuda13" }
```

4) Use the correct Pixi CLI for environment listing

Pixi 0.62.2:

```bash
pixi workspace environment list
pixi workspace environment add cuda13 --feature cuda13 --solve-group cuda13
```

## Notes

Once solved/installed, verify the env is actually using the project-local toolchain:

```bash
pixi run -e cuda13 bash -lc 'which nvcc && nvcc --version | sed -n "1,8p"'
pixi run -e cuda13 bash -lc 'test -f "$CONDA_PREFIX/include/cudnn.h" && echo OK'
```
