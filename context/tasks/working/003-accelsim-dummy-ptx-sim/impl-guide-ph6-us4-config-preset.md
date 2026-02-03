# Implementation Guide: US4 Config Preset Selection (SM80_A100)

**Phase**: 6 | **Feature**: Accel-Sim Dummy CUDA PTX Simulation | **Tasks**: T020–T023

## Goal

Support selecting the single supported simulator preset (`sm80_a100`) and record it in `metadata.json`, while keeping the code structured so future presets could be added safely.

## Public APIs

### T020: Preset mapping/validation

Centralize preset selection logic in `paths.py` so the workflow only deals with validated presets and concrete config paths.

```python
# src/accelsim_test/accelsim_dummy_ptx_sim/paths.py

from __future__ import annotations

from pathlib import Path
from typing import Literal


ConfigPresetCli = Literal[\"sm80_a100\"]
ConfigPreset = Literal[\"SM80_A100\"]


def normalize_config_preset(preset: str) -> ConfigPresetCli:
    if preset != \"sm80_a100\":
        raise ValueError(f\"Unsupported config preset: {preset}\")
    return \"sm80_a100\"


def preset_config_source(*, repo_root: Path, preset: ConfigPresetCli) -> tuple[ConfigPreset, Path]:
    if preset == \"sm80_a100\":
        return \"SM80_A100\", sm80_a100_config_source(repo_root=repo_root)
    raise AssertionError(\"unreachable\")
```

---

### T021: CLI wiring

Ensure `--config-preset` is parsed and passed through unchanged.

```python
# src/accelsim_test/accelsim_dummy_ptx_sim/__main__.py

run.add_argument(\"--config-preset\", default=\"sm80_a100\", choices=[\"sm80_a100\"])
```

---

### T022: Config copy + metadata recording

In the workflow:

- resolve `(config_preset, config_source_path)` via `paths.preset_config_source`,
- copy to `<run_dir>/gpgpusim.config`,
- record both the preset name and source path in metadata.

```python
# src/accelsim_test/accelsim_dummy_ptx_sim/workflow.py

from __future__ import annotations

import shutil
from pathlib import Path

from . import paths


def copy_config_preset(*, repo_root: Path, run_dir: Path, preset: str) -> tuple[str, Path]:
    preset_cli = paths.normalize_config_preset(preset)
    preset_name, src = paths.preset_config_source(repo_root=repo_root, preset=preset_cli)
    dst = run_dir / \"gpgpusim.config\"
    shutil.copy2(src, dst)
    return preset_name, dst
```

---

### T023: Unit tests for preset mapping/validation

```python
# tests/unit/test_accelsim_dummy_ptx_sim_presets.py

from __future__ import annotations

import pytest

from accelsim_test.accelsim_dummy_ptx_sim import paths


def test_normalize_config_preset_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        paths.normalize_config_preset(\"sm70_v100\")
```

---

## Phase Integration

```mermaid
graph LR
    CLI[--config-preset] --> P[paths.preset_config_source];
    P --> CFG[run/gpgpusim.config];
    CFG --> META[metadata.json];
```

## Testing

### Test Input

- Unit tests only.

### Test Procedure

```bash
pixi run pytest -q tests/unit/test_accelsim_dummy_ptx_sim_presets.py
```

### Test Output

- `N passed`

## References

- Research: `specs/003-accelsim-dummy-ptx-sim/research.md` (Decision 2)
- Contracts: `specs/003-accelsim-dummy-ptx-sim/contracts/cli.md`

## Implementation Summary

TBD after implementation.

