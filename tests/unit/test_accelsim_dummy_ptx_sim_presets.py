from __future__ import annotations

from pathlib import Path

import pytest

from accelsim_test.accelsim_dummy_ptx_sim import paths


def test_normalize_config_preset_accepts_sm80_a100() -> None:
    assert paths.normalize_config_preset("sm80_a100") == "sm80_a100"


def test_normalize_config_preset_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        paths.normalize_config_preset("sm70_v100")


def test_preset_config_source_sm80_a100() -> None:
    repo = Path("/tmp/fake-repo")
    preset_name, src = paths.preset_config_source(repo_root=repo, preset="sm80_a100")
    assert preset_name == "SM80_A100"
    assert str(src).endswith(
        "extern/tracked/accel-sim-framework/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM80_A100/gpgpusim.config"
    )

