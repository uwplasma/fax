from __future__ import annotations

import pytest

from fax.config import ConfigError, FaxConfig, load_config


def test_load_minimal_config(tmp_path):
    config_path = tmp_path / "case.toml"
    config_path.write_text(
        """
[run]
case = "unit"
model = "continuum"

[grid]
radial_points = 16
m_max = 3
n_max = 4
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.run.case == "unit"
    assert config.run.model == "continuum"
    assert config.grid.radial_points == 16
    assert config.equilibrium.source == "vmec_jax"


def test_rejects_unknown_section():
    with pytest.raises(ConfigError, match="unknown configuration section"):
        FaxConfig.from_mapping({"not_a_section": {}})


def test_rejects_invalid_grid():
    with pytest.raises(ConfigError, match="radial_points"):
        FaxConfig.from_mapping({"grid": {"radial_points": 3}})

