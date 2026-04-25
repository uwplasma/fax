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


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ([], "configuration root"),
        ({"run": {"case": ""}}, "run.case"),
        ({"run": {"model": "ballooning_proxy"}}, "run.model"),
        ({"equilibrium": {"source": "geqdsk"}}, "equilibrium.source"),
        ({"grid": {"m_max": -1}}, "m_max"),
        ({"grid": {"field_periods": 0}}, "field_periods"),
        ({"physics": {"density_scale": 0.0}}, "density_scale"),
        ({"physics": {"fast_ion_model": "unvalidated"}}, "fast_ion_model"),
        ({"numerics": {"precision": "float16"}}, "precision"),
        ({"numerics": {"eigensolver": "ad_hoc"}}, "eigensolver"),
        ({"grid": {"theta_points": 16}}, "unknown key"),
        ({"physics": False}, "PhysicsConfig must be a TOML table"),
        ({"equilibrium": None}, "run"),
    ],
)
def test_config_rejects_unphysical_or_unknown_user_inputs(mapping, message):
    if message == "run":
        config = FaxConfig.from_mapping(mapping)
        assert config.equilibrium.source == "vmec_jax"
        return

    with pytest.raises(ConfigError, match=message):
        FaxConfig.from_mapping(mapping)  # type: ignore[arg-type]


def test_load_config_reports_missing_file_and_bad_toml(tmp_path):
    with pytest.raises(ConfigError, match="configuration file not found"):
        load_config(tmp_path / "missing.toml")

    bad_toml = tmp_path / "bad.toml"
    bad_toml.write_text("[run\ncase = 'bad'\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="failed to parse TOML"):
        load_config(bad_toml)
