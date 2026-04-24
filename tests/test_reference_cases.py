from __future__ import annotations

from fax.config import load_config
from fax.reference_cases import landreman_paul_qa_lowres


def test_landreman_paul_qa_lowres_metadata():
    case = landreman_paul_qa_lowres(root="/tmp/fax-does-not-need-to-exist")

    assert case.case_id == "LandremanPaul2021_QA_lowres"
    assert "quasi-axisymmetric" in case.description
    assert case.vmec_input.name == "input.LandremanPaul2021_QA_lowres"
    assert case.wout is not None
    assert case.wout.name == "wout_LandremanPaul2021_QA_lowres.nc"
    assert any("2108.03711" in ref for ref in case.references)


def test_landreman_paul_qa_lowres_example_loads():
    config = load_config("examples/landreman_paul_qa_lowres.toml")

    assert config.run.case == "LandremanPaul2021_QA_lowres"
    assert config.equilibrium.source == "wout"
    assert config.equilibrium.use_boozer is False
    assert config.grid.field_periods == 2
    assert config.grid.radial_points == 75

