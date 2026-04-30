from __future__ import annotations

import numpy as np
import pytest

from fax.config import load_config
from fax.continuum import continuum_branch_matrix
from fax.equilibrium import from_vmec_wout
from fax.modes import stellarator_mode_family
from fax.reference_cases import available_vmec_equilibria_cases, landreman_paul_qa_lowres, vmec_equilibria_catalog


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


def test_vmec_equilibria_catalog_defines_broad_benchmark_matrix():
    catalog = vmec_equilibria_catalog()
    case_ids = {case.case_id for case in catalog}
    tags = {tag for case in catalog for tag in case.tags}

    assert "hsx_qhs_vac" in case_ids
    assert "w7x_standard" in case_ids
    assert "aries_cs" in case_ids
    assert "iter_model" in case_ids
    assert {"quasi-helical", "quasi-axisymmetric", "reactor", "tokamak"} <= tags
    assert all(case.relative_wout.name.endswith(".nc") for case in catalog)


def test_available_vmec_equilibria_wouts_have_finite_profile_continuum():
    cases = available_vmec_equilibria_cases()
    if len(cases) < 4:
        pytest.skip("local landreman/vmec_equilibria checkout is not available")

    for case in cases[:6]:
        geometry = from_vmec_wout(case.wout_path(), name=case.case_id)
        base_n = -1 if float(np.nanmedian(np.asarray(geometry.iota))) < 0.0 else 1
        family = stellarator_mode_family(
            poloidal_modes=(1, 2, 3),
            base_n=base_n,
            field_periods=geometry.field_periods,
            sidebands=(-1, 0, 1),
        )
        branches = np.asarray(continuum_branch_matrix(geometry, family.as_array()))

        assert branches.shape == (geometry.s.shape[0], len(family.harmonics))
        assert np.isfinite(branches).all()
        assert float(np.nanmax(branches)) > 0.0
        assert float(geometry.major_radius) > float(geometry.minor_radius) > 0.0
