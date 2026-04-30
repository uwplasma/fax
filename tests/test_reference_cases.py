from __future__ import annotations

import numpy as np
import pytest

from fax.boozer import (
    boozer_geometry_jacobian_identity_error,
    normalized_nonaxisymmetric_boozer_energy,
    read_boozmn_geometry,
    slice_boozer_geometry_surfaces,
)
from fax.config import load_config
from fax.continuum import continuum_branch_matrix
from fax.equilibrium import from_vmec_wout
from fax.global_modes import radial_block_problem_from_boozer_geometry, solve_radial_block_eigenmodes
from fax.modes import stellarator_mode_family
from fax.reference_cases import (
    available_vmec_equilibria_boozmn_cases,
    available_vmec_equilibria_cases,
    landreman_paul_qa_lowres,
    vmec_equilibria_catalog,
)


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
    assert "ku_boozer_n4qh_b4_a79a" in case_ids
    assert "iter_model" in case_ids
    assert {"quasi-helical", "quasi-axisymmetric", "reactor", "tokamak"} <= tags
    assert all(case.relative_wout.name.endswith(".nc") for case in catalog)
    assert any(case.relative_boozmn is not None for case in catalog)


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


def test_available_vmec_equilibria_boozmn_cases_feed_global_operator():
    cases = available_vmec_equilibria_boozmn_cases()
    if len(cases) < 2:
        pytest.skip("local landreman/vmec_equilibria boozmn artifacts are not available")

    harmonics = np.asarray([[1.0, 0.0], [2.0, 1.0]])
    for case in cases[:3]:
        boozmn = case.boozmn_path()
        assert boozmn is not None
        geometry = read_boozmn_geometry(boozmn)
        surface_count = min(12, int(geometry.s.shape[0]))
        indices = np.linspace(0, int(geometry.s.shape[0]) - 1, surface_count, dtype=int)
        sampled = slice_boozer_geometry_surfaces(geometry, indices)

        problem = radial_block_problem_from_boozer_geometry(
            sampled,
            harmonics,
            mass_density=3.4e-8,
            radial_stiffness_scale=1.0e-8,
            theta_points=8,
            zeta_points=8,
        )
        result = solve_radial_block_eigenmodes(problem)

        assert bool(np.all(np.isfinite(np.asarray(result.frequencies_hz[:4]))))
        assert float(np.min(np.asarray(result.frequencies_hz[:4]))) > 0.0
        assert float(np.max(np.asarray(result.residuals[:4]))) < 1.0e-9
        assert float(boozer_geometry_jacobian_identity_error(sampled, theta_points=8, zeta_points=8)) < 1.0e-3
        assert float(normalized_nonaxisymmetric_boozer_energy(sampled.as_spectrum())) >= 0.0
