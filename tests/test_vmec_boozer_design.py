from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fax.vmec_boozer_design import (
    VmecBoozerFaxDesignSettings,
    prepare_vmec_boozer_fax_design_context,
    run_vmec_boundary_alfven_gradient_convergence,
    validate_vmec_boundary_alfven_gradient,
    validate_vmec_boundary_frequency_jacobian,
    vmec_boundary_fax_frequencies,
)

INPUT_PATH = Path(__file__).resolve().parents[1] / "examples" / "data" / "input.LandremanPaul2021_QA_lowres"
# Entries of boundary modes with m = 1 and n != 0.  VMEC's m = 1 constraint
# leaves a poloidal-angle gauge that each converged re-solve moves; the implicit
# adjoint holds it fixed, so centered differences of re-solves differ from the
# adjoint by 0.3-1.2% there (vmex.core.implicit, "Gradient checking
# solver-sensitive metrics"); vmex's frozen-path differences match the adjoint
# to 5e-7 for every entry.  The same gauge drift leaves <= 5e-5 on the other
# entries, which are checked at 1e-4 (2.5e-4 for the cotangent pullback).
M1_GAUGE_ENTRIES = [2, 3]


@pytest.mark.requires_vmex
def test_landreman_paul_vmec_boundary_boozer_fax_gradient_matches_finite_difference():
    pytest.importorskip("vmex")
    settings = VmecBoozerFaxDesignSettings(
        input_path=INPUT_PATH,
        parameter_count=1,
        surfaces=(0.25, 0.40, 0.55, 0.70),
        mboz=3,
        nboz=3,
        theta_points=8,
        zeta_points=8,
    )
    result = validate_vmec_boundary_alfven_gradient(
        settings,
        finite_difference_step=1.0e-4,
        descent_step=1.0e-3,
    )

    assert result.parameter_names == ("rc01",)
    assert result.relative_gradient_error < 1.0e-4
    assert float(result.descent_objective) < float(result.objective)
    assert float(result.descent_step_norm) > 0.0
    np.testing.assert_allclose(
        np.asarray(result.gradient),
        np.asarray(result.finite_difference_gradient),
        rtol=1.0e-4,
        atol=1.0e-10,
    )
    assert np.all(np.isfinite(np.asarray(result.initial_frequencies_hz)))
    assert np.all(np.asarray(result.initial_frequencies_hz) > 0.0)


@pytest.mark.requires_vmex
def test_vmec_boundary_fax_frequency_map_uses_selected_boundary_parameters():
    pytest.importorskip("vmex")
    settings = VmecBoozerFaxDesignSettings(
        input_path=INPUT_PATH,
        parameter_count=2,
        surfaces=(0.25, 0.40, 0.55, 0.70),
        mboz=3,
        nboz=3,
        theta_points=8,
        zeta_points=8,
    )
    context = prepare_vmec_boozer_fax_design_context(settings)
    frequencies0 = vmec_boundary_fax_frequencies(np.zeros(2), context)
    frequencies1 = vmec_boundary_fax_frequencies(np.asarray([1.0e-3, -1.0e-3]), context)

    assert context.parameter_names[:2] == ("rc01", "zs01")
    assert frequencies0.shape == (3,)
    assert np.all(np.isfinite(np.asarray(frequencies0)))
    assert np.linalg.norm(np.asarray(frequencies1 - frequencies0)) > 0.0


@pytest.mark.requires_vmex
def test_landreman_paul_vmec_boundary_gradient_convergence_ladder_is_stable():
    pytest.importorskip("vmex")
    settings = VmecBoozerFaxDesignSettings(
        input_path=INPUT_PATH,
        parameter_count=1,
        surfaces=(0.25, 0.40, 0.55, 0.70),
        mboz=3,
        nboz=3,
        theta_points=8,
        zeta_points=8,
    )
    result = run_vmec_boundary_alfven_gradient_convergence(
        settings,
        levels=(
            ("ftol1e-10", {"vmec_ftol": 1.0e-10}),
            ("ftol1e-12", {"vmec_ftol": 1.0e-12}),
        ),
        finite_difference_step=1.0e-4,
        descent_step=1.0e-3,
    )

    assert result.reference_label == "ftol1e-12"
    assert len(result.points) == 2
    assert result.max_relative_gradient_error < 1.0e-4
    assert result.max_gradient_delta_to_reference < 5.0e-6
    assert result.max_objective_delta_to_reference < 1.0e-6
    assert result.max_frequency_delta_to_reference_hz < 20.0
    assert result.points[-1].frequency_delta_to_reference_hz == 0.0
    assert result.as_dict()["points"][0]["label"] == "ftol1e-10"


@pytest.mark.requires_vmex
def test_landreman_paul_vmec_boundary_two_parameter_gradient_matches_finite_difference():
    pytest.importorskip("vmex")
    settings = VmecBoozerFaxDesignSettings(
        input_path=INPUT_PATH,
        parameter_count=2,
        surfaces=(0.25, 0.40, 0.55, 0.70),
        mboz=3,
        nboz=3,
        theta_points=8,
        zeta_points=8,
    )
    result = validate_vmec_boundary_alfven_gradient(
        settings,
        finite_difference_step=1.0e-4,
        descent_step=1.0e-3,
    )

    assert result.parameter_names == ("rc01", "zs01")
    assert result.relative_gradient_error < 1.0e-4
    assert float(result.descent_objective) < float(result.objective)
    np.testing.assert_allclose(
        np.asarray(result.gradient),
        np.asarray(result.finite_difference_gradient),
        rtol=1.0e-4,
        atol=1.0e-10,
    )


@pytest.mark.requires_vmex
def test_landreman_paul_vmec_boundary_six_parameter_gradient_matches_finite_difference():
    pytest.importorskip("vmex")
    settings = VmecBoozerFaxDesignSettings(
        input_path=INPUT_PATH,
        max_mode=1,
        parameter_count=6,
        surfaces=(0.25, 0.40, 0.55, 0.70),
        mboz=3,
        nboz=3,
        theta_points=8,
        zeta_points=8,
    )
    result = validate_vmec_boundary_alfven_gradient(
        settings,
        finite_difference_step=2.0e-5,
        descent_step=1.0e-3,
    )

    assert result.parameter_names == ("rc01", "zs01", "rc1-1", "zs1-1", "rc10", "zs10")
    assert result.relative_gradient_error < 1.0e-2
    assert float(result.descent_objective) < float(result.objective)
    gradient = np.delete(np.asarray(result.gradient), M1_GAUGE_ENTRIES)
    finite_difference = np.delete(np.asarray(result.finite_difference_gradient), M1_GAUGE_ENTRIES)
    np.testing.assert_allclose(gradient, finite_difference, rtol=1.0e-4, atol=1.0e-10)


@pytest.mark.requires_vmex
def test_landreman_paul_vmec_boundary_frequency_jacobian_and_pullback_match_finite_difference():
    pytest.importorskip("vmex")
    settings = VmecBoozerFaxDesignSettings(
        input_path=INPUT_PATH,
        max_mode=1,
        parameter_count=6,
        surfaces=(0.25, 0.40, 0.55, 0.70),
        mboz=3,
        nboz=3,
        theta_points=8,
        zeta_points=8,
    )
    result = validate_vmec_boundary_frequency_jacobian(
        settings,
        finite_difference_step=2.0e-5,
    )

    assert result.parameter_names == ("rc01", "zs01", "rc1-1", "zs1-1", "rc10", "zs10")
    assert result.frequency_jacobian.shape == (3, 6)
    assert result.relative_jacobian_error < 1.0e-2
    assert result.relative_pullback_error < 1.0e-2
    np.testing.assert_allclose(
        np.delete(np.asarray(result.frequency_jacobian), M1_GAUGE_ENTRIES, axis=1),
        np.delete(np.asarray(result.finite_difference_jacobian), M1_GAUGE_ENTRIES, axis=1),
        rtol=1.0e-4,
        atol=2.0e-3,
    )
    np.testing.assert_allclose(
        np.delete(np.asarray(result.pullback), M1_GAUGE_ENTRIES),
        np.delete(np.asarray(result.finite_difference_pullback), M1_GAUGE_ENTRIES),
        rtol=2.5e-4,
        atol=2.0e-3,
    )
