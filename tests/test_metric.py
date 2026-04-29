from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fax.continuum import continuum_branch_matrix
from fax.equilibrium import analytic_circular_tokamak, analytic_cylinder
from fax.metric import (
    diagonal_continuum_frequencies_hz,
    generalized_eigen_residual_norms,
    generalized_hermitian_eigenvalue_sensitivity,
    generalized_hermitian_eigensystem,
    generalized_hermitian_eigenvalues,
    generalized_hermitian_isolated_eigenvalue,
    metric_continuum_frequencies_hz,
    metric_continuum_matrices,
    metric_from_equilibrium,
    metric_from_vmec_wout,
    parallel_wavenumber,
)
from fax.modes import stellarator_mode_family
from fax.reference_cases import landreman_paul_qa_lowres
from fax.validation import finite_difference_derivative, hermitian_relative_error


def test_metric_from_equilibrium_matches_profile_continuum_limit():
    geometry = analytic_circular_tokamak(radial_points=24, iota_axis=0.2, iota_edge=0.7)
    harmonics = jnp.asarray([[1, 0], [2, 1], [3, 2]], dtype=float)

    metric_geometry = metric_from_equilibrium(geometry, theta_points=32, zeta_points=32)
    metric_frequencies = diagonal_continuum_frequencies_hz(metric_geometry, harmonics)
    profile_frequencies = continuum_branch_matrix(geometry, harmonics)

    np.testing.assert_allclose(np.asarray(metric_frequencies), np.asarray(profile_frequencies), rtol=2.0e-5)


def test_metric_geometry_pytree_roundtrip_preserves_continuum_limit():
    geometry = analytic_circular_tokamak(radial_points=12, iota_axis=0.2, iota_edge=0.6)
    harmonics = jnp.asarray([[1, 0], [2, 1]], dtype=float)
    metric_geometry = metric_from_equilibrium(geometry, theta_points=16, zeta_points=16)

    leaves, treedef = jax.tree_util.tree_flatten(metric_geometry)
    reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
    original = diagonal_continuum_frequencies_hz(metric_geometry, harmonics)
    roundtripped = diagonal_continuum_frequencies_hz(reconstructed, harmonics)

    assert reconstructed.name == metric_geometry.name
    assert reconstructed.field_periods == metric_geometry.field_periods
    np.testing.assert_allclose(np.asarray(roundtripped), np.asarray(original), rtol=1.0e-12)


def test_metric_matrices_are_hermitian_and_positive_mass():
    geometry = analytic_circular_tokamak(radial_points=18)
    metric_geometry = metric_from_equilibrium(geometry)
    family = stellarator_mode_family(poloidal_modes=(1, 2, 3), base_n=1, field_periods=geometry.field_periods)

    matrices = metric_continuum_matrices(metric_geometry, family.as_array())

    assert matrices.stiffness.shape == (18, len(family.harmonics), len(family.harmonics))
    assert float(hermitian_relative_error(matrices.stiffness[5])) < 1.0e-12
    assert float(hermitian_relative_error(matrices.mass[5])) < 1.0e-12
    assert np.all(np.asarray(jnp.real(jnp.diagonal(matrices.mass, axis1=-2, axis2=-1))) > 0.0)


def test_coupled_metric_frequencies_match_diagonal_limit_for_uniform_geometry():
    geometry = analytic_cylinder(radial_points=6, iota=0.37, major_radius=4.5)
    metric_geometry = metric_from_equilibrium(geometry, theta_points=64, zeta_points=64)
    harmonics = jnp.asarray([[1, 0], [2, 1], [3, 2]], dtype=float)

    diagonal = diagonal_continuum_frequencies_hz(metric_geometry, harmonics)
    coupled = metric_continuum_frequencies_hz(metric_geometry, harmonics)

    np.testing.assert_allclose(np.sort(np.asarray(coupled), axis=1), np.sort(np.asarray(diagonal), axis=1), rtol=5.0e-6)


def test_generalized_eigenvalues_satisfy_known_diagonal_problem():
    stiffness = jnp.asarray([[[2.0, 0.0], [0.0, 12.0]]])
    mass = jnp.asarray([[[2.0, 0.0], [0.0, 3.0]]])

    eigenvalues = generalized_hermitian_eigenvalues(stiffness, mass)

    np.testing.assert_allclose(np.asarray(eigenvalues[0]), np.asarray([1.0, 4.0]), rtol=1.0e-6)
    np.testing.assert_allclose(
        np.asarray(generalized_hermitian_eigenvalues(stiffness[0], mass[0])),
        np.asarray([1.0, 4.0]),
        rtol=1.0e-6,
    )


def test_generalized_eigenvalue_sensitivity_matches_autodiff_and_finite_difference():
    stiffness = jnp.asarray([[4.0, 0.15], [0.15, 9.0]])
    mass = jnp.asarray([[2.0, 0.05], [0.05, 1.2]])
    stiffness_tangent = jnp.asarray([[0.3, -0.04], [-0.04, 0.8]])
    mass_tangent = jnp.asarray([[0.1, 0.02], [0.02, -0.03]])
    eigensystem = generalized_hermitian_eigensystem(stiffness, mass)

    def eigenvalue(alpha):
        return generalized_hermitian_eigenvalues(stiffness + alpha * stiffness_tangent, mass + alpha * mass_tangent)[0]

    sensitivity = generalized_hermitian_eigenvalue_sensitivity(
        stiffness,
        mass,
        eigensystem.eigenvalues[0],
        3.7 * eigensystem.eigenvectors[:, 0],
        stiffness_tangent,
        mass_tangent,
    )
    autodiff = jax.grad(eigenvalue)(0.0)
    step = 2.0e-5
    finite_difference = (eigenvalue(step) - eigenvalue(-step)) / (2.0 * step)

    np.testing.assert_allclose(float(sensitivity), float(autodiff), rtol=2.0e-10)
    np.testing.assert_allclose(float(sensitivity), float(finite_difference), rtol=2.0e-7)


def test_generalized_isolated_eigenvalue_custom_jvp_matches_adjoint_formula():
    stiffness = jnp.asarray([[4.0, 0.15], [0.15, 9.0]])
    mass = jnp.asarray([[2.0, 0.05], [0.05, 1.2]])
    stiffness_tangent = jnp.asarray([[0.3, -0.04], [-0.04, 0.8]])
    mass_tangent = jnp.asarray([[0.1, 0.02], [0.02, -0.03]])
    eigensystem = generalized_hermitian_eigensystem(stiffness, mass)
    expected = generalized_hermitian_eigenvalue_sensitivity(
        stiffness,
        mass,
        eigensystem.eigenvalues[1],
        eigensystem.eigenvectors[:, 1],
        stiffness_tangent,
        mass_tangent,
    )

    def eigenvalue(alpha):
        return generalized_hermitian_isolated_eigenvalue(
            stiffness + alpha * stiffness_tangent,
            mass + alpha * mass_tangent,
            1,
        )

    value = generalized_hermitian_isolated_eigenvalue(stiffness, mass, 1)
    jit_value = jax.jit(lambda k, m: generalized_hermitian_isolated_eigenvalue(k, m, 1))(stiffness, mass)
    custom_jvp = jax.grad(eigenvalue)(0.0)
    step = 2.0e-5
    finite_difference = (eigenvalue(step) - eigenvalue(-step)) / (2.0 * step)

    np.testing.assert_allclose(float(value), float(eigensystem.eigenvalues[1]), rtol=1.0e-12)
    np.testing.assert_allclose(float(jit_value), float(value), rtol=1.0e-12)
    np.testing.assert_allclose(float(custom_jvp), float(expected), rtol=2.0e-10)
    np.testing.assert_allclose(float(custom_jvp), float(finite_difference), rtol=2.0e-7)


def test_generalized_eigenvalue_sensitivity_contracts():
    stiffness = jnp.eye(2)
    mass = jnp.eye(2)
    eigensystem = generalized_hermitian_eigensystem(stiffness, mass)

    with pytest.raises(ValueError, match="matching shapes"):
        generalized_hermitian_eigenvalue_sensitivity(
            stiffness,
            jnp.eye(3),
            eigensystem.eigenvalues[0],
            eigensystem.eigenvectors[:, 0],
            stiffness,
            mass,
        )
    with pytest.raises(ValueError, match="tangent matrices"):
        generalized_hermitian_eigenvalue_sensitivity(
            stiffness,
            mass,
            eigensystem.eigenvalues[0],
            eigensystem.eigenvectors[:, 0],
            jnp.eye(3),
            mass,
        )
    with pytest.raises(ValueError, match="one eigenvector"):
        generalized_hermitian_eigenvalue_sensitivity(
            stiffness,
            mass,
            eigensystem.eigenvalues[0],
            eigensystem.eigenvectors,
            stiffness,
            mass,
        )


def test_generalized_eigenpair_residuals_are_small():
    geometry = analytic_circular_tokamak(radial_points=10, iota_axis=0.22, iota_edge=0.62)
    metric_geometry = metric_from_equilibrium(geometry, theta_points=32, zeta_points=32)
    harmonics = jnp.asarray([[1, 0], [2, 2], [3, 3]], dtype=float)
    matrices = metric_continuum_matrices(metric_geometry, harmonics)

    eigensystem = generalized_hermitian_eigensystem(matrices.stiffness, matrices.mass)
    residuals = generalized_eigen_residual_norms(matrices.stiffness, matrices.mass, eigensystem)

    assert residuals.shape == (10, 3)
    assert float(jnp.max(residuals)) < 1.0e-5


def test_metric_density_scale_gradient_matches_finite_difference():
    metric_geometry = metric_from_equilibrium(analytic_cylinder(radial_points=16, iota=0.25, major_radius=5.0))
    harmonics = jnp.asarray([[2, 1]], dtype=float)

    def mean_frequency(scale):
        return jnp.mean(diagonal_continuum_frequencies_hz(metric_geometry.with_density_scale(scale), harmonics))

    scale0 = 1.3
    autodiff = float(jax.grad(mean_frequency)(scale0))
    finite_difference = finite_difference_derivative(lambda x: float(mean_frequency(x)), scale0, step=1.0e-2)
    expected = float(-0.5 * mean_frequency(scale0) / scale0)

    np.testing.assert_allclose(autodiff, expected, rtol=3.0e-5)
    np.testing.assert_allclose(finite_difference, expected, rtol=5.0e-4)


def test_parallel_wavenumber_shape_and_large_aspect_values():
    geometry = analytic_cylinder(radial_points=4, iota=0.5, major_radius=2.0)
    metric_geometry = metric_from_equilibrium(geometry, theta_points=8, zeta_points=8)
    harmonics = jnp.asarray([[1, 0], [2, 1]], dtype=float)

    k_parallel = parallel_wavenumber(metric_geometry, harmonics)

    assert k_parallel.shape == (4, 2, 8, 8)
    np.testing.assert_allclose(np.asarray(k_parallel[:, 1]), 0.0, atol=1.0e-12)


@pytest.mark.requires_vmec_jax
def test_landreman_paul_metric_adapter_smoke():
    case = landreman_paul_qa_lowres()
    if not case.is_available():
        pytest.skip(f"missing local Landreman-Paul QA paths: {case.missing_paths()}")

    metric_geometry = metric_from_vmec_wout(case.wout, theta_points=12, zeta_points=10, name=case.case_id)  # type: ignore[arg-type]

    assert metric_geometry.name == case.case_id
    assert metric_geometry.field_periods == 2
    assert metric_geometry.jacobian.shape == (50, 12, 10)
    assert metric_geometry.magnetic_field.shape == metric_geometry.jacobian.shape
    assert float(jnp.min(metric_geometry.jacobian)) > 0.0
    assert float(jnp.min(metric_geometry.magnetic_field)) > 0.0
    assert dict(metric_geometry.metadata)["metric_adapter"] == "vmec_wout_nyquist"
