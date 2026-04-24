from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fax.continuum import (
    alfven_speed,
    continuum_branch_matrix,
    large_aspect_k_parallel,
    local_continuum_angular_frequency,
    local_continuum_frequency_hz,
    nearest_rational_surface,
)
from fax.equilibrium import analytic_circular_tokamak, analytic_cylinder
from fax.validation import finite_difference_derivative


def test_large_aspect_k_parallel_limit():
    iota = jnp.asarray([0.25, 0.5, 0.75])
    k_parallel = large_aspect_k_parallel(iota, m=2, n=1, major_radius=4.0)

    np.testing.assert_allclose(np.asarray(k_parallel), np.asarray([0.125, 0.0, -0.125]))


def test_local_continuum_matches_closed_form_cylinder():
    geometry = analytic_cylinder(
        radial_points=4,
        iota=0.25,
        magnetic_field=3.0,
        number_density=2.0e20,
        major_radius=5.0,
    )

    omega = local_continuum_angular_frequency(geometry, m=2, n=1)
    expected = alfven_speed(geometry.magnetic_field, geometry.mass_density) * abs(1.0 - 2.0 * 0.25) / 5.0

    np.testing.assert_allclose(np.asarray(omega), np.asarray(expected), rtol=2.0e-6)
    np.testing.assert_allclose(np.asarray(local_continuum_frequency_hz(geometry, m=2, n=1)), np.asarray(omega / (2 * np.pi)))


def test_continuum_branch_matrix_shape_and_values():
    geometry = analytic_cylinder(radial_points=5, iota=0.5, magnetic_field=1.0, major_radius=2.0)
    harmonics = jnp.asarray([[1, 0], [2, 1], [3, 2]], dtype=float)

    branches = continuum_branch_matrix(geometry, harmonics)

    assert branches.shape == (5, 3)
    np.testing.assert_allclose(np.asarray(branches[:, 1]), 0.0, atol=1.0e-12)


def test_nearest_rational_surface_for_linear_iota_profile():
    geometry = analytic_circular_tokamak(radial_points=101, iota_axis=0.2, iota_edge=0.8)

    surface, distance = nearest_rational_surface(geometry, m=2, n=1)

    assert abs(float(surface) - 0.5) < 0.02
    assert float(distance) < 0.01


def test_density_scale_gradient_matches_finite_difference_and_formula():
    geometry = analytic_cylinder(radial_points=16, iota=0.25, magnetic_field=3.0, major_radius=5.0)

    def omega_at_scale(scale):
        scaled = geometry.with_density_scale(scale)
        return jnp.mean(local_continuum_angular_frequency(scaled, m=2, n=1))

    scale0 = 1.3
    autodiff = float(jax.grad(omega_at_scale)(scale0))
    finite_difference = finite_difference_derivative(lambda x: float(omega_at_scale(x)), scale0, step=1.0e-2)
    expected = float(-0.5 * omega_at_scale(scale0) / scale0)

    np.testing.assert_allclose(autodiff, expected, rtol=3.0e-6)
    np.testing.assert_allclose(finite_difference, expected, rtol=3.0e-4)


def test_nearest_rational_surface_rejects_m_zero():
    with pytest.raises(ValueError, match="m must be nonzero"):
        nearest_rational_surface(analytic_cylinder(), m=0, n=1)
