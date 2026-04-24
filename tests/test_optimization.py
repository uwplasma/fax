from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from fax.equilibrium import analytic_circular_tokamak
from fax.optimization import frequency_window_penalty, radial_mode_avoidance_objective, scaled_profile_geometry


def test_frequency_window_penalty_is_larger_inside_window():
    inside = frequency_window_penalty(jnp.asarray([101_000.0, 105_000.0]), lower_hz=100_000.0, upper_hz=110_000.0, softness_hz=500.0)
    outside = frequency_window_penalty(jnp.asarray([80_000.0, 140_000.0]), lower_hz=100_000.0, upper_hz=110_000.0, softness_hz=500.0)

    assert float(inside) > float(outside)


def test_scaled_profile_geometry_keeps_shapes():
    geometry = analytic_circular_tokamak(radial_points=18)

    adjusted = scaled_profile_geometry(geometry, density_scale=1.2, iota_shift=0.01, field_scale=0.95)

    assert adjusted.s.shape == geometry.s.shape
    np.testing.assert_allclose(np.asarray(adjusted.iota), np.asarray(geometry.iota) + 0.01)


def test_radial_mode_avoidance_objective_is_differentiable():
    geometry = analytic_circular_tokamak(radial_points=24, iota_axis=0.28, iota_edge=0.68)
    parameters = jnp.asarray([0.0, 0.0, 0.0])

    def objective(params):
        return radial_mode_avoidance_objective(
            params,
            geometry=geometry,
            harmonic=(2, 1),
            avoid_window_hz=(80_000.0, 180_000.0),
        )
    value = objective(parameters)
    gradient = jax.grad(objective)(parameters)

    assert float(value) >= 0.0
    assert gradient.shape == (3,)
    assert np.all(np.isfinite(np.asarray(gradient)))
