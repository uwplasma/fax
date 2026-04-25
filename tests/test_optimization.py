from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from fax.boozer import BoozerGeometry
from fax.equilibrium import analytic_circular_tokamak
from fax.optimization import (
    boozer_global_mode_avoidance_objective,
    boozer_pipeline_mode_avoidance_objective,
    frequency_window_penalty,
    radial_mode_avoidance_objective,
    scaled_profile_geometry,
    shifted_boozer_geometry,
)


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


def _synthetic_boozer_geometry() -> BoozerGeometry:
    s = np.linspace(0.2, 0.8, 5)
    minor = 0.35 * s
    magnetic_field = 2.7
    major_radius = 5.5
    numerator = magnetic_field * major_radius
    zeros = np.zeros((s.size, 2))
    return BoozerGeometry(
        source="synthetic_optimization_boozer",
        s=s,
        xm=np.asarray([0.0, 1.0]),
        xn=np.asarray([0.0, 0.0]),
        bmnc=np.column_stack([np.full_like(s, magnetic_field), np.zeros_like(s)]),
        bmns=zeros,
        rmnc=np.column_stack([np.full_like(s, major_radius), minor]),
        rmns=zeros,
        zmns=np.column_stack([np.zeros_like(s), minor]),
        zmnc=zeros,
        pmns=zeros,
        pmnc=zeros,
        gmnc=np.column_stack([np.full_like(s, numerator / magnetic_field**2), np.zeros_like(s)]),
        gmns=zeros,
        iota=np.full_like(s, 0.61),
        boozer_i=np.zeros_like(s),
        boozer_g=np.full_like(s, numerator),
        field_periods=1,
    )


def test_boozer_global_mode_avoidance_objective_is_differentiable():
    geometry = _synthetic_boozer_geometry()
    shifted = shifted_boozer_geometry(geometry, iota_shift=0.01)
    params = jnp.asarray([0.0, 0.0, 0.0])

    def objective(x):
        return boozer_global_mode_avoidance_objective(
            x,
            geometry=geometry,
            harmonics=jnp.asarray([[1.0, 0.0], [2.0, 1.0]]),
            mass_density=3.4e-8,
            avoid_window_hz=(50_000.0, 350_000.0),
            theta_points=12,
            zeta_points=12,
            mode_count=3,
        )

    value = objective(params)
    gradient = jax.grad(objective)(params)

    np.testing.assert_allclose(np.asarray(shifted.iota), np.asarray(geometry.iota) + 0.01)
    assert float(value) >= 0.0
    assert gradient.shape == (3,)
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_boozer_pipeline_mode_avoidance_objective_differentiates_through_builder():
    geometry = _synthetic_boozer_geometry()
    params = jnp.asarray([0.0, 0.0, 0.0])

    def geometry_builder(x):
        field_scale = 1.0 + 0.03 * x[1]
        return replace(
            geometry,
            iota=geometry.iota + 0.02 * x[0],
            bmnc=geometry.bmnc * field_scale,
            boozer_g=geometry.boozer_g * field_scale,
        )

    def density_builder(x, built_geometry):
        return jnp.full_like(built_geometry.s, 3.4e-8 * jnp.exp(x[2]))

    def objective(x):
        return boozer_pipeline_mode_avoidance_objective(
            x,
            geometry_builder=geometry_builder,
            harmonics=jnp.asarray([[1.0, 0.0], [2.0, 1.0]]),
            mass_density=density_builder,
            radial_stiffness_scale=lambda x, _geometry: jnp.exp(0.1 * x[1]),
            avoid_window_hz=(50_000.0, 350_000.0),
            theta_points=12,
            zeta_points=12,
            mode_count=3,
        )

    value = objective(params)
    gradient = jax.grad(objective)(params)

    assert float(value) >= 0.0
    assert gradient.shape == (3,)
    assert np.all(np.isfinite(np.asarray(gradient)))
