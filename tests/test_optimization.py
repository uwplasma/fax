from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from fax.boozer import BoozerGeometry
from fax.equilibrium import analytic_circular_tokamak
from fax.optimization import (
    boozer_fast_ion_growth_screen,
    boozer_fast_ion_risk_objective,
    boozer_global_cluster_mean_avoidance_objective,
    boozer_global_cluster_mean_frequency_hz,
    boozer_global_cluster_mean_omega2,
    boozer_global_isolated_mode_avoidance_objective,
    boozer_global_isolated_mode_frequency_hz,
    boozer_global_isolated_mode_omega2,
    boozer_global_mode_avoidance_objective,
    boozer_global_mode_frequency_jacobian,
    boozer_global_mode_frequency_vector,
    boozer_global_mode_inverse_design,
    boozer_pipeline_cluster_mean_avoidance_objective,
    boozer_pipeline_isolated_mode_avoidance_objective,
    boozer_pipeline_mode_avoidance_objective,
    frequency_window_penalty,
    gradient_descent_minimize,
    linearized_boozer_global_mode_uncertainty,
    linearized_radial_mode_uncertainty,
    radial_fast_ion_growth_screen,
    radial_fast_ion_risk_objective,
    radial_mode_frequency_jacobian,
    radial_mode_frequency_vector,
    radial_mode_inverse_design,
    radial_mode_avoidance_objective,
    scaled_profile_geometry,
    shifted_boozer_geometry,
)
from fax.fast_ions import FastIonPopulation


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


def test_radial_frequency_jacobian_has_physical_signs():
    geometry = analytic_circular_tokamak(radial_points=32, iota_axis=0.26, iota_edge=0.72)
    parameters = jnp.zeros((3,))

    frequencies = radial_mode_frequency_vector(
        parameters,
        geometry=geometry,
        harmonic=(2, 1),
        mode_count=4,
        radial_coupling_fraction=0.02,
    )
    jacobian = radial_mode_frequency_jacobian(
        parameters,
        geometry=geometry,
        harmonic=(2, 1),
        mode_count=4,
        radial_coupling_fraction=0.02,
    )

    assert frequencies.shape == (4,)
    assert jacobian.shape == (4, 3)
    assert np.all(np.asarray(jacobian[:, 0]) < 0.0)
    assert np.all(np.asarray(jacobian[:, 2]) > 0.0)
    np.testing.assert_allclose(np.asarray(jacobian[:, 2]), np.asarray(frequencies), rtol=5.0e-6)


def test_linearized_radial_uncertainty_has_ordered_bands():
    geometry = analytic_circular_tokamak(radial_points=32, iota_axis=0.26, iota_edge=0.72)
    estimate = linearized_radial_mode_uncertainty(
        jnp.zeros((3,)),
        covariance=jnp.diag(jnp.asarray([0.04, 0.005, 0.02]) ** 2),
        geometry=geometry,
        harmonic=(2, 1),
        mode_count=4,
        sigma_multiplier=2.0,
        radial_coupling_fraction=0.02,
    )

    assert estimate.jacobian_hz_per_parameter.shape == (4, 3)
    assert np.all(np.asarray(estimate.sigma_hz) > 0.0)
    assert np.all(np.asarray(estimate.lower_hz) < np.asarray(estimate.mean_hz))
    assert np.all(np.asarray(estimate.upper_hz) > np.asarray(estimate.mean_hz))


def test_radial_mode_inverse_design_reduces_frequency_mismatch():
    geometry = analytic_circular_tokamak(radial_points=28, iota_axis=0.26, iota_edge=0.72)
    initial = jnp.zeros((3,))
    target = 1.03 * radial_mode_frequency_vector(
        initial,
        geometry=geometry,
        harmonic=(2, 1),
        mode_count=3,
        radial_coupling_fraction=0.02,
    )

    result = radial_mode_inverse_design(
        initial,
        geometry=geometry,
        harmonic=(2, 1),
        target_frequencies_hz=target,
        radial_coupling_fraction=0.02,
        step_size=0.2,
        steps=12,
    )

    assert result.parameter_history.shape == (13, 3)
    assert float(result.objective_history[-1]) < 0.05 * float(result.objective_history[0])
    assert np.all(np.isfinite(np.asarray(result.parameters)))


def _fast_ion_population_for_geometry(geometry):
    s = jnp.asarray(geometry.s)
    return FastIonPopulation(
        density_fraction=0.03,
        pressure_gradient=-jnp.exp(-((s - 0.45) / 0.12) ** 2),
        transit_frequency_hz=jnp.full_like(s, 83_000.0),
        precession_frequency_hz=jnp.full_like(s, 5_000.0),
        resonance_width_hz=jnp.full_like(s, 4_000.0),
    )


def test_radial_fast_ion_screen_and_risk_objective_are_differentiable():
    geometry = analytic_circular_tokamak(radial_points=30, iota_axis=0.26, iota_edge=0.72)
    population = _fast_ion_population_for_geometry(geometry)
    parameters = jnp.zeros((4,))

    screen = radial_fast_ion_growth_screen(
        parameters,
        geometry=geometry,
        harmonic=(2, 1),
        population=population,
        toroidal_mode=1,
        transit_harmonics=(1,),
        mode_count=3,
        drive_scale_hz=1_000.0,
        damping_rate_hz=5.0,
        radial_coupling_fraction=0.02,
    )

    def objective(params):
        return radial_fast_ion_risk_objective(
            params,
            geometry=geometry,
            harmonic=(2, 1),
            population=population,
            toroidal_mode=1,
            transit_harmonics=(1,),
            avoid_window_hz=(70_000.0, 120_000.0),
            mode_count=3,
            drive_scale_hz=1_000.0,
            damping_rate_hz=5.0,
            radial_coupling_fraction=0.02,
        )

    gradient = jax.grad(objective)(parameters)
    optimized = gradient_descent_minimize(objective, parameters, step_size=0.1, steps=8)

    assert screen.net_growth_rate_hz.shape == (3,)
    assert np.max(np.asarray(screen.net_growth_rate_hz)) > 0.0
    assert gradient.shape == (4,)
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(optimized.objective_history[-1]) < float(optimized.objective_history[0])


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


def test_boozer_global_frequency_uq_and_inverse_design_are_differentiable():
    geometry = _synthetic_boozer_geometry()
    harmonics = jnp.asarray([[1.0, 0.0], [2.0, 1.0]])
    mass_density = 3.4e-8
    parameters = jnp.zeros((3,))
    kwargs = {
        "geometry": geometry,
        "harmonics": harmonics,
        "mass_density": mass_density,
        "mode_count": 3,
        "theta_points": 10,
        "zeta_points": 10,
        "radial_stiffness_scale": 1.0e-4,
    }

    frequencies = boozer_global_mode_frequency_vector(parameters, **kwargs)
    jacobian = boozer_global_mode_frequency_jacobian(parameters, **kwargs)
    uncertainty = linearized_boozer_global_mode_uncertainty(
        parameters,
        covariance=jnp.diag(jnp.asarray([0.03, 0.01, 0.08]) ** 2),
        **kwargs,
    )
    target = 1.02 * frequencies
    design = boozer_global_mode_inverse_design(
        parameters,
        target_frequencies_hz=target,
        step_size=0.05,
        steps=8,
        **kwargs,
    )

    assert frequencies.shape == (3,)
    assert jacobian.shape == (3, 3)
    assert uncertainty.sigma_hz.shape == (3,)
    assert np.all(np.asarray(uncertainty.upper_hz) > np.asarray(uncertainty.lower_hz))
    assert float(design.objective_history[-1]) < float(design.objective_history[0])


def test_boozer_isolated_mode_custom_jvp_design_gradient_matches_finite_difference():
    geometry = _synthetic_boozer_geometry()
    harmonics = jnp.asarray([[1.0, 0.0], [2.0, 1.0]])
    parameters = jnp.asarray([0.03, -0.004, 0.02])
    kwargs = {
        "geometry": geometry,
        "harmonics": harmonics,
        "mass_density": 3.4e-8,
        "mode_index": 0,
        "theta_points": 10,
        "zeta_points": 10,
        "radial_stiffness_scale": 1.0e-4,
    }

    def frequency(params):
        return boozer_global_isolated_mode_frequency_hz(params, **kwargs)

    isolated_frequency = frequency(parameters)
    isolated_omega2 = boozer_global_isolated_mode_omega2(parameters, **kwargs)
    dense_frequency = boozer_global_mode_frequency_vector(
        parameters,
        geometry=geometry,
        harmonics=harmonics,
        mass_density=3.4e-8,
        mode_count=1,
        theta_points=10,
        zeta_points=10,
        radial_stiffness_scale=1.0e-4,
    )[0]
    gradient = jax.grad(frequency)(parameters)
    step = 2.0e-4
    finite_difference = []
    for index in range(parameters.shape[0]):
        direction = jnp.zeros_like(parameters).at[index].set(step)
        finite_difference.append((frequency(parameters + direction) - frequency(parameters - direction)) / (2.0 * step))
    finite_difference = jnp.asarray(finite_difference)
    objective_gradient = jax.grad(
        lambda params: boozer_global_isolated_mode_avoidance_objective(
            params,
            avoid_window_hz=(50_000.0, 350_000.0),
            **kwargs,
        )
    )(parameters)
    jit_frequency = jax.jit(frequency)(parameters)

    np.testing.assert_allclose(float(isolated_frequency), float(dense_frequency), rtol=1.0e-10)
    np.testing.assert_allclose(float(isolated_omega2), float((2.0 * jnp.pi * dense_frequency) ** 2), rtol=1.0e-10)
    np.testing.assert_allclose(np.asarray(gradient), np.asarray(finite_difference), rtol=4.0e-3, atol=5.0e-2)
    np.testing.assert_allclose(float(jit_frequency), float(isolated_frequency), rtol=1.0e-12)
    assert objective_gradient.shape == (3,)
    assert np.all(np.isfinite(np.asarray(objective_gradient)))


def test_boozer_cluster_mean_objective_uses_subspace_gradient():
    geometry = _synthetic_boozer_geometry()
    harmonics = jnp.asarray([[1.0, 0.0], [2.0, 1.0]])
    parameters = jnp.asarray([0.03, -0.004, 0.02])
    kwargs = {
        "geometry": geometry,
        "harmonics": harmonics,
        "mass_density": 3.4e-8,
        "start_index": 0,
        "stop_index": 2,
        "theta_points": 10,
        "zeta_points": 10,
        "radial_stiffness_scale": 1.0e-4,
    }

    def omega2(params):
        return boozer_global_cluster_mean_omega2(params, **kwargs)

    cluster_omega2 = omega2(parameters)
    cluster_frequency = boozer_global_cluster_mean_frequency_hz(parameters, **kwargs)
    gradient = jax.grad(omega2)(parameters)
    step = 2.0e-4
    finite_difference = []
    for index in range(parameters.shape[0]):
        direction = jnp.zeros_like(parameters).at[index].set(step)
        finite_difference.append((omega2(parameters + direction) - omega2(parameters - direction)) / (2.0 * step))
    finite_difference = jnp.asarray(finite_difference)
    objective_gradient = jax.grad(
        lambda params: boozer_global_cluster_mean_avoidance_objective(
            params,
            avoid_window_hz=(50_000.0, 350_000.0),
            **kwargs,
        )
    )(parameters)

    assert float(cluster_omega2) > 0.0
    np.testing.assert_allclose(float(cluster_frequency), float(jnp.sqrt(cluster_omega2) / (2.0 * jnp.pi)), rtol=1.0e-12)
    np.testing.assert_allclose(np.asarray(gradient), np.asarray(finite_difference), rtol=4.0e-3)
    assert objective_gradient.shape == (3,)
    assert np.all(np.isfinite(np.asarray(objective_gradient)))


def test_boozer_fast_ion_risk_objective_reduces_with_gradient_steps():
    geometry = _synthetic_boozer_geometry()
    harmonics = jnp.asarray([[1.0, 0.0], [2.0, 1.0]])
    s = jnp.asarray(geometry.s)
    population = FastIonPopulation(
        density_fraction=0.03,
        pressure_gradient=-jnp.exp(-((s - 0.5) / 0.2) ** 2),
        transit_frequency_hz=jnp.full_like(s, 250_000.0),
        precession_frequency_hz=jnp.full_like(s, 10_000.0),
        resonance_width_hz=jnp.full_like(s, 15_000.0),
    )
    parameters = jnp.zeros((4,))

    def objective(params):
        return boozer_fast_ion_risk_objective(
            params,
            geometry=geometry,
            harmonics=harmonics,
            mass_density=3.4e-8,
            population=population,
            toroidal_mode=1,
            transit_harmonics=(1,),
            avoid_window_hz=(50_000.0, 350_000.0),
            mode_count=3,
            drive_scale_hz=1_000.0,
            damping_rate_hz=5.0,
            theta_points=10,
            zeta_points=10,
            radial_stiffness_scale=1.0e-4,
            regularization=0.1,
        )

    screen = boozer_fast_ion_growth_screen(
        parameters,
        geometry=geometry,
        harmonics=harmonics,
        mass_density=3.4e-8,
        population=population,
        toroidal_mode=1,
        transit_harmonics=(1,),
        mode_count=3,
        drive_scale_hz=1_000.0,
        damping_rate_hz=5.0,
        theta_points=10,
        zeta_points=10,
        radial_stiffness_scale=1.0e-4,
    )
    gradient = jax.grad(objective)(parameters)
    optimized = gradient_descent_minimize(objective, parameters, step_size=0.05, steps=6, gradient_clip_norm=1.0)

    assert screen.net_growth_rate_hz.shape == (3,)
    assert gradient.shape == (4,)
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(optimized.objective_history[-1]) < float(optimized.objective_history[0])


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


def test_boozer_pipeline_isolated_objective_differentiates_through_builder():
    geometry = _synthetic_boozer_geometry()
    params = jnp.asarray([0.01, -0.02, 0.03])

    def geometry_builder(x):
        field_scale = 1.0 + 0.02 * x[1]
        return replace(
            geometry,
            iota=geometry.iota + 0.015 * x[0],
            bmnc=geometry.bmnc * field_scale,
            boozer_g=geometry.boozer_g * field_scale,
        )

    def density_builder(x, built_geometry):
        return jnp.full_like(built_geometry.s, 3.4e-8 * jnp.exp(x[2]))

    def objective(x):
        return boozer_pipeline_isolated_mode_avoidance_objective(
            x,
            geometry_builder=geometry_builder,
            harmonics=jnp.asarray([[1.0, 0.0], [2.0, 1.0]]),
            mass_density=density_builder,
            radial_stiffness_scale=lambda x, _geometry: 1.0e-4 * jnp.exp(0.1 * x[1]),
            avoid_window_hz=(50_000.0, 350_000.0),
            mode_index=0,
            theta_points=12,
            zeta_points=12,
        )

    value = objective(params)
    gradient = jax.grad(objective)(params)

    assert float(value) >= 0.0
    assert gradient.shape == (3,)
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_boozer_pipeline_cluster_mean_objective_differentiates_through_builder():
    geometry = _synthetic_boozer_geometry()
    params = jnp.asarray([0.01, -0.02, 0.03])

    def geometry_builder(x):
        field_scale = 1.0 + 0.02 * x[1]
        return replace(
            geometry,
            iota=geometry.iota + 0.015 * x[0],
            bmnc=geometry.bmnc * field_scale,
            boozer_g=geometry.boozer_g * field_scale,
        )

    def density_builder(x, built_geometry):
        return jnp.full_like(built_geometry.s, 3.4e-8 * jnp.exp(x[2]))

    def objective(x):
        return boozer_pipeline_cluster_mean_avoidance_objective(
            x,
            geometry_builder=geometry_builder,
            harmonics=jnp.asarray([[1.0, 0.0], [2.0, 1.0]]),
            mass_density=density_builder,
            radial_stiffness_scale=lambda x, _geometry: 1.0e-4 * jnp.exp(0.1 * x[1]),
            avoid_window_hz=(50_000.0, 350_000.0),
            start_index=0,
            stop_index=2,
            theta_points=12,
            zeta_points=12,
        )

    value = objective(params)
    gradient = jax.grad(objective)(params)

    assert float(value) >= 0.0
    assert gradient.shape == (3,)
    assert np.all(np.isfinite(np.asarray(gradient)))
