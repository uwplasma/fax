"""Differentiable design objectives for continuum and eigenmode screens."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp

from .boozer import BoozerGeometry
from .equilibrium import EquilibriumGeometry
from .fast_ions import FastIonModeDriveResult, FastIonPopulation, eigenfunction_fast_ion_drive
from .global_modes import (
    radial_block_problem_from_boozer_geometry,
    radial_problem_from_equilibrium,
    solve_radial_block_eigenmodes,
    solve_radial_eigenmodes,
)


def frequency_window_penalty(
    frequencies_hz: Any,
    *,
    lower_hz: float,
    upper_hz: float,
    softness_hz: float,
) -> jax.Array:
    """Smoothly penalize frequencies inside a target exclusion window."""
    frequencies = jnp.asarray(frequencies_hz)
    softness = jnp.maximum(float(softness_hz), 1.0e-30)
    above_lower = jax.nn.sigmoid((frequencies - float(lower_hz)) / softness)
    below_upper = jax.nn.sigmoid((float(upper_hz) - frequencies) / softness)
    return jnp.mean(above_lower * below_upper)


def scaled_profile_geometry(
    geometry: EquilibriumGeometry,
    *,
    density_scale: Any = 1.0,
    iota_shift: Any = 0.0,
    field_scale: Any = 1.0,
) -> EquilibriumGeometry:
    """Return a differentiably perturbed profile geometry."""
    return replace(
        geometry,
        iota=geometry.iota + jnp.asarray(iota_shift),
        magnetic_field=geometry.magnetic_field * jnp.asarray(field_scale),
        mass_density=geometry.mass_density * jnp.asarray(density_scale),
    )


def shifted_boozer_geometry(
    geometry: BoozerGeometry,
    *,
    iota_shift: Any = 0.0,
) -> BoozerGeometry:
    """Return a Boozer geometry with differentiably shifted rotational transform."""
    return replace(geometry, iota=geometry.iota + jnp.asarray(iota_shift))


@dataclass(frozen=True)
class LinearizedUncertaintyEstimate:
    """Linear uncertainty propagation for differentiable mode frequencies."""

    mean_hz: jax.Array
    sigma_hz: jax.Array
    lower_hz: jax.Array
    upper_hz: jax.Array
    jacobian_hz_per_parameter: jax.Array
    covariance: jax.Array


@dataclass(frozen=True)
class GradientDescentResult:
    """History from a small differentiable inverse-design optimization."""

    parameters: jax.Array
    objective_history: jax.Array
    gradient_norm_history: jax.Array
    parameter_history: jax.Array


def radial_mode_frequency_vector(
    parameters: Any,
    *,
    geometry: EquilibriumGeometry,
    harmonic: tuple[int, int],
    mode_count: int = 6,
    radial_coupling_fraction: float = 0.025,
) -> jax.Array:
    """Return low-order radial eigenfrequencies for profile design variables.

    ``parameters`` are ``[log_density_scale, iota_shift, log_field_scale]``.
    This gives sensitivity, inverse-design, and UQ workflows a compact
    differentiable frequency map before the full VMEC/Boozer path is available.
    """
    params = jnp.asarray(parameters)
    adjusted = scaled_profile_geometry(
        geometry,
        density_scale=jnp.exp(params[0]),
        iota_shift=params[1],
        field_scale=jnp.exp(params[2]),
    )
    problem = radial_problem_from_equilibrium(
        adjusted,
        harmonic=harmonic,
        radial_coupling_fraction=radial_coupling_fraction,
    )
    return solve_radial_eigenmodes(problem).frequencies_hz[: int(mode_count)]


def radial_mode_frequency_jacobian(
    parameters: Any,
    *,
    geometry: EquilibriumGeometry,
    harmonic: tuple[int, int],
    mode_count: int = 6,
    radial_coupling_fraction: float = 0.025,
) -> jax.Array:
    """Return ``d f_mode / d parameter`` for radial profile variables."""

    def frequency_map(params: jax.Array) -> jax.Array:
        return radial_mode_frequency_vector(
            params,
            geometry=geometry,
            harmonic=harmonic,
            mode_count=mode_count,
            radial_coupling_fraction=radial_coupling_fraction,
        )

    return jax.jacfwd(frequency_map)(jnp.asarray(parameters))


def linearized_radial_mode_uncertainty(
    parameters: Any,
    *,
    covariance: Any,
    geometry: EquilibriumGeometry,
    harmonic: tuple[int, int],
    mode_count: int = 6,
    sigma_multiplier: float = 2.0,
    radial_coupling_fraction: float = 0.025,
) -> LinearizedUncertaintyEstimate:
    """Propagate parameter covariance to mode-frequency uncertainty.

    The returned interval is the local linearized band
    ``f +/- sigma_multiplier * sqrt(J C J^T)``. It is useful for fast
    reviewer-facing screens and for deciding which uncertain design variables
    need full Monte Carlo or adjoint follow-up.
    """
    params = jnp.asarray(parameters)
    covariance_array = jnp.asarray(covariance)
    mean = radial_mode_frequency_vector(
        params,
        geometry=geometry,
        harmonic=harmonic,
        mode_count=mode_count,
        radial_coupling_fraction=radial_coupling_fraction,
    )
    jacobian = radial_mode_frequency_jacobian(
        params,
        geometry=geometry,
        harmonic=harmonic,
        mode_count=mode_count,
        radial_coupling_fraction=radial_coupling_fraction,
    )
    variance = jnp.einsum("ip,pq,iq->i", jacobian, covariance_array, jacobian)
    sigma = jnp.sqrt(jnp.maximum(variance, 0.0))
    width = float(sigma_multiplier) * sigma
    return LinearizedUncertaintyEstimate(
        mean_hz=mean,
        sigma_hz=sigma,
        lower_hz=mean - width,
        upper_hz=mean + width,
        jacobian_hz_per_parameter=jacobian,
        covariance=covariance_array,
    )


def gradient_descent_minimize(
    objective: Callable[[jax.Array], jax.Array],
    initial_parameters: Any,
    *,
    step_size: float = 0.1,
    steps: int = 32,
    gradient_clip_norm: float | None = 10.0,
) -> GradientDescentResult:
    """Run a small transparent gradient-descent loop for examples and tests."""
    params = jnp.asarray(initial_parameters)
    history = []
    gradient_norms = []
    parameter_history = [params]
    value_and_grad = jax.value_and_grad(objective)
    for _ in range(int(steps)):
        value, gradient = value_and_grad(params)
        gradient_norm = jnp.linalg.norm(gradient)
        if gradient_clip_norm is not None:
            scale = jnp.minimum(1.0, float(gradient_clip_norm) / jnp.maximum(gradient_norm, 1.0e-30))
            gradient = gradient * scale
        history.append(value)
        gradient_norms.append(gradient_norm)
        params = params - float(step_size) * gradient
        parameter_history.append(params)
    return GradientDescentResult(
        parameters=params,
        objective_history=jnp.asarray(history),
        gradient_norm_history=jnp.asarray(gradient_norms),
        parameter_history=jnp.asarray(parameter_history),
    )


def radial_mode_inverse_design(
    initial_parameters: Any,
    *,
    geometry: EquilibriumGeometry,
    harmonic: tuple[int, int],
    target_frequencies_hz: Any,
    mode_count: int | None = None,
    step_size: float = 0.1,
    steps: int = 32,
    regularization: float = 1.0e-3,
    parameter_prior: Any | None = None,
    radial_coupling_fraction: float = 0.025,
) -> GradientDescentResult:
    """Fit profile parameters to target radial eigenfrequencies.

    This is a lightweight inverse-design primitive: it is deliberately simple,
    deterministic, and easy to audit. Larger design studies can replace the
    optimizer while reusing the same differentiable frequency map.
    """
    targets = jnp.asarray(target_frequencies_hz)
    count = int(mode_count if mode_count is not None else targets.shape[0])
    prior = jnp.zeros_like(jnp.asarray(initial_parameters)) if parameter_prior is None else jnp.asarray(parameter_prior)
    scale = jnp.maximum(jnp.mean(jnp.abs(targets[:count])), 1.0)

    def objective(params: jax.Array) -> jax.Array:
        frequencies = radial_mode_frequency_vector(
            params,
            geometry=geometry,
            harmonic=harmonic,
            mode_count=count,
            radial_coupling_fraction=radial_coupling_fraction,
        )
        mismatch = (frequencies - targets[:count]) / scale
        return jnp.mean(mismatch**2) + float(regularization) * jnp.mean((params - prior) ** 2)

    return gradient_descent_minimize(
        objective,
        initial_parameters,
        step_size=step_size,
        steps=steps,
        gradient_clip_norm=5.0,
    )


def radial_fast_ion_growth_screen(
    parameters: Any,
    *,
    geometry: EquilibriumGeometry,
    harmonic: tuple[int, int],
    population: FastIonPopulation,
    toroidal_mode: int,
    transit_harmonics: tuple[int, ...],
    mode_count: int = 4,
    drive_scale_hz: Any = 1.0,
    damping_rate_hz: Any = 0.0,
    radial_coupling_fraction: float = 0.025,
) -> FastIonModeDriveResult:
    """Project a reduced fast-ion drive screen onto differentiable eigenmodes.

    ``parameters`` may contain three or four entries. The first three are the
    profile variables used by :func:`radial_mode_frequency_vector`; an optional
    fourth entry is a log scale on fast-ion density fraction.
    """
    params = jnp.asarray(parameters)
    adjusted = scaled_profile_geometry(
        geometry,
        density_scale=jnp.exp(params[0]),
        iota_shift=params[1],
        field_scale=jnp.exp(params[2]),
    )
    problem = radial_problem_from_equilibrium(
        adjusted,
        harmonic=harmonic,
        radial_coupling_fraction=radial_coupling_fraction,
    )
    result = solve_radial_eigenmodes(problem)
    count = int(mode_count)
    interior = jnp.swapaxes(result.eigenvectors[:, :count], 0, 1)
    boundary = jnp.zeros((count, 1), dtype=interior.dtype)
    structures = jnp.concatenate([boundary, interior, boundary], axis=1)[:, :, None]
    density_scale = jnp.exp(params[3]) if params.shape[0] > 3 else 1.0
    scaled_population = replace(population, density_fraction=population.density_fraction * density_scale)
    return eigenfunction_fast_ion_drive(
        result.frequencies_hz[:count],
        structures,
        problem.s,
        toroidal_mode=toroidal_mode,
        transit_harmonics=transit_harmonics,
        population=scaled_population,
        drive_scale_hz=drive_scale_hz,
        damping_rate_hz=damping_rate_hz,
    )


def radial_fast_ion_risk_objective(
    parameters: Any,
    *,
    geometry: EquilibriumGeometry,
    harmonic: tuple[int, int],
    population: FastIonPopulation,
    toroidal_mode: int,
    transit_harmonics: tuple[int, ...],
    avoid_window_hz: tuple[float, float],
    mode_count: int = 4,
    drive_scale_hz: Any = 1.0,
    damping_rate_hz: Any = 0.0,
    growth_softness_hz: float = 5.0,
    frequency_softness_hz: float = 2.0e3,
    regularization: float = 1.0e-3,
    radial_coupling_fraction: float = 0.025,
) -> jax.Array:
    """Combine frequency-window avoidance with perturbative fast-ion risk."""
    params = jnp.asarray(parameters)
    frequencies = radial_mode_frequency_vector(
        params[:3],
        geometry=geometry,
        harmonic=harmonic,
        mode_count=mode_count,
        radial_coupling_fraction=radial_coupling_fraction,
    )
    drive = radial_fast_ion_growth_screen(
        params,
        geometry=geometry,
        harmonic=harmonic,
        population=population,
        toroidal_mode=toroidal_mode,
        transit_harmonics=transit_harmonics,
        mode_count=mode_count,
        drive_scale_hz=drive_scale_hz,
        damping_rate_hz=damping_rate_hz,
        radial_coupling_fraction=radial_coupling_fraction,
    )
    window = frequency_window_penalty(
        frequencies,
        lower_hz=float(avoid_window_hz[0]),
        upper_hz=float(avoid_window_hz[1]),
        softness_hz=float(frequency_softness_hz),
    )
    growth_penalty = jnp.mean(jax.nn.softplus(drive.net_growth_rate_hz / float(growth_softness_hz)))
    return window + growth_penalty + float(regularization) * jnp.mean(params**2)


def radial_mode_avoidance_objective(
    parameters: Any,
    *,
    geometry: EquilibriumGeometry,
    harmonic: tuple[int, int],
    avoid_window_hz: tuple[float, float],
    softness_hz: float = 2.0e3,
    regularization: float = 1.0e-3,
) -> jax.Array:
    """Return a differentiable objective for avoiding a frequency window.

    ``parameters`` are ``[log_density_scale, iota_shift, log_field_scale]``.
    Log scaling keeps density and field positive in unconstrained optimizers.
    """
    params = jnp.asarray(parameters)
    adjusted = scaled_profile_geometry(
        geometry,
        density_scale=jnp.exp(params[0]),
        iota_shift=params[1],
        field_scale=jnp.exp(params[2]),
    )
    problem = radial_problem_from_equilibrium(adjusted, harmonic=harmonic)
    frequencies = solve_radial_eigenmodes(problem).frequencies_hz[:6]
    penalty = frequency_window_penalty(
        frequencies,
        lower_hz=float(avoid_window_hz[0]),
        upper_hz=float(avoid_window_hz[1]),
        softness_hz=float(softness_hz),
    )
    return penalty + float(regularization) * jnp.sum(params**2)


def boozer_global_mode_avoidance_objective(
    parameters: Any,
    *,
    geometry: BoozerGeometry,
    harmonics: Any,
    mass_density: Any,
    avoid_window_hz: tuple[float, float],
    theta_points: int = 16,
    zeta_points: int = 16,
    mode_count: int = 6,
    softness_hz: float = 2.0e3,
    regularization: float = 1.0e-3,
    boundary_condition: str = "dirichlet",
) -> jax.Array:
    """Return a differentiable Boozer global-mode avoidance objective.

    ``parameters`` are ``[log_density_scale, iota_shift, log_radial_stiffness_scale]``.
    The objective recomputes the local 3-D Boozer matrices, assembles the global
    radial block operator, and penalizes eigenfrequencies inside
    ``avoid_window_hz``. When the input ``geometry`` comes from differentiable
    VMEC/BOOZ_XFORM adapters, this is the objective layer used by end-to-end
    optimization workflows.
    """
    params = jnp.asarray(parameters)
    adjusted = shifted_boozer_geometry(geometry, iota_shift=params[1])
    problem = radial_block_problem_from_boozer_geometry(
        adjusted,
        harmonics,
        mass_density=jnp.asarray(mass_density) * jnp.exp(params[0]),
        radial_stiffness_method="metric",
        radial_stiffness_scale=jnp.exp(params[2]),
        theta_points=theta_points,
        zeta_points=zeta_points,
        boundary_condition=boundary_condition,
    )
    frequencies = solve_radial_block_eigenmodes(problem).frequencies_hz[: int(mode_count)]
    penalty = frequency_window_penalty(
        frequencies,
        lower_hz=float(avoid_window_hz[0]),
        upper_hz=float(avoid_window_hz[1]),
        softness_hz=float(softness_hz),
    )
    return penalty + float(regularization) * jnp.sum(params**2)


def boozer_pipeline_mode_avoidance_objective(
    parameters: Any,
    *,
    geometry_builder: Callable[[jax.Array], BoozerGeometry],
    harmonics: Any,
    mass_density: Any,
    avoid_window_hz: tuple[float, float],
    radial_stiffness_scale: Any = 1.0,
    theta_points: int = 16,
    zeta_points: int = 16,
    mode_count: int = 6,
    softness_hz: float = 2.0e3,
    regularization: float = 1.0e-3,
    boundary_condition: str = "dirichlet",
) -> jax.Array:
    """Return an objective for differentiable VMEC/Boozer-to-FAX pipelines.

    ``geometry_builder(parameters)`` must return a JAX-pytree
    :class:`~fax.boozer.BoozerGeometry`. In production this callable can wrap a
    differentiable ``vmec_jax``/``booz_xform_jax`` path. In tests it can be a
    small analytic geometry builder. ``mass_density`` and
    ``radial_stiffness_scale`` may be arrays/scalars or callables with
    signature ``(parameters, geometry)``.

    This function intentionally keeps the external-equilibrium adapter outside
    FAX's core operator. The contract is simple: if the builder is
    differentiable, the frequency-window objective remains differentiable.
    """
    params = jnp.asarray(parameters)
    geometry = geometry_builder(params)
    density = _evaluate_parameter_source(mass_density, params, geometry)
    stiffness_scale = _evaluate_parameter_source(radial_stiffness_scale, params, geometry)
    problem = radial_block_problem_from_boozer_geometry(
        geometry,
        harmonics,
        mass_density=density,
        radial_stiffness_method="metric",
        radial_stiffness_scale=stiffness_scale,
        theta_points=theta_points,
        zeta_points=zeta_points,
        boundary_condition=boundary_condition,
    )
    frequencies = solve_radial_block_eigenmodes(problem).frequencies_hz[: int(mode_count)]
    penalty = frequency_window_penalty(
        frequencies,
        lower_hz=float(avoid_window_hz[0]),
        upper_hz=float(avoid_window_hz[1]),
        softness_hz=float(softness_hz),
    )
    return penalty + float(regularization) * jnp.sum(params**2)


def _evaluate_parameter_source(source: Any, parameters: jax.Array, geometry: BoozerGeometry) -> Any:
    if callable(source):
        return source(parameters, geometry)
    return source
