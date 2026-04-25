"""Differentiable design objectives for continuum and eigenmode screens."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp

from .boozer import BoozerGeometry
from .equilibrium import EquilibriumGeometry
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
