"""Differentiable design objectives for continuum and eigenmode screens."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp

from .equilibrium import EquilibriumGeometry
from .global_modes import radial_problem_from_equilibrium, solve_radial_eigenmodes


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
