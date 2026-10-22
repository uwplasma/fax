"""Reduced energetic-particle drive screen (interim until the perturbative model, plan phase P4).

A mode of frequency ``f`` and toroidal number ``n`` resonates with fast ions
when ``f - n f_precession - l f_transit ~ 0``. The resonance is a Lorentzian
of width ``width_hz``; the drive density is ``-n_f dp_f/ds`` times the
resonance, averaged over the mode energy ``sum_h |xi_h(s)|^2``::

    gamma_k = drive_scale_hz * < D(s, f_k) >_k - damping_hz.

The result ranks modes; it is not a calibrated growth rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class FastIons:
    """Fast-ion parameters; each may be a scalar or a radial profile on the mode grid."""

    density_fraction: Any = 0.01
    pressure_gradient: Any = -1.0
    transit_hz: Any = 100.0e3
    precession_hz: Any = 5.0e3
    width_hz: Any = 10.0e3


jax.tree_util.register_dataclass(
    FastIons, data_fields=["density_fraction", "pressure_gradient", "transit_hz", "precession_hz", "width_hz"], meta_fields=[]
)


def resonance(f_hz: Any, n: int, l: int, ions: FastIons) -> jax.Array:  # noqa: E741
    """Lorentzian weight of ``f - n f_precession - l f_transit``."""
    mismatch = jnp.asarray(f_hz) - n * jnp.asarray(ions.precession_hz) - l * jnp.asarray(ions.transit_hz)
    return 1.0 / (1.0 + (mismatch / jnp.asarray(ions.width_hz)) ** 2)


def growth_rate(
    frequencies_hz: Any,
    structures: Any,
    s: Any,
    ions: FastIons,
    *,
    n: int,
    transit_harmonics: tuple[int, ...] = (-1, 0, 1),
    drive_scale_hz: float = 1.0e4,
    damping_hz: Any = 0.0,
) -> jax.Array:
    """Mode-energy-weighted drive minus damping for each mode, in 1/s."""
    weight = jnp.sum(jnp.abs(jnp.asarray(structures)) ** 2, axis=-1)
    weight = weight / _integrate(weight, s)[:, None]
    f = jnp.asarray(frequencies_hz)[:, None]
    res = sum(resonance(f, n, l, ions) for l in transit_harmonics)  # noqa: E741
    density = -jnp.asarray(ions.density_fraction) * jnp.asarray(ions.pressure_gradient) * res
    return drive_scale_hz * _integrate(weight * density, s) - jnp.asarray(damping_hz)


def _integrate(y: jax.Array, s: Any) -> jax.Array:
    """Trapezoidal rule along the last axis (``jnp.trapezoid`` is missing from older JAX)."""
    return 0.5 * jnp.sum((y[..., 1:] + y[..., :-1]) * jnp.diff(jnp.asarray(s)), axis=-1)
