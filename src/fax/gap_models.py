"""Analytic Alfvén gap estimates for validation and design screens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from .equilibrium import EquilibriumGeometry
from .units import MU0


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CoupledGapEstimate:
    """Analytic estimate for a coupled shear-Alfven continuum gap."""

    family: str
    m: int
    n: int
    delta_m: int
    surface: jax.Array
    iota: jax.Array
    center_frequency_hz: jax.Array
    width_frequency_hz: jax.Array
    lower_frequency_hz: jax.Array
    upper_frequency_hz: jax.Array

    def tree_flatten(self):
        children = (
            self.surface,
            self.iota,
            self.center_frequency_hz,
            self.width_frequency_hz,
            self.lower_frequency_hz,
            self.upper_frequency_hz,
        )
        return children, (self.family, self.m, self.n, self.delta_m)

    @classmethod
    def tree_unflatten(cls, aux: tuple[Any, ...], children: tuple[Any, ...]) -> "CoupledGapEstimate":
        family, m, n, delta_m = aux
        surface, iota, center, width, lower, upper = children
        return cls(
            family=family,
            m=m,
            n=n,
            delta_m=delta_m,
            surface=surface,
            iota=iota,
            center_frequency_hz=center,
            width_frequency_hz=width,
            lower_frequency_hz=lower,
            upper_frequency_hz=upper,
        )


def coupled_gap_estimate(
    geometry: EquilibriumGeometry,
    *,
    m: int,
    n: int,
    delta_m: int,
    coupling_strength: float,
    family: str,
) -> CoupledGapEstimate:
    """Estimate a gap opened by coupling harmonics ``m`` and ``m + delta_m``.

    The crossing condition solves
    ``n - m*iota = -(n - (m + delta_m)*iota)``, giving
    ``iota = 2*n/(2*m + delta_m)``. The center frequency is the local
    shear-Alfven frequency at that crossing. ``coupling_strength`` is a
    dimensionless fractional width parameter used for analytic large-aspect
    ratio gates and early design screens.
    """
    if int(delta_m) <= 0:
        raise ValueError("delta_m must be positive")
    if float(coupling_strength) < 0.0:
        raise ValueError("coupling_strength must be non-negative")
    target_iota = 2.0 * float(n) / (2.0 * float(m) + float(delta_m))
    index = jnp.argmin(jnp.abs(geometry.iota - target_iota))
    iota = geometry.iota[index]
    magnetic_field = geometry.magnetic_field[index]
    mass_density = geometry.mass_density[index]
    v_alfven = magnetic_field / jnp.sqrt(MU0 * mass_density)
    k_parallel = jnp.abs(float(n) - float(m) * iota) / geometry.major_radius
    center_hz = v_alfven * k_parallel / (2.0 * jnp.pi)
    width_hz = float(coupling_strength) * center_hz
    return CoupledGapEstimate(
        family=str(family),
        m=int(m),
        n=int(n),
        delta_m=int(delta_m),
        surface=geometry.s[index],
        iota=iota,
        center_frequency_hz=center_hz,
        width_frequency_hz=width_hz,
        lower_frequency_hz=jnp.maximum(center_hz - 0.5 * width_hz, 0.0),
        upper_frequency_hz=center_hz + 0.5 * width_hz,
    )


def tae_gap_estimate(
    geometry: EquilibriumGeometry,
    *,
    m: int,
    n: int,
    inverse_aspect_ratio: float | None = None,
) -> CoupledGapEstimate:
    """Estimate a toroidicity-induced Alfvén eigenmode gap."""
    epsilon = float(inverse_aspect_ratio) if inverse_aspect_ratio is not None else float(geometry.minor_radius / geometry.major_radius)
    return coupled_gap_estimate(
        geometry,
        m=m,
        n=n,
        delta_m=1,
        coupling_strength=epsilon,
        family="TAE",
    )


def eae_gap_estimate(
    geometry: EquilibriumGeometry,
    *,
    m: int,
    n: int,
    ellipticity: float = 0.05,
) -> CoupledGapEstimate:
    """Estimate an ellipticity-induced Alfvén eigenmode gap."""
    return coupled_gap_estimate(
        geometry,
        m=m,
        n=n,
        delta_m=2,
        coupling_strength=float(ellipticity),
        family="EAE",
    )


def hae_gap_estimate(
    geometry: EquilibriumGeometry,
    *,
    m: int,
    n: int,
    helical_coupling: float = 0.05,
    delta_m: int = 1,
) -> CoupledGapEstimate:
    """Estimate a helicity-induced Alfvén eigenmode gap."""
    return coupled_gap_estimate(
        geometry,
        m=m,
        n=n,
        delta_m=int(delta_m),
        coupling_strength=float(helical_coupling),
        family="HAE",
    )
