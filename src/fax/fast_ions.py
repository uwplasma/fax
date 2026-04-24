"""Reduced energetic-particle resonance diagnostics.

These functions implement transparent, differentiable resonance metrics used
for early design screening. They are not a substitute for a full drift-kinetic
or gyro-fluid energetic-particle model, but the resonance condition and signs
match the standard energetic-particle drive picture used before coupling to
FAR3D/CAS3D-K class models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FastIonPopulation:
    """Minimal energetic-particle population for resonance screening."""

    density_fraction: Any
    pressure_gradient: Any
    transit_frequency_hz: Any
    precession_frequency_hz: Any
    resonance_width_hz: Any

    def __post_init__(self) -> None:
        for name in (
            "density_fraction",
            "pressure_gradient",
            "transit_frequency_hz",
            "precession_frequency_hz",
            "resonance_width_hz",
        ):
            object.__setattr__(self, name, jnp.asarray(getattr(self, name)))

    def tree_flatten(self):
        children = (
            self.density_fraction,
            self.pressure_gradient,
            self.transit_frequency_hz,
            self.precession_frequency_hz,
            self.resonance_width_hz,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux: None, children: tuple[Any, ...]) -> "FastIonPopulation":
        density_fraction, pressure_gradient, transit, precession, width = children
        return cls(
            density_fraction=density_fraction,
            pressure_gradient=pressure_gradient,
            transit_frequency_hz=transit,
            precession_frequency_hz=precession,
            resonance_width_hz=width,
        )


def resonance_mismatch_hz(
    mode_frequency_hz: Any,
    *,
    toroidal_mode: int,
    transit_harmonic: int,
    population: FastIonPopulation,
) -> jax.Array:
    """Return ``f_mode - n*f_precession - l*f_transit``."""
    return (
        jnp.asarray(mode_frequency_hz)
        - float(toroidal_mode) * population.precession_frequency_hz
        - float(transit_harmonic) * population.transit_frequency_hz
    )


def lorentzian_resonance_weight(mismatch_hz: Any, width_hz: Any) -> jax.Array:
    """Return a normalized Lorentzian resonance weight."""
    width = jnp.maximum(jnp.asarray(width_hz), 1.0e-30)
    return 1.0 / (1.0 + (jnp.asarray(mismatch_hz) / width) ** 2)


def reduced_fast_ion_drive(
    mode_frequency_hz: Any,
    *,
    toroidal_mode: int,
    transit_harmonic: int,
    population: FastIonPopulation,
) -> jax.Array:
    """Return a reduced energetic-particle drive indicator.

    Positive outward pressure gradient in this sign convention is stabilizing;
    negative pressure gradient can drive the mode. The returned value is
    dimensionless and intended for ranking/design screens.
    """
    mismatch = resonance_mismatch_hz(
        mode_frequency_hz,
        toroidal_mode=toroidal_mode,
        transit_harmonic=transit_harmonic,
        population=population,
    )
    resonance = lorentzian_resonance_weight(mismatch, population.resonance_width_hz)
    return -population.density_fraction * population.pressure_gradient * resonance


def resonance_scan(
    mode_frequency_hz: Any,
    *,
    toroidal_mode: int,
    transit_harmonics: tuple[int, ...],
    population: FastIonPopulation,
) -> jax.Array:
    """Return reduced drive values for a set of transit harmonics."""
    return jnp.asarray(
        [
            reduced_fast_ion_drive(
                mode_frequency_hz,
                toroidal_mode=toroidal_mode,
                transit_harmonic=harmonic,
                population=population,
            )
            for harmonic in transit_harmonics
        ]
    )
