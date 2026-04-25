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


@dataclass(frozen=True)
class FastIonModeDriveResult:
    """Mode-resolved energetic-particle drive and damping diagnostics."""

    drive_indicator: jax.Array
    damping_rate_hz: jax.Array
    net_growth_rate_hz: jax.Array
    resonance_overlap: jax.Array
    mode_weights: jax.Array


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


def eigenfunction_fast_ion_drive(
    frequencies_hz: Any,
    mode_structures: Any,
    s: Any,
    *,
    toroidal_mode: int,
    transit_harmonics: tuple[int, ...],
    population: FastIonPopulation,
    drive_scale_hz: Any = 1.0,
    damping_rate_hz: Any = 0.0,
) -> FastIonModeDriveResult:
    """Project reduced fast-ion drive onto radial eigenfunction structures.

    The reduced resonance drive is evaluated on the radial population profile
    and weighted by ``sum_h |xi_h(s)|^2`` for each eigenmode. The returned
    ``net_growth_rate_hz`` is a signed screening metric,

    ``gamma = drive_scale_hz * <drive>_xi - damping_rate_hz``.

    This is still a perturbative diagnostic rather than a drift-kinetic
    closure, but it ties the standard resonance condition to the global
    eigenfunctions used by the current FAX operator.
    """
    frequencies = jnp.asarray(frequencies_hz)
    structures = jnp.asarray(mode_structures)
    radial_grid = jnp.asarray(s)
    mode_weights = jnp.sum(jnp.abs(structures) ** 2, axis=-1)
    norm = jnp.maximum(_trapz(mode_weights, radial_grid), 1.0e-30)

    drive_rows = []
    overlap_rows = []
    for frequency in frequencies:
        harmonic_drive = []
        harmonic_overlap = []
        for harmonic in transit_harmonics:
            mismatch = resonance_mismatch_hz(
                frequency,
                toroidal_mode=toroidal_mode,
                transit_harmonic=harmonic,
                population=population,
            )
            harmonic_overlap.append(lorentzian_resonance_weight(mismatch, population.resonance_width_hz))
            harmonic_drive.append(
                reduced_fast_ion_drive(
                    frequency,
                    toroidal_mode=toroidal_mode,
                    transit_harmonic=harmonic,
                    population=population,
                )
            )
        drive_rows.append(jnp.sum(jnp.stack(harmonic_drive), axis=0))
        overlap_rows.append(jnp.max(jnp.stack(harmonic_overlap), axis=0))

    drive_density = jnp.stack(drive_rows)
    overlap_density = jnp.stack(overlap_rows)
    weighted_drive = _trapz(mode_weights * drive_density, radial_grid) / norm
    weighted_overlap = _trapz(mode_weights * overlap_density, radial_grid) / norm
    damping = jnp.broadcast_to(jnp.asarray(damping_rate_hz), weighted_drive.shape)
    net_growth = jnp.asarray(drive_scale_hz) * weighted_drive - damping
    return FastIonModeDriveResult(
        drive_indicator=weighted_drive,
        damping_rate_hz=damping,
        net_growth_rate_hz=net_growth,
        resonance_overlap=weighted_overlap,
        mode_weights=mode_weights,
    )


def _trapz(values: jax.Array, x: jax.Array) -> jax.Array:
    dx = x[1:] - x[:-1]
    return jnp.sum(0.5 * (values[..., 1:] + values[..., :-1]) * dx, axis=-1)
