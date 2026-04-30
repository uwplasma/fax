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


@dataclass(frozen=True)
class GyroLandauCalibrationResult:
    """Least-squares calibration result for reduced gyro-Landau coefficients."""

    closure: "GyroLandauClosure"
    predicted_growth_rate_hz: jax.Array
    residual_hz: jax.Array
    rms_residual_hz: jax.Array
    design_matrix: jax.Array


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GyroLandauClosure:
    """Reduced FAR3D-style gyro-Landau closure coefficients.

    FAR3D evolves energetic-particle density and parallel-momentum moments and
    introduces wave-particle resonance physics through Landau-closure terms.
    FAX keeps this as a calibrated perturbative closure around fixed
    eigenfunctions: the Lorentzian part is the resonant drive/damping channel,
    the dispersive part changes sign across the resonance, and
    ``thermal_landau_damping_hz`` is a transparent damping offset.

    This is still a reduced closure, not a full CAS3D-K/FAR3D replacement, but
    it exposes the coefficients that must be calibrated in future parity runs.
    """

    resonant_coefficient: Any = 1.0
    dispersive_coefficient: Any = 0.0
    diamagnetic_frequency_hz: Any = 0.0
    thermal_landau_damping_hz: Any = 0.0

    def __post_init__(self) -> None:
        for name in (
            "resonant_coefficient",
            "dispersive_coefficient",
            "diamagnetic_frequency_hz",
            "thermal_landau_damping_hz",
        ):
            object.__setattr__(self, name, jnp.asarray(getattr(self, name)))

    def tree_flatten(self):
        children = (
            self.resonant_coefficient,
            self.dispersive_coefficient,
            self.diamagnetic_frequency_hz,
            self.thermal_landau_damping_hz,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux: None, children: tuple[Any, ...]) -> "GyroLandauClosure":
        resonant, dispersive, diamagnetic, damping = children
        return cls(
            resonant_coefficient=resonant,
            dispersive_coefficient=dispersive,
            diamagnetic_frequency_hz=diamagnetic,
            thermal_landau_damping_hz=damping,
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


def gyro_landau_closure_drive(
    mode_frequency_hz: Any,
    *,
    toroidal_mode: int,
    transit_harmonic: int,
    population: FastIonPopulation,
    closure: GyroLandauClosure,
) -> jax.Array:
    """Return a calibrated gyro-Landau energetic-particle drive density.

    The resonant term is even in the frequency mismatch and peaks at the
    transit/precession resonance. The dispersive term is odd in the mismatch
    and represents the non-reactive branch shift retained by two-moment
    Landau closures. The diamagnetic factor keeps the pressure-gradient drive
    tied to an externally calibrated drift/diamagnetic frequency scale.
    """
    mismatch = resonance_mismatch_hz(
        mode_frequency_hz,
        toroidal_mode=toroidal_mode,
        transit_harmonic=transit_harmonic,
        population=population,
    )
    width = jnp.maximum(population.resonance_width_hz, 1.0e-30)
    normalized_mismatch = mismatch / width
    denominator = 1.0 + normalized_mismatch**2
    resonant = closure.resonant_coefficient / denominator
    dispersive = closure.dispersive_coefficient * normalized_mismatch / denominator
    frequency_scale = jnp.maximum(jnp.abs(jnp.asarray(mode_frequency_hz)), width)
    diamagnetic_factor = 1.0 + closure.diamagnetic_frequency_hz / frequency_scale
    return -population.density_fraction * population.pressure_gradient * diamagnetic_factor * (resonant + dispersive)


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


def gyro_landau_eigenfunction_drive(
    frequencies_hz: Any,
    mode_structures: Any,
    s: Any,
    *,
    toroidal_mode: int,
    transit_harmonics: tuple[int, ...],
    population: FastIonPopulation,
    closure: GyroLandauClosure,
    drive_scale_hz: Any = 1.0,
    damping_rate_hz: Any = 0.0,
) -> FastIonModeDriveResult:
    """Project the reduced gyro-Landau closure onto global eigenfunctions.

    The output has the same contract as :func:`eigenfunction_fast_ion_drive`,
    but the drive density includes calibrated resonant, dispersive,
    diamagnetic, and thermal-Landau damping terms. This is the first
    FAR3D/CAS3D-K-facing closure API: future executable parity should fit the
    closure coefficients rather than changing the sign convention silently.
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
                gyro_landau_closure_drive(
                    frequency,
                    toroidal_mode=toroidal_mode,
                    transit_harmonic=harmonic,
                    population=population,
                    closure=closure,
                )
            )
        drive_rows.append(jnp.sum(jnp.stack(harmonic_drive), axis=0))
        overlap_rows.append(jnp.max(jnp.stack(harmonic_overlap), axis=0))

    drive_density = jnp.stack(drive_rows)
    overlap_density = jnp.stack(overlap_rows)
    weighted_drive = _trapz(mode_weights * drive_density, radial_grid) / norm
    weighted_overlap = _trapz(mode_weights * overlap_density, radial_grid) / norm
    damping = jnp.broadcast_to(jnp.asarray(damping_rate_hz) + closure.thermal_landau_damping_hz, weighted_drive.shape)
    net_growth = jnp.asarray(drive_scale_hz) * weighted_drive - damping
    return FastIonModeDriveResult(
        drive_indicator=weighted_drive,
        damping_rate_hz=damping,
        net_growth_rate_hz=net_growth,
        resonance_overlap=weighted_overlap,
        mode_weights=mode_weights,
    )


def fit_gyro_landau_eigenfunction_closure(
    frequencies_hz: Any,
    mode_structures: Any,
    s: Any,
    observed_growth_rate_hz: Any,
    *,
    toroidal_mode: int,
    transit_harmonics: tuple[int, ...],
    population: FastIonPopulation,
    diamagnetic_frequency_hz: Any = 0.0,
    drive_scale_hz: Any = 1.0,
    damping_rate_hz: Any = 0.0,
    ridge: float = 0.0,
) -> GyroLandauCalibrationResult:
    """Fit reduced gyro-Landau coefficients to observed mode growth rates.

    The fit is linear in ``C0``, ``C1``, and ``gamma_L`` when the diamagnetic
    frequency is fixed:

    ``gamma = drive_scale * (C0 A + C1 B) - damping - gamma_L``.

    This gives FAR3D/CAS3D-K parity studies a reproducible calibration layer
    before FAX has a first-principles drift-kinetic closure. The solve is a
    small differentiable normal-equation solve with optional ridge
    regularization.
    """
    frequencies = jnp.asarray(frequencies_hz)
    structures = jnp.asarray(mode_structures)
    radial_grid = jnp.asarray(s)
    observed = jnp.asarray(observed_growth_rate_hz)
    mode_weights = jnp.sum(jnp.abs(structures) ** 2, axis=-1)
    norm = jnp.maximum(_trapz(mode_weights, radial_grid), 1.0e-30)
    diamagnetic = jnp.asarray(diamagnetic_frequency_hz)

    resonant_rows = []
    dispersive_rows = []
    for frequency in frequencies:
        resonant_density = []
        dispersive_density = []
        for harmonic in transit_harmonics:
            mismatch = resonance_mismatch_hz(
                frequency,
                toroidal_mode=toroidal_mode,
                transit_harmonic=harmonic,
                population=population,
            )
            width = jnp.maximum(population.resonance_width_hz, 1.0e-30)
            normalized_mismatch = mismatch / width
            denominator = 1.0 + normalized_mismatch**2
            frequency_scale = jnp.maximum(jnp.abs(frequency), width)
            diamagnetic_factor = 1.0 + diamagnetic / frequency_scale
            base = -population.density_fraction * population.pressure_gradient * diamagnetic_factor
            resonant_density.append(base / denominator)
            dispersive_density.append(base * normalized_mismatch / denominator)
        resonant_rows.append(jnp.sum(jnp.stack(resonant_density), axis=0))
        dispersive_rows.append(jnp.sum(jnp.stack(dispersive_density), axis=0))

    resonant_projection = _trapz(mode_weights * jnp.stack(resonant_rows), radial_grid) / norm
    dispersive_projection = _trapz(mode_weights * jnp.stack(dispersive_rows), radial_grid) / norm
    matrix = jnp.stack(
        [
            jnp.asarray(drive_scale_hz) * resonant_projection,
            jnp.asarray(drive_scale_hz) * dispersive_projection,
            -jnp.ones_like(resonant_projection),
        ],
        axis=1,
    )
    rhs = observed + jnp.asarray(damping_rate_hz)
    normal = matrix.T @ matrix + jnp.asarray(ridge, dtype=matrix.dtype) * jnp.eye(matrix.shape[1], dtype=matrix.dtype)
    coefficients = jnp.linalg.solve(normal, matrix.T @ rhs)
    predicted = matrix @ coefficients - jnp.asarray(damping_rate_hz)
    residual = predicted - observed
    closure = GyroLandauClosure(
        resonant_coefficient=coefficients[0],
        dispersive_coefficient=coefficients[1],
        diamagnetic_frequency_hz=diamagnetic,
        thermal_landau_damping_hz=coefficients[2],
    )
    return GyroLandauCalibrationResult(
        closure=closure,
        predicted_growth_rate_hz=predicted,
        residual_hz=residual,
        rms_residual_hz=jnp.sqrt(jnp.mean(residual**2)),
        design_matrix=matrix,
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
