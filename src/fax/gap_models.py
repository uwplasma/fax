"""Analytic Alfvén gap estimates for validation and design screens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from .continuum import continuum_branch_matrix, local_continuum_angular_frequency
from .equilibrium import EquilibriumGeometry, analytic_circular_tokamak
from .units import MU0, PROTON_MASS, ev_to_joule


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


@dataclass(frozen=True)
class CoupledGapBranches:
    """Two-branch avoided-crossing model for a named Alfvén gap."""

    estimate: CoupledGapEstimate
    uncoupled_a_frequency_hz: jax.Array
    uncoupled_b_frequency_hz: jax.Array
    lower_frequency_hz: jax.Array
    upper_frequency_hz: jax.Array
    coupling_strength: float


@dataclass(frozen=True)
class BasicAlfvenGapMap:
    """Small analytic continuum/gap demo case used in examples and docs."""

    geometry: EquilibriumGeometry
    q: jax.Array
    continuum_harmonics: jax.Array
    continuum_branches_hz: jax.Array
    tae: CoupledGapBranches
    eae: CoupledGapBranches
    bae_frequency_hz: jax.Array
    bae_surface: jax.Array
    bae_q: jax.Array
    bae_frequency_at_surface_hz: jax.Array


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


def coupled_gap_branch_map(
    geometry: EquilibriumGeometry,
    *,
    m: int,
    n: int,
    delta_m: int,
    coupling_strength: float,
    family: str,
) -> CoupledGapBranches:
    """Return a two-branch avoided crossing for harmonics ``m`` and ``m+delta_m``.

    The uncoupled branches use the large-aspect-ratio continuum
    ``omega=v_A |n-m*iota|/R``. The coupled branches diagonalize a minimal
    Hermitian ``omega^2`` matrix with fractional off-diagonal coupling
    ``coupling_strength * omega_m * omega_{m+delta_m}``. This reproduces the
    textbook TAE/EAE avoided-crossing picture quickly enough for examples and
    validation smoke tests; it is not a replacement for the full continuum
    operator.
    """
    estimate = coupled_gap_estimate(
        geometry,
        m=m,
        n=n,
        delta_m=delta_m,
        coupling_strength=coupling_strength,
        family=family,
    )
    omega_a = local_continuum_angular_frequency(geometry, m=m, n=n)
    omega_b = local_continuum_angular_frequency(geometry, m=m + int(delta_m), n=n)
    omega_a2 = omega_a**2
    omega_b2 = omega_b**2
    off_diagonal = float(coupling_strength) * omega_a * omega_b
    center = 0.5 * (omega_a2 + omega_b2)
    split = jnp.sqrt(0.25 * (omega_a2 - omega_b2) ** 2 + off_diagonal**2)
    lower = jnp.sqrt(jnp.maximum(center - split, 0.0)) / (2.0 * jnp.pi)
    upper = jnp.sqrt(jnp.maximum(center + split, 0.0)) / (2.0 * jnp.pi)
    return CoupledGapBranches(
        estimate=estimate,
        uncoupled_a_frequency_hz=omega_a / (2.0 * jnp.pi),
        uncoupled_b_frequency_hz=omega_b / (2.0 * jnp.pi),
        lower_frequency_hz=lower,
        upper_frequency_hz=upper,
        coupling_strength=float(coupling_strength),
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


def tae_gap_branch_map(
    geometry: EquilibriumGeometry,
    *,
    m: int,
    n: int,
    inverse_aspect_ratio: float | None = None,
) -> CoupledGapBranches:
    """Return a minimal toroidicity-induced ``m,m+1`` gap map."""
    epsilon = float(inverse_aspect_ratio) if inverse_aspect_ratio is not None else float(geometry.minor_radius / geometry.major_radius)
    return coupled_gap_branch_map(
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


def eae_gap_branch_map(
    geometry: EquilibriumGeometry,
    *,
    m: int,
    n: int,
    ellipticity: float = 0.05,
) -> CoupledGapBranches:
    """Return a minimal ellipticity-induced ``m,m+2`` gap map."""
    return coupled_gap_branch_map(
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


def bae_accumulation_frequency_hz(
    geometry: EquilibriumGeometry,
    *,
    ion_temperature_ev: float,
    electron_temperature_ev: float,
    ion_mass: float = PROTON_MASS,
) -> jax.Array:
    """Return the fluid-order BAE accumulation-frequency estimate.

    The implemented normalization is
    ``omega_BAE = sqrt(7/4 + Te/Ti) * v_ti / (q R)`` with
    ``v_ti=sqrt(Ti/mi)`` and ``q=1/iota``. This is a low-frequency,
    finite-beta/acoustic gap marker for examples and gates; kinetic, diamagnetic
    and shaping corrections belong in later BAE validation layers.
    """
    if float(ion_temperature_ev) <= 0.0:
        raise ValueError("ion_temperature_ev must be positive")
    if float(electron_temperature_ev) < 0.0:
        raise ValueError("electron_temperature_ev must be non-negative")
    if float(ion_mass) <= 0.0:
        raise ValueError("ion_mass must be positive")
    tau = float(electron_temperature_ev) / float(ion_temperature_ev)
    ion_thermal_speed = jnp.sqrt(ev_to_joule(float(ion_temperature_ev)) / float(ion_mass))
    q_profile = 1.0 / jnp.maximum(jnp.abs(geometry.iota), 1.0e-12)
    omega = jnp.sqrt(1.75 + tau) * ion_thermal_speed / (q_profile * geometry.major_radius)
    return omega / (2.0 * jnp.pi)


def basic_alfven_gap_map(
    *,
    radial_points: int = 240,
    ion_temperature_ev: float = 2_000.0,
    electron_temperature_ev: float = 2_000.0,
) -> BasicAlfvenGapMap:
    """Return a fast analytic BAE/TAE/EAE continuum example.

    The case is a large-aspect-ratio circular tokamak profile with ``n=1`` and
    ``q`` spanning the usual ``q=3/2`` TAE and ``q=2`` EAE/BAE reference
    surfaces. It is designed for fast user education and regression figures,
    not for device-specific stability claims.
    """
    geometry = analytic_circular_tokamak(
        radial_points=radial_points,
        iota_axis=1.0 / 1.05,
        iota_edge=1.0 / 3.2,
        magnetic_field_axis=3.0,
        number_density=1.0e20,
        major_radius=3.0,
        minor_radius=0.75,
    )
    harmonics = jnp.asarray([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
    branches = continuum_branch_matrix(geometry, harmonics)
    tae = tae_gap_branch_map(geometry, m=1, n=1, inverse_aspect_ratio=0.10)
    eae = eae_gap_branch_map(geometry, m=1, n=1, ellipticity=0.06)
    bae_frequency = bae_accumulation_frequency_hz(
        geometry,
        ion_temperature_ev=ion_temperature_ev,
        electron_temperature_ev=electron_temperature_ev,
    )
    q_profile = 1.0 / geometry.iota
    bae_index = jnp.argmin(jnp.abs(q_profile - 2.0))
    return BasicAlfvenGapMap(
        geometry=geometry,
        q=q_profile,
        continuum_harmonics=harmonics,
        continuum_branches_hz=branches,
        tae=tae,
        eae=eae,
        bae_frequency_hz=bae_frequency,
        bae_surface=geometry.s[bae_index],
        bae_q=q_profile[bae_index],
        bae_frequency_at_surface_hz=bae_frequency[bae_index],
    )
