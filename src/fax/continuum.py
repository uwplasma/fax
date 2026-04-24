"""Local shear-Alfven continuum formulas and diagnostics."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .equilibrium import EquilibriumGeometry
from .units import MU0


def alfven_speed(magnetic_field: Any, mass_density: Any) -> jax.Array:
    """Return the Alfvén speed ``B / sqrt(mu0 * rho)`` in m/s."""
    return jnp.asarray(magnetic_field) / jnp.sqrt(MU0 * jnp.asarray(mass_density))


def large_aspect_k_parallel(iota: Any, *, m: int, n: int, major_radius: Any) -> jax.Array:
    """Return the large-aspect-ratio field-line wavenumber ``(n - m*iota)/R``.

    This is a limiting formula used for analytic tests and first smoke checks.
    Full stellarator continuum operators must replace it with metric-aware
    VMEC/Boozer operators.
    """
    return (float(n) - float(m) * jnp.asarray(iota)) / jnp.asarray(major_radius)


def local_continuum_angular_frequency(
    geometry: EquilibriumGeometry,
    *,
    m: int,
    n: int,
) -> jax.Array:
    """Return ``omega_A(s) = v_A(s) |k_parallel(s)|`` in rad/s."""
    speed = alfven_speed(geometry.magnetic_field, geometry.mass_density)
    k_parallel = large_aspect_k_parallel(geometry.iota, m=m, n=n, major_radius=geometry.major_radius)
    return speed * jnp.abs(k_parallel)


def local_continuum_frequency_hz(
    geometry: EquilibriumGeometry,
    *,
    m: int,
    n: int,
) -> jax.Array:
    """Return the local shear-Alfven continuum frequency in hertz."""
    return local_continuum_angular_frequency(geometry, m=m, n=n) / (2.0 * jnp.pi)


def continuum_branch_matrix(geometry: EquilibriumGeometry, harmonics: Any) -> jax.Array:
    """Return local continuum frequencies for harmonics ``[(m, n), ...]``.

    The returned array has shape ``(radial_points, harmonic_count)``.
    """
    harmonic_array = jnp.asarray(harmonics)
    if harmonic_array.ndim != 2 or harmonic_array.shape[1] != 2:
        raise ValueError("harmonics must have shape (count, 2) with columns (m, n)")

    def _branch(harmonic: jax.Array) -> jax.Array:
        m = harmonic[0]
        n = harmonic[1]
        speed = alfven_speed(geometry.magnetic_field, geometry.mass_density)
        k_parallel = (n - m * geometry.iota) / geometry.major_radius
        return speed * jnp.abs(k_parallel) / (2.0 * jnp.pi)

    return jax.vmap(_branch)(harmonic_array).T


def nearest_rational_surface(geometry: EquilibriumGeometry, *, m: int, n: int) -> tuple[jax.Array, jax.Array]:
    """Return ``(s, distance)`` for the nearest ``n/m`` resonance in iota."""
    if int(m) == 0:
        raise ValueError("m must be nonzero for a rational surface")
    distance = jnp.abs(geometry.iota - (float(n) / float(m)))
    index = jnp.argmin(distance)
    return geometry.s[index], distance[index]

