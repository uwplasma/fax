"""Readers and comparison metrics for external Alfvén codes.

The functions here intentionally read plain-text artifacts produced by
STELLGAP/AE3D style workflows instead of depending on those Fortran codes at
runtime. This makes cross-code validation reproducible in local tests while
keeping the core FAX package pure Python/JAX.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .boozer import boozer_3d_continuum_frequencies_hz, read_boozmn_geometry


@dataclass(frozen=True)
class StellgapContinuumData:
    """STELLGAP continuum points from ``alfven_post`` or ``alfven_spec``."""

    s: jax.Array
    frequency_khz: jax.Array
    m: jax.Array
    n: jax.Array


@dataclass(frozen=True)
class Ae3dEigenvalueData:
    """AE3D eigenvalue table from ``egn_values.dat``."""

    frequency_khz: jax.Array
    residual_real: jax.Array
    residual_imag: jax.Array


@dataclass(frozen=True)
class Ae3dResidualSummary:
    """Residual and frequency range summary for an AE3D eigenvalue table."""

    count: int
    min_frequency_khz: float
    max_frequency_khz: float
    p95_residual: float
    max_residual: float


@dataclass(frozen=True)
class CrossCodeComparison:
    """Summary metric for an external-code comparison."""

    count: int
    mean_relative_error: float
    p95_relative_error: float
    max_relative_error: float


def read_stellgap_alfven_post(path: str | Path) -> StellgapContinuumData:
    """Read STELLGAP ``alfven_post`` data.

    ``alfven_post`` starts with a one-line point count, followed by columns
    ``s, frequency[kHz], dominant_m, dominant_n``.
    """
    raw = np.loadtxt(Path(path), skiprows=1)
    return StellgapContinuumData(
        s=jnp.asarray(raw[:, 0]),
        frequency_khz=jnp.asarray(raw[:, 1]),
        m=jnp.asarray(raw[:, 2], dtype=int),
        n=jnp.asarray(raw[:, 3], dtype=int),
    )


def read_ae3d_eigenvalues(path: str | Path) -> Ae3dEigenvalueData:
    """Read AE3D ``egn_values.dat`` data."""
    raw = np.loadtxt(Path(path))
    return Ae3dEigenvalueData(
        frequency_khz=jnp.asarray(raw[:, 0]),
        residual_real=jnp.asarray(raw[:, 1]),
        residual_imag=jnp.asarray(raw[:, 2]),
    )


def ae3d_eigenvalue_residual_summary(path: str | Path) -> Ae3dResidualSummary:
    """Return residual summary metrics for public AE3D ``egn_values.dat`` data."""
    data = read_ae3d_eigenvalues(path)
    residual = jnp.sqrt(data.residual_real**2 + data.residual_imag**2)
    residual_np = np.asarray(residual)
    frequency_np = np.asarray(data.frequency_khz)
    return Ae3dResidualSummary(
        count=int(frequency_np.shape[0]),
        min_frequency_khz=float(np.min(frequency_np)),
        max_frequency_khz=float(np.max(frequency_np)),
        p95_residual=float(np.percentile(residual_np, 95.0)),
        max_residual=float(np.max(residual_np)),
    )


def stellgap_ctok_local_continuum_error(
    *,
    boozmn_path: str | Path,
    alfven_post_path: str | Path,
    mass_density: Any,
    toroidal_mode: int = 2,
    poloidal_modes: tuple[int, ...] = tuple(range(13)),
    theta_points: int = 32,
    zeta_points: int = 32,
    nearest_surface_tolerance: float = 2.0e-3,
) -> CrossCodeComparison:
    """Compare FAX local Boozer branches with the STELLGAP CTOK test output.

    The comparison sorts frequencies on each matched surface before computing a
    relative L2 error. This avoids relying on branch labels near crossings while
    still checking the continuum spectrum produced by the same Boozer geometry.
    """
    geometry = read_boozmn_geometry(boozmn_path)
    reference = read_stellgap_alfven_post(alfven_post_path)
    harmonics = jnp.asarray([[float(m), float(toroidal_mode)] for m in poloidal_modes])
    fax_khz = (
        boozer_3d_continuum_frequencies_hz(
            geometry,
            harmonics,
            mass_density=mass_density,
            theta_points=theta_points,
            zeta_points=zeta_points,
        )
        / 1.0e3
    )
    reference_s = np.asarray(reference.s)
    reference_frequency = np.asarray(reference.frequency_khz)
    unique_s = np.unique(reference_s)
    errors: list[float] = []
    for surface_index, surface in enumerate(np.asarray(geometry.s)):
        nearest_index = int(np.argmin(np.abs(unique_s - surface)))
        if abs(float(unique_s[nearest_index] - surface)) > float(nearest_surface_tolerance):
            continue
        surface_mask = reference_s == unique_s[nearest_index]
        reference_branch = np.sort(reference_frequency[surface_mask])
        if reference_branch.shape[0] != len(poloidal_modes):
            continue
        fax_branch = np.sort(np.asarray(fax_khz[surface_index]))
        denominator = max(float(np.linalg.norm(reference_branch)), 1.0e-30)
        errors.append(float(np.linalg.norm(fax_branch - reference_branch) / denominator))
    if not errors:
        raise ValueError("no matching STELLGAP/FAX surfaces found")
    error_array = np.asarray(errors)
    return CrossCodeComparison(
        count=int(error_array.shape[0]),
        mean_relative_error=float(np.mean(error_array)),
        p95_relative_error=float(np.percentile(error_array, 95.0)),
        max_relative_error=float(np.max(error_array)),
    )
