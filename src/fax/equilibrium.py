"""Equilibrium geometry containers and reference adapters.

The objects here are deliberately small. They give continuum and validation
kernels a JAX-transformable profile representation, while Boozer-coordinate
geometry lives in :mod:`fax.boozer`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .units import ion_mass_density


DEFAULT_ION_DENSITY = 1.0e20
"""Default ion number density in m^-3 for analytic continuum examples."""


def _metadata_tuple(mapping: dict[str, Any] | tuple[tuple[str, Any], ...] | None) -> tuple[tuple[str, Any], ...]:
    if mapping is None:
        return ()
    if isinstance(mapping, tuple):
        return mapping
    return tuple(sorted((str(key), value) for key, value in mapping.items()))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EquilibriumGeometry:
    """Minimal JAX pytree for flux-surface profile geometry.

    Parameters
    ----------
    name:
        Human-readable case name.
    s:
        Normalized toroidal-flux coordinate on ``[0, 1]``.
    iota:
        Rotational-transform profile on the same radial grid as ``s``.
    magnetic_field:
        Representative magnetic-field strength in tesla. For profile models,
        this is the flux-surface average or the ``(m,n)=(0,0)`` Fourier
        component of ``|B|``.
    mass_density:
        Ion mass density in kg/m^3 on the same radial grid as ``s``.
    major_radius:
        Representative major radius in meters for large-aspect-ratio limits.
    minor_radius:
        Representative minor radius in meters.
    field_periods:
        Number of toroidal field periods.
    metadata:
        Static provenance and adapter metadata.
    """

    name: str
    s: jax.Array
    iota: jax.Array
    magnetic_field: jax.Array
    mass_density: jax.Array
    major_radius: jax.Array
    minor_radius: jax.Array
    field_periods: int
    metadata: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "s", jnp.asarray(self.s))
        object.__setattr__(self, "iota", jnp.asarray(self.iota))
        object.__setattr__(self, "magnetic_field", jnp.asarray(self.magnetic_field))
        object.__setattr__(self, "mass_density", jnp.asarray(self.mass_density))
        object.__setattr__(self, "major_radius", jnp.asarray(self.major_radius))
        object.__setattr__(self, "minor_radius", jnp.asarray(self.minor_radius))
        object.__setattr__(self, "field_periods", int(self.field_periods))
        object.__setattr__(self, "metadata", _metadata_tuple(self.metadata))

        shape = self.s.shape
        for field_name in ("iota", "magnetic_field", "mass_density"):
            if getattr(self, field_name).shape != shape:
                raise ValueError(f"{field_name} must have the same shape as s")
        if self.s.ndim != 1:
            raise ValueError("s must be one-dimensional")
        if self.field_periods <= 0:
            raise ValueError("field_periods must be positive")

    def tree_flatten(self):
        """Return dynamic JAX children and static auxiliary metadata."""
        children = (self.s, self.iota, self.magnetic_field, self.mass_density, self.major_radius, self.minor_radius)
        aux = (self.name, self.field_periods, self.metadata)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux: tuple[Any, ...], children: tuple[Any, ...]) -> "EquilibriumGeometry":
        """Reconstruct an ``EquilibriumGeometry`` from JAX pytree parts."""
        name, field_periods, metadata = aux
        s, iota, magnetic_field, mass_density, major_radius, minor_radius = children
        return cls(
            name=name,
            s=s,
            iota=iota,
            magnetic_field=magnetic_field,
            mass_density=mass_density,
            major_radius=major_radius,
            minor_radius=minor_radius,
            field_periods=field_periods,
            metadata=metadata,
        )

    def with_mass_density(self, mass_density: Any) -> "EquilibriumGeometry":
        """Return a copy with a replacement mass-density profile."""
        return replace(self, mass_density=jnp.asarray(mass_density))

    def with_density_scale(self, scale: Any) -> "EquilibriumGeometry":
        """Return a copy with ``mass_density`` multiplied by ``scale``."""
        return self.with_mass_density(self.mass_density * jnp.asarray(scale))


def analytic_cylinder(
    *,
    radial_points: int = 64,
    iota: float = 0.4,
    magnetic_field: float = 2.5,
    number_density: float = DEFAULT_ION_DENSITY,
    major_radius: float = 1.0,
    minor_radius: float = 0.3,
) -> EquilibriumGeometry:
    """Return a uniform cylindrical large-aspect-ratio continuum test case."""
    s = jnp.linspace(0.0, 1.0, int(radial_points))
    return EquilibriumGeometry(
        name="analytic_cylinder",
        s=s,
        iota=jnp.full_like(s, float(iota)),
        magnetic_field=jnp.full_like(s, float(magnetic_field)),
        mass_density=jnp.full_like(s, ion_mass_density(number_density)),
        major_radius=jnp.asarray(float(major_radius)),
        minor_radius=jnp.asarray(float(minor_radius)),
        field_periods=1,
        metadata=(("source", "analytic"), ("model", "uniform_cylinder")),
    )


def analytic_circular_tokamak(
    *,
    radial_points: int = 96,
    iota_axis: float = 0.9,
    iota_edge: float = 0.35,
    magnetic_field_axis: float = 5.0,
    number_density: float = DEFAULT_ION_DENSITY,
    major_radius: float = 6.0,
    minor_radius: float = 1.8,
) -> EquilibriumGeometry:
    """Return a smooth circular-tokamak profile for gap and scaling tests."""
    s = jnp.linspace(0.0, 1.0, int(radial_points))
    iota = float(iota_axis) + (float(iota_edge) - float(iota_axis)) * s
    magnetic_field = float(magnetic_field_axis) / (1.0 + 0.08 * s)
    density = ion_mass_density(number_density) * (1.0 - 0.15 * s + 0.02 * s**2)
    return EquilibriumGeometry(
        name="analytic_circular_tokamak",
        s=s,
        iota=iota,
        magnetic_field=magnetic_field,
        mass_density=density,
        major_radius=jnp.asarray(float(major_radius)),
        minor_radius=jnp.asarray(float(minor_radius)),
        field_periods=1,
        metadata=(("source", "analytic"), ("model", "circular_tokamak_profile")),
    )


def _read_scalar(dataset: Any, name: str, default: Any | None = None) -> Any:
    if name not in dataset.variables:
        if default is None:
            raise KeyError(f"missing required VMEC variable {name!r}")
        return default
    value = dataset.variables[name][...]
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    return array


def _normalized_s(dataset: Any, ns: int) -> np.ndarray:
    if "phi" in dataset.variables:
        phi = np.asarray(dataset.variables["phi"][:], dtype=float)
        edge = phi[-1]
        if np.isfinite(edge) and abs(edge) > 0.0:
            return phi / edge
    return np.linspace(0.0, 1.0, ns)


def _axisymmetric_b00(dataset: Any, ns: int) -> np.ndarray:
    if "bmnc" not in dataset.variables:
        b0 = float(_read_scalar(dataset, "b0", 1.0))
        return np.full(ns, b0)

    bmnc = np.asarray(dataset.variables["bmnc"][:], dtype=float)
    xm = np.asarray(dataset.variables.get("xm_nyq", dataset.variables.get("xm"))[:], dtype=float)
    xn = np.asarray(dataset.variables.get("xn_nyq", dataset.variables.get("xn"))[:], dtype=float)
    zero_indices = np.flatnonzero((xm == 0.0) & (xn == 0.0))
    if zero_indices.size == 0:
        profile = np.abs(bmnc[:, 0])
    else:
        profile = np.abs(bmnc[:, int(zero_indices[0])])

    # VMEC full-grid quantities can carry a placeholder value on the magnetic
        # axis. For profile-level checks, copy the first positive neighbor so
    # continuum formulas do not see an unphysical zero magnetic field at s=0.
    positive = np.flatnonzero(profile > 0.0)
    if positive.size > 0 and profile[0] == 0.0:
        profile = profile.copy()
        profile[0] = profile[int(positive[0])]
    return profile


def _representative_radii(dataset: Any, *, major_radius: float, minor_radius: float) -> tuple[float, float]:
    """Return robust representative radii for VMEC files with zero scalar radii."""
    if major_radius > 0.0 and minor_radius > 0.0:
        return major_radius, minor_radius
    if "rmnc" not in dataset.variables or "xm" not in dataset.variables or "xn" not in dataset.variables:
        safe_major = major_radius if major_radius > 0.0 else 1.0
        safe_minor = minor_radius if minor_radius > 0.0 else 0.1 * safe_major
        return safe_major, safe_minor

    rmnc = np.asarray(dataset.variables["rmnc"][:], dtype=float)
    xm = np.asarray(dataset.variables["xm"][:], dtype=float)
    xn = np.asarray(dataset.variables["xn"][:], dtype=float)
    zero_indices = np.flatnonzero((xm == 0.0) & (xn == 0.0))
    if zero_indices.size > 0:
        r00 = np.abs(rmnc[:, int(zero_indices[0])])
        positive = r00[r00 > 0.0]
        if major_radius <= 0.0 and positive.size > 0:
            major_radius = float(positive[-1])

    if minor_radius <= 0.0:
        non_axisymmetric = np.flatnonzero(xm > 0.0)
        edge_amplitudes = np.abs(rmnc[-1, non_axisymmetric]) if non_axisymmetric.size else np.asarray([])
        estimate = float(np.sqrt(np.sum(edge_amplitudes**2))) if edge_amplitudes.size else 0.0
        minor_radius = estimate if estimate > 0.0 else 0.1 * max(major_radius, 1.0)

    return float(major_radius), float(minor_radius)


def from_vmec_wout(
    path: str | Path,
    *,
    number_density: float = DEFAULT_ION_DENSITY,
    name: str | None = None,
) -> EquilibriumGeometry:
    """Read a minimal ``EquilibriumGeometry`` from a VMEC ``wout`` netCDF file.

    This adapter intentionally extracts profile-level quantities for continuum
    and global scalar radial workflows. Full 3-D Boozer geometry should use
    :func:`fax.boozer.read_boozmn_geometry` or the ``booz_xform_jax`` bridge.
    """
    from netCDF4 import Dataset

    wout_path = Path(path).expanduser().resolve()
    with Dataset(wout_path, "r") as dataset:
        ns = int(_read_scalar(dataset, "ns"))
        nfp = int(_read_scalar(dataset, "nfp"))
        mpol = int(_read_scalar(dataset, "mpol", -1))
        ntor = int(_read_scalar(dataset, "ntor", -1))
        s = _normalized_s(dataset, ns)
        iota = np.asarray(dataset.variables["iotaf"][:], dtype=float)
        magnetic_field = _axisymmetric_b00(dataset, ns)
        major_radius = float(_read_scalar(dataset, "Rmajor_p", 1.0))
        minor_radius = float(_read_scalar(dataset, "Aminor_p", 1.0))
        major_radius, minor_radius = _representative_radii(
            dataset,
            major_radius=major_radius,
            minor_radius=minor_radius,
        )

    metadata = {
        "source": "vmec_wout",
        "path": str(wout_path),
        "ns": ns,
        "mpol": mpol,
        "ntor": ntor,
    }
    return EquilibriumGeometry(
        name=name or wout_path.stem.replace("wout_", ""),
        s=jnp.asarray(s),
        iota=jnp.asarray(iota),
        magnetic_field=jnp.asarray(magnetic_field),
        mass_density=jnp.full((ns,), ion_mass_density(number_density)),
        major_radius=jnp.asarray(major_radius),
        minor_radius=jnp.asarray(minor_radius),
        field_periods=nfp,
        metadata=_metadata_tuple(metadata),
    )
