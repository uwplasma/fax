"""Metric-aware continuum matrix scaffolding.

This module is the first angular-grid operator layer in FAX. It evaluates local
shear-Alfven continuum matrices from a flux-surface Jacobian, magnetic-field
strength, density, and field-line derivative coefficients. The implementation
is intentionally compact and validation-oriented: it has an analytic limit that
reduces exactly to :mod:`fax.continuum`, while also accepting VMEC ``wout``
Fourier data for early stellarator smoke tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .equilibrium import DEFAULT_ION_DENSITY, EquilibriumGeometry
from .units import MU0, ion_mass_density


def _metadata_tuple(mapping: dict[str, Any] | tuple[tuple[str, Any], ...] | None) -> tuple[tuple[str, Any], ...]:
    if mapping is None:
        return ()
    if isinstance(mapping, tuple):
        return mapping
    return tuple(sorted((str(key), value) for key, value in mapping.items()))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MetricGeometry:
    """Angular-grid geometry for local continuum matrix assembly.

    Parameters
    ----------
    name:
        Human-readable case name.
    s:
        Normalized toroidal-flux grid.
    theta:
        Poloidal angle grid in radians.
    zeta:
        Toroidal angle grid in radians. For stellarator-symmetric VMEC data this
        spans one field period.
    jacobian:
        Positive flux-surface integration weight with shape
        ``(radial, theta, zeta)``.
    magnetic_field:
        Magnetic-field strength in tesla on the same grid.
    mass_density:
        Ion mass density in kg/m^3 on the radial grid.
    bdotgrad_theta, bdotgrad_zeta:
        Field-line derivative coefficients in inverse meters. The harmonic
        convention used here is ``exp(i * (m * theta - n * zeta))``, giving
        ``k_parallel = n * bdotgrad_zeta - m * bdotgrad_theta``.
    field_periods:
        Number of toroidal field periods.
    metadata:
        Static provenance and adapter metadata.
    """

    name: str
    s: jax.Array
    theta: jax.Array
    zeta: jax.Array
    jacobian: jax.Array
    magnetic_field: jax.Array
    mass_density: jax.Array
    bdotgrad_theta: jax.Array
    bdotgrad_zeta: jax.Array
    field_periods: int
    metadata: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "s", jnp.asarray(self.s))
        object.__setattr__(self, "theta", jnp.asarray(self.theta))
        object.__setattr__(self, "zeta", jnp.asarray(self.zeta))
        object.__setattr__(self, "jacobian", jnp.asarray(self.jacobian))
        object.__setattr__(self, "magnetic_field", jnp.asarray(self.magnetic_field))
        object.__setattr__(self, "mass_density", jnp.asarray(self.mass_density))
        object.__setattr__(self, "bdotgrad_theta", jnp.asarray(self.bdotgrad_theta))
        object.__setattr__(self, "bdotgrad_zeta", jnp.asarray(self.bdotgrad_zeta))
        object.__setattr__(self, "field_periods", int(self.field_periods))
        object.__setattr__(self, "metadata", _metadata_tuple(self.metadata))

        if self.s.ndim != 1:
            raise ValueError("s must be one-dimensional")
        if self.theta.ndim != 1:
            raise ValueError("theta must be one-dimensional")
        if self.zeta.ndim != 1:
            raise ValueError("zeta must be one-dimensional")
        grid_shape = (self.s.shape[0], self.theta.shape[0], self.zeta.shape[0])
        for field_name in ("jacobian", "magnetic_field", "bdotgrad_theta", "bdotgrad_zeta"):
            if getattr(self, field_name).shape != grid_shape:
                raise ValueError(f"{field_name} must have shape {grid_shape}")
        if self.mass_density.shape != self.s.shape:
            raise ValueError("mass_density must have the same shape as s")
        if self.field_periods <= 0:
            raise ValueError("field_periods must be positive")

    def tree_flatten(self):
        """Return dynamic JAX children and static auxiliary metadata."""
        children = (
            self.s,
            self.theta,
            self.zeta,
            self.jacobian,
            self.magnetic_field,
            self.mass_density,
            self.bdotgrad_theta,
            self.bdotgrad_zeta,
        )
        aux = (self.name, self.field_periods, self.metadata)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux: tuple[Any, ...], children: tuple[Any, ...]) -> "MetricGeometry":
        """Reconstruct a ``MetricGeometry`` from JAX pytree parts."""
        name, field_periods, metadata = aux
        s, theta, zeta, jacobian, magnetic_field, mass_density, bdotgrad_theta, bdotgrad_zeta = children
        return cls(
            name=name,
            s=s,
            theta=theta,
            zeta=zeta,
            jacobian=jacobian,
            magnetic_field=magnetic_field,
            mass_density=mass_density,
            bdotgrad_theta=bdotgrad_theta,
            bdotgrad_zeta=bdotgrad_zeta,
            field_periods=field_periods,
            metadata=metadata,
        )

    def with_mass_density(self, mass_density: Any) -> "MetricGeometry":
        """Return a copy with a replacement radial mass-density profile."""
        return MetricGeometry(
            name=self.name,
            s=self.s,
            theta=self.theta,
            zeta=self.zeta,
            jacobian=self.jacobian,
            magnetic_field=self.magnetic_field,
            mass_density=jnp.asarray(mass_density),
            bdotgrad_theta=self.bdotgrad_theta,
            bdotgrad_zeta=self.bdotgrad_zeta,
            field_periods=self.field_periods,
            metadata=self.metadata,
        )

    def with_density_scale(self, scale: Any) -> "MetricGeometry":
        """Return a copy with ``mass_density`` multiplied by ``scale``."""
        return self.with_mass_density(self.mass_density * jnp.asarray(scale))


@dataclass(frozen=True)
class ContinuumMatrices:
    """Generalized local continuum matrices for each flux surface."""

    stiffness: jax.Array
    mass: jax.Array


@dataclass(frozen=True)
class GeneralizedEigensystem:
    """Eigenpairs for ``K x = lambda M x``."""

    eigenvalues: jax.Array
    eigenvectors: jax.Array


def periodic_grid(count: int, *, period: float = 2.0 * np.pi) -> jax.Array:
    """Return a periodic grid without duplicating the endpoint."""
    if int(count) < 1:
        raise ValueError("periodic grids require at least one point")
    return jnp.arange(int(count), dtype=float) * (float(period) / float(count))


def metric_from_equilibrium(
    geometry: EquilibriumGeometry,
    *,
    theta_points: int = 32,
    zeta_points: int = 32,
) -> MetricGeometry:
    """Lift profile geometry to a uniform straight-field-line metric fixture.

    This adapter is an analytic limit: the angular Jacobian and field strength
    are constant on each surface, and the field-line derivative coefficients
    reduce to ``k_parallel = (n - m iota) / R``.
    """
    theta = periodic_grid(theta_points)
    zeta = periodic_grid(zeta_points, period=2.0 * np.pi / float(geometry.field_periods))
    grid_shape = (geometry.s.shape[0], theta.shape[0], zeta.shape[0])
    jacobian_scale = jnp.maximum(jnp.abs(geometry.major_radius * geometry.minor_radius**2), 1.0e-30)
    jacobian = jnp.ones(grid_shape) * jacobian_scale
    magnetic_field = geometry.magnetic_field[:, None, None] * jnp.ones(grid_shape)
    bdotgrad_theta = (geometry.iota / geometry.major_radius)[:, None, None] * jnp.ones(grid_shape)
    bdotgrad_zeta = (1.0 / geometry.major_radius) * jnp.ones(grid_shape)
    return MetricGeometry(
        name=geometry.name,
        s=geometry.s,
        theta=theta,
        zeta=zeta,
        jacobian=jacobian,
        magnetic_field=magnetic_field,
        mass_density=geometry.mass_density,
        bdotgrad_theta=bdotgrad_theta,
        bdotgrad_zeta=bdotgrad_zeta,
        field_periods=geometry.field_periods,
        metadata=geometry.metadata + (("metric_adapter", "uniform_straight_field_line"),),
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


def _optional_coefficients(dataset: Any, name: str, shape: tuple[int, int]) -> np.ndarray:
    if name not in dataset.variables:
        return np.zeros(shape, dtype=float)
    return np.asarray(dataset.variables[name][:], dtype=float)


def _evaluate_vmec_fourier(
    cos_coefficients: Any,
    sin_coefficients: Any,
    xm: Any,
    xn: Any,
    theta: Any,
    zeta: Any,
) -> jax.Array:
    """Evaluate VMEC-style ``cos(m theta - n zeta)`` Fourier fields."""
    cos_array = jnp.asarray(cos_coefficients)
    sin_array = jnp.asarray(sin_coefficients)
    m = jnp.asarray(xm)
    n = jnp.asarray(xn)
    phase = m[:, None, None] * jnp.asarray(theta)[None, :, None] - n[:, None, None] * jnp.asarray(zeta)[None, None, :]
    cos_part = jnp.einsum("sk,ktz->stz", cos_array, jnp.cos(phase))
    sin_part = jnp.einsum("sk,ktz->stz", sin_array, jnp.sin(phase))
    return cos_part + sin_part


def metric_from_vmec_wout(
    path: str | Path,
    *,
    theta_points: int = 32,
    zeta_points: int = 32,
    number_density: float = DEFAULT_ION_DENSITY,
    name: str | None = None,
) -> MetricGeometry:
    """Read a VMEC ``wout`` file into an angular-grid metric fixture.

    The adapter reconstructs ``sqrt(g)``, ``|B|``, and contravariant magnetic
    components from VMEC Nyquist Fourier coefficients. It is suitable for
    smoke tests, symmetry checks, and operator plumbing. Full publication-grade
    stellarator continuum validation still requires the documented STELLGAP/
    AE3D comparison gates.
    """
    from netCDF4 import Dataset

    wout_path = Path(path).expanduser().resolve()
    with Dataset(wout_path, "r") as dataset:
        ns = int(_read_scalar(dataset, "ns"))
        nfp = int(_read_scalar(dataset, "nfp"))
        mpol = int(_read_scalar(dataset, "mpol", -1))
        ntor = int(_read_scalar(dataset, "ntor", -1))
        s = _normalized_s(dataset, ns)
        theta = np.asarray(periodic_grid(theta_points))
        zeta = np.asarray(periodic_grid(zeta_points, period=2.0 * np.pi / float(nfp)))
        xm = np.asarray(dataset.variables["xm_nyq"][:], dtype=float)
        xn = np.asarray(dataset.variables["xn_nyq"][:], dtype=float)
        mode_shape = (ns, xm.shape[0])
        gmnc = np.asarray(dataset.variables["gmnc"][:], dtype=float)
        gmns = _optional_coefficients(dataset, "gmns", mode_shape)
        bmnc = np.asarray(dataset.variables["bmnc"][:], dtype=float)
        bmns = _optional_coefficients(dataset, "bmns", mode_shape)
        bsupumnc = np.asarray(dataset.variables["bsupumnc"][:], dtype=float)
        bsupumns = _optional_coefficients(dataset, "bsupumns", mode_shape)
        bsupvmnc = np.asarray(dataset.variables["bsupvmnc"][:], dtype=float)
        bsupvmns = _optional_coefficients(dataset, "bsupvmns", mode_shape)

    jacobian = jnp.maximum(jnp.abs(_evaluate_vmec_fourier(gmnc, gmns, xm, xn, theta, zeta)), 1.0e-30)
    magnetic_field = jnp.maximum(jnp.abs(_evaluate_vmec_fourier(bmnc, bmns, xm, xn, theta, zeta)), 1.0e-14)
    bsupu = _evaluate_vmec_fourier(bsupumnc, bsupumns, xm, xn, theta, zeta)
    bsupv = _evaluate_vmec_fourier(bsupvmnc, bsupvmns, xm, xn, theta, zeta)
    bdotgrad_theta = bsupu / magnetic_field
    bdotgrad_zeta = bsupv / magnetic_field
    metadata = {
        "source": "vmec_wout",
        "path": str(wout_path),
        "ns": ns,
        "mpol": mpol,
        "ntor": ntor,
        "metric_adapter": "vmec_wout_nyquist",
    }
    return MetricGeometry(
        name=name or wout_path.stem.replace("wout_", ""),
        s=jnp.asarray(s),
        theta=jnp.asarray(theta),
        zeta=jnp.asarray(zeta),
        jacobian=jacobian,
        magnetic_field=magnetic_field,
        mass_density=jnp.full((ns,), ion_mass_density(number_density)),
        bdotgrad_theta=bdotgrad_theta,
        bdotgrad_zeta=bdotgrad_zeta,
        field_periods=nfp,
        metadata=_metadata_tuple(metadata),
    )


def _harmonic_array(harmonics: Any) -> jax.Array:
    harmonic_array = jnp.asarray(harmonics)
    if harmonic_array.ndim != 2 or harmonic_array.shape[1] != 2:
        raise ValueError("harmonics must have shape (count, 2) with columns (m, n)")
    return harmonic_array


def harmonic_phase_factors(metric: MetricGeometry, harmonics: Any) -> jax.Array:
    """Return ``conj(phi_i) * phi_j`` factors for harmonic pairs."""
    harmonic_array = _harmonic_array(harmonics)
    m = harmonic_array[:, 0]
    n = harmonic_array[:, 1]
    delta_m = m[None, :] - m[:, None]
    delta_n = n[None, :] - n[:, None]
    phase = (
        delta_m[:, :, None, None] * metric.theta[None, None, :, None]
        - delta_n[:, :, None, None] * metric.zeta[None, None, None, :]
    )
    return jnp.exp(1j * phase)


def parallel_wavenumber(metric: MetricGeometry, harmonics: Any) -> jax.Array:
    """Return ``k_parallel`` for each harmonic on the angular grid."""
    harmonic_array = _harmonic_array(harmonics)
    m = harmonic_array[:, 0]
    n = harmonic_array[:, 1]
    return (
        n[None, :, None, None] * metric.bdotgrad_zeta[:, None, :, :]
        - m[None, :, None, None] * metric.bdotgrad_theta[:, None, :, :]
    )


def metric_continuum_matrices(metric: MetricGeometry, harmonics: Any) -> ContinuumMatrices:
    """Assemble local Hermitian continuum matrices on each flux surface.

    The generalized problem is ``K x = omega^2 M x``. ``K`` contains the
    weighted field-line-bending energy density ``B^2 k_i k_j / mu0`` and ``M``
    contains the weighted mass density.
    """
    phase = harmonic_phase_factors(metric, harmonics)
    k_parallel = parallel_wavenumber(metric, harmonics)
    stiffness_weight = metric.jacobian[:, None, None, :, :] * metric.magnetic_field[:, None, None, :, :] ** 2 / MU0
    mass_weight = metric.jacobian[:, None, None, :, :] * metric.mass_density[:, None, None, None, None]
    k_pair = k_parallel[:, :, None, :, :] * k_parallel[:, None, :, :, :]
    stiffness = jnp.mean(stiffness_weight * k_pair * phase[None, :, :, :, :], axis=(-2, -1))
    mass = jnp.mean(mass_weight * phase[None, :, :, :, :], axis=(-2, -1))
    stiffness = 0.5 * (stiffness + jnp.conjugate(jnp.swapaxes(stiffness, -1, -2)))
    mass = 0.5 * (mass + jnp.conjugate(jnp.swapaxes(mass, -1, -2)))
    return ContinuumMatrices(stiffness=stiffness, mass=mass)


def diagonal_continuum_frequencies_hz(metric: MetricGeometry, harmonics: Any) -> jax.Array:
    """Return diagonal matrix estimates as frequencies in hertz."""
    matrices = metric_continuum_matrices(metric, harmonics)
    diagonal_omega2 = jnp.real(jnp.diagonal(matrices.stiffness, axis1=-2, axis2=-1)) / jnp.real(
        jnp.diagonal(matrices.mass, axis1=-2, axis2=-1)
    )
    return jnp.sqrt(jnp.maximum(diagonal_omega2, 0.0)) / (2.0 * jnp.pi)


def generalized_hermitian_eigensystem(stiffness: Any, mass: Any, *, mass_floor: float = 1.0e-30) -> GeneralizedEigensystem:
    """Return eigenpairs of ``K x = lambda M x`` for Hermitian matrices."""
    stiffness_array = jnp.asarray(stiffness)
    mass_array = jnp.asarray(mass)
    if stiffness_array.shape != mass_array.shape:
        raise ValueError("stiffness and mass must have matching shapes")
    if stiffness_array.ndim not in (2, 3):
        raise ValueError("stiffness and mass must be rank-2 matrices or rank-3 batches")
    eye = jnp.eye(mass_array.shape[-1], dtype=mass_array.dtype)

    def _solve_one(stiffness_one: jax.Array, mass_one: jax.Array) -> tuple[jax.Array, jax.Array]:
        regularized_mass = mass_one + float(mass_floor) * eye
        factor = jnp.linalg.cholesky(regularized_mass)
        inverse_factor = jnp.linalg.inv(factor)
        standard = inverse_factor @ stiffness_one @ jnp.conjugate(inverse_factor.T)
        standard = 0.5 * (standard + jnp.conjugate(standard.T))
        eigenvalues, standard_eigenvectors = jnp.linalg.eigh(standard)
        eigenvectors = jnp.linalg.solve(jnp.conjugate(factor.T), standard_eigenvectors)
        return eigenvalues, eigenvectors

    if stiffness_array.ndim == 2:
        eigenvalues, eigenvectors = _solve_one(stiffness_array, mass_array)
        return GeneralizedEigensystem(eigenvalues=eigenvalues, eigenvectors=eigenvectors)
    eigenvalues, eigenvectors = jax.vmap(_solve_one)(stiffness_array, mass_array)
    return GeneralizedEigensystem(eigenvalues=eigenvalues, eigenvectors=eigenvectors)


def generalized_hermitian_eigenvalues(stiffness: Any, mass: Any, *, mass_floor: float = 1.0e-30) -> jax.Array:
    """Return eigenvalues of ``K x = lambda M x`` for Hermitian matrices."""
    return generalized_hermitian_eigensystem(stiffness, mass, mass_floor=mass_floor).eigenvalues


def generalized_hermitian_eigenvalue_sensitivity(
    stiffness: Any,
    mass: Any,
    eigenvalue: Any,
    eigenvector: Any,
    stiffness_tangent: Any,
    mass_tangent: Any,
    *,
    floor: float = 1.0e-30,
) -> jax.Array:
    """Return first-order sensitivity of an isolated Hermitian eigenvalue.

    For ``K x = lambda M x`` and an isolated eigenpair, the derivative in a
    parameter direction is

    ``d lambda = x.H @ (dK - lambda dM) @ x / (x.H @ M @ x)``.

    The denominator is kept explicitly so callers may pass eigenvectors from
    FAX, SciPy, AE3D artifacts, or matrix-free prototypes without relying on a
    particular normalization convention. Repeated or nearly crossing
    eigenvalues require a subspace derivative and should not use this
    single-eigenvector formula for publication claims.
    """
    stiffness_array = jnp.asarray(stiffness)
    mass_array = jnp.asarray(mass)
    stiffness_tangent_array = jnp.asarray(stiffness_tangent)
    mass_tangent_array = jnp.asarray(mass_tangent)
    vector = jnp.asarray(eigenvector)
    if stiffness_array.shape != mass_array.shape:
        raise ValueError("stiffness and mass must have matching shapes")
    if stiffness_tangent_array.shape != stiffness_array.shape or mass_tangent_array.shape != mass_array.shape:
        raise ValueError("tangent matrices must match stiffness and mass shapes")
    if stiffness_array.ndim != 2 or vector.ndim != 1 or vector.shape[0] != stiffness_array.shape[0]:
        raise ValueError("sensitivity expects rank-2 matrices and one eigenvector")
    tangent_action = (stiffness_tangent_array - jnp.asarray(eigenvalue) * mass_tangent_array) @ vector
    numerator = jnp.vdot(vector, tangent_action)
    denominator = jnp.vdot(vector, mass_array @ vector)
    denominator = jnp.where(jnp.abs(denominator) < float(floor), float(floor) + 0.0j, denominator)
    return jnp.real(numerator / denominator)


def generalized_eigen_residual_norms(stiffness: Any, mass: Any, eigensystem: GeneralizedEigensystem, *, floor: float = 1.0e-30) -> jax.Array:
    """Return relative residual norms for ``Kx = lambda Mx`` eigenpairs."""
    stiffness_array = jnp.asarray(stiffness)
    mass_array = jnp.asarray(mass)
    eigenvalues = jnp.asarray(eigensystem.eigenvalues)
    eigenvectors = jnp.asarray(eigensystem.eigenvectors)

    def _residual_one(stiffness_one: jax.Array, mass_one: jax.Array, values: jax.Array, vectors: jax.Array) -> jax.Array:
        stiffness_vectors = stiffness_one @ vectors
        mass_vectors = mass_one @ vectors
        residual = stiffness_vectors - mass_vectors * values[None, :]
        numerator = jnp.linalg.norm(residual, axis=0)
        denominator = jnp.maximum(
            jnp.linalg.norm(stiffness_vectors, axis=0) + jnp.abs(values) * jnp.linalg.norm(mass_vectors, axis=0),
            float(floor),
        )
        return numerator / denominator

    if stiffness_array.ndim == 2:
        return _residual_one(stiffness_array, mass_array, eigenvalues, eigenvectors)
    return jax.vmap(_residual_one)(stiffness_array, mass_array, eigenvalues, eigenvectors)


def metric_continuum_frequencies_hz(metric: MetricGeometry, harmonics: Any, *, mass_floor: float = 1.0e-30) -> jax.Array:
    """Return coupled-matrix continuum frequencies in hertz."""
    matrices = metric_continuum_matrices(metric, harmonics)
    omega2 = generalized_hermitian_eigenvalues(matrices.stiffness, matrices.mass, mass_floor=mass_floor)
    return jnp.sqrt(jnp.maximum(jnp.real(omega2), 0.0)) / (2.0 * jnp.pi)
