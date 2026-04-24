"""Optional Boozer-coordinate adapters.

This module keeps Boozer coupling optional. It can read precomputed ``boozmn``
files directly and, when ``booz_xform_jax`` is installed, run the local JAX
transform from a VMEC ``wout`` file. FAX continuum operators still need a
dedicated Boozer metric assembly layer before Boozer spectra become the default
publication path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .metric import ContinuumMatrices, generalized_hermitian_eigenvalues
from .units import MU0


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BoozerSpectrum:
    """Boozer-space magnetic-field spectrum."""

    source: str
    s: jax.Array
    xm: jax.Array
    xn: jax.Array
    bmnc: jax.Array
    bmns: jax.Array | None
    field_periods: int
    metadata: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "s", jnp.asarray(self.s))
        object.__setattr__(self, "xm", jnp.asarray(self.xm))
        object.__setattr__(self, "xn", jnp.asarray(self.xn))
        object.__setattr__(self, "bmnc", jnp.asarray(self.bmnc))
        if self.bmns is not None:
            object.__setattr__(self, "bmns", jnp.asarray(self.bmns))
        object.__setattr__(self, "field_periods", int(self.field_periods))
        object.__setattr__(self, "metadata", tuple((str(key), value) for key, value in self.metadata))
        if self.bmnc.ndim != 2:
            raise ValueError("bmnc must have shape (surfaces, modes)")
        if self.bmnc.shape[0] != self.s.shape[0]:
            raise ValueError("bmnc first axis must match s")
        if self.bmnc.shape[1] != self.xm.shape[0] or self.xm.shape != self.xn.shape:
            raise ValueError("Boozer mode arrays must match bmnc mode axis")

    def tree_flatten(self):
        """Return JAX pytree children and static metadata."""
        children = (self.s, self.xm, self.xn, self.bmnc, self.bmns)
        return children, (self.source, self.field_periods, self.metadata)

    @classmethod
    def tree_unflatten(cls, aux: tuple[Any, ...], children: tuple[Any, ...]) -> "BoozerSpectrum":
        """Reconstruct a Boozer spectrum from pytree parts."""
        source, field_periods, metadata = aux
        s, xm, xn, bmnc, bmns = children
        return cls(
            source=source,
            s=s,
            xm=xm,
            xn=xn,
            bmnc=bmnc,
            bmns=bmns,
            field_periods=field_periods,
            metadata=metadata,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BoozerGeometry:
    """Boozer-coordinate geometry and field harmonics from a ``boozmn`` file.

    The arrays use the FAX convention ``cos(m theta_B - n zeta_B)`` and
    ``sin(m theta_B - n zeta_B)`` with radial surface-major mode ordering.  The
    ``gmnc``/``gmns`` fields are the Boozer-Jacobian harmonics written by
    ``booz_xform_jax`` and VMEC/BOOZ_XFORM-style tools.
    """

    source: str
    s: jax.Array
    xm: jax.Array
    xn: jax.Array
    bmnc: jax.Array
    bmns: jax.Array | None
    rmnc: jax.Array | None
    rmns: jax.Array | None
    zmns: jax.Array | None
    zmnc: jax.Array | None
    pmns: jax.Array | None
    pmnc: jax.Array | None
    gmnc: jax.Array | None
    gmns: jax.Array | None
    iota: jax.Array
    boozer_i: jax.Array
    boozer_g: jax.Array
    field_periods: int
    metadata: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "s", jnp.asarray(self.s))
        object.__setattr__(self, "xm", jnp.asarray(self.xm))
        object.__setattr__(self, "xn", jnp.asarray(self.xn))
        object.__setattr__(self, "bmnc", jnp.asarray(self.bmnc))
        for name in ("bmns", "rmnc", "rmns", "zmns", "zmnc", "pmns", "pmnc", "gmnc", "gmns"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, jnp.asarray(value))
        object.__setattr__(self, "iota", jnp.asarray(self.iota))
        object.__setattr__(self, "boozer_i", jnp.asarray(self.boozer_i))
        object.__setattr__(self, "boozer_g", jnp.asarray(self.boozer_g))
        object.__setattr__(self, "field_periods", int(self.field_periods))
        object.__setattr__(self, "metadata", tuple((str(key), value) for key, value in self.metadata))
        if self.bmnc.ndim != 2:
            raise ValueError("bmnc must have shape (surfaces, modes)")
        if self.bmnc.shape[0] != self.s.shape[0]:
            raise ValueError("bmnc first axis must match s")
        if self.xm.shape != self.xn.shape or self.bmnc.shape[1] != self.xm.shape[0]:
            raise ValueError("Boozer mode arrays must match bmnc mode axis")
        profile_shape = (int(self.s.shape[0]),)
        for name in ("iota", "boozer_i", "boozer_g"):
            if getattr(self, name).shape != profile_shape:
                raise ValueError(f"{name} must have shape {profile_shape}")
        for name in ("bmns", "rmnc", "rmns", "zmns", "zmnc", "pmns", "pmnc", "gmnc", "gmns"):
            value = getattr(self, name)
            if value is not None and value.shape != self.bmnc.shape:
                raise ValueError(f"{name} must have shape {self.bmnc.shape}")

    def tree_flatten(self):
        """Return JAX pytree children and static metadata."""
        children = (
            self.s,
            self.xm,
            self.xn,
            self.bmnc,
            self.bmns,
            self.rmnc,
            self.rmns,
            self.zmns,
            self.zmnc,
            self.pmns,
            self.pmnc,
            self.gmnc,
            self.gmns,
            self.iota,
            self.boozer_i,
            self.boozer_g,
        )
        return children, (self.source, self.field_periods, self.metadata)

    @classmethod
    def tree_unflatten(cls, aux: tuple[Any, ...], children: tuple[Any, ...]) -> "BoozerGeometry":
        """Reconstruct a Boozer geometry from pytree parts."""
        source, field_periods, metadata = aux
        (
            s,
            xm,
            xn,
            bmnc,
            bmns,
            rmnc,
            rmns,
            zmns,
            zmnc,
            pmns,
            pmnc,
            gmnc,
            gmns,
            iota,
            boozer_i,
            boozer_g,
        ) = children
        return cls(
            source=source,
            s=s,
            xm=xm,
            xn=xn,
            bmnc=bmnc,
            bmns=bmns,
            rmnc=rmnc,
            rmns=rmns,
            zmns=zmns,
            zmnc=zmnc,
            pmns=pmns,
            pmnc=pmnc,
            gmnc=gmnc,
            gmns=gmns,
            iota=iota,
            boozer_i=boozer_i,
            boozer_g=boozer_g,
            field_periods=field_periods,
            metadata=metadata,
        )

    def as_spectrum(self) -> BoozerSpectrum:
        """Return the magnetic-field part as a :class:`BoozerSpectrum`."""
        return BoozerSpectrum(
            source=self.source,
            s=self.s,
            xm=self.xm,
            xn=self.xn,
            bmnc=self.bmnc,
            bmns=self.bmns,
            field_periods=self.field_periods,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class BoozerGeometryGrid:
    """Sampled Boozer geometry fields on ``(s, theta_B, zeta_B)`` grids."""

    theta: jax.Array
    zeta: jax.Array
    magnetic_field: jax.Array
    signed_jacobian: jax.Array
    integration_jacobian: jax.Array
    grad_s_squared: jax.Array
    geometric_jacobian: jax.Array
    phi: jax.Array


def read_boozmn(path: str | Path) -> BoozerSpectrum:
    """Read a minimal Boozer magnetic-field spectrum from a ``boozmn`` netCDF file."""
    from netCDF4 import Dataset

    boozmn_path = Path(path).expanduser().resolve()
    with Dataset(boozmn_path, "r") as dataset:
        nfp = int(_read_first_scalar(dataset, ("nfp_b", "nfp"), 1))
        xm = np.asarray(dataset.variables["ixm_b"][:], dtype=float)
        xn = np.asarray(dataset.variables["ixn_b"][:], dtype=float)
        bmnc = _surface_mode_array(dataset.variables["bmnc_b"][:])
        bmns = _surface_mode_array(dataset.variables["bmns_b"][:]) if "bmns_b" in dataset.variables else None
        if "jlist" in dataset.variables and "ns_b" in dataset.variables:
            jlist = np.asarray(dataset.variables["jlist"][:], dtype=float)
            ns_b = float(_read_scalar(dataset, "ns_b", np.max(jlist)))
            s = (jlist - 1.0) / max(ns_b - 1.0, 1.0)
        else:
            s = np.linspace(0.0, 1.0, bmnc.shape[0])
    return BoozerSpectrum(
        source="boozmn",
        s=jnp.asarray(s),
        xm=jnp.asarray(xm),
        xn=jnp.asarray(xn),
        bmnc=jnp.asarray(bmnc),
        bmns=jnp.asarray(bmns) if bmns is not None else None,
        field_periods=nfp,
        metadata=(("path", str(boozmn_path)),),
    )


def read_boozmn_geometry(path: str | Path) -> BoozerGeometry:
    """Read Boozer field, embedding, and Jacobian harmonics from ``boozmn``.

    Missing stellarator-asymmetry arrays are represented as zero arrays so that
    symmetric and asymmetric files can be handled by the same reconstruction
    code.  Profile quantities are sampled on the ``jlist`` surfaces used by the
    harmonic arrays.
    """
    from netCDF4 import Dataset

    boozmn_path = Path(path).expanduser().resolve()
    with Dataset(boozmn_path, "r") as dataset:
        nfp = int(_read_first_scalar(dataset, ("nfp_b", "nfp"), 1))
        xm = np.asarray(dataset.variables["ixm_b"][:], dtype=float)
        xn = np.asarray(dataset.variables["ixn_b"][:], dtype=float)
        bmnc = _surface_mode_array(dataset.variables["bmnc_b"][:])
        zeros = np.zeros_like(bmnc)
        bmns = _optional_surface_mode_array(dataset, "bmns_b", zeros)
        rmnc = _optional_surface_mode_array(dataset, "rmnc_b", zeros)
        rmns = _optional_surface_mode_array(dataset, "rmns_b", zeros)
        zmns = _optional_surface_mode_array(dataset, "zmns_b", zeros)
        zmnc = _optional_surface_mode_array(dataset, "zmnc_b", zeros)
        pmns = _optional_surface_mode_array(dataset, "pmns_b", zeros)
        pmnc = _optional_surface_mode_array(dataset, "pmnc_b", zeros)
        gmnc = _optional_surface_mode_array(dataset, "gmn_b", zeros)
        gmns = _optional_surface_mode_array(dataset, "gmns_b", zeros)
        jlist = _read_jlist(dataset, bmnc.shape[0])
        ns_b = float(_read_scalar(dataset, "ns_b", np.max(jlist)))
        s = (jlist.astype(float) - 1.0) / max(ns_b - 1.0, 1.0)
        profile_indices = np.asarray(jlist, dtype=int) - 1
        iota = _profile_on_jlist(dataset, "iota_b", profile_indices, default=0.0)
        boozer_i = _profile_on_jlist(dataset, "buco_b", profile_indices, default=0.0)
        boozer_g = _profile_on_jlist(dataset, "bvco_b", profile_indices, default=0.0)

    return BoozerGeometry(
        source="boozmn",
        s=jnp.asarray(s),
        xm=jnp.asarray(xm),
        xn=jnp.asarray(xn),
        bmnc=jnp.asarray(bmnc),
        bmns=jnp.asarray(bmns),
        rmnc=jnp.asarray(rmnc),
        rmns=jnp.asarray(rmns),
        zmns=jnp.asarray(zmns),
        zmnc=jnp.asarray(zmnc),
        pmns=jnp.asarray(pmns),
        pmnc=jnp.asarray(pmnc),
        gmnc=jnp.asarray(gmnc),
        gmns=jnp.asarray(gmns),
        iota=jnp.asarray(iota),
        boozer_i=jnp.asarray(boozer_i),
        boozer_g=jnp.asarray(boozer_g),
        field_periods=nfp,
        metadata=(("path", str(boozmn_path)),),
    )


def run_booz_xform_jax_from_wout(
    path: str | Path,
    *,
    mboz: int = 8,
    nboz: int = 8,
    surfaces: tuple[float, ...] = (0.25, 0.5, 0.75),
    jit: bool = False,
) -> BoozerSpectrum:
    """Run ``booz_xform_jax`` from a VMEC ``wout`` file and return spectra."""
    try:
        from booz_xform_jax import Booz_xform
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("booz_xform_jax is required for run_booz_xform_jax_from_wout") from exc

    wout_path = Path(path).expanduser().resolve()
    transform = Booz_xform()
    transform.read_wout(str(wout_path))
    transform.mboz = int(mboz)
    transform.nboz = int(nboz)
    transform.compute_surfs = []
    transform.register_surfaces([float(surface) for surface in surfaces])
    output = transform.run_jax(jit=bool(jit))
    return BoozerSpectrum(
        source="booz_xform_jax",
        s=jnp.asarray(np.asarray(transform.s_in)[transform.compute_surfs]),
        xm=jnp.asarray(output["ixm_b"]),
        xn=jnp.asarray(output["ixn_b"]),
        bmnc=jnp.asarray(output["bmnc_b"]),
        bmns=jnp.asarray(output["bmns_b"]) if "bmns_b" in output else None,
        field_periods=int(transform.nfp),
        metadata=(
            ("wout", str(wout_path)),
            ("mboz", int(mboz)),
            ("nboz", int(nboz)),
            ("surfaces", tuple(float(surface) for surface in surfaces)),
        ),
    )


def normalized_nonaxisymmetric_boozer_energy(spectrum: BoozerSpectrum) -> jax.Array:
    """Return non-axisymmetric ``|B|`` spectral energy normalized by total energy."""
    nonaxisymmetric = (jnp.asarray(spectrum.xn) != 0).astype(spectrum.bmnc.dtype)
    total = jnp.sum(jnp.asarray(spectrum.bmnc) ** 2)
    if spectrum.bmns is not None:
        total = total + jnp.sum(jnp.asarray(spectrum.bmns) ** 2)
    numerator = jnp.sum(jnp.asarray(spectrum.bmnc) ** 2 * nonaxisymmetric[None, :])
    if spectrum.bmns is not None:
        numerator = numerator + jnp.sum(jnp.asarray(spectrum.bmns) ** 2 * nonaxisymmetric[None, :])
    return numerator / jnp.maximum(total, 1.0e-30)


def boozer_angle_grid(*, theta_points: int = 64, zeta_points: int = 64) -> tuple[jax.Array, jax.Array]:
    """Return a periodic tensor-product Boozer angle grid.

    The returned arrays have shape ``(theta_points, zeta_points)`` and cover
    ``[0, 2*pi)`` without duplicating the periodic endpoint.
    """
    theta = 2.0 * jnp.pi * jnp.arange(int(theta_points), dtype=float) / float(theta_points)
    zeta = 2.0 * jnp.pi * jnp.arange(int(zeta_points), dtype=float) / float(zeta_points)
    theta_grid, zeta_grid = jnp.meshgrid(theta, zeta, indexing="ij")
    return theta_grid, zeta_grid


def boozer_harmonic_phase(theta: Any, zeta: Any, xm: Any, xn: Any) -> jax.Array:
    """Return phases ``m*theta - n*zeta`` for Boozer harmonics."""
    theta_array = jnp.asarray(theta)
    zeta_array = jnp.asarray(zeta)
    return theta_array[..., None] * jnp.asarray(xm) - zeta_array[..., None] * jnp.asarray(xn)


def reconstruct_boozer_modes(cos_coefficients: Any, sin_coefficients: Any | None, xm: Any, xn: Any, theta: Any, zeta: Any) -> jax.Array:
    """Reconstruct scalar Boozer harmonics on a supplied angle grid.

    ``cos_coefficients`` may have shape ``(modes,)`` for one surface or
    ``(surfaces, modes)`` for a radial stack.  ``sin_coefficients`` follows the
    same convention and may be ``None`` for stellarator-symmetric cosine-only
    fields.
    """
    phase = boozer_harmonic_phase(theta, zeta, xm, xn)
    cos_coefficients_array = jnp.asarray(cos_coefficients)
    field = _combine_mode_basis(cos_coefficients_array, jnp.cos(phase))
    if sin_coefficients is not None:
        field = field + _combine_mode_basis(jnp.asarray(sin_coefficients), jnp.sin(phase))
    return field


def reconstruct_boozer_modes_with_derivatives(
    cos_coefficients: Any,
    sin_coefficients: Any | None,
    xm: Any,
    xn: Any,
    theta: Any,
    zeta: Any,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Reconstruct a Boozer scalar and exact angular derivatives."""
    phase = boozer_harmonic_phase(theta, zeta, xm, xn)
    m = jnp.asarray(xm)
    n = jnp.asarray(xn)
    cos_coefficients_array = jnp.asarray(cos_coefficients)
    cos_phase = jnp.cos(phase)
    sin_phase = jnp.sin(phase)
    field = _combine_mode_basis(cos_coefficients_array, cos_phase)
    dtheta = _combine_mode_basis(-cos_coefficients_array * m, sin_phase)
    dzeta = _combine_mode_basis(cos_coefficients_array * n, sin_phase)
    if sin_coefficients is not None:
        sin_coefficients_array = jnp.asarray(sin_coefficients)
        field = field + _combine_mode_basis(sin_coefficients_array, sin_phase)
        dtheta = dtheta + _combine_mode_basis(sin_coefficients_array * m, cos_phase)
        dzeta = dzeta + _combine_mode_basis(-sin_coefficients_array * n, cos_phase)
    return field, dtheta, dzeta


def reconstruct_boozer_field(
    spectrum: BoozerSpectrum,
    *,
    surface_index: int = 0,
    theta: Any | None = None,
    zeta: Any | None = None,
    theta_points: int = 64,
    zeta_points: int = 64,
) -> jax.Array:
    """Reconstruct ``|B|(theta,zeta)`` from one radial surface of a spectrum."""
    theta_grid, zeta_grid = boozer_angle_grid(theta_points=theta_points, zeta_points=zeta_points) if theta is None or zeta is None else (theta, zeta)
    phase = boozer_harmonic_phase(theta_grid, zeta_grid, spectrum.xm, spectrum.xn)
    bmnc = spectrum.bmnc[int(surface_index)]
    field = jnp.sum(bmnc * jnp.cos(phase), axis=-1)
    if spectrum.bmns is not None:
        field = field + jnp.sum(spectrum.bmns[int(surface_index)] * jnp.sin(phase), axis=-1)
    return field


def boozer_geometry_field_and_jacobian(
    geometry: BoozerGeometry,
    *,
    theta: Any | None = None,
    zeta: Any | None = None,
    theta_points: int = 64,
    zeta_points: int = 64,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return ``theta``, ``zeta``, ``|B|``, and signed Boozer Jacobian grids."""
    if geometry.gmnc is None:
        raise ValueError("BoozerGeometry.gmnc is required to reconstruct the Boozer Jacobian")
    theta_grid, zeta_grid = boozer_angle_grid(theta_points=theta_points, zeta_points=zeta_points) if theta is None or zeta is None else (theta, zeta)
    field = reconstruct_boozer_modes(geometry.bmnc, geometry.bmns, geometry.xm, geometry.xn, theta_grid, zeta_grid)
    jacobian = reconstruct_boozer_modes(geometry.gmnc, geometry.gmns, geometry.xm, geometry.xn, theta_grid, zeta_grid)
    return theta_grid, zeta_grid, field, jacobian


def boozer_geometry_grid(
    geometry: BoozerGeometry,
    *,
    theta_points: int = 64,
    zeta_points: int = 64,
    floor: float = 1.0e-30,
) -> BoozerGeometryGrid:
    """Sample real Boozer embedding metrics from ``boozmn`` harmonics.

    The returned ``grad_s_squared`` is computed from the physical embedding
    derivatives with respect to the normalized radial coordinate ``s``.  It is a
    local geometry weight for surface continuum matrices, not yet a substitute
    for a full global radial VMEC flux-coordinate operator.
    """
    for name in ("rmnc", "rmns", "zmns", "zmnc", "pmns", "pmnc", "gmnc"):
        if getattr(geometry, name) is None:
            raise ValueError(f"BoozerGeometry.{name} is required to build the embedding grid")
    theta, zeta, magnetic_field, signed_jacobian = boozer_geometry_field_and_jacobian(
        geometry,
        theta_points=theta_points,
        zeta_points=zeta_points,
    )
    rmnc = _require_surface_modes(geometry.rmnc, "rmnc")
    rmns = _require_surface_modes(geometry.rmns, "rmns")
    zmns = _require_surface_modes(geometry.zmns, "zmns")
    zmnc = _require_surface_modes(geometry.zmnc, "zmnc")
    pmns = _require_surface_modes(geometry.pmns, "pmns")
    pmnc = _require_surface_modes(geometry.pmnc, "pmnc")
    r, dtheta_r, dzeta_r = reconstruct_boozer_modes_with_derivatives(rmnc, rmns, geometry.xm, geometry.xn, theta, zeta)
    z, dtheta_z, dzeta_z = reconstruct_boozer_modes_with_derivatives(zmnc, zmns, geometry.xm, geometry.xn, theta, zeta)
    phi_offset, dtheta_phi_offset, dzeta_phi_offset = reconstruct_boozer_modes_with_derivatives(
        pmnc,
        pmns,
        geometry.xm,
        geometry.xn,
        theta,
        zeta,
    )
    dr_ds = reconstruct_boozer_modes(
        _radial_derivative(rmnc, geometry.s),
        _radial_derivative(rmns, geometry.s),
        geometry.xm,
        geometry.xn,
        theta,
        zeta,
    )
    dz_ds = reconstruct_boozer_modes(
        _radial_derivative(zmnc, geometry.s),
        _radial_derivative(zmns, geometry.s),
        geometry.xm,
        geometry.xn,
        theta,
        zeta,
    )
    dphi_offset_ds = reconstruct_boozer_modes(
        _radial_derivative(pmnc, geometry.s),
        _radial_derivative(pmns, geometry.s),
        geometry.xm,
        geometry.xn,
        theta,
        zeta,
    )

    phi = zeta[None, :, :] + phi_offset
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    dtheta_phi = dtheta_phi_offset
    dzeta_phi = 1.0 + dzeta_phi_offset
    ds_phi = dphi_offset_ds

    r_theta = _stack_xyz(
        dtheta_r * cos_phi - r * sin_phi * dtheta_phi,
        dtheta_r * sin_phi + r * cos_phi * dtheta_phi,
        dtheta_z,
    )
    r_zeta = _stack_xyz(
        dzeta_r * cos_phi - r * sin_phi * dzeta_phi,
        dzeta_r * sin_phi + r * cos_phi * dzeta_phi,
        dzeta_z,
    )
    r_s = _stack_xyz(
        dr_ds * cos_phi - r * sin_phi * ds_phi,
        dr_ds * sin_phi + r * cos_phi * ds_phi,
        dz_ds,
    )
    surface_normal = jnp.cross(r_theta, r_zeta)
    geometric_jacobian = jnp.sum(r_s * surface_normal, axis=-1)
    grad_s_squared = jnp.sum(surface_normal**2, axis=-1) / jnp.maximum(geometric_jacobian**2, float(floor))
    return BoozerGeometryGrid(
        theta=theta,
        zeta=zeta,
        magnetic_field=magnetic_field,
        signed_jacobian=signed_jacobian,
        integration_jacobian=jnp.abs(signed_jacobian),
        grad_s_squared=grad_s_squared,
        geometric_jacobian=geometric_jacobian,
        phi=phi,
    )


def boozer_geometry_jacobian_identity_error(
    geometry: BoozerGeometry,
    *,
    theta_points: int = 64,
    zeta_points: int = 64,
    floor: float = 1.0e-30,
) -> jax.Array:
    """Return the ``boozmn`` error in ``sqrt(g) |B|^2 = G + iota I``."""
    _, _, magnetic_field, signed_jacobian = boozer_geometry_field_and_jacobian(
        geometry,
        theta_points=theta_points,
        zeta_points=zeta_points,
    )
    numerator = geometry.boozer_g + geometry.iota * geometry.boozer_i
    expected = jnp.broadcast_to(numerator[:, None, None], magnetic_field.shape)
    residual = signed_jacobian * magnetic_field**2 - expected
    return jnp.linalg.norm(residual) / jnp.maximum(jnp.linalg.norm(expected), float(floor))


def boozer_embedding_jacobian_shape_error(
    geometry: BoozerGeometry,
    *,
    theta_points: int = 64,
    zeta_points: int = 64,
    floor: float = 1.0e-30,
) -> jax.Array:
    """Compare embedding-derived and Boozer-Jacobian harmonic shapes.

    The physical embedding derivative is taken with respect to normalized
    toroidal flux ``s`` and may carry an overall scale/orientation relative to
    the Boozer-Jacobian harmonics.  This gate removes one best-fit scale per
    surface and checks the remaining angular shape.
    """
    grid = boozer_geometry_grid(geometry, theta_points=theta_points, zeta_points=zeta_points, floor=floor)
    geometric = grid.geometric_jacobian
    target = grid.signed_jacobian
    scale = jnp.sum(geometric * target, axis=(1, 2), keepdims=True) / jnp.maximum(jnp.sum(geometric**2, axis=(1, 2), keepdims=True), float(floor))
    residual = scale * geometric - target
    return jnp.linalg.norm(residual) / jnp.maximum(jnp.linalg.norm(target), float(floor))


def project_boozer_field_to_modes(field: Any, theta: Any, zeta: Any, xm: Any, xn: Any, *, floor: float = 1.0e-30) -> tuple[jax.Array, jax.Array]:
    """Project a sampled scalar field onto Boozer cosine and sine modes."""
    field_array = jnp.asarray(field)
    phase = boozer_harmonic_phase(theta, zeta, xm, xn)
    cos_basis = jnp.cos(phase)
    sin_basis = jnp.sin(phase)
    cos_norm = jnp.mean(cos_basis**2, axis=(0, 1))
    sin_norm = jnp.mean(sin_basis**2, axis=(0, 1))
    bmnc = jnp.mean(field_array[..., None] * cos_basis, axis=(0, 1)) / jnp.maximum(cos_norm, float(floor))
    bmns = jnp.where(
        sin_norm > float(floor),
        jnp.mean(field_array[..., None] * sin_basis, axis=(0, 1)) / jnp.maximum(sin_norm, float(floor)),
        0.0,
    )
    return bmnc, bmns


def boozer_projection_error(
    spectrum: BoozerSpectrum,
    *,
    surface_index: int = 0,
    theta_points: int = 64,
    zeta_points: int = 64,
    floor: float = 1.0e-30,
) -> jax.Array:
    """Return relative coefficient error after reconstructing and re-projecting."""
    theta, zeta = boozer_angle_grid(theta_points=theta_points, zeta_points=zeta_points)
    field = reconstruct_boozer_field(spectrum, surface_index=surface_index, theta=theta, zeta=zeta)
    bmnc, bmns = project_boozer_field_to_modes(field, theta, zeta, spectrum.xm, spectrum.xn)
    expected_bmns = jnp.zeros_like(bmns) if spectrum.bmns is None else spectrum.bmns[int(surface_index)]
    numerator = jnp.linalg.norm(bmnc - spectrum.bmnc[int(surface_index)]) + jnp.linalg.norm(bmns - expected_bmns)
    denominator = jnp.maximum(jnp.linalg.norm(spectrum.bmnc[int(surface_index)]) + jnp.linalg.norm(expected_bmns), float(floor))
    return numerator / denominator


def boozer_harmonic_orthogonality_error(xm: Any, xn: Any, *, theta_points: int = 64, zeta_points: int = 64, floor: float = 1.0e-30) -> jax.Array:
    """Return the maximum normalized off-diagonal error of Boozer Fourier bases."""
    theta, zeta = boozer_angle_grid(theta_points=theta_points, zeta_points=zeta_points)
    phase = boozer_harmonic_phase(theta, zeta, xm, xn)
    basis = jnp.concatenate(
        [
            jnp.reshape(jnp.cos(phase), (-1, phase.shape[-1])),
            jnp.reshape(jnp.sin(phase), (-1, phase.shape[-1])),
        ],
        axis=1,
    )
    gram = basis.T @ basis / float(theta_points * zeta_points)
    norms = jnp.sqrt(jnp.maximum(jnp.diag(gram), 0.0))
    normalized = gram / jnp.maximum(norms[:, None] * norms[None, :], float(floor))
    off_diagonal = normalized - jnp.diag(jnp.diag(normalized))
    return jnp.max(jnp.abs(off_diagonal))


def straight_field_line_phase_error(
    harmonics: Any,
    *,
    iota: Any,
    theta0: float = 0.17,
    zeta_points: int = 128,
    floor: float = 1.0e-30,
) -> jax.Array:
    """Check that Boozer harmonic phases vary linearly along field lines.

    With the FAX convention ``exp(i*(m*theta - n*zeta))`` and straight field
    lines ``theta = theta0 + iota*zeta``, each phase has slope
    ``m*iota - n`` with respect to ``zeta``.
    """
    harmonic_array = jnp.asarray(harmonics, dtype=float)
    zeta = 2.0 * jnp.pi * jnp.arange(int(zeta_points), dtype=harmonic_array.dtype) / float(zeta_points)
    theta = float(theta0) + jnp.asarray(iota) * zeta
    phase = theta[:, None] * harmonic_array[:, 0] - zeta[:, None] * harmonic_array[:, 1]
    dzeta = zeta[1] - zeta[0]
    observed = (phase[1:] - phase[:-1]) / dzeta
    expected = harmonic_array[:, 0] * jnp.asarray(iota) - harmonic_array[:, 1]
    numerator = jnp.linalg.norm(observed - expected[None, :])
    denominator = jnp.maximum(jnp.linalg.norm(expected), float(floor))
    return numerator / denominator


def boozer_parallel_wavenumber(harmonics: Any, *, iota: Any, major_radius: float) -> jax.Array:
    """Return large-aspect-ratio ``k_parallel=(n-m*iota)/R`` for Boozer harmonics."""
    harmonic_array = jnp.asarray(harmonics)
    if harmonic_array.ndim != 2 or harmonic_array.shape[1] != 2:
        raise ValueError("harmonics must have shape (count, 2) with columns (m, n)")
    m = harmonic_array[:, 0]
    n = harmonic_array[:, 1]
    iota_array = jnp.asarray(iota)
    return (n[None, :] - m[None, :] * iota_array[:, None]) / float(major_radius) if iota_array.ndim == 1 else (n - m * iota_array) / float(major_radius)


def _surface_parameter(value: Any, surface_count: int, *, name: str) -> jax.Array:
    array = jnp.asarray(value)
    if array.ndim == 0:
        return jnp.full((int(surface_count),), array)
    if array.shape != (int(surface_count),):
        raise ValueError(f"{name} must be a scalar or have shape ({surface_count},)")
    return array


def boozer_jacobian_weight(
    spectrum: BoozerSpectrum,
    *,
    jacobian_numerator: Any = 1.0,
    theta_points: int = 64,
    zeta_points: int = 64,
    floor: float = 1.0e-30,
) -> jax.Array:
    """Return the reduced Boozer Jacobian weight ``J_B=(G+iota*I)/B^2``.

    ``jacobian_numerator`` represents the flux-function numerator
    ``G+iota*I`` on each surface. When only a ``boozmn`` ``|B|`` spectrum is
    available, callers can pass a normalized positive numerator to validate the
    Boozer ``J_B B^2`` identity before full metric data are available.
    """
    theta, zeta = boozer_angle_grid(theta_points=theta_points, zeta_points=zeta_points)
    phase = boozer_harmonic_phase(theta, zeta, spectrum.xm, spectrum.xn)
    field = jnp.einsum("sm,tzm->stz", spectrum.bmnc, jnp.cos(phase))
    if spectrum.bmns is not None:
        field = field + jnp.einsum("sm,tzm->stz", spectrum.bmns, jnp.sin(phase))
    numerator = _surface_parameter(jacobian_numerator, int(spectrum.s.shape[0]), name="jacobian_numerator")
    return numerator[:, None, None] / jnp.maximum(field**2, float(floor))


def boozer_jacobian_identity_error(
    spectrum: BoozerSpectrum,
    *,
    jacobian_numerator: Any = 1.0,
    theta_points: int = 64,
    zeta_points: int = 64,
    floor: float = 1.0e-30,
) -> jax.Array:
    """Return relative error in the Boozer identity ``J_B B^2=G+iota*I``."""
    theta, zeta = boozer_angle_grid(theta_points=theta_points, zeta_points=zeta_points)
    field = jnp.stack(
        [
            reconstruct_boozer_field(spectrum, surface_index=surface_index, theta=theta, zeta=zeta)
            for surface_index in range(int(spectrum.s.shape[0]))
        ]
    )
    numerator = _surface_parameter(jacobian_numerator, int(spectrum.s.shape[0]), name="jacobian_numerator")
    jacobian = numerator[:, None, None] / jnp.maximum(field**2, float(floor))
    residual = jacobian * field**2 - numerator[:, None, None]
    return jnp.linalg.norm(residual) / jnp.maximum(jnp.linalg.norm(numerator), float(floor))


def boozer_continuum_matrices(
    spectrum: BoozerSpectrum,
    harmonics: Any,
    *,
    iota: Any,
    major_radius: float,
    mass_density: Any,
    theta_points: int = 64,
    zeta_points: int = 64,
) -> ContinuumMatrices:
    """Assemble a reduced Boozer-space shear-Alfven continuum matrix.

    This is the first Boozer coupling operator in FAX. It uses the verified
    Boozer scalar-field representation for ``|B|`` and the large-aspect-ratio
    straight-field-line expression ``k_parallel=(n-m*iota)/R``. It is a
    validation bridge toward full Boozer metric operators, not the final 3-D
    stellarator continuum model.
    """
    harmonic_array = jnp.asarray(harmonics)
    if harmonic_array.ndim != 2 or harmonic_array.shape[1] != 2:
        raise ValueError("harmonics must have shape (count, 2) with columns (m, n)")

    theta, zeta = boozer_angle_grid(theta_points=theta_points, zeta_points=zeta_points)
    boozer_phase = boozer_harmonic_phase(theta, zeta, spectrum.xm, spectrum.xn)
    field = jnp.einsum("sm,tzm->stz", spectrum.bmnc, jnp.cos(boozer_phase))
    if spectrum.bmns is not None:
        field = field + jnp.einsum("sm,tzm->stz", spectrum.bmns, jnp.sin(boozer_phase))

    m = harmonic_array[:, 0]
    n = harmonic_array[:, 1]
    delta_m = m[None, :] - m[:, None]
    delta_n = n[None, :] - n[:, None]
    pair_phase = delta_m[:, :, None, None] * theta[None, None, :, :] - delta_n[:, :, None, None] * zeta[None, None, :, :]
    phase_factor = jnp.exp(1j * pair_phase)

    surface_count = int(spectrum.s.shape[0])
    iota_array = _surface_parameter(iota, surface_count, name="iota")
    density_array = _surface_parameter(mass_density, surface_count, name="mass_density")
    k_parallel = boozer_parallel_wavenumber(harmonic_array, iota=iota_array, major_radius=float(major_radius))
    if k_parallel.ndim == 1:
        k_parallel = jnp.broadcast_to(k_parallel[None, :], (surface_count, harmonic_array.shape[0]))
    k_pair = k_parallel[:, :, None, None, None] * k_parallel[:, None, :, None, None]

    stiffness = jnp.mean((field[:, None, None, :, :] ** 2 / MU0) * k_pair * phase_factor[None, :, :, :, :], axis=(-2, -1))
    mass = density_array[:, None, None] * jnp.mean(phase_factor, axis=(-2, -1))[None, :, :]
    stiffness = 0.5 * (stiffness + jnp.conjugate(jnp.swapaxes(stiffness, -1, -2)))
    mass = 0.5 * (mass + jnp.conjugate(jnp.swapaxes(mass, -1, -2)))
    return ContinuumMatrices(stiffness=stiffness, mass=mass)


def boozer_3d_continuum_matrices(
    geometry: BoozerGeometry,
    harmonics: Any,
    *,
    mass_density: Any,
    radial_metric: Any | None = None,
    theta_points: int = 64,
    zeta_points: int = 64,
    floor: float = 1.0e-30,
) -> ContinuumMatrices:
    """Assemble a local 3-D Boozer shear-Alfven continuum matrix.

    This operator uses real ``boozmn`` geometry rather than the large-aspect
    ratio ``(n-m*iota)/R`` bridge.  The straight-field-line derivative is
    ``k_parallel=(n-m*iota)|B|/(G+iota I)`` and the Galerkin weights are
    ``K=<|J| |grad s|^2 B^2 k_i k_j phi_i^* phi_j / mu0>`` and
    ``M=<|J| |grad s|^2 rho phi_i^* phi_j>``.  It is a local
    surface-by-surface continuum operator; a global radial eigenmode operator
    still needs radial coupling and boundary conditions.
    """
    harmonic_array = jnp.asarray(harmonics)
    if harmonic_array.ndim != 2 or harmonic_array.shape[1] != 2:
        raise ValueError("harmonics must have shape (count, 2) with columns (m, n)")
    if radial_metric is None:
        grid = boozer_geometry_grid(geometry, theta_points=theta_points, zeta_points=zeta_points, floor=floor)
        theta = grid.theta
        zeta = grid.zeta
        magnetic_field = grid.magnetic_field
        signed_jacobian = grid.signed_jacobian
        radial_weight = grid.grad_s_squared
    else:
        theta, zeta, magnetic_field, signed_jacobian = boozer_geometry_field_and_jacobian(
            geometry,
            theta_points=theta_points,
            zeta_points=zeta_points,
        )
        radial_weight = _radial_metric_weight(
            geometry,
            radial_metric,
            magnetic_field=magnetic_field,
            floor=floor,
        )
    integration_weight = jnp.abs(signed_jacobian) * jnp.maximum(radial_weight, float(floor))
    surface_count = int(geometry.s.shape[0])
    density_array = _surface_parameter(mass_density, surface_count, name="mass_density")
    numerator = geometry.boozer_g + geometry.iota * geometry.boozer_i
    safe_numerator = jnp.where(jnp.abs(numerator) > float(floor), numerator, jnp.where(numerator < 0.0, -float(floor), float(floor)))

    m = harmonic_array[:, 0]
    n = harmonic_array[:, 1]
    parallel_mismatch = n[None, :] - m[None, :] * geometry.iota[:, None]
    k_parallel = parallel_mismatch[:, :, None, None] * magnetic_field[:, None, :, :] / safe_numerator[:, None, None, None]

    delta_m = m[None, :] - m[:, None]
    delta_n = n[None, :] - n[:, None]
    pair_phase = delta_m[:, :, None, None] * theta[None, None, :, :] - delta_n[:, :, None, None] * zeta[None, None, :, :]
    phase_factor = jnp.exp(1j * pair_phase)

    stiffness = jnp.mean(
        (
            integration_weight[:, None, None, :, :]
            * magnetic_field[:, None, None, :, :] ** 2
            / MU0
            * k_parallel[:, :, None, :, :]
            * k_parallel[:, None, :, :, :]
            * phase_factor[None, :, :, :, :]
        ),
        axis=(-2, -1),
    )
    mass = density_array[:, None, None] * jnp.mean(
        integration_weight[:, None, None, :, :] * phase_factor[None, :, :, :, :],
        axis=(-2, -1),
    )
    stiffness = 0.5 * (stiffness + jnp.conjugate(jnp.swapaxes(stiffness, -1, -2)))
    mass = 0.5 * (mass + jnp.conjugate(jnp.swapaxes(mass, -1, -2)))
    return ContinuumMatrices(stiffness=stiffness, mass=mass)


def boozer_3d_continuum_frequencies_hz(
    geometry: BoozerGeometry,
    harmonics: Any,
    *,
    mass_density: Any,
    radial_metric: Any | None = None,
    theta_points: int = 64,
    zeta_points: int = 64,
) -> jax.Array:
    """Return local 3-D Boozer continuum frequencies in hertz."""
    matrices = boozer_3d_continuum_matrices(
        geometry,
        harmonics,
        mass_density=mass_density,
        radial_metric=radial_metric,
        theta_points=theta_points,
        zeta_points=zeta_points,
    )
    omega2 = generalized_hermitian_eigenvalues(matrices.stiffness, matrices.mass)
    return jnp.sqrt(jnp.maximum(jnp.real(omega2), 0.0)) / (2.0 * jnp.pi)


def boozer_metric_continuum_matrices(
    spectrum: BoozerSpectrum,
    harmonics: Any,
    *,
    iota: Any,
    major_radius: float,
    mass_density: Any,
    jacobian_numerator: Any = 1.0,
    theta_points: int = 64,
    zeta_points: int = 64,
) -> ContinuumMatrices:
    """Assemble a reduced Boozer-continuum matrix with Boozer-Jacobian weights.

    This uses ``J_B=(G+iota*I)/B^2`` in the energy weights,
    ``K=<J_B B^2 k_i k_j phi_i^* phi_j>/mu0`` and
    ``M=<J_B rho phi_i^* phi_j>``. It is a metric/Jacobian validation bridge:
    full stellarator continuum work still needs the complete perpendicular
    metric tensor and radial terms from VMEC/Boozer geometry.
    """
    harmonic_array = jnp.asarray(harmonics)
    if harmonic_array.ndim != 2 or harmonic_array.shape[1] != 2:
        raise ValueError("harmonics must have shape (count, 2) with columns (m, n)")

    theta, zeta = boozer_angle_grid(theta_points=theta_points, zeta_points=zeta_points)
    boozer_phase = boozer_harmonic_phase(theta, zeta, spectrum.xm, spectrum.xn)
    field = jnp.einsum("sm,tzm->stz", spectrum.bmnc, jnp.cos(boozer_phase))
    if spectrum.bmns is not None:
        field = field + jnp.einsum("sm,tzm->stz", spectrum.bmns, jnp.sin(boozer_phase))
    numerator = _surface_parameter(jacobian_numerator, int(spectrum.s.shape[0]), name="jacobian_numerator")
    jacobian = numerator[:, None, None] / jnp.maximum(field**2, 1.0e-30)

    m = harmonic_array[:, 0]
    n = harmonic_array[:, 1]
    delta_m = m[None, :] - m[:, None]
    delta_n = n[None, :] - n[:, None]
    pair_phase = delta_m[:, :, None, None] * theta[None, None, :, :] - delta_n[:, :, None, None] * zeta[None, None, :, :]
    phase_factor = jnp.exp(1j * pair_phase)

    surface_count = int(spectrum.s.shape[0])
    iota_array = _surface_parameter(iota, surface_count, name="iota")
    density_array = _surface_parameter(mass_density, surface_count, name="mass_density")
    k_parallel = boozer_parallel_wavenumber(harmonic_array, iota=iota_array, major_radius=float(major_radius))
    if k_parallel.ndim == 1:
        k_parallel = jnp.broadcast_to(k_parallel[None, :], (surface_count, harmonic_array.shape[0]))
    k_pair = k_parallel[:, :, None, None, None] * k_parallel[:, None, :, None, None]

    stiffness = jnp.mean(
        (jacobian[:, None, None, :, :] * field[:, None, None, :, :] ** 2 / MU0)
        * k_pair
        * phase_factor[None, :, :, :, :],
        axis=(-2, -1),
    )
    mass = density_array[:, None, None] * jnp.mean(
        jacobian[:, None, None, :, :] * phase_factor[None, :, :, :, :],
        axis=(-2, -1),
    )
    stiffness = 0.5 * (stiffness + jnp.conjugate(jnp.swapaxes(stiffness, -1, -2)))
    mass = 0.5 * (mass + jnp.conjugate(jnp.swapaxes(mass, -1, -2)))
    return ContinuumMatrices(stiffness=stiffness, mass=mass)


def boozer_metric_continuum_frequencies_hz(
    spectrum: BoozerSpectrum,
    harmonics: Any,
    *,
    iota: Any,
    major_radius: float,
    mass_density: Any,
    jacobian_numerator: Any = 1.0,
    theta_points: int = 64,
    zeta_points: int = 64,
) -> jax.Array:
    """Return reduced Boozer-Jacobian weighted continuum frequencies in hertz."""
    matrices = boozer_metric_continuum_matrices(
        spectrum,
        harmonics,
        iota=iota,
        major_radius=major_radius,
        mass_density=mass_density,
        jacobian_numerator=jacobian_numerator,
        theta_points=theta_points,
        zeta_points=zeta_points,
    )
    omega2 = generalized_hermitian_eigenvalues(matrices.stiffness, matrices.mass)
    return jnp.sqrt(jnp.maximum(jnp.real(omega2), 0.0)) / (2.0 * jnp.pi)


def boozer_continuum_frequencies_hz(
    spectrum: BoozerSpectrum,
    harmonics: Any,
    *,
    iota: Any,
    major_radius: float,
    mass_density: Any,
    theta_points: int = 64,
    zeta_points: int = 64,
) -> jax.Array:
    """Return coupled reduced Boozer-continuum frequencies in hertz."""
    matrices = boozer_continuum_matrices(
        spectrum,
        harmonics,
        iota=iota,
        major_radius=major_radius,
        mass_density=mass_density,
        theta_points=theta_points,
        zeta_points=zeta_points,
    )
    omega2 = generalized_hermitian_eigenvalues(matrices.stiffness, matrices.mass)
    return jnp.sqrt(jnp.maximum(jnp.real(omega2), 0.0)) / (2.0 * jnp.pi)


def dominant_boozer_modes(spectrum: BoozerSpectrum, *, count: int = 8) -> tuple[tuple[int, int, float], ...]:
    """Return dominant Boozer ``|B|`` modes by maximum radial amplitude."""
    amplitudes = np.max(np.abs(np.asarray(spectrum.bmnc)), axis=0)
    order = np.argsort(-amplitudes)[: int(count)]
    xm = np.asarray(spectrum.xm, dtype=int)
    xn = np.asarray(spectrum.xn, dtype=int)
    return tuple((int(xm[index]), int(xn[index]), float(amplitudes[index])) for index in order)


def _read_scalar(dataset: Any, name: str, default: Any | None = None) -> Any:
    if name not in dataset.variables:
        if default is None:
            raise KeyError(f"missing required Boozer variable {name!r}")
        return default
    value = np.asarray(dataset.variables[name][...])
    return value.item() if value.shape == () else value


def _read_first_scalar(dataset: Any, names: tuple[str, ...], default: Any) -> Any:
    for name in names:
        if name in dataset.variables:
            return _read_scalar(dataset, name)
    return default


def _surface_mode_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("Boozer coefficient arrays must be rank 2")
    return array.T if array.shape[0] > array.shape[1] else array


def _optional_surface_mode_array(dataset: Any, name: str, zeros: np.ndarray) -> np.ndarray:
    if name not in dataset.variables:
        return np.zeros_like(zeros)
    return _surface_mode_array(dataset.variables[name][:])


def _read_jlist(dataset: Any, surface_count: int) -> np.ndarray:
    if "jlist" in dataset.variables:
        return np.asarray(dataset.variables["jlist"][:], dtype=int)
    return np.arange(1, int(surface_count) + 1, dtype=int)


def _profile_on_jlist(dataset: Any, name: str, profile_indices: np.ndarray, *, default: float) -> np.ndarray:
    if name not in dataset.variables:
        return np.full(profile_indices.shape, float(default), dtype=float)
    profile = np.asarray(dataset.variables[name][:], dtype=float)
    return profile[np.asarray(profile_indices, dtype=int)]


def _combine_mode_basis(coefficients: jax.Array, basis: jax.Array) -> jax.Array:
    if coefficients.ndim == 1:
        return jnp.einsum("m,tzm->tz", coefficients, basis)
    if coefficients.ndim == 2:
        return jnp.einsum("sm,tzm->stz", coefficients, basis)
    raise ValueError("Boozer coefficient arrays must have shape (modes,) or (surfaces, modes)")


def _require_surface_modes(value: jax.Array | None, name: str) -> jax.Array:
    if value is None:
        raise ValueError(f"BoozerGeometry.{name} is required")
    return value


def _radial_derivative(values: Any, s: Any, *, floor: float = 1.0e-30) -> jax.Array:
    array = jnp.asarray(values)
    radius = jnp.asarray(s)
    if array.ndim != 2:
        raise ValueError("radial derivative values must have shape (surfaces, modes)")
    surface_count = int(array.shape[0])
    if surface_count < 2:
        raise ValueError("at least two Boozer surfaces are required for radial derivatives")
    if surface_count == 2:
        slope = (array[1] - array[0]) / jnp.maximum(radius[1] - radius[0], float(floor))
        return jnp.stack([slope, slope])
    first = (array[1] - array[0]) / jnp.maximum(radius[1] - radius[0], float(floor))
    interior = (array[2:] - array[:-2]) / jnp.maximum((radius[2:] - radius[:-2])[:, None], float(floor))
    last = (array[-1] - array[-2]) / jnp.maximum(radius[-1] - radius[-2], float(floor))
    return jnp.concatenate([first[None, :], interior, last[None, :]], axis=0)


def _stack_xyz(x: Any, y: Any, z: Any) -> jax.Array:
    return jnp.stack([jnp.asarray(x), jnp.asarray(y), jnp.asarray(z)], axis=-1)


def _radial_metric_weight(
    geometry: BoozerGeometry,
    radial_metric: Any | None,
    *,
    magnetic_field: jax.Array,
    floor: float,
) -> jax.Array:
    if radial_metric is None:
        raise ValueError("radial_metric must be provided when no embedding grid is used")
    metric = jnp.asarray(radial_metric)
    if metric.ndim == 0:
        return jnp.full_like(magnetic_field, metric)
    if metric.shape == (int(geometry.s.shape[0]),):
        return jnp.broadcast_to(metric[:, None, None], magnetic_field.shape)
    if metric.shape == magnetic_field.shape:
        return metric
    raise ValueError(
        "radial_metric must be a scalar, a radial profile with shape "
        f"({int(geometry.s.shape[0])},), or a grid with shape {magnetic_field.shape}"
    )
