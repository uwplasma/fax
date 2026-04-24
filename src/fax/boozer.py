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


def read_boozmn(path: str | Path) -> BoozerSpectrum:
    """Read a minimal Boozer magnetic-field spectrum from a ``boozmn`` netCDF file."""
    from netCDF4 import Dataset

    boozmn_path = Path(path).expanduser().resolve()
    with Dataset(boozmn_path, "r") as dataset:
        nfp = int(_read_scalar(dataset, "nfp", 1))
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


def _surface_mode_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("Boozer coefficient arrays must be rank 2")
    return array.T if array.shape[0] > array.shape[1] else array
