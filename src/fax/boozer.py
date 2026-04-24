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
