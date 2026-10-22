"""Boozer-coordinate geometry: readers, the vmex/booz_xform_jax path, and grids.

Harmonics use ``cos(m theta - n zeta)`` / ``sin(m theta - n zeta)`` with arrays
shaped ``(surfaces, modes)``. Stellarator-asymmetric arrays are stored as
zeros when absent, so every geometry has the same pytree structure.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

_MODES = ("bmnc", "bmns", "rmnc", "rmns", "zmns", "zmnc", "pmns", "pmnc", "gmnc", "gmns")


@dataclass(frozen=True)
class Geometry:
    """Boozer harmonics on a set of flux surfaces ``s`` (normalized toroidal flux).

    ``G`` and ``I`` are the Boozer covariant components, ``B = G grad(zeta) +
    I grad(theta) + K grad(psi)``, so the Jacobian is ``(G + iota I) / B^2``.
    """

    s: jax.Array
    xm: jax.Array
    xn: jax.Array
    bmnc: jax.Array
    bmns: jax.Array
    rmnc: jax.Array
    rmns: jax.Array
    zmns: jax.Array
    zmnc: jax.Array
    pmns: jax.Array
    pmnc: jax.Array
    gmnc: jax.Array
    gmns: jax.Array
    iota: jax.Array
    G: jax.Array
    I: jax.Array  # noqa: E741 - standard Boozer notation
    nfp: int = 1

    @classmethod
    def from_arrays(cls, nfp: int, **arrays: Any) -> Geometry:
        """Build a geometry, filling missing asymmetric harmonics with zeros."""
        zeros = np.zeros_like(np.asarray(arrays["bmnc"], dtype=float))
        values = {f.name: jnp.asarray(arrays.get(f.name, zeros), dtype=float) for f in fields(cls) if f.name != "nfp"}
        return cls(**values, nfp=int(nfp))

    def surfaces(self, index: Any) -> Geometry:
        """Return the geometry restricted to the given surface indices."""
        sliced = {name: getattr(self, name)[index] for name in ("s", "iota", "G", "I", *_MODES)}
        return replace(self, **sliced)

    def full_mesh(self) -> Geometry:
        """Average neighbouring half-mesh surfaces onto the VMEC full mesh, as STELLGAP and AE3D do."""
        return jax.tree_util.tree_map(lambda a, b: 0.5 * (a + b), self.surfaces(slice(None, -1)), self.surfaces(slice(1, None)))


_ARRAYS = ("s", "xm", "xn", *_MODES, "iota", "G", "I")
jax.tree_util.register_dataclass(Geometry, data_fields=list(_ARRAYS), meta_fields=["nfp"])


class SurfaceGrid(NamedTuple):
    """Field and metric sampled on a tensor Boozer grid, shape ``(surfaces, theta, zeta)``.

    ``dB`` stacks ``(d_s, d_theta, d_zeta) |B|`` on a last axis, ``sqrt_g`` is the
    embedding Jacobian of ``(s, theta, zeta)`` and ``metric`` the contravariant
    tensor ``g^ij = grad(u^i) . grad(u^j)``, shape ``(..., 3, 3)``.
    """

    theta: jax.Array
    zeta: jax.Array
    B: jax.Array
    jacobian: jax.Array
    grad_s2: jax.Array
    dB: jax.Array
    sqrt_g: jax.Array
    metric: jax.Array


def read_boozmn(path: str | Path) -> Geometry:
    """Read a BOOZ_XFORM ``boozmn`` netCDF file."""
    from netCDF4 import Dataset

    with Dataset(Path(path).expanduser(), "r") as ds:
        var = ds.variables
        arrays = {name: _surface_major(var[f"{name}_b"][:]) for name in _MODES if f"{name}_b" in var}
        if "gmn_b" in var:
            arrays["gmnc"] = _surface_major(var["gmn_b"][:])
        jlist = np.asarray(var["jlist"][:], dtype=int) if "jlist" in var else np.arange(1, arrays["bmnc"].shape[0] + 1)
        ns = float(np.asarray(var["ns_b"][...])) if "ns_b" in var else float(jlist.max())
        profile = lambda name: np.asarray(var[name][:], dtype=float)[jlist - 1]  # noqa: E731
        nfp = int(np.asarray(var["nfp_b"][...])) if "nfp_b" in var else 1
        return Geometry.from_arrays(
            nfp,
            s=(jlist - 1.5) / (ns - 1.0),  # half-mesh row j; booz_xform writes jlist = surface index + 2
            xm=np.asarray(var["ixm_b"][:], dtype=float),
            xn=np.asarray(var["ixn_b"][:], dtype=float),
            iota=profile("iota_b"),
            G=profile("bvco_b"),
            I=profile("buco_b"),
            **arrays,
        )


def _surface_major(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array.T if array.shape[0] > array.shape[1] else array


def from_booz_xform(output: dict[str, Any], s: Any, nfp: int) -> Geometry:
    """Convert a ``booz_xform_jax`` output dictionary (traceable)."""
    arrays = {name: output[f"{name}_b"] for name in _MODES if f"{name}_b" in output}
    if "gmnc" not in arrays and "gmn_b" in output:
        arrays["gmnc"] = output["gmn_b"]
    zeros = jnp.zeros_like(jnp.asarray(output["bmnc_b"]))
    values = {name: jnp.asarray(arrays.get(name, zeros)) for name in _MODES}
    return Geometry(
        s=jnp.asarray(s),
        xm=jnp.asarray(output["ixm_b"], dtype=float),
        xn=jnp.asarray(output["ixn_b"], dtype=float),
        iota=jnp.asarray(output["iota_b"]),
        G=jnp.asarray(output["bvco_b"]),
        I=jnp.asarray(output["buco_b"]),
        nfp=int(nfp),
        **values,
    )


def from_wout(wout: Any, *, mboz: int = 8, nboz: int = 8, surfaces: tuple[float, ...] = (0.25, 0.5, 0.75)) -> Geometry:
    """Run ``booz_xform_jax`` on a VMEC ``wout`` path or in-memory ``wout`` object."""
    from booz_xform_jax import Booz_xform

    transform = Booz_xform()
    if isinstance(wout, (str, Path)):
        transform.read_wout(str(Path(wout).expanduser()))
    else:
        transform.read_wout_data(wout)
    transform.mboz, transform.nboz = int(mboz), int(nboz)
    transform.compute_surfs = []
    transform.register_surfaces([float(x) for x in surfaces])
    output = transform.run_jax(jit=False)
    s = np.asarray(transform.s_in)[transform.compute_surfs]
    return from_booz_xform(output, s, int(transform.nfp))


def from_vmec_input(path: str | Path, ns: int | None = None, ftol: float = 1e-12, max_iter: int = 5000, **booz: Any) -> Geometry:
    """Solve a fixed-boundary VMEC input with ``vmex`` and transform it.

    ``ns=None`` runs the input's own ``NS_ARRAY`` ladder; an integer runs one stage
    at ``ns`` surfaces to ``ftol`` within ``max_iter`` iterations.
    """
    import vmex

    inp = vmex.VmecInput.from_file(str(Path(path).expanduser()))
    result = vmex.solve_multigrid(inp) if ns is None else vmex.solve_multigrid(inp, [ns], [ftol], [max_iter])
    return from_wout(vmex.wout_from_result(inp, result), **booz)


def circular_torus(
    *,
    s: Any = None,
    major_radius: float = 5.5,
    minor_radius: float = 1.0,
    B0: float = 2.7,
    iota: Any = 0.4,
    epsilon_B: float = 0.0,
    nfp: int = 1,
) -> Geometry:
    """Concentric circular tokamak in Boozer form, to first order in ``eps = epsilon_B sqrt(s)``.

    ``|B| = B0 (1 - eps cos(theta))`` and the surfaces are circles of radius
    ``r = a sqrt(s)`` in the geometric angle ``theta + eps sin(theta)``, which makes
    ``sqrt(g) ~ 1/B^2``. ``epsilon_B = a/R0`` is the ad hoc equilibrium of
    Könies et al. (2018, appendix); ``epsilon_B = 0`` gives a uniform field, for
    which the continuum is ``k_par v_A`` exactly. Surfaces default to the
    midpoints of 24 equal intervals of ``s``.
    """
    s = (jnp.arange(24) + 0.5) / 24 if s is None else jnp.asarray(s, dtype=float)
    xm, xn = jnp.array([0.0, 1.0, 2.0]), jnp.zeros(3)
    root = jnp.sqrt(s)
    zero = jnp.zeros_like(s)
    G = jnp.full_like(s, B0 * major_radius)
    iota = jnp.broadcast_to(jnp.asarray(iota, dtype=float), s.shape)
    bmnc = jnp.stack([jnp.full_like(s, B0), -B0 * epsilon_B * root, zero], axis=1)
    # Jacobian (G + iota I)/B^2 with I = 0, expanded to first order in epsilon_B.
    gmnc = jnp.stack([G / B0**2, 2.0 * G * epsilon_B * root / B0**2, zero], axis=1)
    shift = 0.5 * minor_radius * epsilon_B * s  # r cos(theta + eps sin) ~ r cos - r eps (1 - cos 2 theta) / 2
    return Geometry.from_arrays(
        nfp,
        s=s,
        xm=xm,
        xn=xn,
        bmnc=bmnc,
        gmnc=gmnc,
        rmnc=jnp.stack([major_radius - shift, minor_radius * root, shift], axis=1),
        zmns=jnp.stack([zero, minor_radius * root, shift], axis=1),
        iota=iota,
        G=G,
        I=zero,
    )


def synthesize(cos: Any, sin: Any, xm: Any, xn: Any, theta: Any, zeta: Any) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Evaluate ``sum c cos(m t - n z) + s sin(m t - n z)`` and its exact angle derivatives."""
    phase = theta[..., None] * xm - zeta[..., None] * xn
    c, s_ = jnp.cos(phase), jnp.sin(phase)
    f = jnp.einsum("...m,tzm->...tz", cos, c) + jnp.einsum("...m,tzm->...tz", sin, s_)
    ft = jnp.einsum("...m,tzm->...tz", sin * xm, c) - jnp.einsum("...m,tzm->...tz", cos * xm, s_)
    fz = jnp.einsum("...m,tzm->...tz", cos * xn, s_) - jnp.einsum("...m,tzm->...tz", sin * xn, c)
    return f, ft, fz


def angle_grid(ntheta: int, nzeta: int, nfp: int = 1) -> tuple[jax.Array, jax.Array]:
    """Uniform periodic grid; ``zeta`` covers one full toroidal turn."""
    theta = 2.0 * jnp.pi * jnp.arange(ntheta) / ntheta
    zeta = 2.0 * jnp.pi * jnp.arange(nzeta) / nzeta
    return jnp.meshgrid(theta, zeta, indexing="ij")


def surface_grid(geo: Geometry, ntheta: int = 32, nzeta: int = 32) -> SurfaceGrid:
    """Sample ``|B|``, its derivatives, the Boozer Jacobian and the metric on every surface.

    The covariant basis ``r_s, r_theta, r_zeta`` comes from the cylindrical
    embedding ``(R, phi = zeta + p, Z)``; ``g^ij`` is the inverse of
    ``g_ij = r_i . r_j``. Radial derivatives of the harmonics use second-order
    finite differences in ``s``.
    """
    theta, zeta = angle_grid(ntheta, nzeta)
    ds = lambda a: _d_ds(a, geo.s)
    B, Bt, Bz = synthesize(geo.bmnc, geo.bmns, geo.xm, geo.xn, theta, zeta)
    Bs = synthesize(ds(geo.bmnc), ds(geo.bmns), geo.xm, geo.xn, theta, zeta)[0]
    jac = synthesize(geo.gmnc, geo.gmns, geo.xm, geo.xn, theta, zeta)[0]
    R, Rt, Rz = synthesize(geo.rmnc, geo.rmns, geo.xm, geo.xn, theta, zeta)
    _, Zt, Zz = synthesize(geo.zmnc, geo.zmns, geo.xm, geo.xn, theta, zeta)
    p, pt, pz = synthesize(geo.pmnc, geo.pmns, geo.xm, geo.xn, theta, zeta)
    Rs = synthesize(ds(geo.rmnc), ds(geo.rmns), geo.xm, geo.xn, theta, zeta)[0]
    Zs = synthesize(ds(geo.zmnc), ds(geo.zmns), geo.xm, geo.xn, theta, zeta)[0]
    ps = synthesize(ds(geo.pmnc), ds(geo.pmns), geo.xm, geo.xn, theta, zeta)[0]
    phi = zeta + p
    cp, sp = jnp.cos(phi), jnp.sin(phi)

    def cart(dR: jax.Array, dphi: jax.Array, dZ: jax.Array) -> jax.Array:
        return jnp.stack([dR * cp - R * sp * dphi, dR * sp + R * cp * dphi, dZ], axis=-1)

    basis = jnp.stack([cart(Rs, ps, Zs), cart(Rt, pt, Zt), cart(Rz, 1.0 + pz, Zz)], axis=-2)
    sqrt_g = jnp.linalg.det(basis)
    metric = jnp.linalg.inv(basis @ jnp.swapaxes(basis, -1, -2))
    dB = jnp.stack([Bs, Bt, Bz], axis=-1)
    return SurfaceGrid(theta=theta, zeta=zeta, B=B, jacobian=jac, grad_s2=metric[..., 0, 0], dB=dB, sqrt_g=sqrt_g, metric=metric)


def _d_ds(a: jax.Array, s: jax.Array) -> jax.Array:
    """Radial derivative along axis 0: central differences inside, one-sided at the ends."""
    inner = (a[2:] - a[:-2]) / (s[2:] - s[:-2])[:, None]
    first = (a[1] - a[0]) / (s[1] - s[0])
    last = (a[-1] - a[-2]) / (s[-1] - s[-2])
    return jnp.concatenate([first[None], inner, last[None]])
