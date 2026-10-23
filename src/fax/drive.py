"""Perturbative energetic-particle drive and Landau damping of an Alfvén eigenmode in Boozer geometry.

The growth rate is the power the particles give to a fixed mode over twice its
energy (NOVA-K / CAS3D-K style), ``gamma = -P / (2 W)``, with ``W`` the inertia form
``int rho |grad phi|^2 / (2 B^2) dV`` of :func:`fax.eigen.global_matrices`.

Model: zero orbit width (ZOW), zero Larmor radius, passing guiding centres on the
surfaces of a :class:`fax.Geometry`, low-beta drift ``v_d = m v^2 (1 + xi^2) / (2 Z e B^3)
B x grad B`` with Boozer components ``(B x grad B)^i = eps^ijk B_j d_k B / sqrt(g)``,
``B_theta = I``, ``B_zeta = G`` (``B_s`` dropped). The mode is ``phi = sum_h phi_h(s)
exp[i(m_h theta - n_h zeta - omega t)]`` (the convention of :mod:`fax.eigen`). A passing
orbit of pitch ``lam = mu / E`` follows ``theta - iota zeta = const`` with ``dt/dzeta =
(G + iota I) / (v |xi| B)``; the uniform angles ``vartheta = theta + iota chi``,
``varphi = zeta + chi`` solve ``(iota d_theta + d_zeta) chi = h / <h> - 1``,
``h = 1 / (|xi| B)``. The work ``Ze v_d . E`` decomposed in them resonates at
``omega = |j iota - k| omega_zeta(v, lam)`` for ``j = m + l`` and ``k = n + k' N_fp``:
the toroidal (``l``) and helical (``k'``) sidebands of the B spectrum. With ZOW
``P_zeta = m v_par G / B - Z e psi_p`` and ``E_par = 0`` each harmonic carries
``Q_h F = dF/dE + n_h dF/ds / (omega Z e iota psi')``, and the resonance is integrated
exactly in speed at each pitch (Gauss-Legendre in ``lam``), so ``gamma`` is smooth::

    gamma = pi^2 mu0 m_f^2 / (2 y.H G y) int |psi'| ds int dlam sum_jk v^6 Re[C* Q C] / (|j iota - k| w^2),
    C_h = < (1 + xi^2) / (2 B^3 sqrt(g)) [(I B_zeta - G B_theta) phi_h' + i B_s (G m_h + I n_h) phi_h]
            e^{i(m_h theta - n_h zeta)} e^{-i(j vartheta - k varphi)} >_orbit,

with ``w = omega_zeta / v``, ``B_i = d_i |B|`` and ``v = omega / (|j iota - k| w)``.

Trapped and precessing particles are omitted (no bounce-averaged precession resonance).
``n > 0`` with a peaked density drives (``iota psi' > 0``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .eigen import _quad, global_matrices
from .geometry import Geometry, _d_ds, surface_grid

E = 1.602176634e-19
AMU = 1.66053906660e-27
MU0 = 4e-7 * np.pi


@dataclass(frozen=True)
class FastIons:
    """Isotropic fast ions: ``density`` (m^-3, scalar or profile on ``s``), energies in eV.

    ``kind="maxwellian"`` uses ``temperature_ev``; ``kind="slowing_down"`` uses
    ``F ~ 1/(v^3 + v_c^3)`` below ``birth_ev`` with ``critical_ev`` (the edge delta of
    ``dF/dE`` at ``birth_ev`` is not included).
    """

    density: Any = 1.0e17
    temperature_ev: Any = 100.0e3
    birth_ev: Any = 3.5e6
    critical_ev: Any = 0.5e6
    mass_amu: Any = 2.01410178
    charge: Any = 1.0
    kind: str = field(default="maxwellian", metadata={"static": True})


jax.tree_util.register_dataclass(
    FastIons,
    data_fields=["density", "temperature_ev", "birth_ev", "critical_ev", "mass_amu", "charge"],
    meta_fields=["kind"],
)


def growth_rate(
    omega: Any,
    phi: Any,
    geo: Geometry,
    harmonics: Any,
    ions: FastIons,
    mass_density: Any,
    *,
    sidebands: int = 3,
    helical: int = 1,
    nodes: int = 32,
    ntheta: int = 64,
    nzeta: int = 32,
    resolved: bool = False,
) -> Any:
    """Fast-ion growth rate (1/s) of a mode ``phi`` (shape ``(len(geo.s), H)``, nodal values) at ``omega`` (rad/s).

    ``harmonics`` is a static ``(H, 2)`` list of ``(m, n)``; ``geo.s`` lies strictly inside
    ``(0, 1)`` as for :func:`fax.eigenmodes`. Resonances ``j = m + l``, ``|l| <= sidebands``,
    and ``k = n + k' nfp``, ``|k'| <= helical``, are summed. An axisymmetric case with
    ``helical = 0`` is exact with ``nzeta = 1``. ``resolved=True`` returns the resonances
    ``(j, k)`` and the growth rate of each instead of their sum.
    """
    hm = np.asarray(harmonics, dtype=float)
    m, n = (jnp.asarray(hm[:, i])[:, None, None] for i in (0, 1))
    phi = jnp.asarray(phi)
    g = surface_grid(geo, ntheta, nzeta)
    col = lambda a: jnp.asarray(a)[:, None, None]  # noqa: E731 - per-surface value -> (S, 1, 1)
    D = geo.G + geo.iota * geo.I
    psi = jnp.mean(g.sqrt_g * g.B**2, axis=(1, 2)) / D  # dpsi/ds, psi = toroidal flux / 2 pi
    Bs, Bt, Bz = (g.dB[..., i] for i in range(3))
    e = jnp.exp(1j * (m * g.theta - n * g.zeta)) / (g.B**3 * g.sqrt_g)[:, None]  # (S, H, T, Z)
    kernel = (col(geo.I) * Bz - col(geo.G) * Bt)[:, None] * e * _d_ds(phi, geo.s)[..., None, None]
    kernel = kernel + 1j * Bs[:, None] * (col(geo.G)[:, None] * m + col(geo.I)[:, None] * n) * e * phi[..., None, None]
    # passing pitches lam = mu/E in [0, 1/B_max), Gauss-Legendre; B_max a smooth upper bound (log-sum-exp)
    x, wx = (jnp.asarray(z) for z in np.polynomial.legendre.leggauss(nodes))
    scale = 1e-3 * jnp.mean(g.B, axis=(1, 2), keepdims=True)
    bmax = scale[..., 0] * jax.nn.logsumexp(g.B / scale, axis=(1, 2))[:, None]
    lam = 0.5 * (x + 1.0) / bmax  # (S, L)
    xi = jnp.sqrt(1.0 - lam[..., None, None] * g.B[:, None])  # (S, L, T, Z)
    h = 1.0 / (xi * g.B[:, None])
    hbar = jnp.mean(h, axis=(2, 3))
    kt, kz = (jnp.fft.fftfreq(k, 1.0 / k)[:, None] for k in (ntheta, nzeta))
    kappa = geo.iota[:, None, None, None] * kt + kz.T  # (iota d_theta + d_zeta) -> i kappa
    chi = jnp.real(jnp.fft.ifft2(jnp.fft.fft2(h / hbar[..., None, None] - 1.0) * -1j * kappa / (kappa**2 + 1e-4)))
    weight = (0.5 * (1.0 + xi**2) * h / hbar[..., None, None])[:, :, None] * kernel[:, None]  # (S, L, H, T, Z)
    omega_hat = 1.0 / (jnp.abs(D)[:, None] * hbar)  # omega_zeta / v
    dens = jnp.asarray(ions.density) * jnp.ones_like(geo.s)
    pairs = np.array(
        [
            (j, k)
            for j in range(int(hm[:, 0].min()) - sidebands, int(hm[:, 0].max()) + sidebands + 1)
            for k in sorted({nn + kk * geo.nfp for nn in hm[:, 1] for kk in range(-helical, helical + 1)})
        ],
        dtype=float,
    )

    def resonance(jk: jax.Array) -> jax.Array:
        j, k = jk[0], jk[1]
        kap = (j * geo.iota - k)[:, None]
        phase = jnp.exp(-1j * (kap[..., None, None] * chi + j * g.theta - k * g.zeta)[:, :, None])
        c = jnp.mean(weight * phase, axis=(-2, -1))  # (S, L, H)
        C, N = jnp.sum(c, -1), jnp.sum(c * n[:, 0, 0], -1)
        rate = jnp.maximum(jnp.abs(kap) * omega_hat, 1e-12)
        v = omega / rate
        dfde, f = _distribution(v, ions)
        star = _d_ds(dens[:, None], geo.s) / (omega * ions.charge * E * (geo.iota * psi)[:, None])
        q = dens[:, None] * dfde * jnp.abs(C) ** 2 + star * f * jnp.real(jnp.conj(C) * N)
        return v**6 * q / (rate * omega_hat)

    power = jnp.abs(psi) * jnp.sum(jax.lax.map(jax.checkpoint(resonance), jnp.asarray(pairs)) * wx / (2 * bmax), axis=-1)  # (P, S)
    zero = jnp.zeros((len(pairs), 1))
    radial = _integrate(jnp.concatenate([zero, power, zero], 1), jnp.concatenate([zero[0], geo.s, jnp.ones(1)]))
    inertia = global_matrices(geo, hm, mass_density, ntheta=ntheta, nzeta=nzeta)[1]
    gamma = jnp.pi**2 * MU0 * (ions.mass_amu * AMU) ** 2 / 2 * radial / _quad(phi[..., None], inertia)[0]
    return (pairs.astype(int), gamma) if resolved else jnp.sum(gamma)


def ion_landau_damping(
    omega: Any, phi: Any, geo: Geometry, harmonics: Any, temperature_ev: Any, density: Any, *, mass_amu: Any = 1.00727647, **options: Any
) -> jax.Array:
    """Bulk-ion Landau damping rate (1/s, positive damps): the same power transfer for a flat
    thermal Maxwellian (Fu & Van Dam 1989; Betti & Freidberg 1992). Electron Landau damping
    needs ``E_par`` and is not included. ``options`` go to :func:`growth_rate`."""
    bulk = FastIons(density=density, temperature_ev=temperature_ev, mass_amu=mass_amu)
    return -growth_rate(omega, phi, geo, harmonics, bulk, density * mass_amu * AMU, **options)


def _distribution(v: jax.Array, ions: FastIons) -> tuple[jax.Array, jax.Array]:
    """``(dF/dE, F)`` per unit density at speed ``v``."""
    m = ions.mass_amu * AMU
    if ions.kind == "maxwellian":
        t = ions.temperature_ev * E
        f = (m / (2 * jnp.pi * t)) ** 1.5 * jnp.exp(-0.5 * m * v**2 / t)
        return -f / t, f
    vb, vc = jnp.sqrt(2 * ions.birth_ev * E / m), jnp.sqrt(2 * ions.critical_ev * E / m)
    norm = 3.0 / (4 * jnp.pi * jnp.log1p((vb / vc) ** 3))

    def shape(u: jax.Array) -> jax.Array:  # birth edge smoothed over 2% of v_b so the edge drive stays differentiable
        return norm * 0.5 * jax.scipy.special.erfc((u - vb) / (0.02 * vb)) / (u**3 + vc**3)

    f, dfdv = jax.jvp(shape, (v,), (jnp.ones_like(v),))
    return dfdv / (m * v), f


def _integrate(y: jax.Array, x: Any) -> jax.Array:
    """Trapezoidal rule along the last axis (``jnp.trapezoid`` is missing from older JAX)."""
    return 0.5 * jnp.sum((y[..., 1:] + y[..., :-1]) * jnp.diff(jnp.asarray(x)), axis=-1)
