"""Perturbative energetic-particle drive and Landau damping of an Alfvén eigenmode.

The growth rate is the power the particles give to a fixed mode over twice its
energy (NOVA-K / CAS3D-K style, ``gamma_h = -Im dW_h / (2 omega dK)``)::

    gamma = -P / (2 W),   W = int rho |grad_perp phi|^2 / (2 B^2) d^3x.

Model: large-aspect-ratio circular geometry (``B = B0 (1 - eps cos theta)``), zero
orbit width (ZOW), zero Larmor radius, passing guiding centres with the curvature
plus grad-B drift ``v_d = (v_par^2 + v_perp^2/2) / (Omega R)``. The mode is
``phi = sum_h phi_h(s) exp[i(n_h zeta - m_h theta - omega t)]``. Along a passing orbit
of pitch ``lam = mu B0 / E`` the work ``Z e v_d . E`` is Fourier-decomposed in the
uniform transit angle ``vartheta``; the component ``p = m + l`` resonates at
``omega = (n q - p) omega_theta`` (TAE: ``v_par ~ v_A, v_A/3, v_A/5``). Linearising along
unperturbed orbits with ``dP_zeta/dt = (n/omega) dE/dt`` and the Landau prescription
``<dE dE/dt> -> (pi/2)|A_p|^2 delta(omega - (nq - p) omega_theta)`` gives::

    gamma = pi m_f^2 / (4 R^2) * int r dr sum_p int dlam <b/|xi|> v^6 |A_p|^2 QF/omega / ((nq - p) w_th / v)
                               / int r rho sum_h (|phi_h'|^2 + m_h^2 |phi_h|^2 / r^2) dr,
    A_p = < (1 - lam b/2) [sin(theta) d_r + cos(theta) d_theta / r] phi e^{i[(nq - m) theta - (nq - p) vartheta]} >_orbit,
    QF/omega = dF/dE + (n/omega) dF/dP_zeta = dF/dE - n q / (omega Z e B r) dF/dr,

with ``v = omega / |(nq - p) omega_theta/v|``. The delta function is integrated exactly
over speed at each pitch (Gauss–Legendre in ``lam``, uniform ``theta``), so ``gamma`` is
smooth in every parameter without artificial broadening. Trapped particles are
omitted. Sign convention: ``n > 0`` with an outward-decreasing density is destabilising.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

E = 1.602176634e-19
AMU = 1.66053906660e-27
MU0 = 4e-7 * np.pi


@dataclass(frozen=True)
class Plasma:
    """Large-aspect-ratio circular plasma; ``q`` and ``mass_density`` are scalars or profiles on ``s``."""

    major_radius: Any = 10.0
    minor_radius: Any = 1.0
    b0: Any = 3.0
    q: Any = 1.75
    mass_density: Any = 2.0e19 * 1.00727647 * AMU


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


jax.tree_util.register_dataclass(Plasma, data_fields=["major_radius", "minor_radius", "b0", "q", "mass_density"], meta_fields=[])
jax.tree_util.register_dataclass(
    FastIons,
    data_fields=["density", "temperature_ev", "birth_ev", "critical_ev", "mass_amu", "charge"],
    meta_fields=["kind"],
)


def growth_rate(
    omega: Any, phi: Any, s: Any, harmonics: Any, plasma: Plasma, ions: FastIons, *, sidebands: int = 3, nodes: int = 32, angles: int = 64
) -> jax.Array:
    """Fast-ion growth rate ``gamma_h`` (1/s) of a mode ``phi`` (shape ``(len(s), H)``) at ``omega`` (rad/s).

    ``harmonics`` is a static sequence of ``(m, n)``; Boozer families ``n0 + k Nfp`` enter
    as separate ``n``. Resonances ``p = m + l`` with ``|l| <= sidebands`` are summed. Use ``s > 0``.
    """
    s, phi = jnp.asarray(s), jnp.asarray(phi)
    r = plasma.minor_radius * jnp.sqrt(s)
    q, rho, dens = (jnp.asarray(x) * jnp.ones_like(s) for x in (plasma.q, plasma.mass_density, ions.density))
    dr = jnp.gradient(r)  # chain rule: jnp.gradient has no non-uniform spacing in JAX 0.4.30
    dphi, ddens = jnp.gradient(phi, axis=0) / dr[:, None], jnp.gradient(dens) / dr
    mass, R = ions.mass_amu * AMU, plasma.major_radius
    eps = (r / R)[:, None, None]
    # passing pitches lam = mu B0 / E in [0, 1/(1+eps)), Gauss-Legendre in x = lam (1 + eps)
    x, wx = (jnp.asarray(z) for z in np.polynomial.legendre.leggauss(nodes))
    lam = (0.5 * (x + 1.0))[None, :, None] / (1.0 + eps)
    theta = jnp.linspace(0.0, 2 * jnp.pi, angles, endpoint=False)
    b = 1.0 - eps * jnp.cos(theta)
    xi = jnp.sqrt(1.0 - lam * b)
    loop = 2 * jnp.pi * jnp.mean(1.0 / xi, axis=-1)  # oint dtheta / |xi|
    vartheta = 2 * jnp.pi * (jnp.cumsum(1.0 / xi, axis=-1) - 0.5 / xi - 0.5 / xi[..., :1]) / (angles * loop[..., None] / (2 * jnp.pi))
    omega_hat = 2 * jnp.pi / (q[:, None] * R * loop)  # transit frequency / v
    drift = (1.0 - 0.5 * lam * b) / xi / (angles * loop[..., None] / (2 * jnp.pi))  # orbit-time weight * v_D / v^2
    jac = (0.5 / (1.0 + eps[..., 0])) * wx * jnp.mean(b / xi, axis=-1) * 2 * jnp.pi
    power = 0.0
    for n in sorted({n for _, n in harmonics}):
        ms = [m for m, nn in harmonics if nn == n]
        for p in range(min(ms) - sidebands, max(ms) + sidebands + 1):
            amp = 0.0
            for h, (m, nn) in enumerate(harmonics):
                if nn == n:
                    kick = jnp.sin(theta) * dphi[:, h, None, None] - 1j * m * jnp.cos(theta) * phi[:, h, None, None] / r[:, None, None]
                    phase = (n * q[:, None, None] - m) * theta - (n * q[:, None, None] - p) * vartheta
                    amp = amp + jnp.sum(drift * kick * jnp.exp(1j * phase), axis=-1)
            rate = jnp.abs(n * q[:, None] - p) * omega_hat
            v = omega / jnp.where(rate > 1e-12, rate, 1e-12)
            g = _resonant_qf(v, dens[:, None], -n * q[:, None] / (omega * ions.charge * E * plasma.b0 * r[:, None]) * ddens[:, None], ions)
            power = power + jnp.sum(jac * v**6 * jnp.abs(amp) ** 2 * g / jnp.where(rate > 1e-12, rate, 1e-12), axis=-1)
    energy = sum(jnp.abs(dphi[:, h]) ** 2 + (m * jnp.abs(phi[:, h]) / r) ** 2 for h, (m, _) in enumerate(harmonics))
    return jnp.pi * mass**2 / (4 * R**2) * _integrate(r * power, r) / _integrate(r * rho * energy, r)


def ion_landau_damping(
    omega: Any, phi: Any, s: Any, harmonics: Any, plasma: Plasma, temperature_ev: Any, density: Any, *, mass_amu: Any = 1.00727647
) -> jax.Array:
    """Bulk-ion Landau damping rate (1/s, positive damps): the same power transfer for a flat
    thermal Maxwellian (Fu & Van Dam 1989; Betti & Freidberg 1992). Electron Landau damping
    needs ``E_par`` and is not included."""
    bulk = FastIons(density=density, temperature_ev=temperature_ev, mass_amu=mass_amu)
    return -growth_rate(omega, phi, s, harmonics, plasma, bulk)


def _resonant_qf(v: jax.Array, dens: Any, ddens_term: Any, ions: FastIons) -> jax.Array:
    """``QF/omega = dens dF/dE + ddens_term F`` at speed ``v`` with ``F`` per unit density."""
    m = ions.mass_amu * AMU
    if ions.kind == "maxwellian":
        t = ions.temperature_ev * E
        f = (m / (2 * jnp.pi * t)) ** 1.5 * jnp.exp(-0.5 * m * v**2 / t)
        return f * (-dens / t + ddens_term)
    vb, vc = jnp.sqrt(2 * ions.birth_ev * E / m), jnp.sqrt(2 * ions.critical_ev * E / m)
    norm = 3.0 / (4 * jnp.pi * jnp.log1p((vb / vc) ** 3))

    def shape(u: jax.Array) -> jax.Array:  # birth edge smoothed over 2% of v_b so the edge drive stays differentiable
        return norm * 0.5 * jax.scipy.special.erfc((u - vb) / (0.02 * vb)) / (u**3 + vc**3)

    f, dfdv = jax.jvp(shape, (v,), (jnp.ones_like(v),))
    return dens * dfdv / (m * v) + ddens_term * f


def _integrate(y: jax.Array, x: Any) -> jax.Array:
    """Trapezoidal rule along the last axis (``jnp.trapezoid`` is missing from older JAX)."""
    return 0.5 * jnp.sum((y[..., 1:] + y[..., :-1]) * jnp.diff(jnp.asarray(x)), axis=-1)
