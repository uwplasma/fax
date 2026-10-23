"""Perturbative fast-ion drive in Boozer geometry: Fu-Van Dam scaling, Koenies 2018 TAE, sign rules, gradients."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fax

S = (jnp.arange(200) + 0.5) / 200
R0 = jnp.sqrt((1.75 - 1.71) / 0.16)  # q = (m + 1/2)/n = 1.75 for m = 10, n = 6
TAE = jnp.stack([jnp.exp(-(((jnp.sqrt(S) - R0) / 0.0425) ** 2) / 2)] * 2, axis=1)
HARM = ((10, 6), (11, 6))
ITPA = fax.circular_torus(s=S, major_radius=10.0, minor_radius=1.0, B0=3.0, iota=1 / (1.71 + 0.16 * S), epsilon_B=0.1)
RHO = fax.mass_density(2e19, 1.00727647)
AXI = dict(helical=0, nzeta=1)  # axisymmetric: exact on one toroidal point


def koenies_density(scale=1.0):
    """Koenies et al., NF 58 126027 (2018), eq. 1."""
    c0, c1, c2, c3 = 0.49123, 0.298228, 0.198739, 0.521298
    return scale * 1.44131e17 * c3 * jnp.exp(-(c2 / c1) * jnp.tanh((jnp.sqrt(S) - c0) / c2))


def tae_gamma(t_kev, scale=1.0, omega=4.0e5, n=6, geo=ITPA):
    ions = fax.FastIons(density=koenies_density(scale), temperature_ev=t_kev * 1e3)
    return fax.growth_rate(omega, TAE, geo, ((10 * n // 6, n), (11 * n // 6, n)), ions, RHO, **AXI)


def test_fu_van_dam_maxwellian_scaling():
    """Weak toroidicity (|B| = B0 (1 - eps sqrt(s) cos theta), eps = 1e-4, R0 = 1e4 a), flat Maxwellian, the two
    v_par = v_A resonances (j = m + 1 of m = 10 and j = m - 1 of m = 11): gamma ~ -x (1 + 2 x^2 + 2 x^4) exp(-x^2) at fixed
    beta_h, x = v_A / v_th (Fu & Van Dam 1989; Betti & Freidberg 1992), and the absolute coefficient of the zero-orbit-width
    integral done in closed form with NumPy: kernel ``-(1 + xi^2) eps / (B0 a^2) [sqrt(s) sin(theta) phi' + i m cos(theta) phi /
    (2 sqrt(s))] e^{i m theta}``, orbit average -> ``|C|^2 = (1 + xi^2)^2 eps^2 (phi_r -+ m phi / r)^2 / (16 B0^2 a^4)``,
    resonance at ``v = v_par / xi``, ``int_0^1 2 xi dxi xi^-8 (1 + xi^2)^2 exp(-x^2 / xi^2) = 2 (1 + 2 x^2 + 2 x^4) exp(-x^2) / x^6``,
    psi' = B0 a^2 / 2, and ``W = int rho |grad phi|^2 / (2 B0^2) dV``. The prefactor ``gamma = pi^4 m_f^2 P / W`` (``P`` the radial
    integral) is that of the module docstring, fixed by the Koenies gate below, not derived here. Approximations: O(eps) toroidal
    corrections and the v_A / 3 resonances (excluded, they are ~exp(-9 x^2))."""
    R0, a, B0, eps, io, m_f, vpar, e = 1e4, 1.0, 3.0, 1e-4, 1 / 1.75, 2.01410178 * 1.66053906660e-27, 1e6, 1.602176634e-19
    rho = float(RHO)
    geo = fax.circular_torus(s=S, major_radius=R0, minor_radius=a, B0=B0, iota=io, epsilon_B=eps)
    r, r0, w = np.sqrt(np.asarray(S)), 0.5, 0.1
    f = np.exp(-(((r - r0) / w) ** 2) / 2)
    fr = -(r - r0) / w**2 * f
    trapz = lambda y: np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(r))  # noqa: E731
    W = sum(4 * np.pi**2 * R0 * rho / (2 * B0**2) * trapz(r * (fr**2 + m**2 * f**2 / r**2)) for m in (10, 11))
    kappa = abs(11 * io - 6)  # |j iota - k| of both v_A resonances
    omega = vpar * kappa / R0

    def analytic(x, m, sign):
        v_th = vpar / x
        n = 1e22 / (0.5 * m_f * vpar**2 / x**2 / e)  # fixed beta_h
        radial = trapz(0.5 * B0 * a**2 * eps**2 / (16 * B0**2 * a**4) * (fr - sign * m * f / r) ** 2 * 2 * r)
        return (
            -(np.pi**4)
            * m_f**2
            / W
            * 4
            * n
            * v_th
            * R0**2
            * (1 + 2 * x**2 + 2 * x**4)
            * np.exp(-(x**2))
            / (m_f * np.pi**1.5 * kappa * B0)
            * radial
        )

    x = np.array([1.0, 1.5, 2.0, 2.5])
    gam = []
    for xi in x:
        t_ev = 0.5 * m_f * vpar**2 / xi**2 / e
        ions = fax.FastIons(1e22 / t_ev, t_ev)
        pairs, g = fax.growth_rate(omega, jnp.stack([f, f], 1), geo, HARM, ions, RHO, resolved=True, **AXI)
        by_j = dict(zip(pairs[:, 0].tolist(), np.asarray(g), strict=True))
        gam.append([by_j[11], by_j[10]])
    gam = np.array(gam)
    assert (gam < 0).all() and (np.abs(gam) > 1e-3).all()  # flat density: pure Landau damping, nonzero
    np.testing.assert_allclose(gam[:, 0], [analytic(xi, 10, 1) for xi in x], rtol=1e-3)
    np.testing.assert_allclose(gam[:, 1], [analytic(xi, 11, -1) for xi in x], rtol=1e-3)
    shape = x * (1 + 2 * x**2 + 2 * x**4) * np.exp(-(x**2))
    np.testing.assert_allclose(gam / shape[:, None], (gam[0] / shape[0]) * np.ones((4, 1)), rtol=1e-3)


def test_koenies_2018_tae_growth_rates():
    """ZOW passing drive in the Boozer form of the ad hoc equilibrium vs Koenies et al. (2018), fig. 1a ZLR
    (8-10, 25-33, 38-43 x10^3/s); the circular large-aspect-ratio model gave 4.4, 16.6, 33.1, 65 x10^3/s."""
    gam = np.array([tae_gamma(t) for t in (100.0, 200.0, 300.0, 400.0)])
    np.testing.assert_allclose(gam, [4.4e3, 16.6e3, 33.1e3, 65e3], rtol=0.03)
    assert 0.8 * 40e3 < gam[2] < 1.2 * 40e3


def test_drive_sign_and_flat_profile():
    flat = fax.FastIons(density=1e17, temperature_ev=200e3)
    for ions in (flat, fax.FastIons(density=1e17, kind="slowing_down")):
        g = [fax.growth_rate(4e5, TAE, ITPA, ((10 * n // 6, n), (11 * n // 6, n)), ions, RHO, **AXI) for n in (6, -6)]
        assert g[0] == pytest.approx(g[1]) and g[0] < 0  # no gradient: pure dF/dE (Landau) damping
    assert tae_gamma(200.0, n=6) > 0 > tae_gamma(200.0, n=-6)  # (m, n) -> (-m, -n): counter-propagating
    hollow = fax.FastIons(density=koenies_density()[::-1], temperature_ev=200e3)
    assert fax.growth_rate(4e5, TAE, ITPA, HARM, hollow, RHO, **AXI) < 0  # dF/dP_zeta reversed
    assert fax.ion_landau_damping(4e5, TAE, ITPA, HARM, 20e3, 2e19, **AXI) > 0


@pytest.mark.parametrize("arg", [0, 1, 2])
def test_gradients_match_finite_differences(arg):
    f = lambda a: tae_gamma(*[200.0, 1.0, 4.0e5][:arg], a, *[200.0, 1.0, 4.0e5][arg + 1 :])  # noqa: E731
    x0 = [200.0, 1.0, 4.0e5][arg]
    h = 1e-4 * x0
    assert jax.grad(f)(x0) == pytest.approx((f(x0 + h) - f(x0 - h)) / (2 * h), rel=1e-5)


def test_geometry_and_mode_gradients():
    """``jax.grad`` with respect to every geometry leaf and the mode, along a random direction, vs central FD."""
    f = lambda geo, phi: fax.growth_rate(4e5, phi, geo, HARM, fax.FastIons(koenies_density(), 200e3), RHO, **AXI)  # noqa: E731
    key = jax.random.PRNGKey(1)
    leaves, tree = jax.tree_util.tree_flatten((ITPA, TAE))
    du = [1e-2 * jnp.abs(x).mean() * jax.random.normal(jax.random.fold_in(key, i), x.shape) for i, x in enumerate(leaves)]
    du = jax.tree_util.tree_unflatten(tree, [d * 0 if i in (1, 2) else d for i, d in enumerate(du)])  # xm, xn are integers
    move = lambda t: jax.tree_util.tree_map(lambda x, d: x + t * d, (ITPA, TAE), du)  # noqa: E731
    grad = jax.grad(lambda p: f(*p))((ITPA, TAE))
    ad = sum(jnp.vdot(a, b) for a, b in zip(jax.tree_util.tree_leaves(grad), jax.tree_util.tree_leaves(du), strict=True))
    fd = (f(*move(1e-5)) - f(*move(-1e-5))) / 2e-5
    assert ad == pytest.approx(fd, rel=1e-5)


def test_slowing_down_is_normalized_with_its_smoothed_edge():
    """``int F d^3 v = 1`` for the erfc-smoothed birth edge (plain quadrature), and the edge drive ``dF/dE`` is its derivative."""
    ions = fax.FastIons(kind="slowing_down", birth_ev=300e3, critical_ev=100e3)
    m = ions.mass_amu * 1.66053906660e-27
    vb = np.sqrt(2 * 300e3 * 1.602176634e-19 / m)
    v = jnp.linspace(1e-6, 1.5 * vb, 400001)
    dfde, f = fax.drive._distribution(v, ions)
    assert float(fax.drive._integrate(4 * jnp.pi * v**2 * f, v)) == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(dfde[5:-1], np.gradient(f, v)[5:-1] / (m * v[5:-1]), rtol=1e-6, atol=1e-6 * float(jnp.abs(dfde).max()))
    with pytest.raises(ValueError, match="unknown FastIons kind"):
        fax.FastIons(kind="slowing-down")


def test_duplicated_toroidal_samples_leave_the_drive_unchanged():
    """The passing boundary uses the exact ``max |B|``: ``nzeta`` identical copies of an axisymmetric surface agree to 1e-10."""
    gam = [fax.growth_rate(4e5, TAE, ITPA, HARM, fax.FastIons(koenies_density(), 200e3), RHO, helical=0, nzeta=nz) for nz in (1, 4, 16)]
    assert gam[0] == pytest.approx(gam[1], rel=1e-10) and gam[0] == pytest.approx(gam[2], rel=1e-10)


def test_slowing_down_gradient_is_finite():
    f = lambda w: fax.growth_rate(
        w, TAE, ITPA, HARM, fax.FastIons(koenies_density(), birth_ev=300e3, critical_ev=100e3, kind="slowing_down"), RHO, **AXI
    )  # noqa: E731
    assert jnp.isfinite(f(4e5)) and jnp.isfinite(jax.grad(f)(4e5))


def test_ctok_global_tae_drive():
    """Full chain on CTOK: the FAX TAE (272.8 kHz, n = 2) from the shift-invert solve, then its drive by peaked deuterons."""
    geo = fax.read_boozmn(Path(__file__).parent / "data" / "boozmn_ctok.nc").full_mesh()
    harmonics, rho = fax.mode_family(2, 12), fax.mass_density(1e19, 2.0)
    ev = fax.eigenmodes(geo, harmonics, rho, target_hz=272.8e3, count=4)
    i = int(jnp.argmin(jnp.abs(ev.frequencies_hz - 272.8e3)))
    assert ev.frequencies_hz[i] == pytest.approx(272.8e3, rel=1e-3)
    omega = jnp.sqrt(ev.omega2[i])
    gam = [
        fax.growth_rate(omega, ev.structures[i], geo, harmonics, fax.FastIons(1e17 * (1 - geo.s) ** 2, t), rho, **AXI) for t in (25e3, 50e3)
    ]
    np.testing.assert_allclose(np.array(gam) / omega, [9.19e-4, 3.57e-3], rtol=1e-2)  # gamma / omega
    assert 0 < fax.ion_landau_damping(omega, ev.structures[i], geo, harmonics, 1e3, 1e19, mass_amu=2.0, **AXI) < 1e-4 * omega
