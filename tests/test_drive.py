"""Perturbative fast-ion drive: Fu-Van Dam scaling, Koenies 2018 TAE, sign rules, gradients."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fax

S = jnp.linspace(1e-3, 1.0, 201)
R0 = jnp.sqrt((1.75 - 1.71) / 0.16)  # q = (m + 1/2)/n = 1.75 for m = 10, n = 6
TAE = jnp.stack([jnp.exp(-(((jnp.sqrt(S) - R0) / 0.0425) ** 2) / 2)] * 2, axis=1)
HARM = ((10, 6), (11, 6))
ITPA = fax.Plasma(q=1.71 + 0.16 * S)


def koenies_density(scale=1.0):
    """Koenies et al., NF 58 126027 (2018), eq. 1."""
    c0, c1, c2, c3 = 0.49123, 0.298228, 0.198739, 0.521298
    return scale * 1.44131e17 * c3 * jnp.exp(-(c2 / c1) * jnp.tanh((jnp.sqrt(S) - c0) / c2))


def tae_gamma(t_kev, scale=1.0, omega=4.0e5, n=6):
    ions = fax.FastIons(density=koenies_density(scale), temperature_ev=t_kev * 1e3)
    return fax.growth_rate(omega, TAE, S, ((10 * n // 6, n), (11 * n // 6, n)), ITPA, ions)


def test_fu_van_dam_maxwellian_scaling():
    """Flat Maxwellian, one v_par = v_A resonance: gamma / beta_h ~ -x (1 + 2x^2 + 2x^4) exp(-x^2), x = v_A/v_th
    (Fu & Van Dam, Phys. Fluids B 1, 1949 (1989); Betti & Freidberg, Phys. Fluids B 4, 1465 (1992))."""
    plasma = fax.Plasma(major_radius=1e3, q=10 / 6)
    omega, mass = 1.0e4, 2.01410178 * 1.66053906660e-27
    x = jnp.array([1.0, 1.5, 2.0, 2.5])
    temps = mass * (omega * 10 / 6 * 1e3 / x) ** 2 / 2 / 1.602176634e-19
    gam = jnp.array([fax.growth_rate(omega, TAE[:, :1], S, ((10, 6),), plasma, fax.FastIons(1e17 / t, t)) for t in temps])
    shape = -x * (1 + 2 * x**2 + 2 * x**4) * jnp.exp(-(x**2))
    np.testing.assert_allclose(gam / shape, gam[0] / shape[0], rtol=1e-3)


def test_koenies_2018_tae_growth_rates():
    """ZOW passing drive vs the ZLR curve of Koenies et al. (2018), fig. 1a (8-10, 25-33, 38-43 x10^3/s)."""
    gam = np.array([tae_gamma(t) for t in (100.0, 200.0, 300.0)])
    np.testing.assert_allclose(gam, [4.4e3, 16.6e3, 33.1e3], rtol=0.05)  # this model; see plan.md logbook
    assert 0.8 * 40e3 < gam[2] < 1.2 * 40e3


def test_drive_sign_and_flat_profile():
    flat = fax.FastIons(density=1e17, temperature_ev=200e3)
    for ions in (flat, fax.FastIons(density=1e17, kind="slowing_down")):
        g = [fax.growth_rate(4e5, TAE, S, ((10 * n // 6, n), (11 * n // 6, n)), ITPA, ions) for n in (6, -6)]
        assert g[0] == pytest.approx(g[1]) and g[0] < 0  # no gradient: pure dF/dE (Landau) damping
    assert tae_gamma(200.0, n=6) > 0 > tae_gamma(200.0, n=-6)  # (m, n) -> (-m, -n): counter-propagating
    hollow = fax.FastIons(density=koenies_density()[::-1], temperature_ev=200e3)
    assert fax.growth_rate(4e5, TAE, S, HARM, ITPA, hollow) < 0  # dF/dP_zeta reversed
    assert fax.ion_landau_damping(4e5, TAE, S, HARM, ITPA, 20e3, 2e19) > 0


@pytest.mark.parametrize("arg", [0, 1, 2])
def test_gradients_match_finite_differences(arg):
    f = lambda a: tae_gamma(*[200.0, 1.0, 4.0e5][:arg], a, *[200.0, 1.0, 4.0e5][arg + 1 :])  # noqa: E731
    x0 = [200.0, 1.0, 4.0e5][arg]
    h = 1e-4 * x0
    assert jax.grad(f)(x0) == pytest.approx((f(x0 + h) - f(x0 - h)) / (2 * h), rel=1e-5)


def test_slowing_down_gradient_is_finite():
    f = lambda w: fax.growth_rate(
        w, TAE, S, HARM, ITPA, fax.FastIons(koenies_density(), birth_ev=300e3, critical_ev=100e3, kind="slowing_down")
    )  # noqa: E731
    assert jnp.isfinite(f(4e5)) and jnp.isfinite(jax.grad(f)(4e5))
