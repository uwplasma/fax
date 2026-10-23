"""Continuum gates: analytic uniform-field limit, STELLGAP CTOK parity, gap formulas."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fax
from fax.continuum import MU0

DATA = Path(__file__).parent / "data"


def test_uniform_field_continuum_is_k_parallel_v_alfven():
    geo = fax.circular_torus(iota=jnp.linspace(0.3, 0.6, 24))
    harmonics = fax.mode_family((1, 2), 4)
    rho = fax.mass_density(1e19, 2.0)
    f = fax.continuum(geo, harmonics, rho)
    k = (harmonics[:, 1] - harmonics[:, 0] * geo.iota[:, None]) / 5.5
    exact = jnp.sort(jnp.abs(k) * 2.7 / jnp.sqrt(MU0 * rho) / (2 * jnp.pi), axis=1)
    np.testing.assert_allclose(f, exact, rtol=1e-9)


def test_matrices_are_hermitian_and_resolution_converged():
    geo = fax.read_boozmn(DATA / "boozmn_ctok.nc").surfaces(slice(10, 90, 10))
    h = fax.mode_family(2, 6)
    K, M = fax.continuum_matrices(geo, h, 3.4e-8)
    np.testing.assert_allclose(K, jnp.conj(jnp.swapaxes(K, -1, -2)), atol=1e-12 * float(jnp.abs(K).max()))
    np.testing.assert_allclose(fax.continuum(geo, h, 3.4e-8, 32, 32), fax.continuum(geo, h, 3.4e-8, 64, 64), rtol=1e-9)


def test_stellgap_ctok_parity():
    """Same boozmn_ctok.nc and STELLGAP inputs (n = 2, m = 0..12, deuterium at 1e19 m^-3)."""
    geo = fax.read_boozmn(DATA / "boozmn_ctok.nc")
    khz = np.asarray(jax.jit(lambda g: fax.continuum(g, fax.mode_family(2, 12), fax.mass_density(1e19, 2.0)))(geo)) / 1e3
    ref = np.load(DATA / "stellgap_ctok_continuum.npz")
    # STELLGAP averages half-mesh rows k, k+1 onto full surface k = 2..ns-1 but labels it
    # k/(ns-1) (stellgap rho = nsurf/nsurf_max), one cell outward of the physical (k-1)/(ns-1).
    s_ref = ref["s"] - 1.0 / 100.0  # boozmn_ctok.nc has ns_b = 101
    inside = (geo.s > s_ref[0]) & (geo.s < s_ref[-1])
    expected = np.stack([np.interp(geo.s[inside], s_ref, col) for col in ref["khz"].T], axis=1)
    errors = np.linalg.norm(khz[inside] - expected, axis=1) / np.linalg.norm(expected, axis=1)
    assert errors.size == 97
    assert np.mean(errors) < 7e-3 and np.percentile(errors, 95) < 7e-3


def test_gaps_and_gap_location():
    center, width = fax.gaps(jnp.array([[1.0, 2.0, 5.0, 5.5], [0.0, 3.0, 3.5, 4.0]]))
    np.testing.assert_allclose(center, [3.5, 1.5])
    np.testing.assert_allclose(width, [3.0, 3.0])
    assert fax.gap_location(1, 1, 1) == pytest.approx((2 / 3, 1 / 3))  # TAE: iota = 2n/(2m+1)
    assert fax.gap_location(4, 1, 0, -1, 4)[0] == pytest.approx(2.0)  # MAE: iota = (2n - nfp)/(2m)


def test_elliptic_two_mode_gap_has_full_width_two_f0_c():
    """``circular_torus(iota=1)``, ellipticity kappa on the (0, 1)-(2, 1) crossing: the edge separation is ``2 f0 |c|``,
    ``c = (kappa^2 - 1) / (2 (kappa^2 + 1))`` the complex Fourier coefficient of ``|grad s|^2 / <|grad s|^2> - 1``, and
    ``|c|^2 (iota dm - n)^2`` of the cost is the squared full width in ``w_A`` units (``w_A = B0 / (R0 sqrt(mu0 rho))``)."""
    base, kappa, harmonics, rho = fax.circular_torus(iota=1.0), 1.001, [[0, 1], [2, 1]], fax.mass_density(1e19, 2.0)
    geo = replace(base, zmns=base.zmns.at[:, 1].multiply(kappa))
    f0 = fax.continuum(base, harmonics, rho)[:, 0]  # both branches cross at f0
    c = (kappa**2 - 1) / (2 * (kappa**2 + 1))
    center, width = fax.gaps(fax.continuum(geo, harmonics, rho))
    np.testing.assert_allclose(width, 2 * f0 * c, rtol=1e-6)
    np.testing.assert_allclose(center, f0, rtol=1e-6)
    f_a = 2.7 / (5.5 * jnp.sqrt(MU0 * rho)) / (2 * jnp.pi)
    assert fax.continuum_cost(geo, (0, 0)) / 2 == pytest.approx(jnp.sum((width / f_a) ** 2), rel=1e-2)  # +-(dm, n) twice
