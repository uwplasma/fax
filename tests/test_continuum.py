"""Continuum gates: analytic uniform-field limit, STELLGAP CTOK parity, gap formulas."""

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
    errors = []
    for s, row in zip(np.asarray(geo.s) + 0.005, khz, strict=True):  # STELLGAP labels half-mesh data with full-mesh s
        j = np.argmin(abs(ref["s"] - s))
        if abs(ref["s"][j] - s) < 2e-3:
            errors.append(np.linalg.norm(row - ref["khz"][j]) / np.linalg.norm(ref["khz"][j]))
    assert len(errors) == 97
    assert np.mean(errors) < 5e-3 and np.percentile(errors, 95) < 1.2e-2


def test_gaps_and_gap_location():
    center, width = fax.gaps(jnp.array([[1.0, 2.0, 5.0, 5.5], [0.0, 3.0, 3.5, 4.0]]))
    np.testing.assert_allclose(center, [3.5, 1.5])
    np.testing.assert_allclose(width, [3.0, 3.0])
    assert fax.gap_location(1, 1, 1) == pytest.approx((2 / 3, 1 / 3))  # TAE: iota = 2n/(2m+1)
    assert fax.gap_location(4, 1, 0, -1, 4)[0] == pytest.approx(2.0)  # MAE: iota = (2n - nfp)/(2m)
