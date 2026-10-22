"""Global operator gates: continuum accumulation, AE3D CTOK parity, Könies ITPA TAE, gradients."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fax
from fax.continuum import MU0, generalized_eigh
from fax.eigen import dense, global_matrices

DATA = Path(__file__).parent / "data"


def konies(N):
    """Könies et al. (2018) ITPA TAE: R0 = 10 m, a = 1 m, B0 = 3 T, q = 1.71 + 0.16 r^2, n = 6, hydrogen 2e19."""
    s = (jnp.arange(N) + 0.5) / N
    return fax.circular_torus(s=s, major_radius=10.0, minor_radius=1.0, B0=3.0, iota=1 / (1.71 + 0.16 * s), epsilon_B=0.1)


def test_uniform_field_spectrum_is_a_continuum():
    rho = fax.mass_density(1e19, 2.0)
    edges = np.array([0.4, 0.7]) * 2.7 / 5.5 / np.sqrt(MU0 * rho)  # |n - m iota| v_A / R at s = 0, 1
    spread = []
    for N in (40, 160):
        geo = fax.circular_torus(s=(jnp.arange(N) + 0.5) / N, iota=0.3 + 0.3 * (jnp.arange(N) + 0.5) / N)
        modes = fax.eigenmodes(geo, jnp.array([[1.0, 1.0]]), rho, ntheta=8, nzeta=4)
        w = np.sqrt(np.asarray(modes.omega2))
        assert edges[0] < w.min() < edges[0] * (1 + 0.5 / N) and edges[1] * (1 - 0.5 / N) < w.max() < edges[1]
        jump = np.abs(np.diff(np.asarray(modes.structures[:, :, 0]), axis=1)) ** 2  # singular: phi' localizes
        spread.append(np.median(jump.sum(1) ** 2 / (jump**2).sum(1)))
    assert spread[0] == pytest.approx(spread[1], rel=0.05)  # width in cells is fixed, so it shrinks as 1/N


def test_ae3d_ctok_parity():
    """AE3D's CTOK run (same boozmn, n = 2, m = 0..12, D at 1e19 m^-3) on its 99-point full mesh."""
    geo = fax.read_boozmn(DATA / "boozmn_ctok.nc").full_mesh()
    geo = jax.tree_util.tree_map(lambda a: jnp.concatenate([a, 2 * a[-1:] - a[-2:-1]]) if a.ndim and a.shape[0] == 98 else a, geo)
    F, G = (np.asarray(dense(A)) for A in global_matrices(geo, fax.mode_family(2, 12), fax.mass_density(1e19, 2.0), 32, 4))
    w, x = (np.asarray(a) for a in generalized_eigh(F, G))
    x = x / np.linalg.norm(x, axis=0)
    ref = np.load(DATA / "ae3d_ctok_modes.npz")
    v = ref["vectors"].reshape(len(ref["khz"]), -1).astype(float)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    khz = np.sqrt(w) / 2e3 / np.pi
    overlap = np.abs(v @ x)
    best = overlap.argmax(1)
    rel = khz[best] / ref["khz"] - 1
    np.testing.assert_array_less(np.abs(rel[1:]), 0.015)  # six global modes, overlap > 0.9
    assert overlap[np.arange(len(best)), best].min() > 0.9
    tae = v[0]
    assert np.sqrt((tae @ F @ tae).real / (tae @ G @ tae).real) / 2e3 / np.pi == pytest.approx(ref["khz"][0], rel=0.03)


def test_konies_tae_frequency_and_convergence():
    h, rho = fax.mode_family(6, 16, 5), fax.mass_density(2e19, 1.0)
    omega = []
    for N in (50, 100, 200):
        modes = fax.eigenmodes(konies(N), h, rho, target_hz=6.46e4, count=3, ntheta=32, nzeta=4)
        omega.append(float(np.sqrt(modes.omega2[-1])))
    assert 4.01e5 < omega[-1] < 4.13e5  # CAS3D (shifted equilibrium) and GYGLES (ad hoc, as here)
    assert abs(omega[1] / omega[2] - 1) < 2e-4 and abs(omega[0] / omega[2] - 1) < 5e-4


def test_shift_invert_matches_dense_and_gradients_match_finite_differences():
    h, rho = fax.mode_family(6, 13, 9), fax.mass_density(2e19, 1.0)
    dense_w = fax.eigenmodes(konies(40), h, rho, ntheta=16, nzeta=4).omega2
    target = fax.eigenmodes(konies(40), h, rho, target_hz=6.46e4, count=4, ntheta=16, nzeta=4).omega2
    np.testing.assert_allclose(target, dense_w[np.sort(np.argsort(np.abs(dense_w - (2 * np.pi * 6.46e4) ** 2))[:4])], rtol=1e-10)

    def f(p, **options):
        geo = fax.circular_torus(s=konies(40).s, major_radius=10.0, B0=3.0, iota=1 / (1.71 + 0.16 * konies(40).s) + p[1], epsilon_B=0.1)
        return fax.frequency_hz(geo, h, rho * p[0], ntheta=16, nzeta=4, **options)

    p = jnp.array([1.0, 0.0])
    for options in ({"start": 2, "stop": 5}, {"target_hz": 6.46e4, "count": 4, "start": 3, "stop": 4}):
        grad = jax.grad(f)(p, **options)
        fd = jnp.array([(f(p + e, **options) - f(p - e, **options)) / 2e-6 for e in 1e-6 * jnp.eye(2)])
        np.testing.assert_allclose(grad, fd, rtol=1e-5)
        np.testing.assert_allclose(grad[0], -0.5 * f(p, **options), rtol=1e-10)  # f ~ rho^(-1/2)
