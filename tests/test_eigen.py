"""Global operator gates: exact discrete spectrum and Hellmann-Feynman gradients."""

import jax
import jax.numpy as jnp
import numpy as np

import fax
from fax.eigen import assemble


def test_constant_coefficient_spectrum_is_exact():
    n, w2, c = 30, 4.0, 0.5
    s = jnp.linspace(0.0, 1.0, n + 2)
    one = jnp.ones((n + 2, 1, 1), dtype=complex)
    K, M = assemble(w2 * one, one, c * one, s)
    lam = jnp.linalg.eigvalsh(K)
    j = jnp.arange(1, n + 1)
    exact = w2 + 4 * c * (n + 1) ** 2 * jnp.sin(j * jnp.pi / (2 * (n + 1))) ** 2
    np.testing.assert_allclose(lam, exact, rtol=1e-12)


def test_eigenmodes_are_normalized_and_shaped():
    geo = fax.circular_torus(iota=jnp.linspace(0.3, 0.6, 24), epsilon_B=0.1)
    h = fax.mode_family(1, 3)
    modes = fax.eigenmodes(geo, h, 3.4e-8, ntheta=16, nzeta=8)
    assert modes.structures.shape == (22 * 4, 22, 4)
    x = modes.structures.reshape(88, -1)
    np.testing.assert_allclose(jnp.einsum("ki,ij,kj->k", jnp.conj(x), modes.M, x).real, 1.0, rtol=1e-9)
    assert jnp.all(jnp.diff(modes.frequencies_hz) >= 0)


def test_hellmann_feynman_gradients_match_finite_differences():
    h = fax.mode_family(1, 3)

    def f(p, start=0, stop=1):
        geo = fax.circular_torus(s=jnp.linspace(0.05, 1.0, 16), iota=jnp.linspace(0.3, 0.6, 16) + p[1], epsilon_B=0.1)
        return fax.frequency_hz(geo, h, 3.4e-8 * p[0], start, stop, ntheta=16, nzeta=8)

    p = jnp.array([1.0, 0.0])
    for start, stop in ((0, 1), (2, 5)):
        grad = jax.grad(f)(p, start, stop)
        fd = jnp.array([(f(p + e, start, stop) - f(p - e, start, stop)) / 2e-6 for e in 1e-6 * jnp.eye(2)])
        np.testing.assert_allclose(grad, fd, rtol=1e-5)
        np.testing.assert_allclose(grad[0], -0.5 * f(p, start, stop), rtol=1e-10)  # f ~ rho^(-1/2)
