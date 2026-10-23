"""Global operator gates: continuum accumulation, AE3D CTOK parity, Könies ITPA TAE, gradients."""

import itertools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fax
from fax.continuum import MU0, generalized_eigh
from fax.eigen import Blocks, block_factor, block_solve, dense, equilibrium_currents, global_matrices, targeted_eigenvalues
from fax.geometry import surface_grid

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
    tae = v[0]  # AE3D's 337.9 kHz TAE: FAX 272.8; AE3D's inter-surface iota' terms are wrong (plan.md section 6)
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

    def f(p, finite_beta=False, **options):
        s, pressure = konies(40).s, [7.17e3, -6.811e3, -3.585e2] if finite_beta else None
        geo = fax.circular_torus(s=s, major_radius=10.0, B0=3.0, iota=1 / (1.71 + 0.16 * s) + p[1], epsilon_B=0.1, pressure=pressure)
        return fax.frequency_hz(geo, h, rho * p[0], ntheta=16, nzeta=4, finite_beta=finite_beta, **options)

    p = jnp.array([1.0, 0.0])
    targeted = {"target_hz": 6.46e4, "count": 4, "start": 3, "stop": 4}
    for options in ({"start": 2, "stop": 5}, targeted, {**targeted, "finite_beta": True}):
        grad = jax.grad(f)(p, **options)
        fd = jnp.array([(f(p + e, **options) - f(p - e, **options)) / 2e-7 for e in 1e-7 * jnp.eye(2)])
        np.testing.assert_allclose(grad, fd, rtol=1e-5)
        np.testing.assert_allclose(grad[0], -0.5 * f(p, **options), rtol=1e-10)  # f ~ rho^(-1/2)


def _embedding(geo, n=2, T=32):
    """Grid, spectral/FD gradient, embedding basis, metric and test potential for the quadrature checks."""
    s, xm, xn = (np.asarray(a) for a in (geo.s, geo.xm, geo.xn))
    S, Th, Ze = np.meshgrid(s, *2 * [np.linspace(0, 2 * np.pi, T, endpoint=False)], indexing="ij")

    def field(c, sn):
        ang = xm * Th[..., None] - xn * Ze[..., None]
        return (np.asarray(c)[:, None, None] * np.cos(ang) + np.asarray(sn)[:, None, None] * np.sin(ang)).sum(-1)

    def grad(f):  # d/ds by second-order differences, d/dtheta and d/dzeta spectrally
        k = lambda a: np.fft.fftfreq(T, 1 / T).reshape([-1 if i == a else 1 for i in range(f.ndim)])  # noqa: E731
        spec = [np.fft.ifft(1j * k(a) * np.fft.fft(f, axis=a), axis=a).real for a in (1, 2)]
        return np.stack([np.gradient(f, s, axis=0, edge_order=2), *spec], -1)

    R, Zc, P, B = (field(geo.rmnc, geo.rmns), field(geo.zmnc, geo.zmns), field(geo.pmnc, geo.pmns), field(geo.bmnc, geo.bmns))
    X = np.stack([R * np.cos(Ze + P), R * np.sin(Ze + P), Zc], -1)
    e = np.swapaxes(grad(X), -1, -2)  # covariant basis r_i
    g, iota = e @ np.swapaxes(e, -1, -2), np.asarray(geo.iota)[:, None, None]
    Bz = B / np.sqrt(iota**2 * g[..., 1, 1] + 2 * iota * g[..., 1, 2] + g[..., 2, 2])  # B^theta = iota B^zeta
    f = {3: np.sin(np.pi * s) ** 8, 4: -2 * s * np.sin(np.pi * s) ** 8}
    phi = sum(f[m][:, None, None] * np.cos(m * Th - n * Ze) for m in f)
    u = Bz * (iota * grad(phi)[..., 1] + grad(phi)[..., 2]) / B
    y = np.zeros((len(s), 5))
    for m in f:
        y[:, m] = f[m]
    trapz = lambda a: ((a[1:] + a[:-1]) * np.diff(s)).sum() / 2  # noqa: E731
    return grad, e, g, iota, B, Bz, phi, u, y.ravel(), lambda a: trapz(a.mean((1, 2)))


def test_ctok_bending_energy_matches_real_space_quadrature():
    """F and G against finite differences of phi on the boozmn embedding; B^i from |B|, iota and g_ij only."""
    geo = fax.read_boozmn(DATA / "boozmn_ctok.nc")
    grad, e, g, iota, B, Bz, phi, u, y, integral = _embedding(geo)
    gi, J = np.linalg.inv(g), np.abs(np.linalg.det(e))
    rho = float(fax.mass_density(1e19, 2.0))
    energy = lambda a, w: integral(np.einsum("...i,...ij,...j", grad(a), gi, grad(a)) * w)  # noqa: E731
    F, G = (np.asarray(dense(A)) for A in global_matrices(geo, fax.mode_family(2, 4), rho, 32, 4))
    assert 0.5 * (y @ F @ y).real == pytest.approx(energy(u, J), rel=5e-3)
    assert 0.5 * (y @ G @ y).real == pytest.approx(energy(phi, MU0 * rho * J / B**2), rel=5e-3)


@pytest.mark.parametrize("case", ["ctok", "pressure"])
def test_finite_beta_terms_match_real_space_quadrature(case):
    """Kink and pressure terms against quadrature of J.B (curl of B_i on the grid) and the curvature.

    CTOK takes B_i and kappa = (b.grad) b from the embedding alone; the pressure case (Könies with
    its p(s)) takes B_i = (B_s, I, G) and kappa = grad_perp(mu0 p + B^2/2) / B^2 with the given p.
    """
    pressure = [7.17e3, -6.811e3, -3.585e2]
    geo = fax.read_boozmn(DATA / "boozmn_ctok.nc") if case == "ctok" else konies(128)
    if case == "pressure":
        geo = fax.circular_torus(s=geo.s, major_radius=10.0, B0=3.0, iota=geo.iota, epsilon_B=0.1, pressure=pressure)
    grad, e, g, iota, B, Bz, phi, u, y, integral = _embedding(geo)
    gi, sg = np.linalg.inv(g), np.linalg.det(e)
    ps, Bs, sig = (np.asarray(a) for a in equilibrium_currents(geo, surface_grid(geo, 32, 32)))
    up = lambda a: np.einsum("...i,...ij,...jc->...c", a, gi, e)  # noqa: E731 - covariant components -> vector
    s = np.asarray(geo.s)[:, None, None]
    if case == "ctok":
        Bv = Bz[..., None] * (iota[..., None] * e[..., 1, :] + e[..., 2, :])
        Bc, b = np.einsum("...ij,...j->...i", e, Bv), Bv / B[..., None]
        db = np.stack([grad(b[..., c]) for c in range(3)], -2)
        kappa, ps = (Bz / B)[..., None] * (iota[..., None] * db[..., 1] + db[..., 2]), ps[:, None, None]
    else:
        Bc = np.stack([Bs, *np.broadcast_arrays(*(np.asarray(a)[:, None, None] for a in (geo.I, geo.G)), Bs)[:2]], -1)
        b = up(Bc) / B[..., None]
        np.testing.assert_allclose(ps, MU0 * (pressure[1] + 2 * pressure[2] * s[:, 0, 0]), rtol=0.02)  # solvability
        ps = ps[:, None, None]
        dX = up(grad(B**2 / 2)) + ps[..., None] * up(np.stack([np.ones_like(B), 0 * B, 0 * B], -1))
        kappa = (dX - b * (b * dX).sum(-1, keepdims=True)) / B[..., None] ** 2
    d = [grad(Bc[..., k]) for k in range(3)]
    curl = np.stack([d[2][..., 1] - d[1][..., 2], d[0][..., 2] - d[2][..., 0], d[1][..., 0] - d[0][..., 1]], -1)
    sigma = (curl * Bc).sum(-1) / (sg * B**2)  # mu0 J.B / B^2
    np.testing.assert_allclose(sig[2:-2], sigma[2:-2], atol=2e-3 * np.abs(sigma).max())
    dphi, du = grad(phi), grad(u)
    kink = integral(sigma * (Bc * np.cross(dphi, du)).sum(-1) / B * np.sign(sg))
    drive = (Bc * np.cross(dphi, [1.0, 0, 0])).sum(-1) / (B * sg)  # (b x grad phi).grad s
    pres = integral(-2 * ps * drive * (kappa * np.cross(b, up(dphi))).sum(-1) / B**2 * np.abs(sg))
    rho = float(fax.mass_density(1e19, 2.0))
    F0, F1 = (np.asarray(dense(global_matrices(geo, fax.mode_family(2, 4), rho, 32, 4, finite_beta=x)[0])) for x in (False, True))
    assert 0.5 * (y @ (F1 - F0) @ y).real == pytest.approx(kink + pres, rel=5e-3)
    assert abs(pres) < 1e-4 * abs(kink) if case == "ctok" else abs(pres) > 0.1 * abs(kink)


def test_cylinder_newcomb_m1_and_suydam_threshold():
    """Large-aspect cylinder (epsilon_B = 0) gates of the kink and pressure terms.

    m = n = 1 rigid shift inside q = 1 (q = 0.8 + s): the kink term cancels the bending energy
    (Newcomb, m^2 - 1 = 0). m/n = 11/6 at q = 1.71 + 0.16 s with p = p0 (1 - s): Suydam
    ``r B^2 (q'/q)^2 / (8 mu0) + p' > 0`` gives p0 = 1.05e4 Pa; below it the lowest omega^2 is
    the discretization error of a marginal continuum (O(1/N^2)), above it a converged instability.
    """
    rho = fax.mass_density(2e19, 1.0)
    vacuum = [global_matrices(konies(16), fax.mode_family(6, 13, 9), rho, 16, 4, finite_beta=x) for x in (False, True)]
    jax.tree_util.tree_map(lambda a, b: np.testing.assert_array_equal(a, b), *vacuum)  # no current, no pressure
    s = (jnp.arange(800) + 0.5) / 800
    geo = fax.circular_torus(s=s, major_radius=10.0, B0=3.0, iota=1 / (0.8 + s), current=True)
    F0, F1 = (np.asarray(dense(global_matrices(geo, [[1.0, 1.0]], rho, 8, 4, finite_beta=x)[0])).real for x in (False, True))
    r = np.sqrt(np.asarray(s))
    y = r * np.clip((np.sqrt(0.2) - r) / 0.02 + 0.5, 0, 1)
    assert abs(y @ F1 @ y) < 2e-3 * (y @ F0 @ y)
    lowest = {}
    for N, p0 in itertools.product((400, 1600), (1.0e4, 2.0e4)):
        s = (jnp.arange(N) + 0.5) / N
        geo = fax.circular_torus(s=s, major_radius=10.0, B0=3.0, iota=1 / (1.71 + 0.16 * s), pressure=[p0, -p0])
        lowest[N, p0] = float(fax.eigenmodes(geo, [[11.0, 6.0]], rho, ntheta=8, nzeta=4, finite_beta=True).omega2[0])
    assert 12 < lowest[400, 1e4] / lowest[1600, 1e4] < 20 and lowest[1600, 2e4] / lowest[400, 2e4] == pytest.approx(1, abs=0.05)


def test_unstable_omega2_is_signed_with_finite_gradients():
    """Suydam-unstable cylinder (p0 = 2e4 Pa): omega^2 < 0 is kept, with the growth rate and stable frequency separate."""
    rho, s = fax.mass_density(2e19, 1.0), (jnp.arange(400) + 0.5) / 400

    def modes(r, **kw):
        geo = fax.circular_torus(s=s, major_radius=10.0, B0=3.0, iota=1 / (1.71 + 0.16 * s), pressure=[2e4, -2e4])
        return fax.eigenmodes(geo, [[11.0, 6.0]], r, ntheta=8, nzeta=4, finite_beta=True, **kw)

    ev = modes(rho)
    assert ev.omega2[0] < 0 and ev.frequencies_hz[0] == 0
    assert ev.growth_rate[0] == pytest.approx(float(jnp.sqrt(-ev.omega2[0])))

    def f(r, **kw):
        geo = fax.circular_torus(s=s, major_radius=10.0, B0=3.0, iota=1 / (1.71 + 0.16 * s), pressure=[2e4, -2e4])
        return fax.frequency_hz(geo, [[11.0, 6.0]], r, ntheta=8, nzeta=4, finite_beta=True, **kw)

    assert float(jax.grad(f)(rho)) == 0.0  # stable frequency of an unstable mode: finite, flat
    gamma = f(rho, growth=True)
    assert gamma == pytest.approx(float(ev.growth_rate[0]), rel=1e-10)
    assert jax.grad(f)(rho, growth=True) == pytest.approx(-0.5 * gamma / rho, rel=1e-8)  # gamma ~ rho^(-1/2)
    omega2 = lambda x: jnp.stack([fax.eigen._hz(x), fax.eigen._growth(x)])  # noqa: E731
    for x in (-4.0, 4.0, 0.0):
        assert jnp.isfinite(jax.jacfwd(omega2)(x)).all() and jnp.isfinite(jax.jacrev(omega2)(x)).all()


def test_block_solve_survives_zero_pivot():
    """The invertible [[0, 1], [1, 0]] has a zero first Schur pivot: dense fallback, also under jit and with 2x2 blocks."""
    A = Blocks(jnp.zeros((2, 1, 1)), jnp.ones((1, 1, 1)))
    solve = jax.jit(lambda A, b: block_solve(block_factor(A), b))
    np.testing.assert_allclose(solve(A, jnp.ones((2, 1, 1))), 1.0)
    d = jnp.array([[[0.0, 0.0], [0.0, 0.0]], [[2.0, 0.5], [0.5, 1.0]], [[3.0, 0.0], [0.0, 4.0]]])
    B = Blocks(d, jnp.array([[[1.0, 0.2], [0.3, 1.0]], [[0.4, 0.0], [0.1, 0.6]]]))
    b = jnp.arange(12.0).reshape(3, 2, 2)
    np.testing.assert_allclose(solve(B, b), jnp.linalg.solve(dense(B), b.reshape(6, 2)).reshape(3, 2, 2), atol=1e-12)
    assert not block_factor(A)[3] and block_factor(Blocks(jnp.ones((2, 1, 1)) * 3, A.upper))[3]


def test_shift_invert_status_and_rejection_of_unconverged_gradients():
    h, rho = fax.mode_family(6, 13, 9), fax.mass_density(2e19, 1.0)
    options = dict(target_hz=6.46e4, count=4, ntheta=16, nzeta=4)
    good, poor = (fax.eigenmodes(konies(40), h, rho, iters=iters, **options) for iters in (40, 3))
    assert good.status.converged and good.status.residual < 1e-12 and good.status.orthogonality < 1e-10 and good.status.iterations == 40
    assert not poor.status.converged and poor.status.residual > 1e-6 and poor.status.iterations == 3
    full = fax.eigenmodes(konies(40), h, rho, ntheta=16, nzeta=4)
    assert full.status.converged and full.status.iterations == 0 and full.status.residual < 1e-12

    F, G = global_matrices(konies(40), h, rho, 16, 4)
    target = float((2 * np.pi * 6.46e4) ** 2)
    assert jnp.isfinite(targeted_eigenvalues(F, G, target, 4, 40)).all()
    assert jnp.isnan(targeted_eigenvalues(F, G, target, 4, 3)).all()
    jvp = jax.jvp(lambda f: targeted_eigenvalues(Blocks(f, F.upper), G, target, 4, 3), (F.diag,), (jnp.ones_like(F.diag),))
    assert jnp.isnan(jvp[1]).all()
    run = lambda iters: fax.frequency_hz(konies(40), h, rho, 0, 1, iters=iters, **options)  # noqa: E731
    assert jnp.isfinite(run(40)) and jnp.isnan(run(3))


def test_cluster_mean_first_derivative_at_degeneracy():
    """Mean of a repeated pair: d/dt is the projected trace; only first order is supported there (the
    second derivative is NaN whenever the spectrum has a repeated eigenvalue)."""
    K, M = jnp.diag(jnp.array([1.0, 1.0, 3.0])), jnp.eye(3)
    D = jnp.array([[1.0, 0.5, 0.2], [0.5, 2.0, 0.1], [0.2, 0.1, 0.0]])
    mean = lambda t, a=0, b=2: fax.eigenvalue_mean(K + t * D, M, a, b)  # noqa: E731
    assert jax.grad(mean)(0.0) == pytest.approx(1.5, rel=1e-12)  # (D00 + D11) / 2
    assert jax.grad(mean)(0.0) == pytest.approx((mean(1e-6) - mean(-1e-6)) / 2e-6, rel=1e-6)
    mean = lambda t: fax.eigenvalue_mean(jnp.diag(jnp.array([1.0, 1.1, 3.0])) + t * D, M, 0, 2)  # noqa: E731
    fd = (mean(1e-3) - 2 * mean(0.0) + mean(-1e-3)) / 1e-6  # no repeated eigenvalue anywhere: second order works
    assert jax.grad(jax.grad(mean))(0.0) == pytest.approx(fd, rel=1e-4)
