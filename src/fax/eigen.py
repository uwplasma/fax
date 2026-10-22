"""Global shear-Alfvén eigenmodes (AE3D weak form) and differentiable eigenvalues.

The reduced ideal-MHD weak form of Spong, D'Azevedo & Todo (2010) for the
electrostatic potential ``phi`` is::

    w^2 int mu0 rho / B^2 grad(phi~)* . grad(phi) dV = int grad(u~)* . grad(u) dV,   u = B.grad(phi) / B,

without the pressure-gradient and parallel-current (kink) terms, which are
small for TAE-type modes at low beta. With ``phi = sum_j f_k(s) e_j``, linear
radial hat functions ``f_k`` on the geometry surfaces and
``e_j = exp(i(m_j theta - n_j zeta))``, the Galerkin matrices ``F y = w^2 G y``
are Hermitian block tridiagonal with ``H x H`` blocks. ``phi = 0`` at ``s = 1``
and, for ``m != 0``, at the axis; ``m = 0`` harmonics are flat between the axis
and the first surface (AE3D's regularity condition). Coefficients are averaged
over each radial element, as in AE3D.

Eigenvalue derivatives use the Hermitian identity ``d lam = y.H (dF - lam dG) y``
for ``G``-normalized ``y``. Averaging over a cluster of eigenvalues gives the
trace of the projected perturbation, which stays smooth through crossings
inside the cluster.
"""

from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .continuum import MU0, generalized_eigh
from .geometry import Geometry, _d_ds, surface_grid


class Blocks(NamedTuple):
    """Hermitian block-tridiagonal matrix; ``upper[k]`` couples surfaces ``k`` and ``k + 1``."""

    diag: jax.Array
    upper: jax.Array


class Eigenmodes(NamedTuple):
    """Global eigenmodes: ``structures`` has shape ``(modes, surfaces, harmonics)``, ``G``-normalized."""

    s: jax.Array
    omega2: jax.Array
    frequencies_hz: jax.Array
    structures: jax.Array


def global_matrices(geo: Geometry, harmonics: Any, density: Any, ntheta: int = 32, nzeta: int = 32) -> tuple[Blocks, Blocks]:
    """Bending ``F`` and inertia ``G`` (``w^2`` in rad^2/s^2); surfaces must lie strictly inside ``(0, 1)``."""
    h = jnp.asarray(harmonics, dtype=float)
    m, n = h[:, 0, None, None], h[:, 1, None, None]
    g = surface_grid(geo, ntheta, nzeta)
    e = jnp.exp(1j * (m * g.theta - n * g.zeta))
    zero = jnp.zeros_like(e)
    D, (iota_s, D_s) = geo.G + geo.iota * geo.I, _d_ds(jnp.stack([geo.iota, geo.G + geo.iota * geo.I], 1), geo.s).T
    r = lambda a: a[:, None, None, None]  # noqa: E731 - per-surface value -> (S, H, T, Z)
    k = r(geo.iota) * m - n
    c = 1j * k * g.B[:, None] / r(D)  # u_j = c_j e_j f_j
    c_s = 1j * (r(iota_s) * m * g.B[:, None] + k * (g.dB[:, None, ..., 0] - g.B[:, None] * r(D_s / D))) / r(D)
    c_t, c_z = 1j * k * g.dB[:, None, ..., 1] / r(D), 1j * k * g.dB[:, None, ..., 2] / r(D)
    # Covariant components on (grad s, grad theta, grad zeta) of the f' and f parts of each gradient.
    inertia = jnp.stack([e, zero, zero], -1), 1j * jnp.stack([zero, m * e, -n * e], -1)
    bending = jnp.stack([c * e, 0 * c, 0 * c], -1), jnp.stack([c_s, c_t + 1j * m * c, c_z - 1j * n * c], -1) * e[..., None]
    rho = jnp.broadcast_to(jnp.asarray(density, dtype=float), geo.s.shape)
    volume = jnp.abs(g.sqrt_g)
    pairs = ((volume, bending), (r(MU0 * rho)[:, 0] * volume / g.B**2, inertia))
    F, G = (_assemble(geo.s, h[:, 0] == 0, *_coefficients(g.metric, w, *x)) for w, x in pairs)
    return F, G


def _coefficients(metric: jax.Array, w: jax.Array, X0: Any, X1: Any) -> tuple[jax.Array, ...]:
    """Surface averages ``<w X_a*.X_b>`` for the ``f' f'``, ``f' f`` and ``f f`` products, shape ``(S, H, H)``."""
    shape = jnp.broadcast_shapes(jnp.shape(X0), jnp.shape(X1), (w.shape[0], 1, *w.shape[1:], 3))
    X0, X1 = jnp.broadcast_to(X0, shape), jnp.broadcast_to(X1, shape)
    gw = metric * (w / (w.shape[-1] * w.shape[-2]))[..., None, None]
    dot = lambda P, Q: jnp.einsum("shtzi,stzij,sqtzj->shq", jnp.conj(P), gw, Q)  # noqa: E731
    return dot(X0, X0), dot(X0, X1), dot(X1, X1)


def _assemble(s: jax.Array, flat: jax.Array, A: jax.Array, Bm: jax.Array, D: jax.Array) -> Blocks:
    """Linear-element assembly with element-averaged coefficients on nodes ``[0, s..., 1]``."""
    ext = lambda a: jnp.concatenate([a[:1], a, a[-1:]])  # noqa: E731 - axis and edge elements reuse the end surfaces
    A, Bm, D = (0.5 * (ext(a)[1:] + ext(a)[:-1]) for a in (A, Bm, D))
    h = jnp.diff(jnp.concatenate([jnp.zeros(1), s, jnp.ones(1)]))[:, None, None]
    BH = jnp.conj(jnp.swapaxes(Bm, -1, -2))
    LL, RR = A / h - 0.5 * (Bm + BH) + D * h / 3, A / h + 0.5 * (Bm + BH) + D * h / 3
    LR, RL = -A / h - 0.5 * (Bm - BH) + D * h / 6, -A / h + 0.5 * (Bm - BH) + D * h / 6
    P = jnp.diag(flat.astype(float))
    diag = (RR[:-1] + LL[1:]).at[0].add(RL[0] @ P + P @ LR[0] + P @ LL[0] @ P)
    return Blocks(diag, LR[1:-1])


def dense(A: Blocks) -> jax.Array:
    """Dense ``(N H, N H)`` matrix of a block-tridiagonal one."""
    N, H = A.diag.shape[:2]
    i = jnp.arange(N)
    full = jnp.zeros((N, N, H, H), A.diag.dtype).at[i, i].set(A.diag).at[i[:-1], i[1:]].set(A.upper)
    full = full.at[i[1:], i[:-1]].set(jnp.conj(jnp.swapaxes(A.upper, -1, -2)))
    return jnp.swapaxes(full, 1, 2).reshape(N * H, N * H)


def _matvec(A: Blocks, x: jax.Array) -> jax.Array:
    """``A x`` for ``x`` of shape ``(N, H, p)``."""
    y = jnp.einsum("kij,kjp->kip", A.diag, x)
    y = y.at[:-1].add(jnp.einsum("kij,kjp->kip", A.upper, x[1:]))
    return y.at[1:].add(jnp.einsum("kji,kjp->kip", jnp.conj(A.upper), x[:-1]))


def _quad(y: jax.Array, A: Blocks) -> jax.Array:
    """``y_k.H A y_k`` for each column ``k`` of ``y`` (shape ``(N, H, p)``)."""
    return jnp.real(jnp.einsum("kip,kip->p", jnp.conj(y), _matvec(A, y)))


def block_factor(A: Blocks) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Block LU (Thomas) factors of ``A``, ``O(N H^3)``: inverse pivots, multipliers and upper blocks."""
    zero = jnp.zeros_like(A.diag[:1])
    up_prev, up_next = jnp.concatenate([zero, A.upper]), jnp.concatenate([A.upper, zero])

    def forward(S_inv: jax.Array, xs: tuple) -> tuple:
        d, u = xs
        L = jnp.conj(jnp.swapaxes(u, -1, -2)) @ S_inv
        S_inv = jnp.linalg.inv(d - L @ u)
        return S_inv, (S_inv, L)

    S_inv, L = jax.lax.scan(forward, jnp.zeros_like(A.diag[0]), (A.diag, up_prev))[1]
    return S_inv, L, up_next


def block_solve(factors: tuple[jax.Array, jax.Array, jax.Array], b: jax.Array) -> jax.Array:
    """Solve ``A x = b`` from :func:`block_factor` factors, ``O(N H^2 p)``; ``b`` has shape ``(N, H, p)``."""
    S_inv, L, up_next = factors
    z = jax.lax.scan(lambda z, xs: (xs[1] - xs[0] @ z,) * 2, jnp.zeros_like(b[0]), (L, b))[1]
    backward = lambda x, xs: (xs[0] @ (xs[2] - xs[1] @ x),) * 2  # noqa: E731
    return jax.lax.scan(backward, jnp.zeros_like(b[0]), (S_inv, up_next, z), reverse=True)[1]


def shift_invert(F: Blocks, G: Blocks, target: float, count: int = 8, iters: int = 40) -> tuple[jax.Array, jax.Array]:
    """``count`` eigenpairs of ``F y = lam G y`` nearest ``target`` (ascending), by shift-invert subspace iteration.

    A block of ``2 count + 8`` vectors is iterated with ``(F - target G)^-1 G``
    and Rayleigh-Ritz on ``(F, G)`` extracts the pairs. Eigenvectors have shape
    ``(N, H, count)`` and are ``G``-normalized.
    """
    N, H = F.diag.shape[:2]
    p = min(2 * count + 8, N * H)
    A = block_factor(Blocks(F.diag - target * G.diag, F.upper - target * G.upper))
    step = lambda _, Y: jnp.linalg.qr(block_solve(A, _matvec(G, Y)).reshape(N * H, p))[0].reshape(N, H, p)  # noqa: E731
    Y = jax.lax.fori_loop(0, iters, step, jax.random.normal(jax.random.PRNGKey(0), (N, H, p)).astype(F.diag.dtype))
    Yf = Y.reshape(N * H, p)
    project = lambda M: jnp.conj(Yf.T) @ _matvec(M, Y).reshape(N * H, p)  # noqa: E731
    lam, c = generalized_eigh(project(F), project(G))
    pick = jnp.sort(jnp.argsort(jnp.abs(lam - target))[:count])
    return lam[pick], (Yf @ c[:, pick]).reshape(N, H, count)


@partial(jax.custom_jvp, nondiff_argnums=(2, 3, 4))
def targeted_eigenvalues(F: Blocks, G: Blocks, target: float, count: int = 8, iters: int = 40) -> jax.Array:
    """Differentiable eigenvalues from :func:`shift_invert` (Hellmann-Feynman JVP)."""
    return shift_invert(F, G, target, count, iters)[0]


@targeted_eigenvalues.defjvp
def _targeted_jvp(target: float, count: int, iters: int, primals: tuple, tangents: tuple) -> tuple[jax.Array, jax.Array]:
    (F, G), (dF, dG) = primals, tangents
    lam, y = shift_invert(F, G, target, count, iters)
    return lam, _quad(y, dF) - lam * _quad(y, dG)


@partial(jax.custom_jvp, nondiff_argnums=(2, 3))
def eigenvalue_mean(K: jax.Array, M: jax.Array, start: int = 0, stop: int = 1) -> jax.Array:
    """Mean of dense sorted generalized eigenvalues ``start <= i < stop`` (``stop = start + 1``: one eigenvalue)."""
    return jnp.mean(generalized_eigh(K, M)[0][start:stop])


@eigenvalue_mean.defjvp
def _eigenvalue_mean_jvp(start: int, stop: int, primals: tuple, tangents: tuple) -> tuple[jax.Array, jax.Array]:
    (K, M), (dK, dM) = primals, tangents
    lam, x = generalized_eigh(K, M)
    lam, x = lam[start:stop], x[:, start:stop]
    d = jnp.einsum("ik,ij,jk->k", jnp.conj(x), dK, x) - lam * jnp.einsum("ik,ij,jk->k", jnp.conj(x), dM, x)
    return jnp.mean(lam), jnp.mean(jnp.real(d))


def _hz(omega2: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.maximum(omega2, 0.0)) / (2 * jnp.pi)


def eigenmodes(
    geo: Geometry, harmonics: Any, density: Any, *, target_hz: float | None = None, count: int = 8, iters: int = 40, **grid: Any
) -> Eigenmodes:
    """Global eigenmodes: all of them densely, or ``count`` nearest ``target_hz`` by shift-invert.

    ``grid`` (``ntheta``, ``nzeta``) is passed to :func:`global_matrices`.
    """
    F, G = global_matrices(geo, harmonics, density, **grid)
    N, H = F.diag.shape[:2]
    if target_hz is None:
        omega2, x = generalized_eigh(dense(F), dense(G))
        x = jnp.swapaxes(x, 0, 1).reshape(-1, N, H)
    else:
        omega2, y = shift_invert(F, G, (2 * jnp.pi * target_hz) ** 2, count, iters)
        x = jnp.moveaxis(y, -1, 0)
    return Eigenmodes(geo.s, omega2, _hz(omega2), x)


def frequency_hz(
    geo: Geometry, harmonics: Any, density: Any, start: int = 0, stop: int = 1, *, target_hz: float | None = None, **options: Any
) -> jax.Array:
    """Differentiable ``sqrt(mean omega^2) / 2 pi`` of eigenvalues ``start:stop`` (all modes, or those nearest ``target_hz``).

    ``options`` are ``count`` and ``iters`` for the targeted solve and the grid sizes.
    """
    solver = {key: options.pop(key) for key in ("count", "iters") if key in options}
    F, G = global_matrices(geo, harmonics, density, **options)
    if target_hz is None:
        return _hz(eigenvalue_mean(dense(F), dense(G), start, stop))
    lam = targeted_eigenvalues(F, G, float((2 * jnp.pi * target_hz) ** 2), solver.get("count", 8), solver.get("iters", 40))
    return _hz(jnp.mean(lam[start:stop]))
