"""Global radial eigenmodes and differentiable eigenvalues.

The global operator here is the *interim* reduced radial envelope::

    [K_loc(s) - d_s C(s) d_s] x = w^2 M_loc(s) x,
    C_ij = scale < |J| |grad s|^2 B^2 / mu0 e_i^* e_j >,

discretized with second-order finite differences and Dirichlet or Neumann
ends. It has no continuum resonance, so its frequencies depend on ``scale``;
it is kept as the differentiable plumbing until the AE3D weak form replaces it
(plan phase P2).

Eigenvalue derivatives use the Hermitian identity
``d lam = x.H (dK - lam dM) x`` for ``M``-normalized ``x``. Averaging over a
cluster of sorted eigenvalues gives the trace of the projected perturbation,
which stays smooth through crossings inside the cluster.
"""

from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .continuum import MU0, continuum_matrices, generalized_eigh
from .geometry import Geometry, surface_grid


@partial(jax.custom_jvp, nondiff_argnums=(2, 3))
def eigenvalue_mean(K: jax.Array, M: jax.Array, start: int = 0, stop: int = 1) -> jax.Array:
    """Mean of sorted generalized eigenvalues ``start <= i < stop`` (``stop = start + 1``: one eigenvalue)."""
    return jnp.mean(generalized_eigh(K, M)[0][start:stop])


@eigenvalue_mean.defjvp
def _eigenvalue_mean_jvp(start: int, stop: int, primals: tuple, tangents: tuple) -> tuple[jax.Array, jax.Array]:
    (K, M), (dK, dM) = primals, tangents
    lam, x = generalized_eigh(K, M)
    lam, x = lam[start:stop], x[:, start:stop]
    d = jnp.einsum("ik,ij,jk->k", jnp.conj(x), dK, x) - lam * jnp.einsum("ik,ij,jk->k", jnp.conj(x), dM, x)
    return jnp.mean(lam), jnp.mean(jnp.real(d))


class Eigenmodes(NamedTuple):
    """Global eigenmodes: ``structures`` has shape ``(modes, surfaces, harmonics)``."""

    s: jax.Array
    omega2: jax.Array
    frequencies_hz: jax.Array
    structures: jax.Array
    K: jax.Array
    M: jax.Array


def radial_stiffness(geo: Geometry, harmonics: Any, ntheta: int = 32, nzeta: int = 32) -> jax.Array:
    """Envelope coefficient ``C_ij = < |J| |grad s|^2 B^2 / mu0 e_i^* e_j >`` on each surface."""
    h = jnp.asarray(harmonics, dtype=float)
    grid = surface_grid(geo, ntheta, nzeta)
    dm, dn = h[None, :, 0] - h[:, None, 0], h[None, :, 1] - h[:, None, 1]
    phase = jnp.exp(1j * (dm[..., None, None] * grid.theta - dn[..., None, None] * grid.zeta))
    weight = jnp.abs(grid.jacobian) * grid.grad_s2 * grid.B**2 / MU0
    C = jnp.mean(weight[:, None, None] * phase, axis=(-2, -1))
    return 0.5 * (C + jnp.conj(jnp.swapaxes(C, -1, -2)))


def assemble(K_loc: jax.Array, M_loc: jax.Array, C: jax.Array, s: jax.Array, boundary: str = "dirichlet") -> tuple[jax.Array, jax.Array]:
    """Assemble dense block-tridiagonal global ``(K, M)`` from per-surface blocks."""
    face = 0.5 * (C[1:] + C[:-1]) / jnp.diff(s)[:, None, None] ** 2
    zero = jnp.zeros_like(face[:1])
    left, right = jnp.concatenate([zero, face]), jnp.concatenate([face, zero])
    diag = K_loc + left + right
    keep = slice(1, -1) if boundary == "dirichlet" else slice(None)
    diag, M_blk, upper = diag[keep], M_loc[keep], -right[keep][:-1]
    n, h = diag.shape[0], diag.shape[1]
    i = jnp.arange(n)
    K = jnp.zeros((n, n, h, h), diag.dtype).at[i, i].set(diag).at[i[:-1], i[1:]].set(upper)
    K = K.at[i[1:], i[:-1]].set(jnp.conj(jnp.swapaxes(upper, -1, -2)))
    M = jnp.zeros((n, n, h, h), M_blk.dtype).at[i, i].set(M_blk)
    flat = lambda A: jnp.swapaxes(A, 1, 2).reshape(n * h, n * h)  # noqa: E731
    return flat(K), flat(M)


def global_matrices(
    geo: Geometry,
    harmonics: Any,
    density: Any,
    *,
    scale: float = 1.0e-4,
    boundary: str = "dirichlet",
    ntheta: int = 32,
    nzeta: int = 32,
) -> tuple[jax.Array, jax.Array]:
    """Global ``(K, M)`` for the interim envelope operator."""
    K_loc, M_loc = continuum_matrices(geo, harmonics, density, ntheta, nzeta)
    C = scale * radial_stiffness(geo, harmonics, ntheta, nzeta)
    return assemble(K_loc, M_loc, C, geo.s, boundary)


def eigenmodes(geo: Geometry, harmonics: Any, density: Any, **options: Any) -> Eigenmodes:
    """Solve the global problem densely; ``options`` are passed to :func:`global_matrices`."""
    K, M = global_matrices(geo, harmonics, density, **options)
    omega2, x = generalized_eigh(K, M)
    h = jnp.asarray(harmonics).shape[0]
    s = geo.s[1:-1] if options.get("boundary", "dirichlet") == "dirichlet" else geo.s
    structures = jnp.swapaxes(x, 0, 1).reshape(-1, s.shape[0], h)
    return Eigenmodes(s, omega2, jnp.sqrt(jnp.maximum(omega2, 0.0)) / (2 * jnp.pi), structures, K, M)


def frequency_hz(geo: Geometry, harmonics: Any, density: Any, start: int = 0, stop: int = 1, **options: Any) -> jax.Array:
    """Differentiable frequency of a mode cluster (Hellmann-Feynman JVP)."""
    K, M = global_matrices(geo, harmonics, density, **options)
    return jnp.sqrt(jnp.maximum(eigenvalue_mean(K, M, start, stop), 0.0)) / (2 * jnp.pi)
