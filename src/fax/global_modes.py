"""Differentiable radial shear-Alfven eigenmode operators.

The operators here are intentionally one-dimensional radial envelope models.
They are not a replacement for the full 3-D AE3D/CAS3D/FAR3D class of
calculations, but they provide the first global eigenvalue layer above the
surface-local continuum. The generalized problem is

    K xi = omega^2 M xi,

with Dirichlet radial boundary conditions on the perturbation envelope. The
constant-coefficient limit has an exact finite-difference spectrum, which makes
it a useful reviewer-facing validation gate for radial discretization,
eigenpair residuals, and autodiff.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from .continuum import continuum_branch_matrix
from .equilibrium import EquilibriumGeometry
from .metric import generalized_eigen_residual_norms, generalized_hermitian_eigensystem


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RadialEigenmodeProblem:
    """Finite-difference radial shear-Alfven eigenmode problem.

    Parameters
    ----------
    s:
        Full radial grid including both Dirichlet boundary points.
    continuum_omega:
        Local continuum angular frequency on ``s`` in rad/s.
    radial_coupling:
        Positive radial stiffness coefficient on ``s`` in rad^2/s^2.
    mass_weight:
        Positive generalized mass weight on ``s``.
    metadata:
        Static provenance for reports and tests.
    """

    s: jax.Array
    continuum_omega: jax.Array
    radial_coupling: jax.Array
    mass_weight: jax.Array
    metadata: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "s", jnp.asarray(self.s))
        object.__setattr__(self, "continuum_omega", jnp.asarray(self.continuum_omega))
        object.__setattr__(self, "radial_coupling", jnp.asarray(self.radial_coupling))
        object.__setattr__(self, "mass_weight", jnp.asarray(self.mass_weight))
        object.__setattr__(self, "metadata", tuple((str(key), value) for key, value in self.metadata))
        shape = self.s.shape
        if self.s.ndim != 1:
            raise ValueError("s must be one-dimensional")
        if self.s.shape[0] < 4:
            raise ValueError("at least four radial points are required")
        for name in ("continuum_omega", "radial_coupling", "mass_weight"):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have the same shape as s")

    def tree_flatten(self):
        """Return JAX pytree children and static metadata."""
        children = (self.s, self.continuum_omega, self.radial_coupling, self.mass_weight)
        return children, self.metadata

    @classmethod
    def tree_unflatten(cls, metadata: tuple[tuple[str, Any], ...], children: tuple[Any, ...]) -> "RadialEigenmodeProblem":
        """Reconstruct a problem from pytree parts."""
        s, continuum_omega, radial_coupling, mass_weight = children
        return cls(
            s=s,
            continuum_omega=continuum_omega,
            radial_coupling=radial_coupling,
            mass_weight=mass_weight,
            metadata=metadata,
        )


@dataclass(frozen=True)
class RadialEigenmodeResult:
    """Eigenvalues, eigenvectors, and residuals for a radial mode problem."""

    omega2: jax.Array
    frequencies_hz: jax.Array
    eigenvectors: jax.Array
    residuals: jax.Array
    stiffness: jax.Array
    mass: jax.Array


def radial_problem_from_equilibrium(
    geometry: EquilibriumGeometry,
    *,
    harmonic: tuple[int, int] = (2, 1),
    radial_coupling_fraction: float = 0.025,
) -> RadialEigenmodeProblem:
    """Build a radial eigenmode problem from a profile equilibrium.

    ``radial_coupling_fraction`` controls the radial-envelope stiffness as a
    fraction of the median local continuum ``omega^2``. This keeps the analytic
    fixture dimensionally consistent while later VMEC/Boozer operators replace
    the scalar coefficient with metric-derived radial energy terms.
    """
    harmonics = jnp.asarray([[float(harmonic[0]), float(harmonic[1])]])
    continuum_hz = continuum_branch_matrix(geometry, harmonics)[:, 0]
    continuum_omega = 2.0 * jnp.pi * continuum_hz
    positive_omega2 = jnp.maximum(continuum_omega**2, 1.0)
    coupling = float(radial_coupling_fraction) * jnp.median(positive_omega2)
    return RadialEigenmodeProblem(
        s=geometry.s,
        continuum_omega=continuum_omega,
        radial_coupling=jnp.full_like(geometry.s, coupling),
        mass_weight=jnp.ones_like(geometry.s),
        metadata=(
            ("source", "equilibrium_profile"),
            ("harmonic_m", int(harmonic[0])),
            ("harmonic_n", int(harmonic[1])),
            ("radial_coupling_fraction", float(radial_coupling_fraction)),
        ),
    )


def constant_radial_problem(
    *,
    radial_points: int = 64,
    continuum_frequency_hz: float = 125_000.0,
    radial_coupling_omega2: float = 1.0e10,
) -> RadialEigenmodeProblem:
    """Return the constant-coefficient validation fixture."""
    s = jnp.linspace(0.0, 1.0, int(radial_points))
    omega = 2.0 * jnp.pi * float(continuum_frequency_hz)
    return RadialEigenmodeProblem(
        s=s,
        continuum_omega=jnp.full_like(s, omega),
        radial_coupling=jnp.full_like(s, float(radial_coupling_omega2)),
        mass_weight=jnp.ones_like(s),
        metadata=(("source", "constant_radial_problem"),),
    )


def assemble_radial_matrices(problem: RadialEigenmodeProblem) -> tuple[jax.Array, jax.Array]:
    """Assemble stiffness and mass matrices on interior radial nodes."""
    s = problem.s
    interior_omega = problem.continuum_omega[1:-1]
    interior_mass = problem.mass_weight[1:-1]
    coupling_faces = 0.5 * (problem.radial_coupling[1:] + problem.radial_coupling[:-1])
    left = coupling_faces[:-1]
    right = coupling_faces[1:]
    ds = s[1] - s[0]
    inverse_ds2 = 1.0 / ds**2
    diagonal = interior_mass * interior_omega**2 + (left + right) * inverse_ds2
    off_upper = -right[:-1] * inverse_ds2
    off_lower = -left[1:] * inverse_ds2
    stiffness = jnp.diag(diagonal)
    stiffness = stiffness + jnp.diag(off_upper, k=1) + jnp.diag(off_lower, k=-1)
    mass = jnp.diag(interior_mass)
    return stiffness, mass


def solve_radial_eigenmodes(problem: RadialEigenmodeProblem) -> RadialEigenmodeResult:
    """Solve the radial generalized Hermitian eigenproblem."""
    stiffness, mass = assemble_radial_matrices(problem)
    eigensystem = generalized_hermitian_eigensystem(stiffness, mass)
    omega2 = jnp.real(eigensystem.eigenvalues)
    frequencies_hz = jnp.sqrt(jnp.maximum(omega2, 0.0)) / (2.0 * jnp.pi)
    residuals = generalized_eigen_residual_norms(stiffness, mass, eigensystem)
    return RadialEigenmodeResult(
        omega2=omega2,
        frequencies_hz=frequencies_hz,
        eigenvectors=eigensystem.eigenvectors,
        residuals=residuals,
        stiffness=stiffness,
        mass=mass,
    )


def constant_problem_discrete_omega2(problem: RadialEigenmodeProblem) -> jax.Array:
    """Return the exact discrete spectrum for a constant-coefficient problem."""
    interior_count = problem.s.shape[0] - 2
    mode_indices = jnp.arange(1, interior_count + 1, dtype=problem.s.dtype)
    ds = problem.s[1] - problem.s[0]
    continuum = problem.continuum_omega[0] ** 2
    coupling = problem.radial_coupling[0]
    laplacian = 4.0 * coupling / ds**2 * jnp.sin(mode_indices * jnp.pi / (2.0 * (interior_count + 1))) ** 2
    return continuum + laplacian


def lowest_radial_frequency_hz(
    density_scale: Any,
    *,
    base_problem: RadialEigenmodeProblem,
) -> jax.Array:
    """Return the lowest frequency after scaling continuum and coupling by density."""
    scale = jnp.asarray(density_scale)
    scaled = RadialEigenmodeProblem(
        s=base_problem.s,
        continuum_omega=base_problem.continuum_omega / jnp.sqrt(scale),
        radial_coupling=base_problem.radial_coupling / scale,
        mass_weight=base_problem.mass_weight,
        metadata=base_problem.metadata,
    )
    return solve_radial_eigenmodes(scaled).frequencies_hz[0]


def eigenpair_residual_norm(stiffness: Any, mass: Any, omega2: Any, vector: Any, *, floor: float = 1.0e-30) -> jax.Array:
    """Return a relative residual norm for one radial eigenpair."""
    stiffness_array = jnp.asarray(stiffness)
    mass_array = jnp.asarray(mass)
    vector_array = jnp.asarray(vector)
    residual = stiffness_array @ vector_array - jnp.asarray(omega2) * (mass_array @ vector_array)
    denominator = jnp.maximum(
        jnp.linalg.norm(stiffness_array @ vector_array)
        + jnp.abs(omega2) * jnp.linalg.norm(mass_array @ vector_array),
        float(floor),
    )
    return jnp.linalg.norm(residual) / denominator
