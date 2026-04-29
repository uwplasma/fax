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
import numpy as np
from scipy import linalg, sparse
from scipy.sparse import linalg as sparse_linalg

from .boozer import BoozerGeometry, boozer_3d_continuum_matrices, boozer_geometry_grid
from .continuum import continuum_branch_matrix
from .equilibrium import EquilibriumGeometry
from .metric import (
    generalized_eigen_residual_norms,
    generalized_hermitian_eigensystem,
    generalized_hermitian_eigenvalues,
)
from .units import MU0


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


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RadialBlockEigenmodeProblem:
    """Block radial eigenproblem with coupled angular/Boozer harmonics.

    ``local_stiffness`` and ``local_mass`` contain one Hermitian generalized
    continuum matrix per radial surface. ``radial_stiffness`` is the Hermitian
    face-averaged coefficient used in the finite-difference radial energy
    ``D_s^H C D_s``. The implementation supports Dirichlet radial boundaries
    and zero-gradient Neumann radial boundaries.
    """

    s: jax.Array
    local_stiffness: jax.Array
    local_mass: jax.Array
    radial_stiffness: jax.Array
    harmonics: jax.Array
    boundary_condition: str = "dirichlet"
    metadata: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "s", jnp.asarray(self.s))
        object.__setattr__(self, "local_stiffness", jnp.asarray(self.local_stiffness))
        object.__setattr__(self, "local_mass", jnp.asarray(self.local_mass))
        object.__setattr__(self, "radial_stiffness", jnp.asarray(self.radial_stiffness))
        object.__setattr__(self, "harmonics", jnp.asarray(self.harmonics))
        object.__setattr__(self, "metadata", tuple((str(key), value) for key, value in self.metadata))
        object.__setattr__(self, "boundary_condition", str(self.boundary_condition))
        if self.boundary_condition not in {"dirichlet", "neumann"}:
            raise ValueError("boundary_condition must be 'dirichlet' or 'neumann'")
        if self.s.ndim != 1 or self.s.shape[0] < 4:
            raise ValueError("s must be one-dimensional with at least four radial points")
        if self.harmonics.ndim != 2 or self.harmonics.shape[1] != 2:
            raise ValueError("harmonics must have shape (count, 2)")
        block_shape = (self.s.shape[0], self.harmonics.shape[0], self.harmonics.shape[0])
        for name in ("local_stiffness", "local_mass", "radial_stiffness"):
            if getattr(self, name).shape != block_shape:
                raise ValueError(f"{name} must have shape {block_shape}")

    def tree_flatten(self):
        """Return JAX pytree children and static metadata."""
        children = (self.s, self.local_stiffness, self.local_mass, self.radial_stiffness, self.harmonics)
        return children, (self.boundary_condition, self.metadata)

    @classmethod
    def tree_unflatten(cls, aux: tuple[Any, ...], children: tuple[Any, ...]) -> "RadialBlockEigenmodeProblem":
        """Reconstruct a block radial problem from pytree parts."""
        boundary_condition, metadata = aux
        s, local_stiffness, local_mass, radial_stiffness, harmonics = children
        return cls(
            s=s,
            local_stiffness=local_stiffness,
            local_mass=local_mass,
            radial_stiffness=radial_stiffness,
            harmonics=harmonics,
            boundary_condition=boundary_condition,
            metadata=metadata,
        )


@dataclass(frozen=True)
class RadialBlockEigenmodeResult:
    """Eigenvalues, eigenvectors, mode structures, and residuals for blocks."""

    omega2: jax.Array
    frequencies_hz: jax.Array
    eigenvectors: jax.Array
    mode_structures: jax.Array
    residuals: jax.Array
    stiffness: jax.Array
    mass: jax.Array


@dataclass(frozen=True)
class SparseRadialBlockEigenmodeResult:
    """Sparse/Krylov eigenpairs for block radial problems.

    This result type is intentionally not a JAX-transformable solver result.
    It is a reviewer-facing scale-up/reference path for larger global radial
    matrices. The dense JAX path remains the differentiable validation target.
    """

    omega2: jax.Array
    frequencies_hz: jax.Array
    eigenvectors: jax.Array
    mode_structures: jax.Array
    residuals: jax.Array
    stiffness: Any
    mass: Any


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


def constant_radial_block_problem(
    *,
    radial_points: int = 64,
    continuum_frequency_hz: float | tuple[float, ...] = 125_000.0,
    radial_coupling_omega2: float | tuple[float, ...] = 1.0e10,
    harmonic_count: int | None = None,
    mass_weight: float = 1.0,
    boundary_condition: str = "dirichlet",
) -> RadialBlockEigenmodeProblem:
    """Return a block constant-coefficient global validation fixture."""
    frequencies = jnp.atleast_1d(jnp.asarray(continuum_frequency_hz, dtype=float))
    coupling = jnp.atleast_1d(jnp.asarray(radial_coupling_omega2, dtype=float))
    count = int(harmonic_count if harmonic_count is not None else max(frequencies.shape[0], coupling.shape[0]))
    frequencies = _broadcast_mode_profile(frequencies, count, name="continuum_frequency_hz")
    coupling = _broadcast_mode_profile(coupling, count, name="radial_coupling_omega2")
    s = jnp.linspace(0.0, 1.0, int(radial_points))
    harmonics = jnp.column_stack([jnp.arange(1, count + 1, dtype=float), jnp.zeros((count,), dtype=float)])
    complex_dtype = jnp.result_type(frequencies, coupling, jnp.asarray(1j))
    local_mass_block = jnp.eye(count, dtype=complex_dtype) * complex(float(mass_weight))
    local_stiffness_block = jnp.diag((2.0 * jnp.pi * frequencies) ** 2 * float(mass_weight)).astype(complex_dtype)
    radial_stiffness_block = jnp.diag(coupling * float(mass_weight)).astype(complex_dtype)
    return RadialBlockEigenmodeProblem(
        s=s,
        local_stiffness=jnp.broadcast_to(local_stiffness_block[None, :, :], (int(radial_points), count, count)),
        local_mass=jnp.broadcast_to(local_mass_block[None, :, :], (int(radial_points), count, count)),
        radial_stiffness=jnp.broadcast_to(radial_stiffness_block[None, :, :], (int(radial_points), count, count)),
        harmonics=harmonics,
        boundary_condition=boundary_condition,
        metadata=(("source", "constant_radial_block_problem"),),
    )


def boozer_radial_stiffness_matrices(
    geometry: BoozerGeometry,
    harmonics: Any,
    *,
    theta_points: int = 32,
    zeta_points: int = 32,
) -> jax.Array:
    """Return Boozer metric-derived radial derivative energy blocks.

    The radial envelope term uses the ideal-MHD magnetic energy weight

    ``C_ij = < |J| |grad s|^2 |B|^2 phi_i^* phi_j / mu0 >``.

    This is the metric-derived coefficient used in ``D_s^H C_s D_s`` for the
    current global shear-Alfven envelope operator. It is Hermitian by
    construction and differentiable with respect to the Boozer geometry arrays.
    """
    harmonic_array = jnp.asarray(harmonics)
    if harmonic_array.ndim != 2 or harmonic_array.shape[1] != 2:
        raise ValueError("harmonics must have shape (count, 2) with columns (m, n)")
    grid = boozer_geometry_grid(geometry, theta_points=theta_points, zeta_points=zeta_points)
    theta = grid.theta
    zeta = grid.zeta
    m = harmonic_array[:, 0]
    n = harmonic_array[:, 1]
    delta_m = m[None, :] - m[:, None]
    delta_n = n[None, :] - n[:, None]
    pair_phase = delta_m[:, :, None, None] * theta[None, None, :, :] - delta_n[:, :, None, None] * zeta[None, None, :, :]
    phase_factor = jnp.exp(1j * pair_phase)
    weight = grid.integration_jacobian * grid.grad_s_squared * grid.magnetic_field**2 / MU0
    radial = jnp.mean(weight[:, None, None, :, :] * phase_factor[None, :, :, :, :], axis=(-2, -1))
    return 0.5 * (radial + jnp.conjugate(jnp.swapaxes(radial, -1, -2)))


def radial_block_problem_from_boozer_geometry(
    geometry: BoozerGeometry,
    harmonics: Any,
    *,
    mass_density: Any,
    radial_coupling_fraction: float = 0.01,
    radial_coupling_omega2: Any | None = None,
    radial_stiffness_method: str = "metric",
    radial_stiffness_scale: Any = 1.0,
    boundary_condition: str = "dirichlet",
    theta_points: int = 32,
    zeta_points: int = 32,
) -> RadialBlockEigenmodeProblem:
    """Build a global radial block problem from real 3-D Boozer geometry.

    The local block matrices are the real-geometry Boozer surface matrices. The
    default radial term is the Boozer metric-derived magnetic energy
    ``<|J| |grad s|^2 |B|^2 phi_i^* phi_j / mu0>``. Supplying
    ``radial_coupling_omega2`` or setting ``radial_stiffness_method="mass"``
    uses the older local-mass scaled envelope coefficient, useful for analytic
    sensitivity tests. This closes the global radial operator path for the
    current shear-Alfven/Boozer model; cross-code ideal-MHD validation remains
    a separate acceptance layer.
    """
    harmonic_array = jnp.asarray(harmonics, dtype=float)
    matrices = boozer_3d_continuum_matrices(
        geometry,
        harmonic_array,
        mass_density=mass_density,
        theta_points=theta_points,
        zeta_points=zeta_points,
    )
    method = str(radial_stiffness_method)
    if radial_coupling_omega2 is not None:
        radial_stiffness = _scale_block_profile(matrices.mass, radial_coupling_omega2, name="radial_coupling_omega2")
        method = "mass"
    elif method == "metric":
        radial_stiffness = jnp.asarray(radial_stiffness_scale) * boozer_radial_stiffness_matrices(
            geometry,
            harmonic_array,
            theta_points=theta_points,
            zeta_points=zeta_points,
        )
    elif method == "mass":
        local_eigenvalues = generalized_hermitian_eigenvalues(matrices.stiffness, matrices.mass)
        coupling = float(radial_coupling_fraction) * jnp.median(jnp.maximum(local_eigenvalues, 1.0))
        radial_stiffness = _scale_block_profile(matrices.mass, coupling, name="radial_coupling_omega2")
    else:
        raise ValueError("radial_stiffness_method must be 'metric' or 'mass'")
    return RadialBlockEigenmodeProblem(
        s=geometry.s,
        local_stiffness=matrices.stiffness,
        local_mass=matrices.mass,
        radial_stiffness=radial_stiffness,
        harmonics=harmonic_array,
        boundary_condition=boundary_condition,
        metadata=(
            ("source", "boozmn_global_3d"),
            ("radial_coupling_fraction", float(radial_coupling_fraction)),
            ("radial_stiffness_method", method),
            ("radial_stiffness_scale", "dynamic" if hasattr(radial_stiffness_scale, "shape") else float(radial_stiffness_scale)),
            ("theta_points", int(theta_points)),
            ("zeta_points", int(zeta_points)),
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


def assemble_radial_block_matrices(problem: RadialBlockEigenmodeProblem) -> tuple[jax.Array, jax.Array]:
    """Assemble block radial stiffness and mass matrices on interior nodes."""
    surface_count = int(problem.s.shape[0])
    harmonic_count = int(problem.harmonics.shape[0])
    surface_indices = range(1, surface_count - 1) if problem.boundary_condition == "dirichlet" else range(surface_count)
    active_surfaces = tuple(surface_indices)
    active_count = len(active_surfaces)
    matrix_size = active_count * harmonic_count
    spacing = problem.s[1:] - problem.s[:-1]
    dtype = jnp.result_type(problem.local_stiffness, problem.local_mass, problem.radial_stiffness, jnp.asarray(1j) * problem.s[0])
    stiffness = jnp.zeros((matrix_size, matrix_size), dtype=dtype)
    mass = jnp.zeros((matrix_size, matrix_size), dtype=dtype)
    inverse_face_ds2 = 1.0 / spacing**2
    face_stiffness = 0.5 * (problem.radial_stiffness[1:] + problem.radial_stiffness[:-1])
    for local_index, surface_index in enumerate(active_surfaces):
        row = slice(local_index * harmonic_count, (local_index + 1) * harmonic_count)
        left = (
            face_stiffness[surface_index - 1] * inverse_face_ds2[surface_index - 1]
            if surface_index > 0
            else jnp.zeros((harmonic_count, harmonic_count), dtype=dtype)
        )
        right = (
            face_stiffness[surface_index] * inverse_face_ds2[surface_index]
            if surface_index < surface_count - 1
            else jnp.zeros((harmonic_count, harmonic_count), dtype=dtype)
        )
        block = problem.local_stiffness[surface_index] + left + right
        stiffness = stiffness.at[row, row].add(block)
        mass = mass.at[row, row].add(problem.local_mass[surface_index])
        if local_index > 0 and surface_index > 0:
            col = slice((local_index - 1) * harmonic_count, local_index * harmonic_count)
            stiffness = stiffness.at[row, col].add(-left)
        if local_index < active_count - 1 and surface_index < surface_count - 1:
            col = slice((local_index + 1) * harmonic_count, (local_index + 2) * harmonic_count)
            stiffness = stiffness.at[row, col].add(-right)
    stiffness = 0.5 * (stiffness + jnp.conjugate(stiffness.T))
    mass = 0.5 * (mass + jnp.conjugate(mass.T))
    return stiffness, mass


def assemble_radial_block_sparse_matrices(problem: RadialBlockEigenmodeProblem) -> tuple[Any, Any]:
    """Assemble sparse CSR block radial matrices.

    The sparse matrix uses the same finite-difference energy as
    :func:`assemble_radial_block_matrices` and is validated against both the
    exact constant-coefficient spectrum and the dense JAX assembly. It is meant
    for larger benchmark/parity runs where forming dense matrices is wasteful.
    """
    surface_count = int(problem.s.shape[0])
    harmonic_count = int(problem.harmonics.shape[0])
    active_surfaces = _active_radial_surfaces(problem)
    active_count = len(active_surfaces)
    matrix_size = active_count * harmonic_count
    spacing = np.diff(np.asarray(problem.s, dtype=float))
    inverse_face_ds2 = 1.0 / spacing**2
    face_stiffness = 0.5 * (
        np.asarray(problem.radial_stiffness[1:], dtype=np.complex128)
        + np.asarray(problem.radial_stiffness[:-1], dtype=np.complex128)
    )
    local_stiffness = np.asarray(problem.local_stiffness, dtype=np.complex128)
    local_mass = np.asarray(problem.local_mass, dtype=np.complex128)
    zero = np.zeros((harmonic_count, harmonic_count), dtype=np.complex128)

    stiffness_rows: list[int] = []
    stiffness_cols: list[int] = []
    stiffness_data: list[complex] = []
    mass_rows: list[int] = []
    mass_cols: list[int] = []
    mass_data: list[complex] = []

    def add_block(rows: list[int], cols: list[int], data: list[complex], row_block: int, col_block: int, block: Any) -> None:
        block_array = np.asarray(block, dtype=np.complex128)
        local_rows, local_cols = np.nonzero(np.abs(block_array) > 0.0)
        row_offset = int(row_block) * harmonic_count
        col_offset = int(col_block) * harmonic_count
        rows.extend((row_offset + local_rows).astype(int).tolist())
        cols.extend((col_offset + local_cols).astype(int).tolist())
        data.extend(block_array[local_rows, local_cols].tolist())

    for local_index, surface_index in enumerate(active_surfaces):
        left = (
            face_stiffness[surface_index - 1] * inverse_face_ds2[surface_index - 1]
            if surface_index > 0
            else zero
        )
        right = (
            face_stiffness[surface_index] * inverse_face_ds2[surface_index]
            if surface_index < surface_count - 1
            else zero
        )
        add_block(
            stiffness_rows,
            stiffness_cols,
            stiffness_data,
            local_index,
            local_index,
            local_stiffness[surface_index] + left + right,
        )
        add_block(mass_rows, mass_cols, mass_data, local_index, local_index, local_mass[surface_index])
        if local_index > 0 and surface_index > 0:
            add_block(stiffness_rows, stiffness_cols, stiffness_data, local_index, local_index - 1, -left)
        if local_index < active_count - 1 and surface_index < surface_count - 1:
            add_block(stiffness_rows, stiffness_cols, stiffness_data, local_index, local_index + 1, -right)

    stiffness = sparse.coo_matrix(
        (stiffness_data, (stiffness_rows, stiffness_cols)),
        shape=(matrix_size, matrix_size),
        dtype=np.complex128,
    ).tocsr()
    mass = sparse.coo_matrix(
        (mass_data, (mass_rows, mass_cols)),
        shape=(matrix_size, matrix_size),
        dtype=np.complex128,
    ).tocsr()
    stiffness = (0.5 * (stiffness + stiffness.getH())).tocsr()
    mass = (0.5 * (mass + mass.getH())).tocsr()
    return stiffness, mass


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


def solve_radial_block_eigenmodes(problem: RadialBlockEigenmodeProblem) -> RadialBlockEigenmodeResult:
    """Solve the block radial generalized Hermitian eigenproblem."""
    stiffness, mass = assemble_radial_block_matrices(problem)
    eigensystem = generalized_hermitian_eigensystem(stiffness, mass)
    omega2 = jnp.real(eigensystem.eigenvalues)
    frequencies_hz = jnp.sqrt(jnp.maximum(omega2, 0.0)) / (2.0 * jnp.pi)
    residuals = generalized_eigen_residual_norms(stiffness, mass, eigensystem)
    return RadialBlockEigenmodeResult(
        omega2=omega2,
        frequencies_hz=frequencies_hz,
        eigenvectors=eigensystem.eigenvectors,
        mode_structures=radial_block_mode_structures(problem, eigensystem.eigenvectors),
        residuals=residuals,
        stiffness=stiffness,
        mass=mass,
    )


def solve_radial_block_eigenmodes_sparse(
    problem: RadialBlockEigenmodeProblem,
    *,
    mode_count: int = 16,
    sigma: float | None = 0.0,
    which: str = "LM",
    tolerance: float = 0.0,
    maxiter: int | None = None,
) -> SparseRadialBlockEigenmodeResult:
    """Solve the block radial generalized problem with sparse Krylov methods.

    The default shift-invert request targets the lowest-frequency modes of the
    positive generalized problem. If ``mode_count`` is too close to the matrix
    dimension, SciPy's dense Hermitian generalized solve is used instead; this
    keeps tiny validation cases deterministic.
    """
    stiffness, mass = assemble_radial_block_sparse_matrices(problem)
    requested = int(mode_count)
    if requested < 1:
        raise ValueError("mode_count must be at least 1")
    matrix_dimension = int(stiffness.shape[0])
    if requested >= matrix_dimension:
        raise ValueError("mode_count must be smaller than the matrix dimension")
    if requested >= matrix_dimension - 1:
        eigenvalues, eigenvectors = linalg.eigh(
            stiffness.toarray(),
            mass.toarray(),
            check_finite=False,
            overwrite_a=True,
            overwrite_b=True,
        )
        eigenvalues = eigenvalues[:requested]
        eigenvectors = eigenvectors[:, :requested]
    else:
        eigenvalues, eigenvectors = sparse_linalg.eigsh(
            stiffness,
            k=requested,
            M=mass,
            sigma=sigma,
            which=which,
            tol=float(tolerance),
            maxiter=maxiter,
        )
    order = np.argsort(np.real(eigenvalues))
    omega2 = np.real(eigenvalues[order])
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.complex128)
    residuals = np.asarray(
        [
            _sparse_eigenpair_residual_norm(stiffness, mass, float(omega2[index]), eigenvectors[:, index])
            for index in range(requested)
        ],
        dtype=float,
    )
    eigenvectors_jax = jnp.asarray(eigenvectors)
    omega2_jax = jnp.asarray(omega2)
    return SparseRadialBlockEigenmodeResult(
        omega2=omega2_jax,
        frequencies_hz=jnp.sqrt(jnp.maximum(omega2_jax, 0.0)) / (2.0 * jnp.pi),
        eigenvectors=eigenvectors_jax,
        mode_structures=radial_block_mode_structures(problem, eigenvectors_jax),
        residuals=jnp.asarray(residuals),
        stiffness=stiffness,
        mass=mass,
    )


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


def radial_block_mode_structures(problem: RadialBlockEigenmodeProblem, eigenvectors: Any) -> jax.Array:
    """Return eigenvectors as ``(mode, surface, harmonic)``."""
    vectors = jnp.asarray(eigenvectors)
    harmonic_count = int(problem.harmonics.shape[0])
    mode_count = int(vectors.shape[1])
    if problem.boundary_condition == "neumann":
        full = jnp.reshape(vectors, (int(problem.s.shape[0]), harmonic_count, mode_count))
        return jnp.moveaxis(full, -1, 0)
    interior = jnp.reshape(vectors, (int(problem.s.shape[0]) - 2, harmonic_count, mode_count))
    boundary = jnp.zeros((1, harmonic_count, mode_count), dtype=vectors.dtype)
    full = jnp.concatenate([boundary, interior, boundary], axis=0)
    return jnp.moveaxis(full, -1, 0)


def constant_problem_discrete_omega2(problem: RadialEigenmodeProblem) -> jax.Array:
    """Return the exact discrete spectrum for a constant-coefficient problem."""
    interior_count = problem.s.shape[0] - 2
    mode_indices = jnp.arange(1, interior_count + 1, dtype=problem.s.dtype)
    ds = problem.s[1] - problem.s[0]
    continuum = problem.continuum_omega[0] ** 2
    coupling = problem.radial_coupling[0]
    laplacian = 4.0 * coupling / ds**2 * jnp.sin(mode_indices * jnp.pi / (2.0 * (interior_count + 1))) ** 2
    return continuum + laplacian


def constant_block_problem_discrete_omega2(problem: RadialBlockEigenmodeProblem) -> jax.Array:
    """Return the exact spectrum for a diagonal constant block problem."""
    if problem.boundary_condition == "neumann":
        node_count = problem.s.shape[0]
        mode_indices = jnp.arange(0, node_count, dtype=problem.s.dtype)
        denominator = 2.0 * node_count
    else:
        node_count = problem.s.shape[0] - 2
        mode_indices = jnp.arange(1, node_count + 1, dtype=problem.s.dtype)
        denominator = 2.0 * (node_count + 1)
    ds = problem.s[1] - problem.s[0]
    mass_diagonal = jnp.real(jnp.diagonal(problem.local_mass[0]))
    local_omega2 = jnp.real(jnp.diagonal(problem.local_stiffness[0])) / mass_diagonal
    coupling_omega2 = jnp.real(jnp.diagonal(problem.radial_stiffness[0])) / mass_diagonal
    radial_laplacian = (
        4.0
        * coupling_omega2[:, None]
        / ds**2
        * jnp.sin(mode_indices[None, :] * jnp.pi / denominator) ** 2
    )
    return jnp.sort(jnp.ravel(local_omega2[:, None] + radial_laplacian))


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


def lowest_radial_block_frequency_hz(
    density_scale: Any,
    *,
    base_problem: RadialBlockEigenmodeProblem,
) -> jax.Array:
    """Return the lowest block frequency after scaling the generalized mass."""
    scale = jnp.asarray(density_scale)
    scaled = RadialBlockEigenmodeProblem(
        s=base_problem.s,
        local_stiffness=base_problem.local_stiffness,
        local_mass=base_problem.local_mass * scale,
        radial_stiffness=base_problem.radial_stiffness,
        harmonics=base_problem.harmonics,
        boundary_condition=base_problem.boundary_condition,
        metadata=base_problem.metadata + (("density_scale", "autodiff"),),
    )
    return solve_radial_block_eigenmodes(scaled).frequencies_hz[0]


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


def _sparse_eigenpair_residual_norm(stiffness: Any, mass: Any, omega2: float, vector: Any, *, floor: float = 1.0e-30) -> float:
    vector_array = np.asarray(vector)
    stiffness_vector = stiffness @ vector_array
    mass_vector = mass @ vector_array
    residual = stiffness_vector - float(omega2) * mass_vector
    denominator = max(float(np.linalg.norm(stiffness_vector) + abs(float(omega2)) * np.linalg.norm(mass_vector)), float(floor))
    return float(np.linalg.norm(residual) / denominator)


def _active_radial_surfaces(problem: RadialBlockEigenmodeProblem) -> tuple[int, ...]:
    surface_count = int(problem.s.shape[0])
    if problem.boundary_condition == "dirichlet":
        return tuple(range(1, surface_count - 1))
    return tuple(range(surface_count))


def _broadcast_mode_profile(value: jax.Array, count: int, *, name: str) -> jax.Array:
    if value.shape == (1,):
        return jnp.full((int(count),), value[0])
    if value.shape != (int(count),):
        raise ValueError(f"{name} must be scalar-like or have shape ({count},)")
    return value


def _scale_block_profile(blocks: jax.Array, scale: Any, *, name: str) -> jax.Array:
    scale_array = jnp.asarray(scale)
    if scale_array.ndim == 0:
        return blocks * scale_array
    if scale_array.shape == (int(blocks.shape[0]),):
        return blocks * scale_array[:, None, None]
    if scale_array.shape == blocks.shape:
        return scale_array
    raise ValueError(f"{name} must be scalar, a radial profile, or full block array with shape {blocks.shape}")
