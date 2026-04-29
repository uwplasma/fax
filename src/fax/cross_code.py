"""Readers and comparison metrics for external Alfvén codes.

The functions here intentionally read plain-text artifacts produced by
STELLGAP/AE3D style workflows instead of depending on those Fortran codes at
runtime. This makes cross-code validation reproducible in local tests while
keeping the core FAX package pure Python/JAX.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy import linalg, sparse
from scipy.optimize import linear_sum_assignment

from .boozer import boozer_3d_continuum_frequencies_hz, read_boozmn_geometry


@dataclass(frozen=True)
class StellgapContinuumData:
    """STELLGAP continuum points from ``alfven_post`` or ``alfven_spec``."""

    s: jax.Array
    frequency_khz: jax.Array
    m: jax.Array
    n: jax.Array


@dataclass(frozen=True)
class StellgapSpectrumData:
    """Raw STELLGAP ``alfven_spec`` generalized-eigenvalue rows."""

    s: jax.Array
    alpha_real: jax.Array
    alpha_imag: jax.Array
    beta: jax.Array
    m: jax.Array
    n: jax.Array


@dataclass(frozen=True)
class Ae3dEigenvalueData:
    """AE3D eigenvalue table from ``egn_values.dat``."""

    frequency_khz: jax.Array
    residual_real: jax.Array
    residual_imag: jax.Array


@dataclass(frozen=True)
class Ae3dOmega2Data:
    """AE3D ``omega2.dat`` generalized-eigenvalue rows."""

    index: jax.Array
    alpha_real: jax.Array
    alpha_imag: jax.Array
    beta: jax.Array
    eigenvalue: jax.Array


@dataclass(frozen=True)
class Ae3dResidualSummary:
    """Residual and frequency range summary for an AE3D eigenvalue table."""

    count: int
    min_frequency_khz: float
    max_frequency_khz: float
    p95_residual: float
    max_residual: float


@dataclass(frozen=True)
class Ae3dMatrixSummary:
    """Sparse matrix symmetry and size summary for an AE3D matrix artifact."""

    dimension: int
    nonzero_count: int
    relative_asymmetry: float
    diagonal_min: float
    diagonal_max: float


@dataclass(frozen=True)
class Ae3dEigenmodeData:
    """AE3D eigenmode table from ``egn_mode_asci.dat``."""

    eigenvalue_omega2: jax.Array
    rho: jax.Array
    harmonics: jax.Array
    modes: jax.Array


@dataclass(frozen=True)
class Ae3dMatrixEigenmodeParity:
    """Residual and mode-structure summary for public AE3D matrix artifacts."""

    count: int
    matrix_dimension: int
    surface_count: int
    harmonic_count: int
    max_residual: float
    p95_residual: float
    max_rayleigh_relative_error: float
    p95_rayleigh_relative_error: float
    max_frequency_relative_error: float
    dominant_mode_surface: int
    dominant_mode_harmonic_m: int
    dominant_mode_harmonic_n: int


@dataclass(frozen=True)
class Ae3dDenseSpectrumParity:
    """Dense-solver spectrum parity for public AE3D generalized matrices."""

    count: int
    matrix_dimension: int
    real_eigenvalue_count: int
    reference_eigenvalue_omega2: jax.Array
    computed_eigenvalue_omega2: jax.Array
    eigenvalue_relative_error: jax.Array
    frequency_relative_error: jax.Array
    max_eigenvalue_relative_error: float
    p95_eigenvalue_relative_error: float
    max_frequency_relative_error: float
    p95_frequency_relative_error: float


@dataclass(frozen=True)
class CrossCodeComparison:
    """Summary metric for an external-code comparison."""

    count: int
    mean_relative_error: float
    p95_relative_error: float
    max_relative_error: float


@dataclass(frozen=True)
class CrossCodePostprocessComparison:
    """Summary metric for one external-code artifact-to-artifact comparison."""

    count: int
    mean_relative_error: float
    p95_relative_error: float
    max_relative_error: float


def read_stellgap_alfven_post(path: str | Path) -> StellgapContinuumData:
    """Read STELLGAP ``alfven_post`` data.

    ``alfven_post`` starts with a one-line point count, followed by columns
    ``s, frequency[kHz], dominant_m, dominant_n``.
    """
    raw = np.loadtxt(Path(path), skiprows=1)
    return StellgapContinuumData(
        s=jnp.asarray(raw[:, 0]),
        frequency_khz=jnp.asarray(raw[:, 1]),
        m=jnp.asarray(raw[:, 2], dtype=int),
        n=jnp.asarray(raw[:, 3], dtype=int),
    )


def read_stellgap_alfven_spec(path: str | Path) -> StellgapSpectrumData:
    """Read raw STELLGAP ``alfven_spec`` rows.

    For the general-matrix STELLGAP path, columns are
    ``s, alpha_real, alpha_imag, beta, dominant_m, dominant_n``. The
    post-processing step computes ``sqrt((alpha_real+i alpha_imag)/beta)`` and
    keeps real frequencies on surfaces inside ``s < 0.99``.
    """
    raw = np.loadtxt(Path(path))
    return StellgapSpectrumData(
        s=jnp.asarray(raw[:, 0]),
        alpha_real=jnp.asarray(raw[:, 1]),
        alpha_imag=jnp.asarray(raw[:, 2]),
        beta=jnp.asarray(raw[:, 3]),
        m=jnp.asarray(raw[:, 4], dtype=int),
        n=jnp.asarray(raw[:, 5], dtype=int),
    )


def stellgap_spec_postprocessed_continuum(path: str | Path) -> StellgapContinuumData:
    """Reproduce STELLGAP ``post_process.f`` for the general-matrix path."""
    raw = np.loadtxt(Path(path))
    rows: list[tuple[float, float, int, int]] = []
    for s_value, alpha_real, alpha_imag, beta, m_value, n_value in raw:
        if beta <= 0.0:
            continue
        omega = np.sqrt(complex(alpha_real / beta, alpha_imag / beta))
        if abs(omega.imag) > 0.0:
            continue
        if s_value >= 0.99:
            continue
        rows.append((float(s_value), float(omega.real), int(m_value), int(n_value)))
    if not rows:
        raise ValueError("no real STELLGAP postprocessed continuum rows found")
    processed = np.asarray(rows)
    return StellgapContinuumData(
        s=jnp.asarray(processed[:, 0]),
        frequency_khz=jnp.asarray(processed[:, 1]),
        m=jnp.asarray(processed[:, 2], dtype=int),
        n=jnp.asarray(processed[:, 3], dtype=int),
    )


def stellgap_postprocess_error(*, alfven_spec_path: str | Path, alfven_post_path: str | Path) -> CrossCodePostprocessComparison:
    """Compare FAX's STELLGAP post-processing reproduction with ``alfven_post``."""
    predicted = stellgap_spec_postprocessed_continuum(alfven_spec_path)
    reference = read_stellgap_alfven_post(alfven_post_path)
    predicted_frequency = np.asarray(predicted.frequency_khz)
    reference_frequency = np.asarray(reference.frequency_khz)
    count = min(predicted_frequency.shape[0], reference_frequency.shape[0])
    if count == 0:
        raise ValueError("empty STELLGAP postprocess comparison")
    errors = np.abs(predicted_frequency[:count] - reference_frequency[:count]) / np.maximum(
        np.abs(reference_frequency[:count]),
        1.0e-30,
    )
    return CrossCodePostprocessComparison(
        count=int(count),
        mean_relative_error=float(np.mean(errors)),
        p95_relative_error=float(np.percentile(errors, 95.0)),
        max_relative_error=float(np.max(errors)),
    )


def read_ae3d_eigenvalues(path: str | Path) -> Ae3dEigenvalueData:
    """Read AE3D ``egn_values.dat`` data."""
    raw = np.loadtxt(Path(path))
    return Ae3dEigenvalueData(
        frequency_khz=jnp.asarray(raw[:, 0]),
        residual_real=jnp.asarray(raw[:, 1]),
        residual_imag=jnp.asarray(raw[:, 2]),
    )


def read_ae3d_omega2(path: str | Path) -> Ae3dOmega2Data:
    """Read AE3D ``omega2.dat`` rows.

    Columns are ``index, alpha_real, alpha_imag, beta, eigenvalue``. For
    positive eigenvalues, AE3D's ``egn_values.dat`` reports ``sqrt(eigenvalue)``;
    for negative entries it reports the negative value directly.
    """
    raw = np.loadtxt(Path(path))
    return Ae3dOmega2Data(
        index=jnp.asarray(raw[:, 0], dtype=int),
        alpha_real=jnp.asarray(raw[:, 1]),
        alpha_imag=jnp.asarray(raw[:, 2]),
        beta=jnp.asarray(raw[:, 3]),
        eigenvalue=jnp.asarray(raw[:, 4]),
    )


def ae3d_eigenvalue_residual_summary(path: str | Path) -> Ae3dResidualSummary:
    """Return residual summary metrics for public AE3D ``egn_values.dat`` data."""
    data = read_ae3d_eigenvalues(path)
    residual = jnp.sqrt(data.residual_real**2 + data.residual_imag**2)
    residual_np = np.asarray(residual)
    frequency_np = np.asarray(data.frequency_khz)
    return Ae3dResidualSummary(
        count=int(frequency_np.shape[0]),
        min_frequency_khz=float(np.min(frequency_np)),
        max_frequency_khz=float(np.max(frequency_np)),
        p95_residual=float(np.percentile(residual_np, 95.0)),
        max_residual=float(np.max(residual_np)),
    )


def ae3d_omega2_eigenvalue_error(*, omega2_path: str | Path, egn_values_path: str | Path) -> CrossCodePostprocessComparison:
    """Compare AE3D ``omega2.dat`` eigenvalues with ``egn_values.dat``."""
    omega2 = read_ae3d_omega2(omega2_path)
    eigenvalues = read_ae3d_eigenvalues(egn_values_path)
    omega2_eigenvalue = np.asarray(omega2.eigenvalue)
    predicted = np.where(omega2_eigenvalue > 0.0, np.sqrt(np.maximum(omega2_eigenvalue, 0.0)), omega2_eigenvalue)
    reference = np.asarray(eigenvalues.frequency_khz)
    count = min(predicted.shape[0], reference.shape[0])
    errors = np.abs(predicted[:count] - reference[:count]) / np.maximum(np.abs(reference[:count]), 1.0e-30)
    return CrossCodePostprocessComparison(
        count=int(count),
        mean_relative_error=float(np.mean(errors)),
        p95_relative_error=float(np.percentile(errors, 95.0)),
        max_relative_error=float(np.max(errors)),
    )


def ae3d_matrix_symmetry_summary(path: str | Path) -> Ae3dMatrixSummary:
    """Return sparse symmetry metrics for an AE3D matrix artifact."""
    matrix = read_ae3d_sparse_matrix(path)
    asymmetry = matrix - matrix.T
    matrix_norm = float(np.sqrt(matrix.multiply(matrix).sum()))
    asymmetry_norm = float(np.sqrt(asymmetry.multiply(asymmetry).sum()))
    diagonal = matrix.diagonal()
    return Ae3dMatrixSummary(
        dimension=int(matrix.shape[0]),
        nonzero_count=int(matrix.nnz),
        relative_asymmetry=asymmetry_norm / max(matrix_norm, 1.0e-30),
        diagonal_min=float(np.min(diagonal)),
        diagonal_max=float(np.max(diagonal)),
    )


def read_ae3d_sparse_matrix(path: str | Path) -> sparse.csr_matrix:
    """Read a public AE3D sparse matrix artifact as a SciPy CSR matrix."""
    raw = np.loadtxt(Path(path))
    rows = raw[:, 0].astype(int) - 1
    cols = raw[:, 1].astype(int) - 1
    values = raw[:, 2]
    dimension = int(max(np.max(rows), np.max(cols)) + 1)
    return sparse.coo_matrix((values, (rows, cols)), shape=(dimension, dimension)).tocsr()


def read_ae3d_eigenmodes(path: str | Path, *, max_modes: int | None = None) -> Ae3dEigenmodeData:
    """Read AE3D ``egn_mode_asci.dat`` eigenvalues and mode structures.

    The public ASCII file starts with ``nmat``, ``mn_col``, and ``ns``, followed
    by ``mn_col`` pairs of ``(m, n)``, ``nmat`` eigenvalues, ``ns`` radial
    coordinates, and then mode vectors in ``(mode, surface, harmonic)`` order.
    ``max_modes`` can be used for quick residual gates without loading all
    public eigenvectors.
    """
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        mode_count_total = int(stream.readline())
        harmonic_count = int(stream.readline())
        surface_count = int(stream.readline())
        harmonics = np.asarray(
            [(int(stream.readline()), int(stream.readline())) for _ in range(harmonic_count)],
            dtype=float,
        )
        eigenvalues = np.asarray([float(stream.readline()) for _ in range(mode_count_total)], dtype=float)
        rho = np.asarray([float(stream.readline()) for _ in range(surface_count)], dtype=float)
        selected_modes = mode_count_total if max_modes is None else min(int(max_modes), mode_count_total)
        vector_size = surface_count * harmonic_count
        raw_modes = np.fromiter(
            (float(stream.readline()) for _ in range(selected_modes * vector_size)),
            dtype=float,
            count=selected_modes * vector_size,
        )
    if raw_modes.size != selected_modes * vector_size:
        raise ValueError("AE3D eigenmode file ended before the requested modes were read")
    modes = raw_modes.reshape((selected_modes, surface_count, harmonic_count))
    return Ae3dEigenmodeData(
        eigenvalue_omega2=jnp.asarray(eigenvalues[:selected_modes]),
        rho=jnp.asarray(rho),
        harmonics=jnp.asarray(harmonics),
        modes=jnp.asarray(modes),
    )


def ae3d_matrix_eigenmode_parity(
    *,
    a_matrix_path: str | Path,
    b_matrix_path: str | Path,
    egn_mode_path: str | Path,
    max_modes: int = 16,
) -> Ae3dMatrixEigenmodeParity:
    """Check public AE3D eigenmodes against public AE3D sparse matrices.

    This is an external matrix/eigenmode parity gate: it verifies that the
    eigenvectors stored by AE3D satisfy the generalized problem ``A x = lambda B
    x`` and that the Rayleigh quotients recover the reported eigenvalues. It is
    a prerequisite for FAX/AE3D physics parity, not by itself a claim that FAX's
    reduced radial operator matches AE3D's full operator.
    """
    a_matrix = read_ae3d_sparse_matrix(a_matrix_path)
    b_matrix = read_ae3d_sparse_matrix(b_matrix_path)
    if a_matrix.shape != b_matrix.shape:
        raise ValueError("AE3D A and B matrices must have matching shapes")
    data = read_ae3d_eigenmodes(egn_mode_path, max_modes=max_modes)
    mode_array = np.asarray(data.modes)
    eigenvalues = np.asarray(data.eigenvalue_omega2)
    vector_size = int(mode_array.shape[1] * mode_array.shape[2])
    if a_matrix.shape[0] != vector_size:
        raise ValueError(
            f"AE3D matrix dimension {a_matrix.shape[0]} does not match eigenmode vector size {vector_size}"
        )

    residuals: list[float] = []
    quotient_errors: list[float] = []
    frequency_errors: list[float] = []
    dominant_flat_index = int(np.argmax(np.abs(mode_array[0]))) if mode_array.size else 0
    dominant_surface, dominant_harmonic = np.unravel_index(dominant_flat_index, mode_array[0].shape)
    for index in range(mode_array.shape[0]):
        vector = mode_array[index].reshape(vector_size)
        eigenvalue = float(eigenvalues[index])
        av = a_matrix @ vector
        bv = b_matrix @ vector
        denominator = max(float(np.linalg.norm(av) + abs(eigenvalue) * np.linalg.norm(bv)), 1.0e-30)
        residuals.append(float(np.linalg.norm(av - eigenvalue * bv) / denominator))
        rayleigh = float(np.vdot(vector, av) / np.vdot(vector, bv))
        quotient_errors.append(abs(rayleigh - eigenvalue) / max(abs(eigenvalue), 1.0e-30))
        reference_frequency = np.sqrt(max(eigenvalue, 0.0))
        rayleigh_frequency = np.sqrt(max(rayleigh, 0.0))
        frequency_errors.append(abs(rayleigh_frequency - reference_frequency) / max(abs(reference_frequency), 1.0e-30))

    residual_array = np.asarray(residuals, dtype=float)
    quotient_error_array = np.asarray(quotient_errors, dtype=float)
    frequency_error_array = np.asarray(frequency_errors, dtype=float)
    harmonics = np.asarray(data.harmonics, dtype=int)
    return Ae3dMatrixEigenmodeParity(
        count=int(mode_array.shape[0]),
        matrix_dimension=int(a_matrix.shape[0]),
        surface_count=int(mode_array.shape[1]),
        harmonic_count=int(mode_array.shape[2]),
        max_residual=float(np.max(residual_array)),
        p95_residual=float(np.percentile(residual_array, 95.0)),
        max_rayleigh_relative_error=float(np.max(quotient_error_array)),
        p95_rayleigh_relative_error=float(np.percentile(quotient_error_array, 95.0)),
        max_frequency_relative_error=float(np.max(frequency_error_array)),
        dominant_mode_surface=int(dominant_surface),
        dominant_mode_harmonic_m=int(harmonics[dominant_harmonic, 0]),
        dominant_mode_harmonic_n=int(harmonics[dominant_harmonic, 1]),
    )


def ae3d_dense_spectrum_parity(
    *,
    a_matrix_path: str | Path,
    b_matrix_path: str | Path,
    egn_mode_path: str | Path,
    max_modes: int = 16,
    imaginary_tolerance: float = 1.0e-7,
) -> Ae3dDenseSpectrumParity:
    """Solve public AE3D matrices and compare against AE3D eigenvalues.

    This is an executable external-code parity check for the public AE3D
    generalized eigenproblem. It forms dense copies of the public sparse
    matrices, solves ``A x = lambda B x`` with SciPy/LAPACK, then matches the
    real computed eigenvalues nearest to the first ``max_modes`` AE3D
    eigenvalues stored in ``egn_mode_asci.dat``. It is intentionally separate
    from :func:`ae3d_matrix_eigenmode_parity`, which checks the published
    eigenvectors directly.
    """
    a_matrix = read_ae3d_sparse_matrix(a_matrix_path)
    b_matrix = read_ae3d_sparse_matrix(b_matrix_path)
    if a_matrix.shape != b_matrix.shape:
        raise ValueError("AE3D A and B matrices must have matching shapes")

    data = read_ae3d_eigenmodes(egn_mode_path, max_modes=max_modes)
    reference = np.asarray(data.eigenvalue_omega2, dtype=float)
    raw_eigenvalues = linalg.eig(
        a_matrix.toarray(),
        b_matrix.toarray(),
        left=False,
        right=False,
        check_finite=False,
        overwrite_a=True,
        overwrite_b=True,
    )
    finite = np.isfinite(raw_eigenvalues)
    real_scale = np.maximum(np.abs(raw_eigenvalues.real), 1.0)
    real_mask = finite & (np.abs(raw_eigenvalues.imag) <= float(imaginary_tolerance) * real_scale)
    real_eigenvalues = raw_eigenvalues[real_mask].real
    if real_eigenvalues.shape[0] < reference.shape[0]:
        raise ValueError("not enough real AE3D generalized eigenvalues for spectrum parity")

    cost = np.abs(real_eigenvalues[:, None] - reference[None, :])
    row_indices, column_indices = linear_sum_assignment(cost)
    computed = np.empty_like(reference)
    computed[column_indices] = real_eigenvalues[row_indices]
    eigenvalue_error = np.abs(computed - reference) / np.maximum(np.abs(reference), 1.0e-30)
    reference_frequency = np.sqrt(np.maximum(reference, 0.0))
    computed_frequency = np.sqrt(np.maximum(computed, 0.0))
    frequency_error = np.abs(computed_frequency - reference_frequency) / np.maximum(reference_frequency, 1.0e-30)

    return Ae3dDenseSpectrumParity(
        count=int(reference.shape[0]),
        matrix_dimension=int(a_matrix.shape[0]),
        real_eigenvalue_count=int(real_eigenvalues.shape[0]),
        reference_eigenvalue_omega2=jnp.asarray(reference),
        computed_eigenvalue_omega2=jnp.asarray(computed),
        eigenvalue_relative_error=jnp.asarray(eigenvalue_error),
        frequency_relative_error=jnp.asarray(frequency_error),
        max_eigenvalue_relative_error=float(np.max(eigenvalue_error)),
        p95_eigenvalue_relative_error=float(np.percentile(eigenvalue_error, 95.0)),
        max_frequency_relative_error=float(np.max(frequency_error)),
        p95_frequency_relative_error=float(np.percentile(frequency_error, 95.0)),
    )


def stellgap_ctok_local_continuum_error(
    *,
    boozmn_path: str | Path,
    alfven_post_path: str | Path,
    mass_density: Any,
    toroidal_mode: int = 2,
    poloidal_modes: tuple[int, ...] = tuple(range(13)),
    theta_points: int = 32,
    zeta_points: int = 32,
    nearest_surface_tolerance: float = 2.0e-3,
) -> CrossCodeComparison:
    """Compare FAX local Boozer branches with the STELLGAP CTOK test output.

    The comparison sorts frequencies on each matched surface before computing a
    relative L2 error. This avoids relying on branch labels near crossings while
    still checking the continuum spectrum produced by the same Boozer geometry.
    """
    geometry = read_boozmn_geometry(boozmn_path)
    reference = read_stellgap_alfven_post(alfven_post_path)
    harmonics = jnp.asarray([[float(m), float(toroidal_mode)] for m in poloidal_modes])
    fax_khz = (
        boozer_3d_continuum_frequencies_hz(
            geometry,
            harmonics,
            mass_density=mass_density,
            theta_points=theta_points,
            zeta_points=zeta_points,
        )
        / 1.0e3
    )
    reference_s = np.asarray(reference.s)
    reference_frequency = np.asarray(reference.frequency_khz)
    unique_s = np.unique(reference_s)
    errors: list[float] = []
    for surface_index, surface in enumerate(np.asarray(geometry.s)):
        nearest_index = int(np.argmin(np.abs(unique_s - surface)))
        if abs(float(unique_s[nearest_index] - surface)) > float(nearest_surface_tolerance):
            continue
        surface_mask = reference_s == unique_s[nearest_index]
        reference_branch = np.sort(reference_frequency[surface_mask])
        if reference_branch.shape[0] != len(poloidal_modes):
            continue
        fax_branch = np.sort(np.asarray(fax_khz[surface_index]))
        denominator = max(float(np.linalg.norm(reference_branch)), 1.0e-30)
        errors.append(float(np.linalg.norm(fax_branch - reference_branch) / denominator))
    if not errors:
        raise ValueError("no matching STELLGAP/FAX surfaces found")
    error_array = np.asarray(errors)
    return CrossCodeComparison(
        count=int(error_array.shape[0]),
        mean_relative_error=float(np.mean(error_array)),
        p95_relative_error=float(np.percentile(error_array, 95.0)),
        max_relative_error=float(np.max(error_array)),
    )
