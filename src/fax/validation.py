"""Validation metrics and registered physics gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from .continuum import continuum_branch_matrix, local_continuum_frequency_hz
from .equilibrium import analytic_circular_tokamak, analytic_cylinder, from_vmec_wout
from .metric import (
    MetricGeometry,
    diagonal_continuum_frequencies_hz,
    generalized_eigen_residual_norms,
    generalized_hermitian_eigensystem,
    metric_continuum_frequencies_hz,
    metric_continuum_matrices,
    metric_from_equilibrium,
    metric_from_vmec_wout,
)
from .modes import stellarator_mode_family
from .reference_cases import landreman_paul_qa_lowres


def relative_l2_error(actual: Any, expected: Any, *, floor: float = 1.0e-30) -> jnp.ndarray:
    """Return ``||actual - expected||_2 / max(||expected||_2, floor)``."""
    actual_array = jnp.asarray(actual)
    expected_array = jnp.asarray(expected)
    denominator = jnp.maximum(jnp.linalg.norm(expected_array), floor)
    return jnp.linalg.norm(actual_array - expected_array) / denominator


def max_relative_error(actual: Any, expected: Any, *, floor: float = 1.0e-30) -> jnp.ndarray:
    """Return the maximum elementwise relative error."""
    actual_array = jnp.asarray(actual)
    expected_array = jnp.asarray(expected)
    return jnp.max(jnp.abs(actual_array - expected_array) / jnp.maximum(jnp.abs(expected_array), floor))


def hermitian_relative_error(matrix: Any, *, floor: float = 1.0e-30) -> jnp.ndarray:
    """Return ``||A - A.H||_F / max(||A||_F, floor)``."""
    array = jnp.asarray(matrix)
    denominator = jnp.maximum(jnp.linalg.norm(array), floor)
    return jnp.linalg.norm(array - jnp.conjugate(array.T)) / denominator


def finite_difference_derivative(func: Callable[[float], float], x: float, *, step: float = 1.0e-5) -> float:
    """Return a centered finite-difference derivative of a scalar function."""
    h = float(step)
    return (float(func(float(x) + h)) - float(func(float(x) - h))) / (2.0 * h)


def observed_order(error_coarse: float, error_fine: float, refinement: float) -> float:
    """Return the observed convergence order from two errors."""
    return float(jnp.log(float(error_coarse) / float(error_fine)) / jnp.log(float(refinement)))


def interpolate_branch_matrix(actual: Any, source_s: Any, target_s: Any) -> jnp.ndarray:
    """Interpolate branch columns from ``source_s`` to ``target_s``."""
    actual_array = np.asarray(actual, dtype=float)
    source = np.asarray(source_s, dtype=float)
    target = np.asarray(target_s, dtype=float)
    if actual_array.ndim != 2:
        raise ValueError("actual must be a rank-2 branch matrix")
    if actual_array.shape[0] != source.shape[0]:
        raise ValueError("actual first axis must match source_s")
    interpolated = np.column_stack(
        [np.interp(target, source, actual_array[:, branch]) for branch in range(actual_array.shape[1])]
    )
    return jnp.asarray(interpolated)


@dataclass(frozen=True)
class GateResult:
    """Result from one validation gate."""

    gate_id: str
    status: str
    metric: float
    tolerance: float
    message: str

    @property
    def passed(self) -> bool:
        """Return ``True`` when the gate passed."""
        return self.status == "passed"


@dataclass(frozen=True)
class ValidationGate:
    """Metadata and runner for one reviewer-facing validation gate."""

    gate_id: str
    title: str
    layer: str
    quantity: str
    tolerance: float
    citation: str
    command: str
    tags: tuple[str, ...]
    _runner: Callable[[], GateResult] = field(repr=False, compare=False)

    def run(self) -> GateResult:
        """Execute the gate and return a structured result."""
        return self._runner()


def _pass_fail(gate_id: str, metric: float, tolerance: float, message: str) -> GateResult:
    status = "passed" if float(metric) <= float(tolerance) else "failed"
    return GateResult(gate_id=gate_id, status=status, metric=float(metric), tolerance=float(tolerance), message=message)


def _gate_analytic_cylinder_frequency() -> GateResult:
    gate_id = "analytic_cylinder_frequency"
    tolerance = 2.0e-6
    geometry = analytic_cylinder(radial_points=8, iota=0.25, magnetic_field=3.0, major_radius=5.0)
    actual = local_continuum_frequency_hz(geometry, m=2, n=1)
    metric = relative_l2_error(actual, actual[0])
    return _pass_fail(gate_id, float(metric), tolerance, "uniform cylinder continuum branch is radially constant")


def _gate_metric_matches_profile_limit() -> GateResult:
    gate_id = "metric_matches_profile_limit"
    tolerance = 2.0e-5
    geometry = analytic_circular_tokamak(radial_points=32, iota_axis=0.25, iota_edge=0.65)
    harmonics = jnp.asarray([[1, 0], [2, 1], [3, 2]], dtype=float)
    profile = continuum_branch_matrix(geometry, harmonics)
    metric = diagonal_continuum_frequencies_hz(metric_from_equilibrium(geometry), harmonics)
    error = relative_l2_error(metric, profile)
    return _pass_fail(gate_id, float(error), tolerance, "metric matrix diagonal reduces to profile continuum formula")


def _gate_metric_matrix_hermitian() -> GateResult:
    gate_id = "metric_matrix_hermitian"
    tolerance = 1.0e-12
    geometry = analytic_circular_tokamak(radial_points=16)
    family = stellarator_mode_family(poloidal_modes=(1, 2, 3), base_n=1, field_periods=geometry.field_periods)
    matrices = metric_continuum_matrices(metric_from_equilibrium(geometry), family.as_array())
    metric = max(float(hermitian_relative_error(matrices.stiffness[3])), float(hermitian_relative_error(matrices.mass[3])))
    return _pass_fail(gate_id, metric, tolerance, "continuum stiffness and mass matrices are Hermitian")


def _gate_metric_eigensolve_residual() -> GateResult:
    gate_id = "metric_eigensolve_residual"
    tolerance = 1.0e-5
    geometry = analytic_circular_tokamak(radial_points=18, iota_axis=0.24, iota_edge=0.72)
    harmonics = jnp.asarray([[1, 0], [2, 2], [3, 3]], dtype=float)
    matrices = metric_continuum_matrices(metric_from_equilibrium(geometry, theta_points=32, zeta_points=32), harmonics)
    eigensystem = generalized_hermitian_eigensystem(matrices.stiffness, matrices.mass)
    residuals = generalized_eigen_residual_norms(matrices.stiffness, matrices.mass, eigensystem)
    metric = float(jnp.max(residuals))
    return _pass_fail(gate_id, metric, tolerance, "generalized continuum eigenpairs satisfy Kx = lambda Mx")


def _gate_density_scale_gradient() -> GateResult:
    gate_id = "density_scale_gradient"
    tolerance = 5.0e-5
    metric_geometry = metric_from_equilibrium(analytic_cylinder(radial_points=24, iota=0.31, major_radius=4.8))
    harmonics = jnp.asarray([[2, 1]], dtype=float)

    def mean_frequency(scale: float) -> jnp.ndarray:
        return jnp.mean(diagonal_continuum_frequencies_hz(metric_geometry.with_density_scale(scale), harmonics))

    scale0 = 1.4
    autodiff = float(jax.grad(mean_frequency)(scale0))
    expected = float(-0.5 * mean_frequency(scale0) / scale0)
    metric = abs(autodiff - expected) / max(abs(expected), 1.0e-30)
    return _pass_fail(gate_id, metric, tolerance, "autodiff density sensitivity matches analytic rho^(-1/2) scaling")


def _gate_metric_angular_convergence() -> GateResult:
    gate_id = "metric_angular_convergence"
    tolerance = 2.0e-5
    geometry = analytic_circular_tokamak(radial_points=24, iota_axis=0.21, iota_edge=0.74)
    harmonics = jnp.asarray([[1, 0], [2, 1], [3, 2], [4, 3]], dtype=float)
    profile = continuum_branch_matrix(geometry, harmonics)
    coarse = diagonal_continuum_frequencies_hz(metric_from_equilibrium(geometry, theta_points=12, zeta_points=12), harmonics)
    fine = diagonal_continuum_frequencies_hz(metric_from_equilibrium(geometry, theta_points=48, zeta_points=48), harmonics)
    metric = max(float(relative_l2_error(coarse, profile)), float(relative_l2_error(fine, profile)))
    return _pass_fail(gate_id, metric, tolerance, "uniform metric fixture is insensitive to angular grid refinement")


def _gate_metric_radial_convergence() -> GateResult:
    gate_id = "metric_radial_convergence"
    tolerance = 5.0e-3
    harmonics = jnp.asarray([[1, 0], [2, 1], [3, 2]], dtype=float)
    coarse_geometry = analytic_circular_tokamak(radial_points=32, iota_axis=0.24, iota_edge=0.72)
    fine_geometry = analytic_circular_tokamak(radial_points=128, iota_axis=0.24, iota_edge=0.72)
    coarse = metric_continuum_frequencies_hz(
        metric_from_equilibrium(coarse_geometry, theta_points=12, zeta_points=12),
        harmonics,
    )
    fine = metric_continuum_frequencies_hz(
        metric_from_equilibrium(fine_geometry, theta_points=12, zeta_points=12),
        harmonics,
    )
    coarse_on_fine = interpolate_branch_matrix(coarse, coarse_geometry.s, fine_geometry.s)
    metric = relative_l2_error(coarse_on_fine, fine)
    return _pass_fail(
        gate_id,
        float(metric),
        tolerance,
        "analytic metric continuum branches converge under radial refinement",
    )


def _gate_analytic_radial_integral_convergence() -> GateResult:
    gate_id = "analytic_radial_integral_convergence"
    tolerance = 5.0e-4
    harmonics = jnp.asarray([[2, 1]], dtype=float)

    def branch_integral(radial_points: int) -> jnp.ndarray:
        geometry = analytic_circular_tokamak(radial_points=radial_points, iota_axis=0.24, iota_edge=0.72)
        branch = continuum_branch_matrix(geometry, harmonics)[:, 0]
        return jnp.sum(0.5 * (branch[1:] + branch[:-1]) * (geometry.s[1:] - geometry.s[:-1]))

    reference = branch_integral(256)
    fine = branch_integral(64)
    metric = jnp.abs(fine - reference) / jnp.maximum(jnp.abs(reference), 1.0e-30)
    return _pass_fail(gate_id, float(metric), tolerance, "radial quadrature of smooth analytic branch converges")


def _gate_landreman_paul_adapter_smoke() -> GateResult:
    gate_id = "landreman_paul_adapter_smoke"
    tolerance = 0.0
    case = landreman_paul_qa_lowres()
    if not case.is_available():
        return GateResult(
            gate_id=gate_id,
            status="skipped",
            metric=float("nan"),
            tolerance=tolerance,
            message=f"missing local paths: {case.missing_paths()}",
        )
    geometry = from_vmec_wout(case.wout, name=case.case_id)  # type: ignore[arg-type]
    metric_geometry = metric_from_vmec_wout(case.wout, theta_points=16, zeta_points=16, name=case.case_id)  # type: ignore[arg-type]
    finite_count = int(jnp.sum(jnp.isfinite(metric_geometry.magnetic_field)))
    expected_count = int(metric_geometry.magnetic_field.size)
    finite = finite_count == expected_count
    positive_b = bool(jnp.all(metric_geometry.magnetic_field > 0.0))
    shape_match = metric_geometry.s.shape == geometry.s.shape
    passed = finite and positive_b and shape_match and geometry.field_periods == 2
    return GateResult(
        gate_id=gate_id,
        status="passed" if passed else "failed",
        metric=0.0 if passed else 1.0,
        tolerance=tolerance,
        message="Landreman-Paul QA wout profile and metric adapters produce finite positive fields",
    )


def _gate_landreman_paul_continuum_finite() -> GateResult:
    gate_id = "landreman_paul_continuum_finite"
    tolerance = 1.0e-4
    case = landreman_paul_qa_lowres()
    if not case.is_available():
        return GateResult(
            gate_id=gate_id,
            status="skipped",
            metric=float("nan"),
            tolerance=tolerance,
            message=f"missing local paths: {case.missing_paths()}",
        )

    geometry = from_vmec_wout(case.wout, name=case.case_id)  # type: ignore[arg-type]
    family = stellarator_mode_family(
        poloidal_modes=(1, 2, 3),
        base_n=1,
        field_periods=geometry.field_periods,
        sidebands=(-1, 0, 1),
    )
    profile_branches = continuum_branch_matrix(geometry, family.as_array())
    metric_geometry = metric_from_vmec_wout(case.wout, theta_points=16, zeta_points=16, name=case.case_id)  # type: ignore[arg-type]
    matrices = metric_continuum_matrices(metric_geometry, family.as_array())
    eigensystem = generalized_hermitian_eigensystem(matrices.stiffness, matrices.mass)
    frequencies = jnp.sqrt(jnp.maximum(jnp.real(eigensystem.eigenvalues), 0.0)) / (2.0 * jnp.pi)
    residuals = generalized_eigen_residual_norms(matrices.stiffness, matrices.mass, eigensystem)
    finite = bool(jnp.all(jnp.isfinite(profile_branches))) and bool(jnp.all(jnp.isfinite(frequencies)))
    non_negative = bool(jnp.all(profile_branches >= 0.0)) and bool(jnp.all(frequencies >= 0.0))
    residual_p95 = float(jnp.percentile(residuals, 95.0))
    metric = residual_p95 if finite and non_negative else 1.0
    return _pass_fail(
        gate_id,
        metric,
        tolerance,
        "Landreman-Paul QA profile and metric continuum branches are finite with small residuals",
    )


def _gate_landreman_paul_radial_sampling_convergence() -> GateResult:
    gate_id = "landreman_paul_radial_sampling_convergence"
    tolerance = 8.0e-2
    case = landreman_paul_qa_lowres()
    if not case.is_available():
        return GateResult(
            gate_id=gate_id,
            status="skipped",
            metric=float("nan"),
            tolerance=tolerance,
            message=f"missing local paths: {case.missing_paths()}",
        )

    fine = metric_from_vmec_wout(case.wout, theta_points=12, zeta_points=12, name=case.case_id)  # type: ignore[arg-type]
    coarse = _coarsen_metric_geometry(fine, radial_points=24)
    family = stellarator_mode_family(
        poloidal_modes=(1, 2),
        base_n=1,
        field_periods=fine.field_periods,
        sidebands=(0,),
    )
    harmonics = family.as_array()
    coarse_branches = metric_continuum_frequencies_hz(coarse, harmonics)
    fine_branches = metric_continuum_frequencies_hz(fine, harmonics)
    coarse_on_fine = interpolate_branch_matrix(coarse_branches, coarse.s, fine.s)
    metric = relative_l2_error(coarse_on_fine, fine_branches)
    return _pass_fail(
        gate_id,
        float(metric),
        tolerance,
        "Landreman-Paul QA metric branches are stable under radial sampling refinement",
    )


def _coarsen_metric_geometry(metric: MetricGeometry, *, radial_points: int) -> MetricGeometry:
    indices = np.unique(np.rint(np.linspace(0, int(metric.s.shape[0]) - 1, int(radial_points))).astype(int))
    if indices.size < 4:
        raise ValueError("coarsened metric geometry must keep at least four radial points")
    return MetricGeometry(
        name=metric.name,
        s=metric.s[indices],
        theta=metric.theta,
        zeta=metric.zeta,
        jacobian=metric.jacobian[indices],
        magnetic_field=metric.magnetic_field[indices],
        mass_density=metric.mass_density[indices],
        bdotgrad_theta=metric.bdotgrad_theta[indices],
        bdotgrad_zeta=metric.bdotgrad_zeta[indices],
        field_periods=metric.field_periods,
        metadata=metric.metadata + (("radial_sampling_gate_points", int(indices.size)),),
    )


def registered_validation_gates() -> tuple[ValidationGate, ...]:
    """Return the current validation-gate registry."""
    return (
        ValidationGate(
            gate_id="analytic_cylinder_frequency",
            title="Uniform-cylinder continuum frequency",
            layer="analytic",
            quantity="relative L2 branch variation",
            tolerance=2.0e-6,
            citation="Goedbloed, Keppens, Poedts, Advanced Magnetohydrodynamics, local shear-Alfven continuum limit",
            command="python -m pytest tests/test_validation_registry.py",
            tags=("analytic", "continuum"),
            _runner=_gate_analytic_cylinder_frequency,
        ),
        ValidationGate(
            gate_id="metric_matches_profile_limit",
            title="Metric matrix reduces to profile continuum limit",
            layer="operator",
            quantity="relative L2 frequency error",
            tolerance=2.0e-5,
            citation="Large-aspect-ratio straight-field-line limit used as the FAX dense reference fixture",
            command="python -m pytest tests/test_metric.py tests/test_validation_registry.py",
            tags=("analytic", "metric", "continuum"),
            _runner=_gate_metric_matches_profile_limit,
        ),
        ValidationGate(
            gate_id="metric_matrix_hermitian",
            title="Metric continuum matrix Hermiticity",
            layer="operator",
            quantity="relative Frobenius Hermitian error",
            tolerance=1.0e-12,
            citation="Self-adjoint ideal-MHD shear-Alfven energy principle in the incompressible limit",
            command="python -m pytest tests/test_metric.py tests/test_validation_registry.py",
            tags=("metric", "symmetry"),
            _runner=_gate_metric_matrix_hermitian,
        ),
        ValidationGate(
            gate_id="metric_eigensolve_residual",
            title="Metric eigensolve residual",
            layer="operator",
            quantity="maximum relative residual",
            tolerance=1.0e-5,
            citation="Generalized Hermitian eigenproblem K x = omega^2 M x from the shear-Alfven energy principle",
            command="python -m pytest tests/test_metric.py tests/test_validation_registry.py",
            tags=("metric", "eigensolve", "residual"),
            _runner=_gate_metric_eigensolve_residual,
        ),
        ValidationGate(
            gate_id="density_scale_gradient",
            title="Density sensitivity against analytic scaling",
            layer="differentiability",
            quantity="relative gradient error",
            tolerance=5.0e-5,
            citation="JAX reverse-mode autodiff consistency check for omega proportional to rho^(-1/2)",
            command="python -m pytest tests/test_metric.py tests/test_validation_registry.py",
            tags=("autodiff", "continuum"),
            _runner=_gate_density_scale_gradient,
        ),
        ValidationGate(
            gate_id="metric_angular_convergence",
            title="Metric angular convergence",
            layer="convergence",
            quantity="relative frequency error across angular refinement",
            tolerance=2.0e-5,
            citation="Large-aspect-ratio straight-field-line limit used as an angular-quadrature convergence fixture",
            command="python -m pytest tests/test_metric.py tests/test_validation_registry.py",
            tags=("metric", "convergence"),
            _runner=_gate_metric_angular_convergence,
        ),
        ValidationGate(
            gate_id="metric_radial_convergence",
            title="Metric radial convergence",
            layer="convergence",
            quantity="relative frequency error across radial refinement",
            tolerance=5.0e-3,
            citation="Smooth analytic radial-grid convergence check for the shear-Alfven metric-matrix scaffold",
            command="python -m pytest tests/test_metric.py tests/test_validation_registry.py",
            tags=("metric", "convergence"),
            _runner=_gate_metric_radial_convergence,
        ),
        ValidationGate(
            gate_id="analytic_radial_integral_convergence",
            title="Analytic radial integral convergence",
            layer="convergence",
            quantity="relative branch-integral error",
            tolerance=5.0e-4,
            citation="Smooth radial quadrature check for local shear-Alfven continuum branch integrals",
            command="python -m pytest tests/test_validation_registry.py",
            tags=("analytic", "continuum", "convergence"),
            _runner=_gate_analytic_radial_integral_convergence,
        ),
        ValidationGate(
            gate_id="landreman_paul_adapter_smoke",
            title="Landreman-Paul QA VMEC adapter smoke test",
            layer="adapter",
            quantity="finite/positive metric-field check",
            tolerance=0.0,
            citation="Landreman & Paul, Phys. Rev. Lett. 128, 035001 (2022), arXiv:2108.03711",
            command="python -m pytest -m requires_vmec_jax tests/test_metric.py tests/test_validation_registry.py",
            tags=("vmec", "landreman-paul", "adapter"),
            _runner=_gate_landreman_paul_adapter_smoke,
        ),
        ValidationGate(
            gate_id="landreman_paul_continuum_finite",
            title="Landreman-Paul QA continuum finite-output check",
            layer="adapter",
            quantity="p95 generalized-eigen residual",
            tolerance=1.0e-4,
            citation="Landreman & Paul, Phys. Rev. Lett. 128, 035001 (2022), arXiv:2108.03711",
            command="python -m pytest -m requires_vmec_jax tests/test_validation_registry.py",
            tags=("vmec", "landreman-paul", "adapter", "continuum"),
            _runner=_gate_landreman_paul_continuum_finite,
        ),
        ValidationGate(
            gate_id="landreman_paul_radial_sampling_convergence",
            title="Landreman-Paul QA radial sampling convergence",
            layer="adapter",
            quantity="relative branch error under radial sampling refinement",
            tolerance=8.0e-2,
            citation="Landreman & Paul, Phys. Rev. Lett. 128, 035001 (2022), arXiv:2108.03711; adapter sampling smoke test",
            command="python -m pytest -m requires_vmec_jax tests/test_validation_registry.py",
            tags=("vmec", "landreman-paul", "adapter", "convergence"),
            _runner=_gate_landreman_paul_radial_sampling_convergence,
        ),
    )


def run_validation_gate(gate_id: str) -> GateResult:
    """Run a registered validation gate by identifier."""
    registry = {gate.gate_id: gate for gate in registered_validation_gates()}
    if gate_id not in registry:
        raise KeyError(f"unknown validation gate {gate_id!r}")
    return registry[gate_id].run()


def run_validation_suite(*, tags: tuple[str, ...] | None = None, include_skipped: bool = True) -> tuple[GateResult, ...]:
    """Run registered validation gates, optionally filtered by tag."""
    selected_tags = set(tags or ())
    results: list[GateResult] = []
    for gate in registered_validation_gates():
        if selected_tags and not selected_tags.intersection(gate.tags):
            continue
        result = gate.run()
        if include_skipped or result.status != "skipped":
            results.append(result)
    return tuple(results)
