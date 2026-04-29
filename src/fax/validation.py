"""Validation metrics and registered physics gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from .boozer import (
    BoozerGeometry,
    BoozerSpectrum,
    boozer_3d_continuum_frequencies_hz,
    boozer_3d_continuum_matrices,
    boozer_3d_operator_angular_convergence,
    boozer_continuum_frequencies_hz,
    boozer_continuum_matrices,
    boozer_embedding_jacobian_shape_error,
    boozer_geometry_jacobian_identity_error,
    boozer_harmonic_orthogonality_error,
    boozer_jacobian_identity_error,
    boozer_metric_continuum_frequencies_hz,
    boozer_metric_continuum_matrices,
    boozer_projection_error,
    read_boozmn_geometry,
    run_booz_xform_jax_geometry_from_wout,
    run_booz_xform_jax_geometry_from_wout_data,
    run_vmec_jax_boozer_solve_convergence,
    run_vmec_jax_boozer_truncation_convergence,
    run_vmec_jax_fixed_boundary_boozer_geometry,
    straight_field_line_phase_error,
)
from .continuum import continuum_branch_matrix, local_continuum_frequency_hz
from .cross_code import (
    ae3d_dense_spectrum_parity,
    ae3d_eigenvalue_residual_summary,
    ae3d_matrix_eigenmode_parity,
    ae3d_matrix_symmetry_summary,
    ae3d_omega2_eigenvalue_error,
    stellgap_ctok_local_continuum_error,
    stellgap_postprocess_error,
)
from .equilibrium import analytic_circular_tokamak, analytic_cylinder, from_vmec_wout
from .fast_ions import FastIonPopulation, eigenfunction_fast_ion_drive, reduced_fast_ion_drive
from .gap_models import bae_accumulation_frequency_hz, basic_alfven_gap_map, tae_gap_estimate
from .global_modes import (
    RadialBlockEigenmodeProblem,
    boozer_radial_stiffness_matrices,
    constant_block_problem_discrete_omega2,
    constant_problem_discrete_omega2,
    constant_radial_block_problem,
    constant_radial_problem,
    lowest_radial_block_frequency_hz,
    lowest_radial_frequency_hz,
    radial_block_eigenvalue_sensitivity,
    radial_block_isolated_omega2,
    radial_block_problem_from_boozer_geometry,
    solve_radial_block_eigenmodes,
    solve_radial_eigenmodes,
)
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
from .optimization import boozer_global_isolated_mode_frequency_hz
from .reference_cases import landreman_paul_qa_lowres
from .units import MU0, PROTON_MASS

jax.config.update("jax_enable_x64", True)


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


def _gate_global_mode_constant_profile() -> GateResult:
    gate_id = "global_mode_constant_profile"
    tolerance = 1.0e-5
    problem = constant_radial_problem(radial_points=42, continuum_frequency_hz=90_000.0, radial_coupling_omega2=2.5e9)
    result = solve_radial_eigenmodes(problem)
    expected = constant_problem_discrete_omega2(problem)
    metric = relative_l2_error(result.omega2[:8], expected[:8])
    residual = float(jnp.max(result.residuals[:8]))
    return _pass_fail(
        gate_id,
        max(float(metric), residual),
        tolerance,
        "global radial finite-difference eigenvalues match the exact constant-profile spectrum",
    )


def _gate_global_mode_density_gradient() -> GateResult:
    gate_id = "global_mode_density_gradient"
    tolerance = 3.0e-3
    problem = constant_radial_problem(radial_points=34, continuum_frequency_hz=120_000.0, radial_coupling_omega2=1.5e10)

    def frequency(scale: float) -> jnp.ndarray:
        return lowest_radial_frequency_hz(scale, base_problem=problem)

    scale0 = 1.25
    autodiff = float(jax.grad(frequency)(scale0))
    expected = float(-0.5 * frequency(scale0) / scale0)
    metric = abs(autodiff - expected) / max(abs(expected), 1.0e-30)
    return _pass_fail(
        gate_id,
        metric,
        tolerance,
        "global radial eigenmode density sensitivity matches analytic rho^(-1/2) scaling",
    )


def _gate_global_3d_block_constant_spectrum() -> GateResult:
    gate_id = "global_3d_block_constant_spectrum"
    tolerance = 1.0e-5
    problem = constant_radial_block_problem(
        radial_points=24,
        continuum_frequency_hz=(90_000.0, 135_000.0),
        radial_coupling_omega2=(2.0e9, 3.5e9),
    )
    result = solve_radial_block_eigenmodes(problem)
    expected = constant_block_problem_discrete_omega2(problem)
    metric = relative_l2_error(result.omega2[:12], expected[:12])
    residual = float(jnp.max(result.residuals[:12]))
    return _pass_fail(
        gate_id,
        max(float(metric), residual),
        tolerance,
        "block radial 3-D eigenproblem recovers the exact diagonal constant-coefficient spectrum",
    )


def _gate_global_3d_block_neumann_spectrum() -> GateResult:
    gate_id = "global_3d_block_neumann_spectrum"
    tolerance = 1.0e-5
    problem = constant_radial_block_problem(
        radial_points=24,
        continuum_frequency_hz=(90_000.0, 135_000.0),
        radial_coupling_omega2=(2.0e9, 3.5e9),
        boundary_condition="neumann",
    )
    result = solve_radial_block_eigenmodes(problem)
    expected = constant_block_problem_discrete_omega2(problem)
    metric = relative_l2_error(result.omega2[:12], expected[:12])
    residual = float(jnp.max(result.residuals[:12]))
    return _pass_fail(
        gate_id,
        max(float(metric), residual),
        tolerance,
        "block radial 3-D Neumann eigenproblem recovers the exact zero-gradient constant-coefficient spectrum",
    )


def _gate_global_3d_block_density_gradient() -> GateResult:
    gate_id = "global_3d_block_density_gradient"
    tolerance = 1.0e-4
    problem = constant_radial_block_problem(
        radial_points=18,
        continuum_frequency_hz=(95_000.0, 145_000.0),
        radial_coupling_omega2=(2.4e9, 3.1e9),
    )

    def frequency(scale: float) -> jnp.ndarray:
        return lowest_radial_block_frequency_hz(scale, base_problem=problem)

    scale0 = 1.25
    autodiff = float(jax.grad(frequency)(scale0))
    expected = float(-0.5 * frequency(scale0) / scale0)
    metric = abs(autodiff - expected) / max(abs(expected), 1.0e-30)
    return _pass_fail(
        gate_id,
        metric,
        tolerance,
        "block radial 3-D eigenproblem density sensitivity follows rho^(-1/2) scaling",
    )


def _gate_global_3d_block_eigenvalue_sensitivity() -> GateResult:
    gate_id = "global_3d_block_eigenvalue_sensitivity"
    tolerance = 2.0e-6
    problem = constant_radial_block_problem(
        radial_points=18,
        continuum_frequency_hz=(92_000.0, 150_000.0),
        radial_coupling_omega2=(2.2e9, 3.4e9),
    )
    result = solve_radial_block_eigenmodes(problem)
    zero_stiffness = jnp.zeros_like(problem.local_stiffness)
    zero_mass = jnp.zeros_like(problem.local_mass)
    zero_radial = jnp.zeros_like(problem.radial_stiffness)
    mass_tangent = RadialBlockEigenmodeProblem(
        s=problem.s,
        local_stiffness=zero_stiffness,
        local_mass=problem.local_mass,
        radial_stiffness=zero_radial,
        harmonics=problem.harmonics,
        boundary_condition=problem.boundary_condition,
    )
    radial_tangent = RadialBlockEigenmodeProblem(
        s=problem.s,
        local_stiffness=zero_stiffness,
        local_mass=zero_mass,
        radial_stiffness=problem.radial_stiffness,
        harmonics=problem.harmonics,
        boundary_condition=problem.boundary_condition,
    )

    def mass_scaled_omega2(alpha: float) -> jnp.ndarray:
        scaled = RadialBlockEigenmodeProblem(
            s=problem.s,
            local_stiffness=problem.local_stiffness,
            local_mass=(1.0 + alpha) * problem.local_mass,
            radial_stiffness=problem.radial_stiffness,
            harmonics=problem.harmonics,
            boundary_condition=problem.boundary_condition,
        )
        return solve_radial_block_eigenmodes(scaled).omega2[0]

    def radial_scaled_omega2(alpha: float) -> jnp.ndarray:
        scaled = RadialBlockEigenmodeProblem(
            s=problem.s,
            local_stiffness=problem.local_stiffness,
            local_mass=problem.local_mass,
            radial_stiffness=(1.0 + alpha) * problem.radial_stiffness,
            harmonics=problem.harmonics,
            boundary_condition=problem.boundary_condition,
        )
        return solve_radial_block_eigenmodes(scaled).omega2[0]

    mass_adjoint = radial_block_eigenvalue_sensitivity(problem, mass_tangent, result.omega2[0], result.eigenvectors[:, 0])
    radial_adjoint = radial_block_eigenvalue_sensitivity(problem, radial_tangent, result.omega2[0], result.eigenvectors[:, 0])
    mass_autodiff = jax.grad(mass_scaled_omega2)(0.0)
    radial_autodiff = jax.grad(radial_scaled_omega2)(0.0)
    mass_metric = jnp.abs(mass_adjoint - mass_autodiff) / jnp.maximum(jnp.abs(mass_autodiff), 1.0e-30)
    radial_metric = jnp.abs(radial_adjoint - radial_autodiff) / jnp.maximum(jnp.abs(radial_autodiff), 1.0e-30)
    return _pass_fail(
        gate_id,
        float(jnp.maximum(mass_metric, radial_metric)),
        tolerance,
        "block radial Hermitian generalized-eigenvalue adjoint sensitivity matches direct JAX autodiff",
    )


def _gate_global_3d_block_custom_jvp_eigenvalue() -> GateResult:
    gate_id = "global_3d_block_custom_jvp_eigenvalue"
    tolerance = 2.0e-6
    problem = constant_radial_block_problem(
        radial_points=18,
        continuum_frequency_hz=(92_000.0, 150_000.0),
        radial_coupling_omega2=(2.2e9, 3.4e9),
    )
    result = solve_radial_block_eigenmodes(problem)
    tangent = RadialBlockEigenmodeProblem(
        s=problem.s,
        local_stiffness=jnp.zeros_like(problem.local_stiffness),
        local_mass=jnp.zeros_like(problem.local_mass),
        radial_stiffness=problem.radial_stiffness,
        harmonics=problem.harmonics,
        boundary_condition=problem.boundary_condition,
    )

    def omega2(alpha: float) -> jnp.ndarray:
        scaled = RadialBlockEigenmodeProblem(
            s=problem.s,
            local_stiffness=problem.local_stiffness,
            local_mass=problem.local_mass,
            radial_stiffness=(1.0 + alpha) * problem.radial_stiffness,
            harmonics=problem.harmonics,
            boundary_condition=problem.boundary_condition,
        )
        return radial_block_isolated_omega2(scaled, mode_index=0)

    custom_jvp = jax.grad(omega2)(0.0)
    explicit = radial_block_eigenvalue_sensitivity(problem, tangent, result.omega2[0], result.eigenvectors[:, 0])
    metric = jnp.abs(custom_jvp - explicit) / jnp.maximum(jnp.abs(explicit), 1.0e-30)
    return _pass_fail(
        gate_id,
        float(metric),
        tolerance,
        "custom-JVP isolated radial-block eigenvalue gradient matches the explicit adjoint sensitivity",
    )


def _gate_boozer_global_custom_jvp_design_gradient() -> GateResult:
    gate_id = "boozer_global_custom_jvp_design_gradient"
    tolerance = 5.0e-3
    geometry = _synthetic_global_boozer_geometry()
    harmonics = jnp.asarray([[1.0, 0.0], [2.0, 1.0]], dtype=float)
    params = jnp.asarray([0.03, -0.004, 0.02])

    def frequency(parameter_vector: jax.Array) -> jnp.ndarray:
        return boozer_global_isolated_mode_frequency_hz(
            parameter_vector,
            geometry=geometry,
            harmonics=harmonics,
            mass_density=3.4e-8,
            mode_index=0,
            theta_points=10,
            zeta_points=10,
            radial_stiffness_scale=1.0e-4,
        )

    custom_jvp = jax.grad(frequency)(params)
    step = 2.0e-4
    finite_difference = []
    for index in range(int(params.shape[0])):
        direction = jnp.zeros_like(params).at[index].set(step)
        finite_difference.append((frequency(params + direction) - frequency(params - direction)) / (2.0 * step))
    metric = relative_l2_error(custom_jvp, jnp.asarray(finite_difference))
    return _pass_fail(
        gate_id,
        float(metric),
        tolerance,
        "real-Boozer isolated-mode custom-JVP design gradient matches centered finite differences",
    )


def _gate_analytic_tae_gap_center() -> GateResult:
    gate_id = "analytic_tae_gap_center"
    tolerance = 2.0e-2
    geometry = analytic_circular_tokamak(radial_points=256, iota_axis=0.45, iota_edge=0.85, major_radius=6.2)
    estimate = tae_gap_estimate(geometry, m=1, n=1, inverse_aspect_ratio=0.08)
    target_iota = 2.0 / 3.0
    metric = abs(float(estimate.iota) - target_iota) / target_iota
    return _pass_fail(
        gate_id,
        metric,
        tolerance,
        "TAE m/m+1 crossing occurs at the analytic iota=2n/(2m+1) resonance",
    )


def _gate_basic_gap_map_ordering() -> GateResult:
    gate_id = "basic_gap_map_ordering"
    tolerance = 2.0e-2
    demo = basic_alfven_gap_map(radial_points=192)
    tae_q = 1.0 / demo.tae.estimate.iota
    eae_q = 1.0 / demo.eae.estimate.iota
    bae_frequency = demo.bae_frequency_at_surface_hz
    tae_frequency = demo.tae.estimate.center_frequency_hz
    eae_frequency = demo.eae.estimate.center_frequency_hz
    location_error = jnp.maximum(jnp.abs(tae_q - 1.5) / 1.5, jnp.abs(eae_q - 2.0) / 2.0)
    ordering_failure = jnp.where((bae_frequency < tae_frequency) & (tae_frequency < eae_frequency), 0.0, 1.0)
    metric = float(jnp.maximum(location_error, ordering_failure))
    return _pass_fail(
        gate_id,
        metric,
        tolerance,
        "basic gap map places BAE below TAE below EAE with TAE/EAE at the analytic q surfaces",
    )


def _gate_bae_temperature_scaling() -> GateResult:
    gate_id = "bae_temperature_scaling"
    tolerance = 1.0e-6
    geometry = analytic_circular_tokamak(radial_points=48, iota_axis=0.8, iota_edge=0.4, major_radius=3.0)
    cold = bae_accumulation_frequency_hz(geometry, ion_temperature_ev=1_000.0, electron_temperature_ev=1_000.0)
    hot = bae_accumulation_frequency_hz(geometry, ion_temperature_ev=4_000.0, electron_temperature_ev=4_000.0)
    metric = max_relative_error(hot / cold, 2.0)
    return _pass_fail(
        gate_id,
        float(metric),
        tolerance,
        "fluid-order BAE accumulation frequency follows sqrt(T_i) scaling at fixed Te/Ti",
    )


def _gate_fast_ion_resonance_peak() -> GateResult:
    gate_id = "fast_ion_resonance_peak"
    tolerance = 1.0e-12
    population = FastIonPopulation(
        density_fraction=0.04,
        pressure_gradient=-2.0,
        transit_frequency_hz=35_000.0,
        precession_frequency_hz=4_000.0,
        resonance_width_hz=2_500.0,
    )
    resonant_frequency = 2.0 * population.precession_frequency_hz + population.transit_frequency_hz
    off_resonant_frequency = resonant_frequency + 12.0 * population.resonance_width_hz
    resonant_drive = reduced_fast_ion_drive(
        resonant_frequency,
        toroidal_mode=2,
        transit_harmonic=1,
        population=population,
    )
    off_resonant_drive = reduced_fast_ion_drive(
        off_resonant_frequency,
        toroidal_mode=2,
        transit_harmonic=1,
        population=population,
    )
    metric = 0.0 if float(resonant_drive) > float(off_resonant_drive) > 0.0 else 1.0
    return _pass_fail(
        gate_id,
        metric,
        tolerance,
        "reduced fast-ion drive peaks at the transit/precession resonance",
    )


def _gate_fast_ion_eigenfunction_drive_localization() -> GateResult:
    gate_id = "fast_ion_eigenfunction_drive_localization"
    tolerance = 0.35
    s = jnp.linspace(0.0, 1.0, 80)
    center_mode = jnp.exp(-((s - 0.45) / 0.10) ** 2)
    edge_mode = jnp.exp(-((s - 0.82) / 0.08) ** 2)
    mode_structures = jnp.stack([center_mode, edge_mode], axis=0)[:, :, None]
    pressure_gradient = -jnp.exp(-((s - 0.45) / 0.12) ** 2)
    population = FastIonPopulation(
        density_fraction=0.03,
        pressure_gradient=pressure_gradient,
        transit_frequency_hz=jnp.full_like(s, 40_000.0),
        precession_frequency_hz=jnp.full_like(s, 5_000.0),
        resonance_width_hz=jnp.full_like(s, 2_000.0),
    )
    result = eigenfunction_fast_ion_drive(
        jnp.asarray([45_000.0, 45_000.0]),
        mode_structures,
        s,
        toroidal_mode=1,
        transit_harmonics=(1,),
        population=population,
        drive_scale_hz=1_000.0,
        damping_rate_hz=1.0,
    )
    metric = float(result.drive_indicator[1] / jnp.maximum(result.drive_indicator[0], 1.0e-30))
    return _pass_fail(
        gate_id,
        metric,
        tolerance,
        "mode-resolved fast-ion drive weights a resonant pressure-gradient profile by eigenfunction localization",
    )


def _synthetic_boozer_spectrum() -> BoozerSpectrum:
    return BoozerSpectrum(
        source="synthetic_validation",
        s=jnp.asarray([0.25, 0.75]),
        xm=jnp.asarray([0.0, 1.0, 2.0, 1.0]),
        xn=jnp.asarray([0.0, 0.0, 1.0, -1.0]),
        bmnc=jnp.asarray(
            [
                [2.8, 0.10, -0.035, 0.022],
                [2.7, 0.08, -0.025, 0.018],
            ]
        ),
        bmns=jnp.asarray(
            [
                [0.0, 0.025, 0.012, -0.008],
                [0.0, 0.020, 0.010, -0.006],
            ]
        ),
        field_periods=2,
    )


def _constant_boozer_geometry() -> BoozerGeometry:
    magnetic_field = 2.7
    major_radius = 5.5
    iota = 0.61
    numerator = magnetic_field * major_radius
    return BoozerGeometry(
        source="constant_boozer_geometry_validation",
        s=jnp.asarray([0.2, 0.8]),
        xm=jnp.asarray([0.0]),
        xn=jnp.asarray([0.0]),
        bmnc=jnp.asarray([[magnetic_field], [magnetic_field]]),
        bmns=None,
        rmnc=None,
        rmns=None,
        zmns=None,
        zmnc=None,
        pmns=None,
        pmnc=None,
        gmnc=jnp.asarray([[numerator / magnetic_field**2], [numerator / magnetic_field**2]]),
        gmns=None,
        iota=jnp.asarray([iota, iota]),
        boozer_i=jnp.asarray([0.0, 0.0]),
        boozer_g=jnp.asarray([numerator, numerator]),
        field_periods=1,
    )


def _synthetic_global_boozer_geometry() -> BoozerGeometry:
    s = jnp.linspace(0.2, 0.8, 5)
    minor_radius = 0.35 * s
    magnetic_field = 2.7
    major_radius = 5.5
    numerator = magnetic_field * major_radius
    zeros = jnp.zeros((int(s.shape[0]), 2))
    return BoozerGeometry(
        source="synthetic_global_boozer_validation",
        s=s,
        xm=jnp.asarray([0.0, 1.0]),
        xn=jnp.asarray([0.0, 0.0]),
        bmnc=jnp.column_stack([jnp.full_like(s, magnetic_field), jnp.zeros_like(s)]),
        bmns=zeros,
        rmnc=jnp.column_stack([jnp.full_like(s, major_radius), minor_radius]),
        rmns=zeros,
        zmns=jnp.column_stack([jnp.zeros_like(s), minor_radius]),
        zmnc=zeros,
        pmns=zeros,
        pmnc=zeros,
        gmnc=jnp.column_stack([jnp.full_like(s, numerator / magnetic_field**2), jnp.zeros_like(s)]),
        gmns=zeros,
        iota=jnp.full_like(s, 0.61),
        boozer_i=jnp.zeros_like(s),
        boozer_g=jnp.full_like(s, numerator),
        field_periods=1,
    )


def _booz_xform_fixture(name: str) -> Path | None:
    path = Path("/Users/rogerio/local/booz_xform_jax/tests/test_files") / name
    return path if path.exists() else None


def _wout_like_from_netcdf(path: Path) -> SimpleNamespace:
    from netCDF4 import Dataset

    with Dataset(str(path), "r") as dataset:
        lasym_name = "lasym__logical__"
        if lasym_name in dataset.variables:
            lasym = bool(dataset.variables[lasym_name][...].item())
        else:
            lasym = bool(getattr(dataset, lasym_name, False))

        rmnc = np.asarray(dataset.variables["rmnc"][:])
        zeros = np.zeros_like(rmnc)

        def read(name: str) -> np.ndarray:
            return np.asarray(dataset.variables[name][:]) if name in dataset.variables else zeros

        return SimpleNamespace(
            lasym=lasym,
            nfp=int(dataset.variables["nfp"][...].item()),
            mpol=int(dataset.variables["mpol"][...].item()),
            ntor=int(dataset.variables["ntor"][...].item()),
            xm=np.asarray(dataset.variables["xm"][:], dtype=int),
            xn=np.asarray(dataset.variables["xn"][:], dtype=int),
            xm_nyq=np.asarray(dataset.variables["xm_nyq"][:], dtype=int),
            xn_nyq=np.asarray(dataset.variables["xn_nyq"][:], dtype=int),
            ns=int(dataset.variables["ns"][...].item()),
            rmnc=rmnc,
            rmns=read("rmns") if lasym else zeros,
            zmnc=read("zmnc") if lasym else zeros,
            zmns=read("zmns"),
            lmnc=read("lmnc") if lasym else zeros,
            lmns=read("lmns"),
            bmnc=read("bmnc"),
            bmns=read("bmns") if lasym else zeros,
            bsubumnc=read("bsubumnc"),
            bsubumns=read("bsubumns") if lasym else zeros,
            bsubvmnc=read("bsubvmnc"),
            bsubvmns=read("bsubvmns") if lasym else zeros,
            iotas=np.asarray(dataset.variables["iotas"][:]),
        )


def _local_external_file(*parts: str) -> Path | None:
    path = Path("/Users/rogerio/local").joinpath(*parts)
    return path if path.exists() else None


def _gate_boozer_harmonic_projection() -> GateResult:
    gate_id = "boozer_harmonic_projection"
    tolerance = 2.0e-6
    spectrum = _synthetic_boozer_spectrum()
    projection = boozer_projection_error(spectrum, surface_index=1, theta_points=64, zeta_points=64)
    orthogonality = boozer_harmonic_orthogonality_error(spectrum.xm, spectrum.xn, theta_points=64, zeta_points=64)
    metric = max(float(projection), float(orthogonality))
    return _pass_fail(
        gate_id,
        metric,
        tolerance,
        "Boozer Fourier harmonics reconstruct and project back to known coefficients",
    )


def _gate_boozer_straight_field_line_phase() -> GateResult:
    gate_id = "boozer_straight_field_line_phase"
    tolerance = 1.0e-4
    spectrum = _synthetic_boozer_spectrum()
    harmonics = jnp.column_stack([spectrum.xm, spectrum.xn])
    metric = straight_field_line_phase_error(harmonics, iota=0.73, zeta_points=128)
    return _pass_fail(
        gate_id,
        float(metric),
        tolerance,
        "Boozer harmonic phases have slope m*iota-n along straight field lines",
    )


def _gate_boozer_constant_field_continuum() -> GateResult:
    gate_id = "boozer_constant_field_continuum"
    tolerance = 1.0e-5
    magnetic_field = 2.7
    mass_density = 3.4e-8
    major_radius = 5.5
    iota = 0.61
    spectrum = BoozerSpectrum(
        source="constant_boozer_validation",
        s=jnp.asarray([0.2, 0.8]),
        xm=jnp.asarray([0.0]),
        xn=jnp.asarray([0.0]),
        bmnc=jnp.asarray([[magnetic_field], [magnetic_field]]),
        bmns=None,
        field_periods=1,
    )
    harmonics = jnp.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    frequencies = boozer_continuum_frequencies_hz(
        spectrum,
        harmonics,
        iota=iota,
        major_radius=major_radius,
        mass_density=mass_density,
    )
    k_parallel = (harmonics[:, 1] - harmonics[:, 0] * iota) / major_radius
    expected = magnetic_field / jnp.sqrt(MU0 * mass_density) * jnp.abs(k_parallel) / (2.0 * jnp.pi)
    metric = relative_l2_error(frequencies[0], jnp.sort(expected))
    return _pass_fail(
        gate_id,
        float(metric),
        tolerance,
        "constant-field Boozer continuum matrix reduces to the analytic profile-limit frequencies",
    )


def _gate_boozer_sideband_coupling_hermitian() -> GateResult:
    gate_id = "boozer_sideband_coupling_hermitian"
    tolerance = 1.0e-6
    spectrum = BoozerSpectrum(
        source="coupled_boozer_validation",
        s=jnp.asarray([0.5]),
        xm=jnp.asarray([0.0, 1.0]),
        xn=jnp.asarray([0.0, 1.0]),
        bmnc=jnp.asarray([[2.8, 0.08]]),
        bmns=None,
        field_periods=1,
    )
    harmonics = jnp.asarray([[1.0, 0.0], [2.0, 1.0]])
    matrices = boozer_continuum_matrices(
        spectrum,
        harmonics,
        iota=0.43,
        major_radius=5.2,
        mass_density=4.1e-8,
        theta_points=96,
        zeta_points=96,
    )
    stiffness = matrices.stiffness[0]
    hermitian_error = hermitian_relative_error(stiffness)
    coupling_ratio = jnp.abs(stiffness[0, 1]) / jnp.maximum(jnp.max(jnp.abs(jnp.diag(stiffness))), 1.0e-30)
    coupling_failure = jnp.maximum(1.0e-3 - coupling_ratio, 0.0)
    metric = float(jnp.maximum(hermitian_error, coupling_failure))
    return _pass_fail(
        gate_id,
        metric,
        tolerance,
        "Boozer B^2 sideband coupling is nonzero and the reduced continuum matrix is Hermitian",
    )


def _gate_boozer_jacobian_identity() -> GateResult:
    gate_id = "boozer_jacobian_identity"
    tolerance = 2.0e-6
    spectrum = _synthetic_boozer_spectrum()
    metric = boozer_jacobian_identity_error(
        spectrum,
        jacobian_numerator=jnp.asarray([1.3, 1.7]),
        theta_points=64,
        zeta_points=64,
    )
    return _pass_fail(
        gate_id,
        float(metric),
        tolerance,
        "Boozer Jacobian weight satisfies J_B |B|^2 = G + iota I on each surface",
    )


def _gate_boozer_metric_constant_field_continuum() -> GateResult:
    gate_id = "boozer_metric_constant_field_continuum"
    tolerance = 1.0e-5
    magnetic_field = 2.7
    mass_density = 3.4e-8
    major_radius = 5.5
    iota = 0.61
    spectrum = BoozerSpectrum(
        source="constant_boozer_metric_validation",
        s=jnp.asarray([0.2]),
        xm=jnp.asarray([0.0]),
        xn=jnp.asarray([0.0]),
        bmnc=jnp.asarray([[magnetic_field]]),
        bmns=None,
        field_periods=1,
    )
    harmonics = jnp.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    frequencies = boozer_metric_continuum_frequencies_hz(
        spectrum,
        harmonics,
        iota=iota,
        major_radius=major_radius,
        mass_density=mass_density,
        jacobian_numerator=1.4,
    )
    k_parallel = (harmonics[:, 1] - harmonics[:, 0] * iota) / major_radius
    expected = magnetic_field / jnp.sqrt(MU0 * mass_density) * jnp.abs(k_parallel) / (2.0 * jnp.pi)
    metric = relative_l2_error(frequencies[0], jnp.sort(expected))
    return _pass_fail(
        gate_id,
        float(metric),
        tolerance,
        "Boozer-Jacobian weighted continuum matrix recovers the constant-field profile-limit frequencies",
    )


def _gate_boozer_metric_mass_coupling() -> GateResult:
    gate_id = "boozer_metric_mass_coupling"
    tolerance = 1.0e-5
    spectrum = BoozerSpectrum(
        source="jacobian_weighted_boozer_validation",
        s=jnp.asarray([0.5]),
        xm=jnp.asarray([0.0, 1.0]),
        xn=jnp.asarray([0.0, 1.0]),
        bmnc=jnp.asarray([[2.8, 0.10]]),
        bmns=None,
        field_periods=1,
    )
    harmonics = jnp.asarray([[1.0, 0.0], [2.0, 1.0]])
    matrices = boozer_metric_continuum_matrices(
        spectrum,
        harmonics,
        iota=0.43,
        major_radius=5.2,
        mass_density=4.1e-8,
        jacobian_numerator=1.6,
        theta_points=96,
        zeta_points=96,
    )
    mass = matrices.mass[0]
    stiffness = matrices.stiffness[0]
    mass_coupling = jnp.abs(mass[0, 1]) / jnp.maximum(jnp.max(jnp.abs(jnp.diag(mass))), 1.0e-30)
    stiffness_coupling = jnp.abs(stiffness[0, 1]) / jnp.maximum(jnp.max(jnp.abs(jnp.diag(stiffness))), 1.0e-30)
    failure = jnp.maximum(1.0e-3 - mass_coupling, 0.0) + jnp.maximum(stiffness_coupling - 1.0e-5, 0.0)
    metric = jnp.maximum(jnp.maximum(hermitian_relative_error(mass), hermitian_relative_error(stiffness)), failure)
    return _pass_fail(
        gate_id,
        float(metric),
        tolerance,
        "Boozer Jacobian weighting gives nonzero mass coupling while preserving J_B |B|^2 stiffness cancellation",
    )


def _gate_boozer_3d_constant_field_continuum() -> GateResult:
    gate_id = "boozer_3d_constant_field_continuum"
    tolerance = 1.0e-5
    magnetic_field = 2.7
    mass_density = 3.4e-8
    major_radius = 5.5
    iota = 0.61
    geometry = _constant_boozer_geometry()
    harmonics = jnp.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    frequencies = boozer_3d_continuum_frequencies_hz(
        geometry,
        harmonics,
        mass_density=mass_density,
        radial_metric=1.0,
        theta_points=32,
        zeta_points=32,
    )
    k_parallel = (harmonics[:, 1] - harmonics[:, 0] * iota) / major_radius
    expected = magnetic_field / jnp.sqrt(MU0 * mass_density) * jnp.abs(k_parallel) / (2.0 * jnp.pi)
    metric = relative_l2_error(frequencies[0], jnp.sort(expected))
    return _pass_fail(
        gate_id,
        float(metric),
        tolerance,
        "3-D Boozer local operator recovers omega=vA|n-m*iota|/R in the constant-field limit",
    )


def _gate_boozmn_geometry_jacobian_identity() -> GateResult:
    gate_id = "boozmn_geometry_jacobian_identity"
    tolerance = 2.0e-5
    fixture = _booz_xform_fixture("boozmn_circular_tokamak.nc")
    if fixture is None:
        return GateResult(
            gate_id=gate_id,
            status="skipped",
            metric=float("nan"),
            tolerance=tolerance,
            message="missing local booz_xform_jax boozmn_circular_tokamak.nc fixture",
        )
    geometry = read_boozmn_geometry(fixture)
    metric = boozer_geometry_jacobian_identity_error(geometry, theta_points=48, zeta_points=48)
    return _pass_fail(
        gate_id,
        float(metric),
        tolerance,
        "real boozmn harmonics satisfy sqrt(g)|B|^2=G+iota I on the sampled surfaces",
    )


def _gate_booz_xform_jax_live_geometry_adapter() -> GateResult:
    gate_id = "booz_xform_jax_live_geometry_adapter"
    tolerance = 5.0e-9
    boozmn = _booz_xform_fixture("boozmn_up_down_asymmetric_tokamak.nc")
    wout = _booz_xform_fixture("wout_up_down_asymmetric_tokamak.nc")
    if boozmn is None or wout is None:
        return GateResult(
            gate_id=gate_id,
            status="skipped",
            metric=float("nan"),
            tolerance=tolerance,
            message="missing local asymmetric booz_xform_jax fixtures",
        )

    reference = read_boozmn_geometry(boozmn)
    try:
        geometry = run_booz_xform_jax_geometry_from_wout(
            wout,
            mboz=48,
            nboz=0,
            surfaces=tuple(float(value) for value in np.asarray(reference.s)),
            jit=False,
        )
    except ImportError as exc:
        return GateResult(
            gate_id=gate_id,
            status="skipped",
            metric=float("nan"),
            tolerance=tolerance,
            message=str(exc),
        )

    errors = [
        relative_l2_error(getattr(geometry, name), getattr(reference, name))
        for name in ("bmnc", "bmns", "rmnc", "rmns", "zmns", "zmnc", "pmns", "pmnc", "gmnc", "gmns")
    ]
    metric = float(jnp.max(jnp.asarray(errors)))
    return _pass_fail(
        gate_id,
        metric,
        tolerance,
        "live booz_xform_jax VMEC-to-Boozer geometry matches the asymmetric BOOZ_XFORM boozmn reference",
    )


def _gate_booz_xform_jax_in_memory_geometry_adapter() -> GateResult:
    gate_id = "booz_xform_jax_in_memory_geometry_adapter"
    tolerance = 5.0e-9
    boozmn = _booz_xform_fixture("boozmn_up_down_asymmetric_tokamak.nc")
    wout = _booz_xform_fixture("wout_up_down_asymmetric_tokamak.nc")
    if boozmn is None or wout is None:
        return GateResult(
            gate_id=gate_id,
            status="skipped",
            metric=float("nan"),
            tolerance=tolerance,
            message="missing local asymmetric booz_xform_jax fixtures",
        )

    reference = read_boozmn_geometry(boozmn)
    try:
        wout_data = _wout_like_from_netcdf(wout)
        geometry = run_booz_xform_jax_geometry_from_wout_data(
            wout_data,
            mboz=48,
            nboz=0,
            surfaces=tuple(float(value) for value in np.asarray(reference.s)),
            jit=False,
        )
    except ImportError as exc:
        return GateResult(
            gate_id=gate_id,
            status="skipped",
            metric=float("nan"),
            tolerance=tolerance,
            message=str(exc),
        )

    errors = [
        relative_l2_error(getattr(geometry, name), getattr(reference, name))
        for name in ("bmnc", "bmns", "rmnc", "rmns", "zmns", "zmnc", "pmns", "pmnc", "gmnc", "gmns")
    ]
    metric = float(jnp.max(jnp.asarray(errors)))
    return _pass_fail(
        gate_id,
        metric,
        tolerance,
        "in-memory VMEC wout_data to booz_xform_jax to FAX geometry matches the asymmetric boozmn reference",
    )


def _gate_boozmn_embedding_jacobian_shape() -> GateResult:
    gate_id = "boozmn_embedding_jacobian_shape"
    tolerance = 4.0e-2
    fixture = _booz_xform_fixture("boozmn_LandremanSenguptaPlunk_section5p3.nc")
    if fixture is None:
        return GateResult(
            gate_id=gate_id,
            status="skipped",
            metric=float("nan"),
            tolerance=tolerance,
            message="missing local booz_xform_jax boozmn_LandremanSenguptaPlunk_section5p3.nc fixture",
        )
    geometry = read_boozmn_geometry(fixture)
    metric = boozer_embedding_jacobian_shape_error(geometry, theta_points=32, zeta_points=32)
    return _pass_fail(
        gate_id,
        float(metric),
        tolerance,
        "embedding-derived Jacobian angular shape matches booz_xform_jax Jacobian harmonics up to one radial scale",
    )


def _gate_boozmn_3d_continuum_finite() -> GateResult:
    gate_id = "boozmn_3d_continuum_finite"
    tolerance = 2.0e-5
    fixture = _booz_xform_fixture("boozmn_LandremanSenguptaPlunk_section5p3.nc")
    if fixture is None:
        return GateResult(
            gate_id=gate_id,
            status="skipped",
            metric=float("nan"),
            tolerance=tolerance,
            message="missing local booz_xform_jax boozmn_LandremanSenguptaPlunk_section5p3.nc fixture",
        )
    geometry = read_boozmn_geometry(fixture)
    harmonics = jnp.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    matrices = boozer_3d_continuum_matrices(
        geometry,
        harmonics,
        mass_density=3.4e-8,
        theta_points=24,
        zeta_points=24,
    )
    frequencies = boozer_3d_continuum_frequencies_hz(
        geometry,
        harmonics,
        mass_density=3.4e-8,
        theta_points=24,
        zeta_points=24,
    )
    finite_failure = 0.0 if bool(jnp.all(jnp.isfinite(frequencies))) and bool(jnp.all(frequencies >= 0.0)) else 1.0
    metric = max(
        float(hermitian_relative_error(matrices.stiffness[0])),
        float(hermitian_relative_error(matrices.mass[0])),
        finite_failure,
    )
    return _pass_fail(
        gate_id,
        metric,
        tolerance,
        "real Boozer geometry produces finite non-negative local 3-D continuum branches and Hermitian matrices",
    )


def _gate_boozmn_global_3d_eigenmode_finite() -> GateResult:
    gate_id = "boozmn_global_3d_eigenmode_finite"
    tolerance = 2.0e-5
    fixture = _booz_xform_fixture("boozmn_LandremanSenguptaPlunk_section5p3.nc")
    if fixture is None:
        return GateResult(
            gate_id=gate_id,
            status="skipped",
            metric=float("nan"),
            tolerance=tolerance,
            message="missing local booz_xform_jax boozmn_LandremanSenguptaPlunk_section5p3.nc fixture",
        )
    geometry = read_boozmn_geometry(fixture)
    harmonics = jnp.asarray([[1.0, 0.0], [2.0, 1.0]])
    problem = radial_block_problem_from_boozer_geometry(
        geometry,
        harmonics,
        mass_density=3.4e-8,
        radial_coupling_fraction=0.004,
        theta_points=16,
        zeta_points=16,
    )
    result = solve_radial_block_eigenmodes(problem)
    finite_failure = 0.0 if bool(jnp.all(jnp.isfinite(result.frequencies_hz[:6]))) else 1.0
    metric = max(float(jnp.max(result.residuals[:6])), finite_failure)
    return _pass_fail(
        gate_id,
        metric,
        tolerance,
        "real boozmn local 3-D matrices assemble into a finite global radial block eigenproblem with small residuals",
    )


def _gate_boozmn_metric_radial_stiffness_hermitian() -> GateResult:
    gate_id = "boozmn_metric_radial_stiffness_hermitian"
    tolerance = 2.0e-5
    fixture = _booz_xform_fixture("boozmn_LandremanSenguptaPlunk_section5p3.nc")
    if fixture is None:
        return GateResult(
            gate_id=gate_id,
            status="skipped",
            metric=float("nan"),
            tolerance=tolerance,
            message="missing local booz_xform_jax boozmn_LandremanSenguptaPlunk_section5p3.nc fixture",
        )
    geometry = read_boozmn_geometry(fixture)
    harmonics = jnp.asarray([[1.0, 0.0], [2.0, 1.0]])
    radial = boozer_radial_stiffness_matrices(geometry, harmonics, theta_points=16, zeta_points=16)
    hermitian_error = float(jnp.max(jax.vmap(hermitian_relative_error)(radial)))
    eigenvalues = jnp.linalg.eigvalsh(radial)
    negative_failure = 0.0 if bool(jnp.min(jnp.real(eigenvalues)) > -1.0e-8 * jnp.max(jnp.abs(eigenvalues))) else 1.0
    metric = max(hermitian_error, negative_failure)
    return _pass_fail(
        gate_id,
        metric,
        tolerance,
        "metric-derived real-boozmn radial stiffness blocks are Hermitian and positive semidefinite",
    )


def _gate_stellgap_ctok_local_continuum() -> GateResult:
    gate_id = "stellgap_ctok_local_continuum"
    tolerance = 4.0e-2
    boozmn = _local_external_file("Stellgap_test_inputs", "boozmn_ctok.nc")
    alfven_post = _local_external_file("Stellgap_test_problem", "alfven_post")
    if boozmn is None or alfven_post is None:
        return GateResult(
            gate_id=gate_id,
            status="skipped",
            metric=float("nan"),
            tolerance=tolerance,
            message="missing local ORNL STELLGAP CTOK test artifacts under /Users/rogerio/local",
        )
    comparison = stellgap_ctok_local_continuum_error(
        boozmn_path=boozmn,
        alfven_post_path=alfven_post,
        mass_density=2.0 * PROTON_MASS * 1.0e19,
        theta_points=32,
        zeta_points=32,
    )
    return _pass_fail(
        gate_id,
        comparison.p95_relative_error,
        tolerance,
        f"FAX local Boozer continuum matches ORNL STELLGAP CTOK test output on {comparison.count} matched surfaces",
    )


def _gate_stellgap_public_postprocess_parity() -> GateResult:
    gate_id = "stellgap_public_postprocess_parity"
    tolerance = 1.0e-5
    alfven_spec = _local_external_file("Stellgap_test_problem", "alfven_spec")
    alfven_post = _local_external_file("Stellgap_test_problem", "alfven_post")
    if alfven_spec is None or alfven_post is None:
        return GateResult(
            gate_id=gate_id,
            status="skipped",
            metric=float("nan"),
            tolerance=tolerance,
            message="missing local ORNL STELLGAP alfven_spec/alfven_post artifacts under /Users/rogerio/local",
        )
    comparison = stellgap_postprocess_error(alfven_spec_path=alfven_spec, alfven_post_path=alfven_post)
    return _pass_fail(
        gate_id,
        comparison.max_relative_error,
        tolerance,
        f"FAX reproduces STELLGAP post_process.f frequencies for {comparison.count} public CTOK rows",
    )


def _gate_ae3d_public_eigenvalue_artifact() -> GateResult:
    gate_id = "ae3d_public_eigenvalue_artifact"
    tolerance = 1.0e-2
    egn_values = _local_external_file("AE3D_test_problem", "egn_values.dat")
    if egn_values is None:
        return GateResult(
            gate_id=gate_id,
            status="skipped",
            metric=float("nan"),
            tolerance=tolerance,
            message="missing local ORNL AE3D public eigenvalue answer artifact under /Users/rogerio/local",
        )
    summary = ae3d_eigenvalue_residual_summary(egn_values)
    return _pass_fail(
        gate_id,
        summary.max_residual,
        tolerance,
        (
            "ORNL AE3D public eigenvalue table is readable and has small reported residuals "
            f"for {summary.count} modes over [{summary.min_frequency_khz:.3g}, {summary.max_frequency_khz:.3g}] kHz"
        ),
    )


def _gate_ae3d_public_omega2_eigenvalue_parity() -> GateResult:
    gate_id = "ae3d_public_omega2_eigenvalue_parity"
    tolerance = 1.0e-5
    omega2 = _local_external_file("AE3D_test_problem", "omega2.dat")
    egn_values = _local_external_file("AE3D_test_problem", "egn_values.dat")
    if omega2 is None or egn_values is None:
        return GateResult(
            gate_id=gate_id,
            status="skipped",
            metric=float("nan"),
            tolerance=tolerance,
            message="missing local ORNL AE3D omega2/egn_values artifacts under /Users/rogerio/local",
        )
    comparison = ae3d_omega2_eigenvalue_error(omega2_path=omega2, egn_values_path=egn_values)
    return _pass_fail(
        gate_id,
        comparison.max_relative_error,
        tolerance,
        f"FAX reproduces AE3D omega2-to-egn_values convention for {comparison.count} public CTOK modes",
    )


def _gate_ae3d_public_matrix_symmetry() -> GateResult:
    gate_id = "ae3d_public_matrix_symmetry"
    tolerance = 1.0e-5
    a_matrix = _local_external_file("AE3D_test_problem", "a_matrix.dat")
    b_matrix = _local_external_file("AE3D_test_problem", "b_matrix.dat")
    if a_matrix is None or b_matrix is None:
        return GateResult(
            gate_id=gate_id,
            status="skipped",
            metric=float("nan"),
            tolerance=tolerance,
            message="missing local ORNL AE3D sparse matrix artifacts under /Users/rogerio/local",
        )
    a_summary = ae3d_matrix_symmetry_summary(a_matrix)
    b_summary = ae3d_matrix_symmetry_summary(b_matrix)
    metric = max(a_summary.relative_asymmetry, b_summary.relative_asymmetry)
    return _pass_fail(
        gate_id,
        metric,
        tolerance,
        (
            "ORNL AE3D public sparse matrices are symmetric to the expected tolerance "
            f"with dimension {a_summary.dimension} and {a_summary.nonzero_count}/{b_summary.nonzero_count} nonzeros"
        ),
    )


def _gate_ae3d_public_matrix_eigenmode_parity() -> GateResult:
    gate_id = "ae3d_public_matrix_eigenmode_parity"
    tolerance = 5.0e-6
    a_matrix = _local_external_file("AE3D_test_problem", "a_matrix.dat")
    b_matrix = _local_external_file("AE3D_test_problem", "b_matrix.dat")
    egn_mode = _local_external_file("AE3D_test_problem", "egn_mode_asci.dat")
    if a_matrix is None or b_matrix is None or egn_mode is None:
        return GateResult(
            gate_id=gate_id,
            status="skipped",
            metric=float("nan"),
            tolerance=tolerance,
            message="missing local ORNL AE3D sparse matrix/eigenmode artifacts under /Users/rogerio/local",
        )
    parity = ae3d_matrix_eigenmode_parity(
        a_matrix_path=a_matrix,
        b_matrix_path=b_matrix,
        egn_mode_path=egn_mode,
        max_modes=16,
    )
    metric = max(parity.max_residual, parity.max_rayleigh_relative_error)
    return _pass_fail(
        gate_id,
        metric,
        tolerance,
        (
            "ORNL AE3D public eigenvectors satisfy A x = lambda B x for "
            f"{parity.count} modes; dominant first-mode component is "
            f"(m,n)=({parity.dominant_mode_harmonic_m},{parity.dominant_mode_harmonic_n}) "
            f"near radial index {parity.dominant_mode_surface}"
        ),
    )


def _gate_ae3d_public_dense_spectrum_parity() -> GateResult:
    gate_id = "ae3d_public_dense_spectrum_parity"
    tolerance = 2.0e-6
    a_matrix = _local_external_file("AE3D_test_problem", "a_matrix.dat")
    b_matrix = _local_external_file("AE3D_test_problem", "b_matrix.dat")
    egn_mode = _local_external_file("AE3D_test_problem", "egn_mode_asci.dat")
    if a_matrix is None or b_matrix is None or egn_mode is None:
        return GateResult(
            gate_id=gate_id,
            status="skipped",
            metric=float("nan"),
            tolerance=tolerance,
            message="missing local ORNL AE3D sparse matrix/eigenmode artifacts under /Users/rogerio/local",
        )
    parity = ae3d_dense_spectrum_parity(
        a_matrix_path=a_matrix,
        b_matrix_path=b_matrix,
        egn_mode_path=egn_mode,
        max_modes=16,
    )
    return _pass_fail(
        gate_id,
        parity.max_eigenvalue_relative_error,
        tolerance,
        (
            "SciPy/LAPACK dense solve of ORNL AE3D public A x = lambda B x "
            f"recovers {parity.count} published omega^2 eigenvalues in a "
            f"{parity.matrix_dimension}x{parity.matrix_dimension} problem"
        ),
    )


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


def _gate_landreman_paul_vmec_jax_boozer_in_memory() -> GateResult:
    gate_id = "landreman_paul_vmec_jax_boozer_in_memory"
    tolerance = 1.0e-2
    case = landreman_paul_qa_lowres()
    if not case.vmec_input.exists():
        return GateResult(
            gate_id=gate_id,
            status="skipped",
            metric=float("nan"),
            tolerance=tolerance,
            message=f"missing Landreman-Paul VMEC input: {case.vmec_input}",
        )

    try:
        geometry = run_vmec_jax_fixed_boundary_boozer_geometry(
            case.vmec_input,
            mboz=4,
            nboz=4,
            surfaces=(0.25, 0.5),
            max_iter=1,
            theta_points=12,
            zeta_points=4,
            jit=False,
        )
    except ImportError as exc:
        return GateResult(
            gate_id=gate_id,
            status="skipped",
            metric=float("nan"),
            tolerance=tolerance,
            message=str(exc),
        )

    harmonics = jnp.asarray([[1.0, 0.0], [2.0, 1.0]])
    frequencies = boozer_3d_continuum_frequencies_hz(
        geometry,
        harmonics,
        mass_density=3.4e-8,
        theta_points=8,
        zeta_points=8,
    )
    identity_error = float(boozer_geometry_jacobian_identity_error(geometry, theta_points=12, zeta_points=8))
    finite = bool(jnp.all(jnp.isfinite(frequencies)))
    non_negative = bool(jnp.all(frequencies >= 0.0))
    metric = identity_error if finite and non_negative and geometry.field_periods == 2 else 1.0
    return _pass_fail(
        gate_id,
        metric,
        tolerance,
        "one-iteration VMEC-JAX to Boozer/FAX in-memory smoke path gives a finite local 3-D continuum",
    )


def _gate_landreman_paul_vmec_jax_boozer_truncation_convergence() -> GateResult:
    gate_id = "landreman_paul_vmec_jax_boozer_truncation_convergence"
    tolerance = 1.0e-2
    case = landreman_paul_qa_lowres()
    if not case.vmec_input.exists():
        return GateResult(
            gate_id=gate_id,
            status="skipped",
            metric=float("nan"),
            tolerance=tolerance,
            message=f"missing Landreman-Paul VMEC input: {case.vmec_input}",
        )

    try:
        result = run_vmec_jax_boozer_truncation_convergence(
            case.vmec_input,
            coarse_mboz=4,
            coarse_nboz=4,
            reference_mboz=5,
            reference_nboz=5,
            surfaces=(0.25, 0.5),
            max_iter=1,
            theta_points=12,
            zeta_points=4,
            jit=False,
        )
    except ImportError as exc:
        return GateResult(
            gate_id=gate_id,
            status="skipped",
            metric=float("nan"),
            tolerance=tolerance,
            message=str(exc),
        )

    return _pass_fail(
        gate_id,
        result.max_metric,
        tolerance,
        (
            "one-iteration Landreman-Paul VMEC-JAX bridge is stable between "
            "(mboz,nboz)=(4,4) and (5,5) Boozer truncations"
        ),
    )


def _gate_landreman_paul_vmec_jax_boozer_operator_angular_convergence() -> GateResult:
    gate_id = "landreman_paul_vmec_jax_boozer_operator_angular_convergence"
    tolerance = 5.0e-3
    case = landreman_paul_qa_lowres()
    if not case.vmec_input.exists():
        return GateResult(
            gate_id=gate_id,
            status="skipped",
            metric=float("nan"),
            tolerance=tolerance,
            message=f"missing Landreman-Paul VMEC input: {case.vmec_input}",
        )

    try:
        geometry = run_vmec_jax_fixed_boundary_boozer_geometry(
            case.vmec_input,
            mboz=5,
            nboz=5,
            surfaces=(0.25, 0.5),
            max_iter=1,
            theta_points=12,
            zeta_points=4,
            jit=False,
        )
    except ImportError as exc:
        return GateResult(
            gate_id=gate_id,
            status="skipped",
            metric=float("nan"),
            tolerance=tolerance,
            message=str(exc),
        )

    result = boozer_3d_operator_angular_convergence(
        geometry,
        jnp.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]]),
        mass_density=3.4e-8,
        grids=((12, 12), (16, 16), (20, 20)),
        reference_grid=(32, 32),
    )
    return _pass_fail(
        gate_id,
        result.max_metric,
        tolerance,
        "live Landreman-Paul VMEC-JAX Boozer local 3-D operator is stable under angular quadrature refinement",
    )


def _gate_landreman_paul_vmec_jax_solve_convergence() -> GateResult:
    gate_id = "landreman_paul_vmec_jax_solve_convergence"
    tolerance = 4.0e-2
    case = landreman_paul_qa_lowres()
    if case.wout is None or not case.vmec_input.exists() or not case.wout.exists():
        return GateResult(
            gate_id=gate_id,
            status="skipped",
            metric=float("nan"),
            tolerance=tolerance,
            message=f"missing Landreman-Paul VMEC input or reference wout: {case.missing_paths()}",
        )

    try:
        result = run_vmec_jax_boozer_solve_convergence(
            case.vmec_input,
            case.wout,
            max_iters=(1, 6, 12, 24),
            mboz=5,
            nboz=5,
            surfaces=(0.25, 0.5),
            theta_points=None,
            zeta_points=None,
            jit=False,
        )
    except ImportError as exc:
        return GateResult(
            gate_id=gate_id,
            status="skipped",
            metric=float("nan"),
            tolerance=tolerance,
            message=str(exc),
        )

    residual_monotone = bool(jnp.all(jnp.diff(result.fsq_totals) <= 1.0e-12))
    frequency_improves = bool(result.frequency_relative_errors[-1] < result.frequency_relative_errors[0])
    field_improves = bool(jnp.max(result.field_relative_errors[-1]) < jnp.max(result.field_relative_errors[0]))
    metric = result.max_metric if residual_monotone and frequency_improves and field_improves else float("inf")
    return _pass_fail(
        gate_id,
        metric,
        tolerance,
        (
            "Landreman-Paul VMEC-JAX residual ladder propagates to converging Boozer/FAX "
            f"field and continuum errors; residual ratio={result.residual_reduction_ratio:.3e}, "
            f"frequency-error ratio={result.frequency_error_reduction_ratio:.3e}"
        ),
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
            gate_id="global_mode_constant_profile",
            title="Global radial mode constant-profile spectrum",
            layer="global-eigenmode",
            quantity="relative omega^2 error and residual",
            tolerance=1.0e-5,
            citation="Dirichlet finite-difference Sturm-Liouville validation for the radial shear-Alfven envelope operator",
            command="python -m pytest tests/test_global_modes.py tests/test_validation_registry.py",
            tags=("global", "eigenmode", "analytic"),
            _runner=_gate_global_mode_constant_profile,
        ),
        ValidationGate(
            gate_id="global_mode_density_gradient",
            title="Global radial mode density sensitivity",
            layer="differentiability",
            quantity="relative autodiff/analytic gradient error",
            tolerance=3.0e-3,
            citation="JAX autodiff consistency check for the radial shear-Alfven generalized eigenproblem rho^(-1/2) scaling",
            command="python -m pytest tests/test_global_modes.py tests/test_validation_registry.py",
            tags=("global", "eigenmode", "autodiff"),
            _runner=_gate_global_mode_density_gradient,
        ),
        ValidationGate(
            gate_id="global_3d_block_constant_spectrum",
            title="Global radial 3-D block constant spectrum",
            layer="global-eigenmode",
            quantity="relative omega^2 error and residual",
            tolerance=1.0e-5,
            citation="Dirichlet block finite-difference validation for the global radial Boozer-harmonic shear-Alfven envelope operator",
            command="python -m pytest tests/test_global_modes.py tests/test_validation_registry.py",
            tags=("global", "eigenmode", "3d", "analytic"),
            _runner=_gate_global_3d_block_constant_spectrum,
        ),
        ValidationGate(
            gate_id="global_3d_block_neumann_spectrum",
            title="Global radial 3-D block Neumann spectrum",
            layer="global-eigenmode",
            quantity="relative omega^2 error and residual",
            tolerance=1.0e-5,
            citation="Zero-gradient block finite-difference validation for the global radial Boozer-harmonic shear-Alfven envelope operator",
            command="python -m pytest tests/test_global_modes.py tests/test_validation_registry.py",
            tags=("global", "eigenmode", "3d", "analytic", "boundary"),
            _runner=_gate_global_3d_block_neumann_spectrum,
        ),
        ValidationGate(
            gate_id="global_3d_block_density_gradient",
            title="Global radial 3-D block density sensitivity",
            layer="differentiability",
            quantity="relative autodiff/analytic gradient error",
            tolerance=1.0e-4,
            citation="JAX autodiff consistency check for the block radial generalized eigenproblem rho^(-1/2) scaling",
            command="python -m pytest tests/test_global_modes.py tests/test_validation_registry.py",
            tags=("global", "eigenmode", "3d", "autodiff"),
            _runner=_gate_global_3d_block_density_gradient,
        ),
        ValidationGate(
            gate_id="global_3d_block_eigenvalue_sensitivity",
            title="Global radial 3-D block eigenvalue adjoint",
            layer="differentiability",
            quantity="relative adjoint/autodiff gradient error",
            tolerance=2.0e-6,
            citation="Hermitian generalized-eigenvalue sensitivity x^H(dK-lambda dM)x/(x^H M x) for isolated block radial shear-Alfven eigenpairs",
            command="python -m pytest tests/test_metric.py tests/test_global_modes.py tests/test_validation_registry.py",
            tags=("global", "eigenmode", "3d", "autodiff", "adjoint"),
            _runner=_gate_global_3d_block_eigenvalue_sensitivity,
        ),
        ValidationGate(
            gate_id="global_3d_block_custom_jvp_eigenvalue",
            title="Global radial 3-D block custom-JVP eigenvalue",
            layer="differentiability",
            quantity="relative custom-JVP/adjoint gradient error",
            tolerance=2.0e-6,
            citation="Custom-JVP isolated Hermitian generalized-eigenvalue derivative for block radial shear-Alfven eigenpairs",
            command="python -m pytest tests/test_metric.py tests/test_global_modes.py tests/test_validation_registry.py",
            tags=("global", "eigenmode", "3d", "autodiff", "adjoint", "custom-jvp"),
            _runner=_gate_global_3d_block_custom_jvp_eigenvalue,
        ),
        ValidationGate(
            gate_id="boozer_global_custom_jvp_design_gradient",
            title="Boozer global custom-JVP design gradient",
            layer="differentiability",
            quantity="relative custom-JVP/finite-difference gradient error",
            tolerance=5.0e-3,
            citation="Custom-JVP isolated Hermitian generalized-eigenvalue derivative applied to Boozer global-mode design variables",
            command="python -m pytest tests/test_optimization.py tests/test_validation_registry.py",
            tags=("boozer", "global", "eigenmode", "autodiff", "custom-jvp", "optimization"),
            _runner=_gate_boozer_global_custom_jvp_design_gradient,
        ),
        ValidationGate(
            gate_id="analytic_tae_gap_center",
            title="Analytic TAE gap-center crossing",
            layer="gap-physics",
            quantity="relative rotational-transform crossing error",
            tolerance=2.0e-2,
            citation="Large-aspect-ratio TAE m/m+1 crossing condition iota=2n/(2m+1)",
            command="python -m pytest tests/test_gap_models.py tests/test_validation_registry.py",
            tags=("gap", "tae", "analytic"),
            _runner=_gate_analytic_tae_gap_center,
        ),
        ValidationGate(
            gate_id="basic_gap_map_ordering",
            title="Basic BAE/TAE/EAE gap-map ordering",
            layer="gap-physics",
            quantity="relative q-location error and frequency ordering",
            tolerance=2.0e-2,
            citation="Large-aspect-ratio continuum: TAE m/m+1 gap, EAE m/m+2 gap, and finite-beta BAE accumulation scale",
            command="python -m pytest tests/test_gap_models.py tests/test_examples.py tests/test_validation_registry.py",
            tags=("gap", "tae", "eae", "bae", "analytic"),
            _runner=_gate_basic_gap_map_ordering,
        ),
        ValidationGate(
            gate_id="bae_temperature_scaling",
            title="BAE accumulation temperature scaling",
            layer="gap-physics",
            quantity="relative sqrt(T_i) scaling error",
            tolerance=1.0e-6,
            citation="Fluid-order beta-induced Alfvén-acoustic accumulation frequency omega_BAE=sqrt(7/4+Te/Ti) vti/(qR)",
            command="python -m pytest tests/test_gap_models.py tests/test_validation_registry.py",
            tags=("gap", "bae", "analytic"),
            _runner=_gate_bae_temperature_scaling,
        ),
        ValidationGate(
            gate_id="fast_ion_resonance_peak",
            title="Reduced fast-ion resonance peak",
            layer="fast-ion",
            quantity="resonant drive ordering",
            tolerance=1.0e-12,
            citation="Energetic-particle wave-particle resonance condition omega=n omega_phi + l omega_theta",
            command="python -m pytest tests/test_fast_ions.py tests/test_validation_registry.py",
            tags=("fast-ion", "resonance", "analytic"),
            _runner=_gate_fast_ion_resonance_peak,
        ),
        ValidationGate(
            gate_id="fast_ion_eigenfunction_drive_localization",
            title="Mode-resolved fast-ion drive localization",
            layer="fast-ion",
            quantity="edge/center drive ratio",
            tolerance=3.5e-1,
            citation="Perturbative energetic-particle drive weighted by global eigenfunction intensity and resonance overlap",
            command="python -m pytest tests/test_fast_ions.py tests/test_validation_registry.py",
            tags=("fast-ion", "eigenfunction", "resonance", "analytic"),
            _runner=_gate_fast_ion_eigenfunction_drive_localization,
        ),
        ValidationGate(
            gate_id="boozer_harmonic_projection",
            title="Boozer harmonic projection",
            layer="boozer",
            quantity="relative projection and orthogonality error",
            tolerance=2.0e-6,
            citation="Boozer straight-field-line Fourier representation used by BOOZ_XFORM and stellarator continuum calculations",
            command="python -m pytest tests/test_boozer.py tests/test_validation_registry.py",
            tags=("boozer", "harmonics", "analytic"),
            _runner=_gate_boozer_harmonic_projection,
        ),
        ValidationGate(
            gate_id="boozer_straight_field_line_phase",
            title="Boozer straight-field-line phase",
            layer="boozer",
            quantity="relative phase-slope error",
            tolerance=1.0e-4,
            citation="Straight-field-line Boozer convention theta=theta0+iota*zeta and phase m*theta-n*zeta",
            command="python -m pytest tests/test_boozer.py tests/test_validation_registry.py",
            tags=("boozer", "straight-field-line", "analytic"),
            _runner=_gate_boozer_straight_field_line_phase,
        ),
        ValidationGate(
            gate_id="boozer_constant_field_continuum",
            title="Boozer constant-field continuum limit",
            layer="boozer",
            quantity="relative frequency error",
            tolerance=1.0e-5,
            citation="Boozer-space reduced continuum matrix must recover omega=vA|n-m*iota|/R in the constant-field limit",
            command="python -m pytest tests/test_boozer.py tests/test_validation_registry.py",
            tags=("boozer", "continuum", "analytic"),
            _runner=_gate_boozer_constant_field_continuum,
        ),
        ValidationGate(
            gate_id="boozer_sideband_coupling_hermitian",
            title="Boozer sideband coupling Hermiticity",
            layer="boozer",
            quantity="Hermitian error and sideband-coupling threshold",
            tolerance=1.0e-6,
            citation="Boozer B^2 Fourier harmonics couple shear-Alfven sidebands through the Hermitian energy principle",
            command="python -m pytest tests/test_boozer.py tests/test_validation_registry.py",
            tags=("boozer", "continuum", "coupling"),
            _runner=_gate_boozer_sideband_coupling_hermitian,
        ),
        ValidationGate(
            gate_id="boozer_jacobian_identity",
            title="Boozer Jacobian identity",
            layer="boozer",
            quantity="relative J_B |B|^2 flux-function error",
            tolerance=2.0e-6,
            citation="Boozer-coordinate Jacobian identity sqrt(g)=(G+iota*I)/B^2",
            command="python -m pytest tests/test_boozer.py tests/test_validation_registry.py",
            tags=("boozer", "jacobian", "metric"),
            _runner=_gate_boozer_jacobian_identity,
        ),
        ValidationGate(
            gate_id="boozer_metric_constant_field_continuum",
            title="Boozer metric constant-field continuum limit",
            layer="boozer",
            quantity="relative frequency error",
            tolerance=1.0e-5,
            citation="Boozer-Jacobian weighted reduced continuum matrix must recover omega=vA|n-m*iota|/R",
            command="python -m pytest tests/test_boozer.py tests/test_validation_registry.py",
            tags=("boozer", "jacobian", "continuum", "analytic"),
            _runner=_gate_boozer_metric_constant_field_continuum,
        ),
        ValidationGate(
            gate_id="boozer_metric_mass_coupling",
            title="Boozer Jacobian mass coupling",
            layer="boozer",
            quantity="Hermitian error and weighted-coupling thresholds",
            tolerance=1.0e-5,
            citation="Boozer Jacobian weighting puts |B| variation into the mass weight while J_B B^2 is a flux function",
            command="python -m pytest tests/test_boozer.py tests/test_validation_registry.py",
            tags=("boozer", "jacobian", "continuum", "coupling"),
            _runner=_gate_boozer_metric_mass_coupling,
        ),
        ValidationGate(
            gate_id="boozer_3d_constant_field_continuum",
            title="3-D Boozer constant-field continuum limit",
            layer="boozer",
            quantity="relative frequency error",
            tolerance=1.0e-5,
            citation="Boozer straight-field-line derivative k_parallel=(n-m*iota)|B|/(G+iota I) recovers the large-aspect-ratio analytic limit",
            command="python -m pytest tests/test_boozer.py tests/test_validation_registry.py",
            tags=("boozer", "continuum", "analytic", "3d"),
            _runner=_gate_boozer_3d_constant_field_continuum,
        ),
        ValidationGate(
            gate_id="boozmn_geometry_jacobian_identity",
            title="Real boozmn Jacobian identity",
            layer="boozer",
            quantity="relative sqrt(g)|B|^2 flux-function error",
            tolerance=2.0e-5,
            citation="Boozer-coordinate Jacobian identity sqrt(g)=(G+iota I)/B^2; checked against booz_xform_jax boozmn fixtures",
            command="python -m pytest -m requires_booz_xform_jax tests/test_boozer.py tests/test_validation_registry.py",
            tags=("boozer", "jacobian", "boozmn", "fixture"),
            _runner=_gate_boozmn_geometry_jacobian_identity,
        ),
        ValidationGate(
            gate_id="booz_xform_jax_live_geometry_adapter",
            title="Live booz_xform_jax geometry adapter",
            layer="boozer",
            quantity="relative asymmetric harmonic parity error",
            tolerance=5.0e-9,
            citation="BOOZ_XFORM asymmetric boozmn spectra provide a regression target for the live VMEC-to-Boozer JAX adapter",
            command="python -m pytest -m requires_booz_xform_jax tests/test_boozer.py tests/test_validation_registry.py",
            tags=("boozer", "jacobian", "boozmn", "booz_xform_jax", "asymmetry"),
            _runner=_gate_booz_xform_jax_live_geometry_adapter,
        ),
        ValidationGate(
            gate_id="booz_xform_jax_in_memory_geometry_adapter",
            title="In-memory booz_xform_jax geometry adapter",
            layer="boozer",
            quantity="relative asymmetric harmonic parity error",
            tolerance=5.0e-9,
            citation="In-memory VMEC wout objects should produce the same Boozer harmonics as file-backed BOOZ_XFORM/booz_xform_jax references",
            command="python -m pytest -m requires_booz_xform_jax tests/test_boozer.py tests/test_validation_registry.py",
            tags=("boozer", "jacobian", "boozmn", "booz_xform_jax", "vmec_jax", "in-memory"),
            _runner=_gate_booz_xform_jax_in_memory_geometry_adapter,
        ),
        ValidationGate(
            gate_id="boozmn_embedding_jacobian_shape",
            title="Real boozmn embedding-Jacobian shape",
            layer="boozer",
            quantity="relative shape error after per-surface scaling",
            tolerance=4.0e-2,
            citation="Boozer embedding and Jacobian harmonics from booz_xform_jax/BOOZ_XFORM must agree up to normalized-radial-coordinate scaling",
            command="python -m pytest -m requires_booz_xform_jax tests/test_boozer.py tests/test_validation_registry.py",
            tags=("boozer", "metric", "boozmn", "fixture"),
            _runner=_gate_boozmn_embedding_jacobian_shape,
        ),
        ValidationGate(
            gate_id="boozmn_3d_continuum_finite",
            title="Real boozmn 3-D local continuum smoke test",
            layer="boozer",
            quantity="Hermitian error and finite-output check",
            tolerance=2.0e-5,
            citation="Local 3-D Boozer shear-Alfven Galerkin matrix with real booz_xform_jax geometry harmonics",
            command="python -m pytest -m requires_booz_xform_jax tests/test_boozer.py tests/test_validation_registry.py",
            tags=("boozer", "continuum", "boozmn", "fixture", "3d"),
            _runner=_gate_boozmn_3d_continuum_finite,
        ),
        ValidationGate(
            gate_id="boozmn_global_3d_eigenmode_finite",
            title="Real boozmn global radial 3-D eigenmode smoke test",
            layer="global-eigenmode",
            quantity="maximum residual and finite-output check",
            tolerance=2.0e-5,
            citation="Block radial generalized eigenproblem built from local 3-D Boozer shear-Alfven matrices and Dirichlet radial boundaries",
            command="python -m pytest -m requires_booz_xform_jax tests/test_boozer.py tests/test_validation_registry.py",
            tags=("boozer", "global", "eigenmode", "boozmn", "fixture", "3d"),
            _runner=_gate_boozmn_global_3d_eigenmode_finite,
        ),
        ValidationGate(
            gate_id="boozmn_metric_radial_stiffness_hermitian",
            title="Real boozmn metric radial stiffness",
            layer="global-eigenmode",
            quantity="Hermitian error and positive-semidefinite check",
            tolerance=2.0e-5,
            citation="Ideal-MHD radial derivative energy <|J||grad s|^2|B|^2 phi_i^* phi_j/mu0> in Boozer coordinates",
            command="python -m pytest -m requires_booz_xform_jax tests/test_boozer.py tests/test_validation_registry.py",
            tags=("boozer", "global", "metric", "boozmn", "fixture", "3d"),
            _runner=_gate_boozmn_metric_radial_stiffness_hermitian,
        ),
        ValidationGate(
            gate_id="stellgap_ctok_local_continuum",
            title="STELLGAP CTOK local continuum comparison",
            layer="cross-code",
            quantity="p95 relative branch error",
            tolerance=4.0e-2,
            citation="ORNL STELLGAP CTOK public test problem compared against FAX real-boozmn local 3-D Boozer continuum",
            command="python -m pytest tests/test_validation_registry.py",
            tags=("cross-code", "stellgap", "boozer", "continuum"),
            _runner=_gate_stellgap_ctok_local_continuum,
        ),
        ValidationGate(
            gate_id="stellgap_public_postprocess_parity",
            title="STELLGAP public postprocess parity",
            layer="cross-code",
            quantity="maximum alfven_spec-to-alfven_post relative error",
            tolerance=1.0e-5,
            citation="ORNL STELLGAP public CTOK alfven_spec/alfven_post artifacts and STELLGAP post_process.f convention",
            command="python -m pytest tests/test_validation_registry.py",
            tags=("cross-code", "stellgap", "artifact", "postprocess"),
            _runner=_gate_stellgap_public_postprocess_parity,
        ),
        ValidationGate(
            gate_id="ae3d_public_eigenvalue_artifact",
            title="AE3D public eigenvalue artifact residual",
            layer="cross-code",
            quantity="maximum reported eigenpair residual",
            tolerance=1.0e-2,
            citation="ORNL AE3D public test-problem answer table; parser/residual smoke gate before full FAX-vs-AE3D parity",
            command="python -m pytest tests/test_validation_registry.py",
            tags=("cross-code", "ae3d", "artifact", "eigenmode"),
            _runner=_gate_ae3d_public_eigenvalue_artifact,
        ),
        ValidationGate(
            gate_id="ae3d_public_omega2_eigenvalue_parity",
            title="AE3D public omega2 eigenvalue convention",
            layer="cross-code",
            quantity="maximum omega2-to-egn_values relative error",
            tolerance=1.0e-5,
            citation="ORNL AE3D public omega2.dat and egn_values.dat artifacts; positive entries are square-rooted and negative entries are preserved",
            command="python -m pytest tests/test_validation_registry.py",
            tags=("cross-code", "ae3d", "artifact", "eigenvalue"),
            _runner=_gate_ae3d_public_omega2_eigenvalue_parity,
        ),
        ValidationGate(
            gate_id="ae3d_public_matrix_symmetry",
            title="AE3D public sparse matrix symmetry",
            layer="cross-code",
            quantity="relative sparse matrix asymmetry",
            tolerance=1.0e-5,
            citation="ORNL AE3D public a_matrix.dat/b_matrix.dat generalized-eigenproblem artifacts",
            command="python -m pytest tests/test_validation_registry.py",
            tags=("cross-code", "ae3d", "artifact", "matrix"),
            _runner=_gate_ae3d_public_matrix_symmetry,
        ),
        ValidationGate(
            gate_id="ae3d_public_matrix_eigenmode_parity",
            title="AE3D public matrix/eigenmode parity",
            layer="cross-code",
            quantity="maximum generalized eigenpair residual/Rayleigh error",
            tolerance=5.0e-6,
            citation="ORNL AE3D public a_matrix.dat/b_matrix.dat/egn_mode_asci.dat artifacts; Spong, D'Azevedo & Todo, Phys. Plasmas 17, 022106 (2010)",
            command="python -m pytest tests/test_validation_registry.py",
            tags=("cross-code", "ae3d", "matrix", "eigenmode", "parity"),
            _runner=_gate_ae3d_public_matrix_eigenmode_parity,
        ),
        ValidationGate(
            gate_id="ae3d_public_dense_spectrum_parity",
            title="AE3D public dense spectrum parity",
            layer="cross-code",
            quantity="maximum dense-solver eigenvalue relative error",
            tolerance=2.0e-6,
            citation="ORNL AE3D public a_matrix.dat/b_matrix.dat/egn_mode_asci.dat artifacts; independent SciPy/LAPACK generalized eigensolve",
            command="python -m pytest tests/test_validation_registry.py tests/test_cross_code.py",
            tags=("cross-code", "ae3d", "matrix", "spectrum", "parity"),
            _runner=_gate_ae3d_public_dense_spectrum_parity,
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
            gate_id="landreman_paul_vmec_jax_boozer_in_memory",
            title="Landreman-Paul QA vmec_jax-to-Boozer bridge",
            layer="adapter",
            quantity="relative Boozer Jacobian identity error",
            tolerance=1.0e-2,
            citation=(
                "Landreman & Paul, Phys. Rev. Lett. 128, 035001 (2022); "
                "one-iteration vmec_jax/booz_xform_jax/FAX in-memory integration smoke gate"
            ),
            command="python -m pytest -m requires_vmec_jax tests/test_boozer.py tests/test_validation_registry.py",
            tags=("vmec_jax", "booz_xform_jax", "landreman-paul", "adapter", "boozer", "in-memory"),
            _runner=_gate_landreman_paul_vmec_jax_boozer_in_memory,
        ),
        ValidationGate(
            gate_id="landreman_paul_vmec_jax_boozer_truncation_convergence",
            title="Landreman-Paul QA vmec_jax Boozer truncation convergence",
            layer="convergence",
            quantity="maximum field/frequency/Jacobian truncation error",
            tolerance=1.0e-2,
            citation=(
                "Landreman & Paul, Phys. Rev. Lett. 128, 035001 (2022); "
                "Boozer truncation convergence smoke gate on the live vmec_jax/booz_xform_jax/FAX bridge"
            ),
            command="python -m pytest -m requires_vmec_jax tests/test_boozer.py tests/test_validation_registry.py",
            tags=("vmec_jax", "booz_xform_jax", "landreman-paul", "convergence", "boozer", "in-memory"),
            _runner=_gate_landreman_paul_vmec_jax_boozer_truncation_convergence,
        ),
        ValidationGate(
            gate_id="landreman_paul_vmec_jax_boozer_operator_angular_convergence",
            title="Landreman-Paul QA vmec_jax Boozer operator angular convergence",
            layer="convergence",
            quantity="relative local 3-D frequency error and Hermitian error",
            tolerance=5.0e-3,
            citation=(
                "Landreman & Paul, Phys. Rev. Lett. 128, 035001 (2022); "
                "Boozer local-operator angular quadrature convergence smoke gate on the live bridge"
            ),
            command="python -m pytest -m requires_vmec_jax tests/test_boozer.py tests/test_validation_registry.py",
            tags=("vmec_jax", "booz_xform_jax", "landreman-paul", "convergence", "boozer", "operator"),
            _runner=_gate_landreman_paul_vmec_jax_boozer_operator_angular_convergence,
        ),
        ValidationGate(
            gate_id="landreman_paul_vmec_jax_solve_convergence",
            title="Landreman-Paul QA vmec_jax solve convergence ladder",
            layer="convergence",
            quantity="final VMEC residual ratio and Boozer/FAX reference error",
            tolerance=4.0e-2,
            citation=(
                "Landreman & Paul, Phys. Rev. Lett. 128, 035001 (2022); "
                "VMEC-JAX residual convergence propagated through booz_xform_jax and FAX local 3-D continuum"
            ),
            command="python -m pytest -m requires_vmec_jax tests/test_validation_registry.py",
            tags=("vmec_jax", "booz_xform_jax", "landreman-paul", "convergence", "boozer", "solve"),
            _runner=_gate_landreman_paul_vmec_jax_solve_convergence,
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
