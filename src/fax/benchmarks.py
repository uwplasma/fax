"""Benchmark helpers for local performance and physics-parity packets."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import platform
import resource
import sys
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class BenchmarkResult:
    """Timing summary for a callable benchmark."""

    name: str
    repeats: int
    warmups: int
    min_seconds: float
    median_seconds: float
    max_seconds: float


@dataclass(frozen=True)
class BenchmarkFamilyResult:
    """Timing and physics-gate summary for one named benchmark family.

    A benchmark family is useful only when the timed kernel is tied to an
    accuracy or parity metric. The ``gate_passed`` flag prevents standalone
    runtime numbers from being interpreted as physics validation.
    """

    family: str
    label: str
    status: str
    gate_passed: bool
    physics_metric: str
    metric_value: float | None
    tolerance: float | None
    cold_seconds: float | None
    warm_result: BenchmarkResult | None
    process_max_rss_bytes: int | None
    details: dict[str, Any]
    environment: dict[str, Any]


BENCHMARK_FAMILY_DESCRIPTIONS: dict[str, str] = {
    "analytic-local": "profile-limit continuum matrix with analytic density-scaling gate",
    "global-block": "global radial block eigensolve with exact finite-difference spectrum gate",
    "global-matrix-free": "JAX matrix-free global radial operator with dense/Rayleigh parity gates",
    "global-krylov": "sparse Krylov global radial block eigensolve with exact-spectrum gate",
    "ae3d-artifact": "public AE3D generalized matrix dense-spectrum parity gate",
    "stellgap-ctok": "public STELLGAP CTOK local Boozer continuum parity gate",
    "stellgap-executable": "bounded local STELLGAP CTOK executable attempt with runtime/status capture",
    "vmec-boozer": "live vmec_jax solve through booz_xform_jax and FAX local 3-D continuum gate",
}


def _block_until_ready(value: Any) -> Any:
    if hasattr(value, "block_until_ready"):
        return value.block_until_ready()
    if isinstance(value, (tuple, list)):
        for item in value:
            _block_until_ready(item)
    if isinstance(value, dict):
        for item in value.values():
            _block_until_ready(item)
    if is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            _block_until_ready(getattr(value, item.name))
    return value


def time_callable(
    name: str,
    func: Callable[[], Any],
    *,
    warmups: int = 1,
    repeats: int = 5,
) -> BenchmarkResult:
    """Time a no-argument callable with JAX ``block_until_ready`` support."""
    for _ in range(int(warmups)):
        _block_until_ready(func())

    timings: list[float] = []
    for _ in range(int(repeats)):
        start = time.perf_counter()
        _block_until_ready(func())
        timings.append(time.perf_counter() - start)

    data = np.asarray(timings, dtype=float)
    return BenchmarkResult(
        name=name,
        repeats=int(repeats),
        warmups=int(warmups),
        min_seconds=float(np.min(data)),
        median_seconds=float(np.median(data)),
        max_seconds=float(np.max(data)),
    )


def known_benchmark_families() -> tuple[str, ...]:
    """Return the named benchmark families available from the CLI."""
    return tuple(BENCHMARK_FAMILY_DESCRIPTIONS)


def benchmark_environment() -> dict[str, Any]:
    """Return reproducibility metadata for local benchmark packets."""
    try:
        x64_enabled = bool(jax.config.read("jax_enable_x64"))
    except Exception:  # pragma: no cover - defensive for older JAX
        x64_enabled = None
    try:
        devices = [str(device) for device in jax.local_devices()]
    except Exception:  # pragma: no cover - defensive for unusual backends
        devices = []
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "jax_backend": jax.default_backend(),
        "jax_enable_x64": x64_enabled,
        "jax_device_count": len(devices),
        "jax_devices": devices,
    }


def run_named_benchmark_family(
    family: str,
    *,
    repeats: int = 3,
    warmups: int = 1,
) -> BenchmarkFamilyResult:
    """Run one named physics benchmark family.

    The external-code families return ``status='skipped'`` when the public
    artifacts are not present on the local machine. This keeps the benchmark
    packet portable while allowing Rogerio's local clones to provide parity
    timings when available.
    """
    family = str(family)
    repeats = int(repeats)
    warmups = int(warmups)
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if warmups < 0:
        raise ValueError("warmups must be non-negative")
    if family == "analytic-local":
        return _run_analytic_local_family(repeats=repeats, warmups=warmups)
    if family == "global-block":
        return _run_global_block_family(repeats=repeats, warmups=warmups)
    if family == "global-matrix-free":
        return _run_global_matrix_free_family(repeats=repeats, warmups=warmups)
    if family == "global-krylov":
        return _run_global_krylov_family(repeats=repeats, warmups=warmups)
    if family == "ae3d-artifact":
        return _run_ae3d_artifact_family(repeats=repeats, warmups=warmups)
    if family == "stellgap-ctok":
        return _run_stellgap_ctok_family(repeats=repeats, warmups=warmups)
    if family == "stellgap-executable":
        return _run_stellgap_executable_family(repeats=repeats, warmups=warmups)
    if family == "vmec-boozer":
        return _run_vmec_boozer_family(repeats=repeats, warmups=warmups)
    known = ", ".join(known_benchmark_families())
    raise ValueError(f"unknown benchmark family {family!r}; expected one of: {known}")


def _timed_family_call(
    name: str,
    func: Callable[[], Any],
    *,
    repeats: int,
    warmups: int,
) -> tuple[float, BenchmarkResult, Any]:
    start = time.perf_counter()
    cold_value = func()
    _block_until_ready(cold_value)
    cold_seconds = time.perf_counter() - start
    warm_result = time_callable(name, func, warmups=warmups, repeats=repeats)
    return cold_seconds, warm_result, cold_value


def process_max_rss_bytes() -> int:
    """Return current process maximum resident set size in bytes."""
    max_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return max_rss if sys.platform == "darwin" else max_rss * 1024


def _family_result(
    *,
    family: str,
    label: str,
    status: str,
    gate_passed: bool,
    physics_metric: str,
    metric_value: float | None,
    tolerance: float | None,
    cold_seconds: float | None,
    warm_result: BenchmarkResult | None,
    details: dict[str, Any] | None = None,
) -> BenchmarkFamilyResult:
    return BenchmarkFamilyResult(
        family=family,
        label=label,
        status=status,
        gate_passed=bool(gate_passed),
        physics_metric=physics_metric,
        metric_value=metric_value,
        tolerance=tolerance,
        cold_seconds=cold_seconds,
        warm_result=warm_result,
        process_max_rss_bytes=process_max_rss_bytes(),
        details=dict(details or {}),
        environment=benchmark_environment(),
    )


def _run_analytic_local_family(*, repeats: int, warmups: int) -> BenchmarkFamilyResult:
    from .continuum import continuum_branch_matrix
    from .equilibrium import analytic_circular_tokamak
    from .modes import stellarator_mode_family

    geometry = analytic_circular_tokamak(radial_points=160, iota_axis=0.24, iota_edge=0.82)
    family = stellarator_mode_family(
        poloidal_modes=range(1, 7),
        base_n=1,
        field_periods=geometry.field_periods,
        sidebands=(-1, 0, 1),
    )
    harmonics = family.as_array()

    def kernel():
        return continuum_branch_matrix(geometry, harmonics)

    cold_seconds, warm_result, branches = _timed_family_call(
        "analytic_local_continuum",
        kernel,
        repeats=repeats,
        warmups=warmups,
    )

    density_scale = 1.37
    scaled = continuum_branch_matrix(geometry.with_density_scale(density_scale), harmonics)
    expected = jnp.asarray(branches) / jnp.sqrt(density_scale)
    error = float(jnp.max(jnp.abs(scaled - expected) / jnp.maximum(jnp.abs(expected), 1.0e-30)))
    tolerance = 5.0e-12
    return _family_result(
        family="analytic-local",
        label=BENCHMARK_FAMILY_DESCRIPTIONS["analytic-local"],
        status="completed",
        gate_passed=error < tolerance,
        physics_metric="max density-scaling relative error",
        metric_value=error,
        tolerance=tolerance,
        cold_seconds=cold_seconds,
        warm_result=warm_result,
        details={
            "radial_points": int(geometry.s.shape[0]),
            "harmonic_count": len(family.harmonics),
            "frequency_min_hz": float(jnp.min(branches)),
            "frequency_max_hz": float(jnp.max(branches)),
        },
    )


def _run_global_block_family(*, repeats: int, warmups: int) -> BenchmarkFamilyResult:
    from .global_modes import (
        constant_block_problem_discrete_omega2,
        constant_radial_block_problem,
        solve_radial_block_eigenmodes,
    )

    problem = constant_radial_block_problem(
        radial_points=52,
        continuum_frequency_hz=(90_000.0, 135_000.0, 210_000.0),
        radial_coupling_omega2=(2.0e9, 3.5e9, 4.2e9),
        boundary_condition="dirichlet",
    )

    def kernel():
        return solve_radial_block_eigenmodes(problem)

    cold_seconds, warm_result, result = _timed_family_call(
        "global_radial_block",
        kernel,
        repeats=repeats,
        warmups=warmups,
    )

    expected = constant_block_problem_discrete_omega2(problem)
    count = min(24, int(result.omega2.shape[0]))
    error = jnp.max(jnp.abs(result.omega2[:count] - expected[:count]) / jnp.maximum(jnp.abs(expected[:count]), 1.0e-30))
    tolerance = 6.0e-6
    residual_max = float(jnp.max(result.residuals[:count]))
    return _family_result(
        family="global-block",
        label=BENCHMARK_FAMILY_DESCRIPTIONS["global-block"],
        status="completed",
        gate_passed=float(error) < tolerance and residual_max < 1.0e-5,
        physics_metric="max exact discrete-spectrum relative error",
        metric_value=float(error),
        tolerance=tolerance,
        cold_seconds=cold_seconds,
        warm_result=warm_result,
        details={
            "radial_points": int(problem.s.shape[0]),
            "harmonic_count": int(problem.harmonics.shape[0]),
            "matrix_dimension": int(result.stiffness.shape[0]),
            "checked_modes": count,
            "max_residual": residual_max,
        },
    )


def _run_global_krylov_family(*, repeats: int, warmups: int) -> BenchmarkFamilyResult:
    from .global_modes import (
        constant_block_problem_discrete_omega2,
        constant_radial_block_problem,
        solve_radial_block_eigenmodes_sparse,
    )

    problem = constant_radial_block_problem(
        radial_points=220,
        continuum_frequency_hz=(90_000.0, 135_000.0, 210_000.0, 275_000.0),
        radial_coupling_omega2=(2.0e9, 3.5e9, 4.2e9, 5.0e9),
        boundary_condition="dirichlet",
    )
    mode_count = 24

    def kernel():
        return solve_radial_block_eigenmodes_sparse(problem, mode_count=mode_count)

    cold_seconds, warm_result, result = _timed_family_call(
        "global_radial_sparse_krylov",
        kernel,
        repeats=repeats,
        warmups=warmups,
    )

    expected = constant_block_problem_discrete_omega2(problem)
    error = jnp.max(
        jnp.abs(result.omega2[:mode_count] - expected[:mode_count])
        / jnp.maximum(jnp.abs(expected[:mode_count]), 1.0e-30)
    )
    tolerance = 2.0e-5
    residual_max = float(jnp.max(result.residuals[:mode_count]))
    return _family_result(
        family="global-krylov",
        label=BENCHMARK_FAMILY_DESCRIPTIONS["global-krylov"],
        status="completed",
        gate_passed=float(error) < tolerance and residual_max < 2.0e-8,
        physics_metric="max sparse Krylov exact-spectrum relative error",
        metric_value=float(error),
        tolerance=tolerance,
        cold_seconds=cold_seconds,
        warm_result=warm_result,
        details={
            "radial_points": int(problem.s.shape[0]),
            "harmonic_count": int(problem.harmonics.shape[0]),
            "matrix_dimension": int(result.stiffness.shape[0]),
            "matrix_nonzeros": int(result.stiffness.nnz),
            "checked_modes": mode_count,
            "max_residual": residual_max,
        },
    )


def _run_global_matrix_free_family(*, repeats: int, warmups: int) -> BenchmarkFamilyResult:
    from .global_modes import (
        assemble_radial_block_matrices,
        constant_block_problem_discrete_omega2,
        constant_radial_block_problem,
        radial_block_mass_matvec,
        radial_block_rayleigh_omega2,
        radial_block_stiffness_matvec,
    )

    problem = constant_radial_block_problem(
        radial_points=220,
        continuum_frequency_hz=(90_000.0, 135_000.0, 210_000.0, 275_000.0),
        radial_coupling_omega2=(2.0e9, 3.5e9, 4.2e9, 5.0e9),
        boundary_condition="dirichlet",
    )
    stiffness, mass = assemble_radial_block_matrices(problem)
    active_count = int(problem.s.shape[0]) - 2
    harmonic_count = int(problem.harmonics.shape[0])
    envelope = jnp.sin(jnp.arange(1, active_count + 1, dtype=problem.s.dtype) * jnp.pi / (active_count + 1))
    vector_blocks = jnp.zeros((active_count, harmonic_count), dtype=stiffness.dtype).at[:, 0].set(envelope)
    vector = jnp.ravel(vector_blocks)

    @jax.jit
    def kernel():
        return (
            radial_block_stiffness_matvec(problem, vector),
            radial_block_mass_matvec(problem, vector),
            radial_block_rayleigh_omega2(problem, vector),
        )

    cold_seconds, warm_result, result = _timed_family_call(
        "global_radial_matrix_free_apply",
        kernel,
        repeats=repeats,
        warmups=warmups,
    )

    stiffness_vector, mass_vector, rayleigh = result
    dense_stiffness_vector = stiffness @ vector
    dense_mass_vector = mass @ vector
    stiffness_error = jnp.linalg.norm(stiffness_vector - dense_stiffness_vector) / jnp.maximum(
        jnp.linalg.norm(dense_stiffness_vector), 1.0e-30
    )
    mass_error = jnp.linalg.norm(mass_vector - dense_mass_vector) / jnp.maximum(jnp.linalg.norm(dense_mass_vector), 1.0e-30)
    expected = constant_block_problem_discrete_omega2(problem)[0]
    rayleigh_error = jnp.abs(rayleigh - expected) / jnp.maximum(jnp.abs(expected), 1.0e-30)
    error = float(jnp.max(jnp.asarray([stiffness_error, mass_error, rayleigh_error])))
    tolerance = 1.0e-10
    return _family_result(
        family="global-matrix-free",
        label=BENCHMARK_FAMILY_DESCRIPTIONS["global-matrix-free"],
        status="completed",
        gate_passed=error < tolerance,
        physics_metric="max dense-matvec/Rayleigh relative error",
        metric_value=error,
        tolerance=tolerance,
        cold_seconds=cold_seconds,
        warm_result=warm_result,
        details={
            "radial_points": int(problem.s.shape[0]),
            "harmonic_count": harmonic_count,
            "matrix_dimension": int(stiffness.shape[0]),
            "stiffness_matvec_relative_error": float(stiffness_error),
            "mass_matvec_relative_error": float(mass_error),
            "rayleigh_relative_error": float(rayleigh_error),
        },
    )


def _run_ae3d_artifact_family(*, repeats: int, warmups: int) -> BenchmarkFamilyResult:
    from pathlib import Path

    from .cross_code import ae3d_dense_spectrum_parity

    root = Path("/Users/rogerio/local/AE3D_test_problem")
    a_matrix = root / "a_matrix.dat"
    b_matrix = root / "b_matrix.dat"
    egn_mode = root / "egn_mode_asci.dat"
    if not all(path.exists() for path in (a_matrix, b_matrix, egn_mode)):
        return _family_result(
            family="ae3d-artifact",
            label=BENCHMARK_FAMILY_DESCRIPTIONS["ae3d-artifact"],
            status="skipped",
            gate_passed=False,
            physics_metric="max frequency relative error",
            metric_value=None,
            tolerance=1.0e-6,
            cold_seconds=None,
            warm_result=None,
            details={"reason": f"missing public AE3D artifacts under {root}"},
        )

    def kernel():
        return ae3d_dense_spectrum_parity(
            a_matrix_path=a_matrix,
            b_matrix_path=b_matrix,
            egn_mode_path=egn_mode,
            max_modes=16,
        )

    cold_seconds, warm_result, parity = _timed_family_call(
        "ae3d_dense_spectrum_parity",
        kernel,
        repeats=repeats,
        warmups=warmups,
    )
    tolerance = 1.0e-6
    return _family_result(
        family="ae3d-artifact",
        label=BENCHMARK_FAMILY_DESCRIPTIONS["ae3d-artifact"],
        status="completed",
        gate_passed=parity.max_frequency_relative_error < tolerance,
        physics_metric="max frequency relative error",
        metric_value=parity.max_frequency_relative_error,
        tolerance=tolerance,
        cold_seconds=cold_seconds,
        warm_result=warm_result,
        details={
            "matrix_dimension": parity.matrix_dimension,
            "real_eigenvalue_count": parity.real_eigenvalue_count,
            "checked_modes": parity.count,
            "max_eigenvalue_relative_error": parity.max_eigenvalue_relative_error,
            "p95_frequency_relative_error": parity.p95_frequency_relative_error,
        },
    )


def _run_stellgap_ctok_family(*, repeats: int, warmups: int) -> BenchmarkFamilyResult:
    from pathlib import Path

    from .cross_code import stellgap_ctok_local_continuum_error
    from .units import PROTON_MASS

    boozmn = Path("/Users/rogerio/local/Stellgap_test_inputs/boozmn_ctok.nc")
    alfven_post = Path("/Users/rogerio/local/Stellgap_test_problem/alfven_post")
    if not boozmn.exists() or not alfven_post.exists():
        return _family_result(
            family="stellgap-ctok",
            label=BENCHMARK_FAMILY_DESCRIPTIONS["stellgap-ctok"],
            status="skipped",
            gate_passed=False,
            physics_metric="p95 sorted-spectrum relative error",
            metric_value=None,
            tolerance=4.0e-2,
            cold_seconds=None,
            warm_result=None,
            details={"reason": "missing public STELLGAP CTOK boozmn/alfven_post artifacts"},
        )

    def kernel():
        return stellgap_ctok_local_continuum_error(
            boozmn_path=boozmn,
            alfven_post_path=alfven_post,
            mass_density=2.0 * PROTON_MASS * 1.0e19,
            theta_points=32,
            zeta_points=32,
        )

    cold_seconds, warm_result, comparison = _timed_family_call(
        "stellgap_ctok_local_continuum",
        kernel,
        repeats=repeats,
        warmups=warmups,
    )
    tolerance = 4.0e-2
    return _family_result(
        family="stellgap-ctok",
        label=BENCHMARK_FAMILY_DESCRIPTIONS["stellgap-ctok"],
        status="completed",
        gate_passed=comparison.p95_relative_error < tolerance,
        physics_metric="p95 sorted-spectrum relative error",
        metric_value=comparison.p95_relative_error,
        tolerance=tolerance,
        cold_seconds=cold_seconds,
        warm_result=warm_result,
        details={
            "matched_surface_count": comparison.count,
            "mean_relative_error": comparison.mean_relative_error,
            "max_relative_error": comparison.max_relative_error,
        },
    )


def _run_stellgap_executable_family(*, repeats: int, warmups: int) -> BenchmarkFamilyResult:
    from .cross_code import run_stellgap_ctok_executable

    def kernel():
        return run_stellgap_ctok_executable(timeout_seconds=1800.0)

    start = time.perf_counter()
    first = kernel()
    cold_seconds = time.perf_counter() - start
    if first.status != "completed":
        return _family_result(
            family="stellgap-executable",
            label=BENCHMARK_FAMILY_DESCRIPTIONS["stellgap-executable"],
            status=first.status,
            gate_passed=False,
            physics_metric="external executable completed and produced alfven_spec",
            metric_value=None,
            tolerance=0.0,
            cold_seconds=None if first.status == "skipped" else cold_seconds,
            warm_result=None,
            details={
                "returncode": first.returncode,
                "elapsed_seconds_reported_by_runner": first.elapsed_seconds,
                "output_files": list(first.output_files),
                "stdout_tail": first.stdout_tail,
                "stderr_tail": first.stderr_tail,
                "executable": first.executable,
            },
        )

    warm_result = time_callable("stellgap_ctok_executable", kernel, warmups=warmups, repeats=repeats)
    return _family_result(
        family="stellgap-executable",
        label=BENCHMARK_FAMILY_DESCRIPTIONS["stellgap-executable"],
        status=first.status,
        gate_passed=first.status == "completed",
        physics_metric="external executable completed and produced alfven_spec",
        metric_value=0.0,
        tolerance=0.0,
        cold_seconds=cold_seconds,
        warm_result=warm_result,
        details={
            "returncode": first.returncode,
            "elapsed_seconds_reported_by_runner": first.elapsed_seconds,
            "output_files": list(first.output_files),
            "stdout_tail": first.stdout_tail,
            "stderr_tail": first.stderr_tail,
            "executable": first.executable,
        },
    )


def _run_vmec_boozer_family(*, repeats: int, warmups: int) -> BenchmarkFamilyResult:
    from pathlib import Path

    from .boozer import run_vmec_jax_boozer_solve_convergence
    from .reference_cases import landreman_paul_qa_lowres

    case = landreman_paul_qa_lowres()
    input_path = case.vmec_input
    reference_wout = case.wout
    if input_path is None or reference_wout is None or not Path(input_path).exists() or not Path(reference_wout).exists():
        return _family_result(
            family="vmec-boozer",
            label=BENCHMARK_FAMILY_DESCRIPTIONS["vmec-boozer"],
            status="skipped",
            gate_passed=False,
            physics_metric="VMEC residual/Boozer/FAX convergence max metric",
            metric_value=None,
            tolerance=8.0e-2,
            cold_seconds=None,
            warm_result=None,
            details={"reason": "missing Landreman-Paul vmec_jax input or converged wout"},
        )

    def kernel():
        return run_vmec_jax_boozer_solve_convergence(
            input_path=input_path,
            reference_wout_path=reference_wout,
            max_iters=(1, 6, 12),
            mboz=4,
            nboz=4,
            surfaces=(0.5,),
            field_theta_points=16,
            field_zeta_points=8,
            frequency_theta_points=12,
            frequency_zeta_points=8,
            harmonics=((2, 1), (3, 2)),
            jit=False,
        )

    cold_seconds, warm_result, result = _timed_family_call(
        "vmec_boozer_solve_convergence",
        kernel,
        repeats=repeats,
        warmups=warmups,
    )
    tolerance = 8.0e-2
    metric = result.max_metric
    return _family_result(
        family="vmec-boozer",
        label=BENCHMARK_FAMILY_DESCRIPTIONS["vmec-boozer"],
        status="completed",
        gate_passed=metric < tolerance,
        physics_metric="VMEC residual/Boozer/FAX convergence max metric",
        metric_value=metric,
        tolerance=tolerance,
        cold_seconds=cold_seconds,
        warm_result=warm_result,
        details={
            "max_iters": [int(value) for value in result.max_iters],
            "residual_reduction_ratio": result.residual_reduction_ratio,
            "frequency_error_reduction_ratio": result.frequency_error_reduction_ratio,
            "final_frequency_relative_error": float(result.frequency_relative_errors[-1]),
            "final_field_relative_error": float(jnp.max(result.field_relative_errors[-1])),
        },
    )
