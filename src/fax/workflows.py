"""CLI-facing workflows for validation and continuum artifacts."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import time
from typing import Any, TextIO

import jax
import jax.numpy as jnp
import numpy as np

from .benchmarks import (
    benchmark_environment,
    known_benchmark_families,
    process_max_rss_bytes,
    run_named_benchmark_family,
    time_callable,
)
from .boozer import read_boozmn_geometry
from .config import ConfigError, FaxConfig
from .continuum import continuum_branch_matrix
from .equilibrium import EquilibriumGeometry, analytic_circular_tokamak, from_vmec_wout
from .gaps import ContinuumGapSummary, minimum_gap_summary
from .global_modes import (
    RadialBlockEigenmodeProblem,
    radial_block_problem_from_boozer_geometry,
    radial_problem_from_equilibrium,
    solve_radial_block_eigenmodes,
    solve_radial_block_eigenmodes_sparse,
    solve_radial_eigenmodes,
)
from .metric import (
    MetricGeometry,
    diagonal_continuum_frequencies_hz,
    generalized_eigen_residual_norms,
    generalized_hermitian_eigensystem,
    metric_continuum_matrices,
    metric_from_equilibrium,
    metric_from_vmec_wout,
)
from .modes import ModeFamily, stellarator_mode_family
from .progress import PhaseRecord, ProgressReporter, format_seconds
from .validation import GateResult, registered_validation_gates


@dataclass(frozen=True)
class ValidationRunSummary:
    """Serializable summary for ``fax validate`` runs."""

    case: str
    model: str
    status: str
    message: str
    config: dict[str, Any]
    elapsed_seconds: float
    phases: list[dict[str, Any]]
    gate_count: int
    passed_count: int
    failed_count: int
    skipped_count: int
    tags: list[str]
    output_dir: str | None
    artifacts: dict[str, str]
    results: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-serializable representation."""
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class ContinuumRunSummary:
    """Serializable summary for ``fax continuum`` runs."""

    case: str
    model: str
    status: str
    message: str
    config: dict[str, Any]
    elapsed_seconds: float
    phases: list[dict[str, Any]]
    output_dir: str | None
    artifacts: dict[str, str]
    geometry: dict[str, Any]
    mode_family: dict[str, Any]
    radial_points: int
    harmonic_count: int
    profile_min_frequency_hz: float
    profile_max_frequency_hz: float
    metric_min_frequency_hz: float
    metric_max_frequency_hz: float
    min_gap_hz: float
    min_gap_surface: float
    metric_residual_max: float
    metric_residual_p95: float

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-serializable representation."""
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class ConvergenceRunSummary:
    """Serializable summary for ``fax convergence`` sweeps."""

    case: str
    model: str
    status: str
    message: str
    config: dict[str, Any]
    elapsed_seconds: float
    phases: list[dict[str, Any]]
    output_dir: str | None
    artifacts: dict[str, str]
    geometry: dict[str, Any]
    radial_points: list[int]
    angular_points: list[int]
    mode_max_values: list[int]
    reference: dict[str, Any]
    max_frequency_l2_error: float
    max_residual_p95: float
    records: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-serializable representation."""
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class GlobalEigenmodeRunSummary:
    """Serializable summary for ``fax eigenmodes`` global radial solves."""

    case: str
    model: str
    status: str
    message: str
    config: dict[str, Any]
    elapsed_seconds: float
    phases: list[dict[str, Any]]
    output_dir: str | None
    artifacts: dict[str, str]
    geometry: dict[str, Any]
    mode_family: dict[str, Any]
    operator: str
    radial_points: int
    harmonic_count: int
    boundary_condition: str
    eigensolver: str
    reported_mode_count: int
    min_frequency_hz: float
    max_frequency_hz: float
    residual_max: float
    residual_p95: float

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-serializable representation."""
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class BenchmarkRunSummary:
    """Serializable summary for ``fax benchmark`` runs."""

    case: str
    model: str
    status: str
    message: str
    config: dict[str, Any]
    elapsed_seconds: float
    phases: list[dict[str, Any]]
    output_dir: str | None
    artifacts: dict[str, str]
    cold_seconds: float
    warm_result: dict[str, Any]
    artifact_size_bytes: int
    continuum_summary: dict[str, Any]
    family_results: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-serializable representation."""
        return _json_safe(asdict(self))


def run_validation(
    config: FaxConfig,
    *,
    progress: bool = True,
    stream: TextIO | None = None,
    tags: tuple[str, ...] | None = None,
    write_artifacts: bool = True,
) -> ValidationRunSummary:
    """Run registered validation gates and optionally write report artifacts."""
    config = config.validated()
    _apply_precision(config)
    selected_tags = tuple(tags or ())
    phase_specs = [
        ("Validate configuration", "check TOML schema and selected validation tags", 0.1),
        ("Run validation gates", "execute registered analytic, metric, autodiff, and adapter checks", 0.5),
        ("Write validation artifacts", f"output_dir={config.run.output_dir}", 0.2),
    ]
    reporter = ProgressReporter(
        enabled=progress,
        stream=stream,
        total_estimate_seconds=sum(phase[2] for phase in phase_specs),
    )
    start = time.perf_counter()
    reporter.banner(config)

    _start_phase(reporter, phase_specs[0])
    gate_registry = {gate.gate_id: gate for gate in registered_validation_gates()}
    if selected_tags:
        reporter.log(f"tag filter: {', '.join(selected_tags)}")
    reporter.log(f"registered gates available: {len(gate_registry)}")
    reporter.phase_done(status="done")

    _start_phase(reporter, phase_specs[1])
    selected_tag_set = set(selected_tags)
    selected_gates = [
        gate
        for gate in registered_validation_gates()
        if not selected_tag_set or selected_tag_set.intersection(gate.tags)
    ]
    results_list: list[GateResult] = []
    for index, gate in enumerate(selected_gates, start=1):
        reporter.log(
            f"gate {index}/{len(selected_gates)} START {gate.gate_id} "
            f"[{gate.layer}; tol={gate.tolerance:.3e}]"
        )
        gate_start = time.perf_counter()
        result = gate.run()
        results_list.append(result)
        elapsed = format_seconds(time.perf_counter() - gate_start)
        reporter.log(f"gate {index}/{len(selected_gates)} DONE  {_format_gate_line(result)} in {elapsed}")
    results = tuple(results_list)
    passed_count = sum(result.status == "passed" for result in results)
    failed_count = sum(result.status == "failed" for result in results)
    skipped_count = sum(result.status == "skipped" for result in results)
    run_status = "failed" if failed_count else "passed"
    if not results:
        run_status = "failed"
    reporter.phase_done(status="done" if run_status == "passed" else "failed")

    artifacts: dict[str, str] = {}
    output_dir: str | None = None
    _start_phase(reporter, phase_specs[2])
    if write_artifacts:
        output_path = _output_dir(config)
        output_path.mkdir(parents=True, exist_ok=True)
        output_dir = str(output_path)
        artifacts = {
            "summary_json": str(output_path / "validation_summary.json"),
            "report_markdown": str(output_path / "validation_report.md"),
        }
        reporter.log(f"writing {artifacts['summary_json']}")
        reporter.log(f"writing {artifacts['report_markdown']}")
    else:
        reporter.log("artifact writing disabled")
    reporter.phase_done(status="done")

    message = _validation_message(
        gate_count=len(results),
        passed_count=passed_count,
        failed_count=failed_count,
        skipped_count=skipped_count,
        tags=selected_tags,
    )
    summary = ValidationRunSummary(
        case=config.run.case,
        model=config.run.model,
        status=run_status,
        message=message,
        config=config.to_dict(),
        elapsed_seconds=time.perf_counter() - start,
        phases=_phase_dicts(reporter.records),
        gate_count=len(results),
        passed_count=passed_count,
        failed_count=failed_count,
        skipped_count=skipped_count,
        tags=list(selected_tags),
        output_dir=output_dir,
        artifacts=artifacts,
        results=[_gate_result_dict(result, gate_registry) for result in results],
    )
    if write_artifacts:
        _write_json(Path(artifacts["summary_json"]), summary.to_dict())
        Path(artifacts["report_markdown"]).write_text(
            _validation_report_markdown(summary),
            encoding="utf-8",
        )
    reporter.summary(summary)
    return summary


def run_convergence(
    config: FaxConfig,
    *,
    progress: bool = True,
    stream: TextIO | None = None,
    radial_points: tuple[int, ...] | None = None,
    angular_points: tuple[int, ...] = (12, 16, 24),
    mode_max_values: tuple[int, ...] = (2, 3, 4),
    write_artifacts: bool = True,
) -> ConvergenceRunSummary:
    """Run radial, angular, and Fourier convergence sweeps for continuum models."""
    config = config.validated()
    _apply_precision(config)
    radial_values = _positive_unique_ints(radial_points or (config.grid.radial_points,), name="radial_points")
    if any(value < 4 for value in radial_values):
        raise ValueError("radial_points values must be at least 4")
    angular_values = _positive_unique_ints(angular_points, name="angular_points")
    mode_values = _positive_unique_ints(mode_max_values, name="mode_max_values")
    phase_specs = [
        ("Validate convergence sweep", "check sweep grids and TOML controls", 0.1),
        ("Resolve equilibrium geometry", f"source={config.equilibrium.source}", 0.4),
        ("Run convergence matrix", "sweep radial grid, angular grid, and Fourier truncation", 1.0),
        ("Write convergence artifacts", f"output_dir={config.run.output_dir}", 0.2),
    ]
    reporter = ProgressReporter(
        enabled=progress,
        stream=stream,
        total_estimate_seconds=sum(phase[2] for phase in phase_specs),
    )
    start = time.perf_counter()
    reporter.banner(config)

    _start_phase(reporter, phase_specs[0])
    reporter.log(f"radial points: {radial_values}")
    reporter.log(f"angular points: {angular_values}")
    reporter.log(f"mode maxima: {mode_values}")
    reporter.phase_done(status="done")

    _start_phase(reporter, phase_specs[1])
    reference_config = replace(config, grid=replace(config.grid, radial_points=max(radial_values)))
    profile_geometry, geometry_source = _load_profile_geometry(reference_config, reporter=reporter)
    reporter.log(
        "geometry resolved: "
        f"name={profile_geometry.name}, radial_points={profile_geometry.s.shape[0]}, "
        f"field_periods={profile_geometry.field_periods}"
    )
    reporter.phase_done(status="done")

    _start_phase(reporter, phase_specs[2])
    raw_records: list[dict[str, Any]] = []
    for radial in radial_values:
        radial_config = replace(config, grid=replace(config.grid, radial_points=radial))
        radial_profile, _ = _load_profile_geometry(radial_config, reporter=reporter)
        for mode_max in mode_values:
            for angular in angular_values:
                record_start = time.perf_counter()
                family = _mode_family_for_limits(
                    field_periods=radial_profile.field_periods,
                    m_max=mode_max,
                    n_max=mode_max,
                )
                metric_geometry = _metric_geometry_for_resolution(
                    radial_config,
                    radial_profile,
                    theta_points=angular,
                    zeta_points=angular,
                )
                metric_branches, residuals = _metric_branches_and_residuals(metric_geometry, family.as_array())
                gap_summary = _minimum_gap_or_nan(metric_branches, metric_geometry.s)
                raw_records.append(
                    {
                        "radial_points": int(metric_geometry.s.shape[0]),
                        "theta_points": angular,
                        "zeta_points": angular,
                        "m_max": mode_max,
                        "n_max": mode_max,
                        "harmonic_count": len(family.harmonics),
                        "branch_count": int(metric_branches.shape[1]),
                        "metric_min_frequency_hz": _finite_min(metric_branches),
                        "metric_max_frequency_hz": _finite_max(metric_branches),
                        "min_gap_hz": gap_summary.min_gap_hz,
                        "min_gap_surface": gap_summary.surface,
                        "residual_max": float(jnp.max(residuals)),
                        "residual_p95": _percentile(residuals, 95.0),
                        "runtime_seconds": time.perf_counter() - record_start,
                        "_s": np.asarray(metric_geometry.s),
                        "_frequencies": np.asarray(metric_branches),
                    }
                )
                reporter.log(
                    f"radial={metric_geometry.s.shape[0]}, theta=zeta={angular}, m_max=n_max={mode_max}: "
                    f"branches={metric_branches.shape[1]}, residual_p95={raw_records[-1]['residual_p95']:.3e}"
                )

    reference_record = max(
        raw_records,
        key=lambda item: (int(item["m_max"]), int(item["theta_points"]), int(item["radial_points"])),
    )
    global_reference_frequencies = np.asarray(reference_record["_frequencies"])
    global_reference_s = np.asarray(reference_record["_s"])
    reference_by_mode = {
        mode_max: max(
            (record for record in raw_records if int(record["m_max"]) == mode_max),
            key=lambda item: (int(item["theta_points"]), int(item["radial_points"])),
        )
        for mode_max in mode_values
    }
    records: list[dict[str, Any]] = []
    for record in raw_records:
        frequencies = np.asarray(record["_frequencies"])
        surfaces = np.asarray(record["_s"])
        mode_reference = np.asarray(reference_by_mode[int(record["m_max"])]["_frequencies"])
        mode_reference_s = np.asarray(reference_by_mode[int(record["m_max"])]["_s"])
        branch_count = min(frequencies.shape[1], mode_reference.shape[1])
        frequencies_on_mode_reference = _interpolate_branch_matrix_np(
            frequencies[:, :branch_count],
            surfaces,
            mode_reference_s,
        )
        frequency_error = _relative_l2_np(
            frequencies_on_mode_reference,
            mode_reference[:, :branch_count],
        )
        global_branch_count = min(frequencies.shape[1], global_reference_frequencies.shape[1])
        frequencies_on_global_reference = _interpolate_branch_matrix_np(
            frequencies[:, :global_branch_count],
            surfaces,
            global_reference_s,
        )
        mode_truncation_error = _relative_l2_np(
            frequencies_on_global_reference,
            global_reference_frequencies[:, :global_branch_count],
        )
        public_record = {key: value for key, value in record.items() if key not in {"_s", "_frequencies"}}
        public_record["frequency_l2_error_vs_reference"] = frequency_error
        public_record["mode_truncation_l2_error_vs_global_reference"] = mode_truncation_error
        public_record["reference_branch_count_used"] = branch_count
        public_record["global_reference_branch_count_used"] = global_branch_count
        public_record["reference_radial_points_used"] = int(mode_reference_s.shape[0])
        public_record["global_reference_radial_points_used"] = int(global_reference_s.shape[0])
        records.append(public_record)

    max_frequency_l2_error = max(float(record["frequency_l2_error_vs_reference"]) for record in records)
    max_residual_p95 = max(float(record["residual_p95"]) for record in records)
    reporter.log(
        "reference: "
        f"radial={reference_record['radial_points']}, theta=zeta={reference_record['theta_points']}, "
        f"m_max=n_max={reference_record['m_max']}"
    )
    reporter.log(f"max frequency L2 error against reference: {max_frequency_l2_error:.3e}")
    reporter.log(f"max p95 residual across sweep: {max_residual_p95:.3e}")
    reporter.phase_done(status="done")

    artifacts: dict[str, str] = {}
    output_dir: str | None = None
    _start_phase(reporter, phase_specs[3])
    if write_artifacts:
        output_path = _output_dir(config) / "convergence"
        output_path.mkdir(parents=True, exist_ok=True)
        output_dir = str(output_path)
        artifacts = {
            "summary_json": str(output_path / "convergence_summary.json"),
            "metrics_csv": str(output_path / "convergence_metrics.csv"),
            "report_markdown": str(output_path / "convergence_report.md"),
        }
        reporter.log(f"writing {artifacts['metrics_csv']}")
        reporter.log(f"writing {artifacts['summary_json']}")
        reporter.log(f"writing {artifacts['report_markdown']}")
    else:
        reporter.log("artifact writing disabled")
    reporter.phase_done(status="done")

    reference = {
        key: value
        for key, value in reference_record.items()
        if key
        in {
            "theta_points",
            "zeta_points",
            "radial_points",
            "m_max",
            "n_max",
            "harmonic_count",
            "branch_count",
            "metric_min_frequency_hz",
            "metric_max_frequency_hz",
            "min_gap_hz",
            "min_gap_surface",
            "residual_max",
            "residual_p95",
        }
    }
    metric_geometry = _metric_geometry_for_resolution(
        reference_config,
        profile_geometry,
        theta_points=max(angular_values),
        zeta_points=max(angular_values),
    )
    summary = ConvergenceRunSummary(
        case=config.run.case,
        model=config.run.model,
        status="completed",
        message="Completed continuum radial/angular/Fourier convergence sweep.",
        config=config.to_dict(),
        elapsed_seconds=time.perf_counter() - start,
        phases=_phase_dicts(reporter.records),
        output_dir=output_dir,
        artifacts=artifacts,
        geometry=_geometry_dict(profile_geometry, metric_geometry, source=geometry_source),
        radial_points=list(radial_values),
        angular_points=list(angular_values),
        mode_max_values=list(mode_values),
        reference=reference,
        max_frequency_l2_error=max_frequency_l2_error,
        max_residual_p95=max_residual_p95,
        records=records,
    )
    if write_artifacts:
        _write_convergence_csv(Path(artifacts["metrics_csv"]), records)
        _write_json(Path(artifacts["summary_json"]), summary.to_dict())
        Path(artifacts["report_markdown"]).write_text(_convergence_report_markdown(summary), encoding="utf-8")
    reporter.summary(summary)
    return summary


def run_global_eigenmodes(
    config: FaxConfig,
    *,
    progress: bool = True,
    stream: TextIO | None = None,
    write_artifacts: bool = True,
    max_modes: int = 8,
) -> GlobalEigenmodeRunSummary:
    """Run the validated global radial eigenmode workflow and write artifacts.

    Analytic and VMEC-``wout`` configurations use the scalar radial envelope
    operator. Configurations with ``equilibrium.boozmn`` use the real-Boozer
    3-D radial block operator with the metric-derived radial stiffness.
    """
    config = config.validated()
    _apply_precision(config)
    reported_mode_limit = int(max_modes)
    if reported_mode_limit < 1:
        raise ValueError("max_modes must be at least 1")
    phase_specs = [
        ("Validate global solve", "check TOML controls and requested eigensolver", 0.1),
        ("Build global operator", "resolve geometry, harmonics, mass, and radial stiffness", 0.8),
        ("Solve global eigenproblem", "compute frequencies, eigenvectors, and residuals", 0.6),
        ("Write global artifacts", f"output_dir={config.run.output_dir}", 0.2),
    ]
    reporter = ProgressReporter(
        enabled=progress,
        stream=stream,
        total_estimate_seconds=sum(phase[2] for phase in phase_specs),
    )
    start = time.perf_counter()
    reporter.banner(config)

    _start_phase(reporter, phase_specs[0])
    reporter.log(f"requested model: {config.run.model}")
    reporter.log(f"eigensolver: {config.numerics.eigensolver}")
    reporter.log(f"reported mode limit: {reported_mode_limit}")
    reporter.phase_done(status="done")

    _start_phase(reporter, phase_specs[1])
    problem: Any
    if config.equilibrium.boozmn:
        geometry_path = Path(config.equilibrium.boozmn).expanduser()
        reporter.log(f"reading Boozer geometry: {geometry_path}")
        boozer_geometry = read_boozmn_geometry(geometry_path)
        family = _mode_family_from_config(config, field_periods=boozer_geometry.field_periods)
        harmonics = family.as_array()
        theta_points = _angular_points(max(config.grid.m_max, config.grid.n_max))
        zeta_points = theta_points
        mass_density = 3.4e-8 * float(config.physics.density_scale)
        problem = radial_block_problem_from_boozer_geometry(
            boozer_geometry,
            harmonics,
            mass_density=mass_density,
            radial_stiffness_method="metric",
            theta_points=theta_points,
            zeta_points=zeta_points,
            boundary_condition="dirichlet",
        )
        operator = "boozmn_global_3d_radial_block"
        geometry_dict: dict[str, Any] = {
            "source": "boozmn",
            "path": str(geometry_path),
            "name": boozer_geometry.source,
            "field_periods": int(boozer_geometry.field_periods),
            "radial_points": int(boozer_geometry.s.shape[0]),
            "theta_points": theta_points,
            "zeta_points": zeta_points,
            "density_scale": float(config.physics.density_scale),
            "mass_density_kg_per_m3": mass_density,
            "metadata": [[key, value] for key, value in boozer_geometry.metadata],
        }
        reporter.log(
            "Boozer operator: "
            f"surfaces={problem.s.shape[0]}, harmonics={problem.harmonics.shape[0]}, "
            f"theta=zeta={theta_points}"
        )
    elif config.equilibrium.use_boozer:
        raise ConfigError("fax eigenmodes with equilibrium.use_boozer=true requires equilibrium.boozmn")
    else:
        profile_geometry, geometry_source = _load_profile_geometry(config, reporter=reporter)
        family = _mode_family_from_config(config, field_periods=profile_geometry.field_periods)
        selected_harmonic = _primary_mode_family_harmonic(family)
        problem = radial_problem_from_equilibrium(
            profile_geometry,
            harmonic=(int(selected_harmonic[0]), int(selected_harmonic[1])),
        )
        operator = "profile_radial_envelope"
        geometry_dict = {
            "source": geometry_source,
            "name": profile_geometry.name,
            "field_periods": int(profile_geometry.field_periods),
            "radial_points": int(profile_geometry.s.shape[0]),
            "selected_harmonic": [int(selected_harmonic[0]), int(selected_harmonic[1])],
            "density_scale": float(config.physics.density_scale),
            "metadata": [[key, value] for key, value in profile_geometry.metadata],
        }
        reporter.log(
            "profile radial operator: "
            f"surfaces={problem.s.shape[0]}, harmonic=({int(selected_harmonic[0])},{int(selected_harmonic[1])})"
        )
    reporter.log(f"operator: {operator}")
    reporter.phase_done(status="done")

    _start_phase(reporter, phase_specs[2])
    result: Any
    if isinstance(problem, RadialBlockEigenmodeProblem):
        matrix_dimension = (int(problem.s.shape[0]) - 2) * int(problem.harmonics.shape[0])
        if config.numerics.eigensolver == "dense" or matrix_dimension <= 1:
            result = solve_radial_block_eigenmodes(problem)
        else:
            sparse_count = min(reported_mode_limit, matrix_dimension - 1)
            result = solve_radial_block_eigenmodes_sparse(problem, mode_count=sparse_count)
        frequencies = result.frequencies_hz
        omega2 = result.omega2
        residuals = result.residuals
        structures = result.mode_structures
        harmonics = problem.harmonics
        radial_grid = problem.s
        boundary_condition = problem.boundary_condition
    else:
        result = solve_radial_eigenmodes(problem)
        frequencies = result.frequencies_hz
        omega2 = result.omega2
        residuals = result.residuals
        selected_harmonic = _primary_mode_family_harmonic(family)
        harmonics = jnp.asarray([[float(selected_harmonic[0]), float(selected_harmonic[1])]])
        radial_grid = problem.s
        structures = _scalar_radial_mode_structures(radial_grid, result.eigenvectors)
        boundary_condition = "dirichlet"
    reported_count = min(reported_mode_limit, int(frequencies.shape[0]))
    frequencies_reported = frequencies[:reported_count]
    omega2_reported = omega2[:reported_count]
    residuals_reported = residuals[:reported_count]
    structures_reported = structures[:reported_count]
    residual_max = float(jnp.max(residuals_reported))
    residual_p95 = _percentile(residuals_reported, 95.0)
    reporter.log(f"reported modes: {reported_count}")
    reporter.log(f"frequency range: {_frequency_range_text(frequencies_reported)}")
    reporter.log(f"global eigen residuals: max={residual_max:.3e}, p95={residual_p95:.3e}")
    reporter.phase_done(status="done")

    artifacts: dict[str, str] = {}
    output_dir: str | None = None
    _start_phase(reporter, phase_specs[3])
    if write_artifacts:
        output_path = _output_dir(config) / "eigenmodes"
        output_path.mkdir(parents=True, exist_ok=True)
        output_dir = str(output_path)
        artifacts = {
            "summary_json": str(output_path / "global_eigenmode_summary.json"),
            "modes_csv": str(output_path / "global_eigenmodes.csv"),
            "structures_csv": str(output_path / "global_mode_structures.csv"),
            "metrics_json": str(output_path / "global_eigenmode_metrics.json"),
        }
        _write_global_modes_csv(
            Path(artifacts["modes_csv"]),
            omega2=omega2_reported,
            frequencies_hz=frequencies_reported,
            residuals=residuals_reported,
        )
        _write_global_structures_csv(
            Path(artifacts["structures_csv"]),
            s=radial_grid,
            harmonics=harmonics,
            mode_structures=structures_reported,
        )
        reporter.log(f"writing {artifacts['modes_csv']}")
        reporter.log(f"writing {artifacts['structures_csv']}")
        reporter.log(f"writing {artifacts['metrics_json']}")
        reporter.log(f"writing {artifacts['summary_json']}")
    else:
        reporter.log("artifact writing disabled")
    reporter.phase_done(status="done")

    summary = GlobalEigenmodeRunSummary(
        case=config.run.case,
        model=config.run.model,
        status="completed",
        message="Completed global radial eigenmode solve.",
        config=config.to_dict(),
        elapsed_seconds=time.perf_counter() - start,
        phases=_phase_dicts(reporter.records),
        output_dir=output_dir,
        artifacts=artifacts,
        geometry=geometry_dict,
        mode_family=_mode_family_dict_from_harmonics(harmonics, family=family),
        operator=operator,
        radial_points=int(radial_grid.shape[0]),
        harmonic_count=int(harmonics.shape[0]),
        boundary_condition=boundary_condition,
        eigensolver=config.numerics.eigensolver,
        reported_mode_count=reported_count,
        min_frequency_hz=_finite_min(frequencies_reported),
        max_frequency_hz=_finite_max(frequencies_reported),
        residual_max=residual_max,
        residual_p95=residual_p95,
    )
    if write_artifacts:
        _write_json(
            Path(artifacts["metrics_json"]),
            _global_eigenmode_metrics_dict(
                summary=summary,
                omega2=omega2_reported,
                frequencies_hz=frequencies_reported,
                residuals=residuals_reported,
                mode_structures=structures_reported,
            ),
        )
        _write_json(Path(artifacts["summary_json"]), summary.to_dict())
    reporter.summary(summary)
    return summary


def run_benchmark(
    config: FaxConfig,
    *,
    progress: bool = True,
    stream: TextIO | None = None,
    repeats: int = 3,
    warmups: int = 1,
    families: tuple[str, ...] = (),
    write_artifacts: bool = True,
) -> BenchmarkRunSummary:
    """Benchmark the current continuum workflow and record artifact sizes."""
    config = config.validated()
    _apply_precision(config)
    repeats = int(repeats)
    warmups = int(warmups)
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if warmups < 0:
        raise ValueError("warmups must be non-negative")
    selected_families = _normalize_benchmark_families(families)

    phase_specs = [
        ("Validate benchmark", f"repeats={repeats}, warmups={warmups}", 0.1),
        ("Cold continuum run", "measure first full workflow and write continuum artifacts", 0.8),
        ("Warm continuum repeats", "repeat continuum workflow without artifact writes", 1.0),
        (
            "Named benchmark families",
            "run selected physics/parity benchmark packets" if selected_families else "no additional families selected",
            max(0.1, 0.8 * len(selected_families)),
        ),
        ("Write benchmark artifacts", f"output_dir={config.run.output_dir}", 0.2),
    ]
    reporter = ProgressReporter(
        enabled=progress,
        stream=stream,
        total_estimate_seconds=sum(phase[2] for phase in phase_specs),
    )
    start = time.perf_counter()
    reporter.banner(config)

    _start_phase(reporter, phase_specs[0])
    reporter.log(f"benchmark repeats: {repeats}")
    reporter.log(f"benchmark warmups: {warmups}")
    if selected_families:
        reporter.log(f"additional benchmark families: {', '.join(selected_families)}")
    reporter.log(f"jax backend: {benchmark_environment()['jax_backend']}")
    reporter.phase_done(status="done")

    benchmark_root = _output_dir(config) / "benchmark"
    continuum_artifact_dir = benchmark_root / "continuum_artifacts"
    continuum_config = replace(config, run=replace(config.run, output_dir=str(continuum_artifact_dir)))

    _start_phase(reporter, phase_specs[1])
    cold_start = time.perf_counter()
    continuum_summary = run_continuum(continuum_config, progress=False, write_artifacts=True)
    cold_seconds = time.perf_counter() - cold_start
    artifact_size_bytes = _directory_size_bytes(continuum_artifact_dir)
    reporter.log(f"cold continuum workflow: {cold_seconds:.3f}s")
    reporter.log(f"continuum artifact size: {artifact_size_bytes} bytes")
    reporter.phase_done(status="done")

    _start_phase(reporter, phase_specs[2])
    warm_result = time_callable(
        "continuum_warm_no_artifacts",
        lambda: run_continuum(continuum_config, progress=False, write_artifacts=False),
        warmups=warmups,
        repeats=repeats,
    )
    reporter.log(f"warm median runtime: {warm_result.median_seconds:.3f}s")
    reporter.phase_done(status="done")

    _start_phase(reporter, phase_specs[3])
    family_results = [
        _continuum_family_result(
            cold_seconds=cold_seconds,
            warm_result=warm_result,
            artifact_size_bytes=artifact_size_bytes,
            continuum_summary=continuum_summary,
        )
    ]
    if selected_families:
        for family in selected_families:
            reporter.log(f"family START {family}")
            family_start = time.perf_counter()
            result = run_named_benchmark_family(family, repeats=repeats, warmups=warmups)
            family_results.append(_json_safe(asdict(result)))
            metric = "n/a" if result.metric_value is None else f"{result.metric_value:.3e}"
            tolerance = "n/a" if result.tolerance is None else f"{result.tolerance:.3e}"
            reporter.log(
                f"family DONE  {family}: status={result.status}, gate={result.gate_passed}, "
                f"metric={metric}, tol={tolerance}, elapsed={format_seconds(time.perf_counter() - family_start)}"
            )
    else:
        reporter.log("no additional benchmark families selected")
    reporter.phase_done(status="done")

    artifacts: dict[str, str] = {}
    output_dir: str | None = None
    _start_phase(reporter, phase_specs[4])
    if write_artifacts:
        benchmark_root.mkdir(parents=True, exist_ok=True)
        output_dir = str(benchmark_root)
        artifacts = {
            "summary_json": str(benchmark_root / "benchmark_summary.json"),
            "metrics_csv": str(benchmark_root / "benchmark_metrics.csv"),
        }
        reporter.log(f"writing {artifacts['summary_json']}")
        reporter.log(f"writing {artifacts['metrics_csv']}")
    else:
        reporter.log("artifact writing disabled")
    reporter.phase_done(status="done")

    summary = BenchmarkRunSummary(
        case=config.run.case,
        model=config.run.model,
        status="completed",
        message="Completed continuum workflow benchmark.",
        config=config.to_dict(),
        elapsed_seconds=time.perf_counter() - start,
        phases=_phase_dicts(reporter.records),
        output_dir=output_dir,
        artifacts=artifacts,
        cold_seconds=cold_seconds,
        warm_result=asdict(warm_result),
        artifact_size_bytes=artifact_size_bytes,
        continuum_summary=continuum_summary.to_dict(),
        family_results=family_results,
    )
    if write_artifacts:
        _write_json(Path(artifacts["summary_json"]), summary.to_dict())
        _write_benchmark_csv(Path(artifacts["metrics_csv"]), summary)
    reporter.summary(summary)
    return summary


def run_continuum(
    config: FaxConfig,
    *,
    progress: bool = True,
    stream: TextIO | None = None,
    write_artifacts: bool = True,
) -> ContinuumRunSummary:
    """Compute profile and metric continuum operators and write artifacts."""
    config = config.validated()
    _apply_precision(config)
    phase_specs = [
        ("Validate configuration", "check continuum-compatible controls", 0.1),
        ("Resolve equilibrium geometry", f"source={config.equilibrium.source}", 0.4),
        ("Build mode family", "construct stellarator sideband harmonics", 0.1),
        ("Assemble continuum operators", "profile branches, metric matrices, and dense eigensolve", 0.6),
        ("Write continuum artifacts", f"output_dir={config.run.output_dir}", 0.2),
    ]
    reporter = ProgressReporter(
        enabled=progress,
        stream=stream,
        total_estimate_seconds=sum(phase[2] for phase in phase_specs),
    )
    start = time.perf_counter()
    reporter.banner(config)

    _start_phase(reporter, phase_specs[0])
    reporter.log(f"requested model: {config.run.model}")
    reporter.log(f"density scale: {config.physics.density_scale:g}")
    reporter.phase_done(status="done")

    _start_phase(reporter, phase_specs[1])
    profile_geometry, metric_geometry, geometry_source = _load_geometries(config, reporter=reporter)
    reporter.log(
        "geometry resolved: "
        f"name={profile_geometry.name}, radial_points={profile_geometry.s.shape[0]}, "
        f"field_periods={profile_geometry.field_periods}"
    )
    reporter.phase_done(status="done")

    _start_phase(reporter, phase_specs[2])
    family = _mode_family_from_config(config, field_periods=profile_geometry.field_periods)
    reporter.log(f"harmonics: {family.labels()[:4]}{' ...' if len(family.harmonics) > 4 else ''}")
    reporter.log(f"harmonic count: {len(family.harmonics)}")
    reporter.phase_done(status="done")

    _start_phase(reporter, phase_specs[3])
    harmonics = family.as_array()
    profile_branches = continuum_branch_matrix(profile_geometry, harmonics)
    metric_diagonal_branches = diagonal_continuum_frequencies_hz(metric_geometry, harmonics)
    matrices = metric_continuum_matrices(metric_geometry, harmonics)
    eigensystem = generalized_hermitian_eigensystem(matrices.stiffness, matrices.mass)
    residuals = generalized_eigen_residual_norms(matrices.stiffness, matrices.mass, eigensystem)
    metric_branches = jnp.sqrt(jnp.maximum(jnp.real(eigensystem.eigenvalues), 0.0)) / (2.0 * jnp.pi)
    gap_summary = _minimum_gap_or_nan(metric_branches, metric_geometry.s)
    metric_residual_max = float(jnp.max(residuals))
    metric_residual_p95 = _percentile(residuals, 95.0)
    reporter.log(f"profile frequency range: {_frequency_range_text(profile_branches)}")
    reporter.log(f"metric frequency range: {_frequency_range_text(metric_branches)}")
    reporter.log(f"minimum metric branch gap: {gap_summary.min_gap_hz:.6g} Hz at s={gap_summary.surface:.4f}")
    reporter.log(f"generalized-eigen residuals: max={metric_residual_max:.3e}, p95={metric_residual_p95:.3e}")
    reporter.phase_done(status="done")

    artifacts: dict[str, str] = {}
    output_dir: str | None = None
    _start_phase(reporter, phase_specs[4])
    if write_artifacts:
        output_path = _output_dir(config)
        output_path.mkdir(parents=True, exist_ok=True)
        output_dir = str(output_path)
        artifacts = {
            "summary_json": str(output_path / "continuum_summary.json"),
            "branches_csv": str(output_path / "continuum_branches.csv"),
            "metrics_json": str(output_path / "continuum_metrics.json"),
        }
        _write_continuum_csv(
            Path(artifacts["branches_csv"]),
            profile_geometry=profile_geometry,
            family=family,
            profile_branches=profile_branches,
            metric_diagonal_branches=metric_diagonal_branches,
            metric_branches=metric_branches,
        )
        reporter.log(f"writing {artifacts['branches_csv']}")
        reporter.log(f"writing {artifacts['metrics_json']}")
        reporter.log(f"writing {artifacts['summary_json']}")
    else:
        reporter.log("artifact writing disabled")
    reporter.phase_done(status="done")

    geometry_dict = _geometry_dict(profile_geometry, metric_geometry, source=geometry_source)
    mode_family_dict = _mode_family_dict(family)
    summary = ContinuumRunSummary(
        case=config.run.case,
        model=config.run.model,
        status="completed",
        message="Computed profile and metric-matrix continuum models.",
        config=config.to_dict(),
        elapsed_seconds=time.perf_counter() - start,
        phases=_phase_dicts(reporter.records),
        output_dir=output_dir,
        artifacts=artifacts,
        geometry=geometry_dict,
        mode_family=mode_family_dict,
        radial_points=int(profile_geometry.s.shape[0]),
        harmonic_count=len(family.harmonics),
        profile_min_frequency_hz=_finite_min(profile_branches),
        profile_max_frequency_hz=_finite_max(profile_branches),
        metric_min_frequency_hz=_finite_min(metric_branches),
        metric_max_frequency_hz=_finite_max(metric_branches),
        min_gap_hz=gap_summary.min_gap_hz,
        min_gap_surface=gap_summary.surface,
        metric_residual_max=metric_residual_max,
        metric_residual_p95=metric_residual_p95,
    )
    if write_artifacts:
        metrics = _continuum_metrics_dict(
            summary=summary,
            gap_summary=gap_summary,
            profile_branches=profile_branches,
            metric_diagonal_branches=metric_diagonal_branches,
            metric_branches=metric_branches,
            residuals=residuals,
        )
        _write_json(Path(artifacts["metrics_json"]), metrics)
        _write_json(Path(artifacts["summary_json"]), summary.to_dict())
    reporter.summary(summary)
    return summary


def _load_geometries(
    config: FaxConfig,
    *,
    reporter: ProgressReporter,
) -> tuple[EquilibriumGeometry, MetricGeometry, str]:
    theta_points = _angular_points(config.grid.m_max)
    zeta_points = _angular_points(config.grid.n_max)
    profile, source = _load_profile_geometry(config, reporter=reporter)
    metric = _metric_geometry_for_resolution(config, profile, theta_points=theta_points, zeta_points=zeta_points)
    return profile, metric, source


def _load_profile_geometry(
    config: FaxConfig,
    *,
    reporter: ProgressReporter,
) -> tuple[EquilibriumGeometry, str]:
    if config.equilibrium.wout:
        reporter.log(f"reading VMEC wout: {config.equilibrium.wout}")
        profile = from_vmec_wout(config.equilibrium.wout, name=config.run.case)
        source = "vmec_wout"
        profile = _resample_profile_geometry(profile, config.grid.radial_points)
    elif config.equilibrium.source == "analytic":
        reporter.log("using analytic circular-tokamak fixture")
        profile = analytic_circular_tokamak(radial_points=config.grid.radial_points)
        profile = _with_config_field_periods(profile, config)
        source = "analytic_circular_tokamak"
    else:
        reporter.log("direct VMEC execution from TOML is not selected here; using analytic fixture")
        profile = analytic_circular_tokamak(radial_points=config.grid.radial_points)
        profile = _with_config_field_periods(profile, config)
        source = "analytic_fixture_for_vmec_source"

    if config.physics.density_scale != 1.0:
        profile = profile.with_density_scale(config.physics.density_scale)
    return profile, source


def _metric_geometry_for_resolution(
    config: FaxConfig,
    profile: EquilibriumGeometry,
    *,
    theta_points: int,
    zeta_points: int,
) -> MetricGeometry:
    if config.equilibrium.wout:
        metric = metric_from_vmec_wout(
            config.equilibrium.wout,
            theta_points=theta_points,
            zeta_points=zeta_points,
            name=config.run.case,
        )
        metric = _resample_metric_geometry(metric, config.grid.radial_points)
        if config.physics.density_scale != 1.0:
            metric = metric.with_density_scale(config.physics.density_scale)
        return metric
    return metric_from_equilibrium(profile, theta_points=theta_points, zeta_points=zeta_points)


def _resample_profile_geometry(profile: EquilibriumGeometry, radial_points: int) -> EquilibriumGeometry:
    target_count = int(radial_points)
    if int(profile.s.shape[0]) == target_count:
        return profile
    target_s = _uniform_radial_grid(profile.s, target_count)
    metadata = profile.metadata + (
        ("radial_adapter", "linear_s_interpolation"),
        ("radial_points", target_count),
    )
    return EquilibriumGeometry(
        name=profile.name,
        s=jnp.asarray(target_s),
        iota=jnp.asarray(_interpolate_radial_array_np(profile.iota, profile.s, target_s)),
        magnetic_field=jnp.asarray(_interpolate_radial_array_np(profile.magnetic_field, profile.s, target_s)),
        mass_density=jnp.asarray(_interpolate_radial_array_np(profile.mass_density, profile.s, target_s)),
        major_radius=profile.major_radius,
        minor_radius=profile.minor_radius,
        field_periods=profile.field_periods,
        metadata=metadata,
    )


def _resample_metric_geometry(metric: MetricGeometry, radial_points: int) -> MetricGeometry:
    target_count = int(radial_points)
    if int(metric.s.shape[0]) == target_count:
        return metric
    target_s = _uniform_radial_grid(metric.s, target_count)
    metadata = metric.metadata + (
        ("radial_adapter", "linear_s_interpolation"),
        ("radial_points", target_count),
    )
    return MetricGeometry(
        name=metric.name,
        s=jnp.asarray(target_s),
        theta=metric.theta,
        zeta=metric.zeta,
        jacobian=jnp.asarray(_interpolate_radial_array_np(metric.jacobian, metric.s, target_s)),
        magnetic_field=jnp.asarray(_interpolate_radial_array_np(metric.magnetic_field, metric.s, target_s)),
        mass_density=jnp.asarray(_interpolate_radial_array_np(metric.mass_density, metric.s, target_s)),
        bdotgrad_theta=jnp.asarray(_interpolate_radial_array_np(metric.bdotgrad_theta, metric.s, target_s)),
        bdotgrad_zeta=jnp.asarray(_interpolate_radial_array_np(metric.bdotgrad_zeta, metric.s, target_s)),
        field_periods=metric.field_periods,
        metadata=metadata,
    )


def _uniform_radial_grid(source_s: Any, radial_points: int) -> np.ndarray:
    source = np.asarray(source_s, dtype=float)
    if source.ndim != 1 or source.size < 2:
        raise ValueError("source radial grid must be one-dimensional with at least two points")
    if int(radial_points) < 4:
        raise ValueError("radial_points must be at least 4")
    return np.linspace(float(source[0]), float(source[-1]), int(radial_points))


def _interpolate_radial_array_np(values: Any, source_s: Any, target_s: Any) -> np.ndarray:
    array = np.asarray(values)
    source = np.asarray(source_s, dtype=float)
    target = np.asarray(target_s, dtype=float)
    if array.shape[0] != source.shape[0]:
        raise ValueError("radial array first axis must match source_s")
    if array.shape[0] == target.shape[0] and np.allclose(source, target):
        return np.array(array, copy=True)

    flat = array.reshape(array.shape[0], -1)
    interpolated = np.empty((target.shape[0], flat.shape[1]), dtype=flat.dtype)
    for column in range(flat.shape[1]):
        if np.iscomplexobj(flat):
            real = np.interp(target, source, np.real(flat[:, column]))
            imag = np.interp(target, source, np.imag(flat[:, column]))
            interpolated[:, column] = real + 1j * imag
        else:
            interpolated[:, column] = np.interp(target, source, flat[:, column])
    return interpolated.reshape((target.shape[0],) + array.shape[1:])


def _with_config_field_periods(geometry: EquilibriumGeometry, config: FaxConfig) -> EquilibriumGeometry:
    if config.grid.field_periods is None:
        return geometry
    return replace(geometry, field_periods=int(config.grid.field_periods))


def _apply_precision(config: FaxConfig) -> None:
    jax.config.update("jax_enable_x64", config.numerics.precision == "float64")


def _mode_family_from_config(config: FaxConfig, *, field_periods: int) -> ModeFamily:
    m_ceiling = max(1, int(config.grid.m_max))
    n_ceiling = max(0, int(config.grid.n_max))
    return _mode_family_for_limits(field_periods=field_periods, m_max=m_ceiling, n_max=n_ceiling)


def _mode_family_for_limits(*, field_periods: int, m_max: int, n_max: int) -> ModeFamily:
    m_ceiling = max(1, int(m_max))
    n_ceiling = max(0, int(n_max))
    base_n = 1 if n_ceiling >= 1 else 0
    sidebands = tuple(
        k for k in (-1, 0, 1) if abs(base_n + k * int(field_periods)) <= n_ceiling
    )
    if not sidebands:
        sidebands = (0,)
    return stellarator_mode_family(
        poloidal_modes=range(1, m_ceiling + 1),
        base_n=base_n,
        field_periods=field_periods,
        sidebands=sidebands,
    )


def _angular_points(max_mode: int) -> int:
    return max(16, 4 * max(1, int(max_mode)) + 8)


def _start_phase(reporter: ProgressReporter, spec: tuple[str, str, float]) -> None:
    name, detail, estimate = spec
    reporter.phase_start(name, detail=detail, estimate_seconds=estimate)


def _phase_dicts(records: list[PhaseRecord]) -> list[dict[str, Any]]:
    return [asdict(record) for record in records]


def _positive_unique_ints(values: tuple[int, ...], *, name: str) -> tuple[int, ...]:
    parsed = tuple(sorted(set(int(value) for value in values)))
    if not parsed:
        raise ValueError(f"{name} must contain at least one value")
    if any(value <= 0 for value in parsed):
        raise ValueError(f"{name} values must be positive")
    return parsed


def _output_dir(config: FaxConfig) -> Path:
    return Path(config.run.output_dir).expanduser()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _write_convergence_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("cannot write empty convergence records")
    headers = tuple(records[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _write_benchmark_csv(path: Path, summary: BenchmarkRunSummary) -> None:
    warm = summary.warm_result
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "value"])
        writer.writerow(["cold_seconds", summary.cold_seconds])
        writer.writerow(["warm_min_seconds", warm["min_seconds"]])
        writer.writerow(["warm_median_seconds", warm["median_seconds"]])
        writer.writerow(["warm_max_seconds", warm["max_seconds"]])
        writer.writerow(["warm_repeats", warm["repeats"]])
        writer.writerow(["warmups", warm["warmups"]])
        writer.writerow(["artifact_size_bytes", summary.artifact_size_bytes])
        writer.writerow([])
        writer.writerow(["family", "status", "gate_passed", "metric", "tolerance", "cold_seconds", "warm_median_seconds"])
        for result in summary.family_results:
            warm_family = result.get("warm_result") or {}
            writer.writerow(
                [
                    result.get("family"),
                    result.get("status"),
                    result.get("gate_passed"),
                    result.get("metric_value"),
                    result.get("tolerance"),
                    result.get("cold_seconds"),
                    warm_family.get("median_seconds"),
                ]
            )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _normalize_benchmark_families(families: tuple[str, ...]) -> tuple[str, ...]:
    known = set(known_benchmark_families())
    normalized: list[str] = []
    for raw in families:
        family = str(raw)
        if family == "continuum":
            continue
        if family not in known:
            choices = ", ".join(("continuum",) + known_benchmark_families())
            raise ValueError(f"unknown benchmark family {family!r}; expected one of: {choices}")
        if family not in normalized:
            normalized.append(family)
    return tuple(normalized)


def _continuum_family_result(
    *,
    cold_seconds: float,
    warm_result: Any,
    artifact_size_bytes: int,
    continuum_summary: ContinuumRunSummary,
) -> dict[str, Any]:
    tolerance = 1.0e-6
    metric_value = float(continuum_summary.metric_residual_p95)
    return _json_safe(
        {
            "family": "continuum",
            "label": "public continuum workflow with generalized-eigen residual gate",
            "status": "completed",
            "gate_passed": bool(metric_value < tolerance),
            "physics_metric": "p95 generalized-eigen residual",
            "metric_value": metric_value,
            "tolerance": tolerance,
            "cold_seconds": float(cold_seconds),
            "warm_result": asdict(warm_result),
            "process_max_rss_bytes": process_max_rss_bytes(),
            "details": {
                "artifact_size_bytes": int(artifact_size_bytes),
                "radial_points": int(continuum_summary.radial_points),
                "harmonic_count": int(continuum_summary.harmonic_count),
                "min_gap_hz": float(continuum_summary.min_gap_hz),
                "metric_residual_max": float(continuum_summary.metric_residual_max),
            },
            "environment": benchmark_environment(),
        }
    )


def _convergence_report_markdown(summary: ConvergenceRunSummary) -> str:
    lines = [
        f"# FAX Convergence Report: {summary.case}",
        "",
        f"- Status: `{summary.status}`",
        f"- Message: {summary.message}",
        f"- Radial points: {summary.radial_points}",
        f"- Angular points: {summary.angular_points}",
        f"- Mode maxima: {summary.mode_max_values}",
        f"- Reference: radial={summary.reference['radial_points']}, "
        f"theta=zeta={summary.reference['theta_points']}, "
        f"m_max=n_max={summary.reference['m_max']}",
        f"- Max frequency L2 error: {summary.max_frequency_l2_error:.6g}",
        f"- Max residual p95: {summary.max_residual_p95:.6g}",
        "",
        "| radial | theta | m_max | branches | freq L2 error | residual p95 | min gap [Hz] | runtime [s] |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in summary.records:
        lines.append(
            "| {radial_points} | {theta_points} | {m_max} | {branch_count} | "
            "{frequency_l2_error_vs_reference:.6g} | {residual_p95:.6g} | "
            "{min_gap_hz:.6g} | {runtime_seconds:.4g} |".format(**record)
        )
    lines.append("")
    return "\n".join(lines)


def _gate_result_dict(result: GateResult, registry: dict[str, Any]) -> dict[str, Any]:
    gate = registry.get(result.gate_id)
    payload = asdict(result)
    if gate is not None:
        payload.update(
            {
                "title": gate.title,
                "layer": gate.layer,
                "quantity": gate.quantity,
                "citation": gate.citation,
                "command": gate.command,
                "tags": list(gate.tags),
            }
        )
    return _json_safe(payload)


def _format_gate_line(result: GateResult) -> str:
    metric = "nan" if not math.isfinite(float(result.metric)) else f"{result.metric:.3e}"
    return (
        f"{result.gate_id}: {result.status} | metric={metric} | "
        f"tolerance={result.tolerance:.3e} | {result.message}"
    )


def _validation_message(
    *,
    gate_count: int,
    passed_count: int,
    failed_count: int,
    skipped_count: int,
    tags: tuple[str, ...],
) -> str:
    tag_text = f" for tags={','.join(tags)}" if tags else ""
    if gate_count == 0:
        return f"No validation gates matched{tag_text}."
    if failed_count:
        return (
            f"{failed_count} of {gate_count} validation gates failed{tag_text}; "
            f"{passed_count} passed and {skipped_count} skipped."
        )
    return f"{passed_count} validation gates passed{tag_text}; {skipped_count} skipped."


def _validation_report_markdown(summary: ValidationRunSummary) -> str:
    lines = [
        f"# FAX Validation Report: {summary.case}",
        "",
        f"- Status: `{summary.status}`",
        f"- Message: {summary.message}",
        f"- Gate count: {summary.gate_count}",
        f"- Passed: {summary.passed_count}",
        f"- Failed: {summary.failed_count}",
        f"- Skipped: {summary.skipped_count}",
        "",
        "| gate | status | metric | tolerance | layer | citation |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for result in summary.results:
        lines.append(
            "| {gate_id} | {status} | {metric} | {tolerance} | {layer} | {citation} |".format(
                gate_id=result["gate_id"],
                status=result["status"],
                metric=_markdown_number(result["metric"]),
                tolerance=_markdown_number(result["tolerance"]),
                layer=result.get("layer", ""),
                citation=str(result.get("citation", "")).replace("|", "\\|"),
            )
        )
    lines.extend(
        [
            "",
            "## Commands",
            "",
        ]
    )
    for result in summary.results:
        command = result.get("command")
        if command:
            lines.append(f"- `{result['gate_id']}`: `{command}`")
    lines.append("")
    return "\n".join(lines)


def _markdown_number(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_continuum_csv(
    path: Path,
    *,
    profile_geometry: EquilibriumGeometry,
    family: ModeFamily,
    profile_branches: Any,
    metric_diagonal_branches: Any,
    metric_branches: Any,
) -> None:
    labels = tuple(_label_slug(label) for label in family.labels())
    metric_labels = tuple(f"metric_eigen_{index:03d}_hz" for index in range(np.asarray(metric_branches).shape[1]))
    headers = (
        ["s", "iota"]
        + [f"profile_{label}_hz" for label in labels]
        + [f"metric_diagonal_{label}_hz" for label in labels]
        + list(metric_labels)
    )
    s = np.asarray(profile_geometry.s)
    iota = np.asarray(profile_geometry.iota)
    profile = np.asarray(profile_branches)
    diagonal = np.asarray(metric_diagonal_branches)
    coupled = np.asarray(metric_branches)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        for index in range(s.shape[0]):
            writer.writerow(
                [float(s[index]), float(iota[index])]
                + [float(value) for value in profile[index]]
                + [float(value) for value in diagonal[index]]
                + [float(value) for value in coupled[index]]
            )


def _scalar_radial_mode_structures(s: Any, eigenvectors: Any) -> jax.Array:
    vectors = jnp.asarray(eigenvectors)
    mode_count = int(vectors.shape[1])
    interior = jnp.moveaxis(vectors, 1, 0)[:, :, None]
    boundary = jnp.zeros((mode_count, 1, 1), dtype=vectors.dtype)
    if int(s.shape[0]) == int(vectors.shape[0]):
        return interior
    return jnp.concatenate([boundary, interior, boundary], axis=1)


def _write_global_modes_csv(
    path: Path,
    *,
    omega2: Any,
    frequencies_hz: Any,
    residuals: Any,
) -> None:
    omega2_array = np.asarray(omega2, dtype=float)
    frequency_array = np.asarray(frequencies_hz, dtype=float)
    residual_array = np.asarray(residuals, dtype=float)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["mode_index", "omega2_rad2_per_s2", "frequency_hz", "residual"])
        for index in range(frequency_array.shape[0]):
            writer.writerow(
                [
                    index,
                    float(omega2_array[index]),
                    float(frequency_array[index]),
                    float(residual_array[index]),
                ]
            )


def _write_global_structures_csv(
    path: Path,
    *,
    s: Any,
    harmonics: Any,
    mode_structures: Any,
) -> None:
    radial_grid = np.asarray(s, dtype=float)
    harmonic_array = np.asarray(harmonics, dtype=float)
    structures = np.asarray(mode_structures)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "mode_index",
                "radial_index",
                "s",
                "harmonic_index",
                "m",
                "n",
                "amplitude_real",
                "amplitude_imag",
                "amplitude_abs",
            ]
        )
        for mode_index in range(structures.shape[0]):
            for radial_index in range(radial_grid.shape[0]):
                for harmonic_index in range(harmonic_array.shape[0]):
                    value = structures[mode_index, radial_index, harmonic_index]
                    writer.writerow(
                        [
                            mode_index,
                            radial_index,
                            float(radial_grid[radial_index]),
                            harmonic_index,
                            int(harmonic_array[harmonic_index, 0]),
                            int(harmonic_array[harmonic_index, 1]),
                            float(np.real(value)),
                            float(np.imag(value)),
                            float(np.abs(value)),
                        ]
                    )


def _mode_family_dict_from_harmonics(harmonics: Any, *, family: ModeFamily) -> dict[str, Any]:
    harmonic_array = np.asarray(harmonics, dtype=float)
    return {
        "base_n": family.base_n,
        "field_periods": family.field_periods,
        "sidebands": list(family.sidebands),
        "harmonics": [[int(m), int(n)] for m, n in harmonic_array],
        "labels": [f"m={int(m)}, n={int(n)}" for m, n in harmonic_array],
    }


def _global_eigenmode_metrics_dict(
    *,
    summary: GlobalEigenmodeRunSummary,
    omega2: Any,
    frequencies_hz: Any,
    residuals: Any,
    mode_structures: Any,
) -> dict[str, Any]:
    structures = jnp.asarray(mode_structures)
    radial_envelope = jnp.sqrt(jnp.sum(jnp.abs(structures) ** 2, axis=-1))
    harmonic_power = jnp.sum(jnp.abs(structures) ** 2, axis=1)
    return {
        "case": summary.case,
        "status": summary.status,
        "operator": summary.operator,
        "omega2_range_rad2_per_s2": [_finite_min(omega2), _finite_max(omega2)],
        "frequency_range_hz": [_finite_min(frequencies_hz), _finite_max(frequencies_hz)],
        "residual_max": float(jnp.max(residuals)),
        "residual_mean": float(jnp.mean(residuals)),
        "residual_p95": _percentile(residuals, 95.0),
        "radial_envelope_max_by_mode": [float(value) for value in jnp.max(radial_envelope, axis=1)],
        "harmonic_power_fraction": np.asarray(
            harmonic_power / jnp.maximum(jnp.sum(harmonic_power, axis=1, keepdims=True), 1.0e-30)
        ).tolist(),
    }


def _label_slug(label: str) -> str:
    return (
        label.replace(" ", "")
        .replace("=", "")
        .replace(",", "_")
        .replace("-", "minus")
    )


def _geometry_dict(profile: EquilibriumGeometry, metric: MetricGeometry, *, source: str) -> dict[str, Any]:
    return {
        "source": source,
        "name": profile.name,
        "field_periods": profile.field_periods,
        "radial_points": int(profile.s.shape[0]),
        "theta_points": int(metric.theta.shape[0]),
        "zeta_points": int(metric.zeta.shape[0]),
        "major_radius_m": float(profile.major_radius),
        "minor_radius_m": float(profile.minor_radius),
        "profile_metadata": [[key, value] for key, value in profile.metadata],
        "metric_metadata": [[key, value] for key, value in metric.metadata],
    }


def _mode_family_dict(family: ModeFamily) -> dict[str, Any]:
    return {
        "base_n": family.base_n,
        "field_periods": family.field_periods,
        "sidebands": list(family.sidebands),
        "harmonics": [[int(m), int(n)] for m, n in family.harmonics],
        "labels": list(family.labels()),
    }


def _primary_mode_family_harmonic(family: ModeFamily) -> tuple[int, int]:
    for harmonic in family.harmonics:
        if int(harmonic[1]) == int(family.base_n):
            return harmonic
    return family.harmonics[0]


def _continuum_metrics_dict(
    *,
    summary: ContinuumRunSummary,
    gap_summary: ContinuumGapSummary,
    profile_branches: Any,
    metric_diagonal_branches: Any,
    metric_branches: Any,
    residuals: Any,
) -> dict[str, Any]:
    return {
        "case": summary.case,
        "status": summary.status,
        "profile_frequency_range_hz": [_finite_min(profile_branches), _finite_max(profile_branches)],
        "metric_diagonal_frequency_range_hz": [
            _finite_min(metric_diagonal_branches),
            _finite_max(metric_diagonal_branches),
        ],
        "metric_frequency_range_hz": [_finite_min(metric_branches), _finite_max(metric_branches)],
        "minimum_gap": asdict(gap_summary),
        "metric_residual_max": float(jnp.max(residuals)),
        "metric_residual_mean": float(jnp.mean(residuals)),
        "metric_residual_p95": _percentile(residuals, 95.0),
    }


def _metric_branches_and_residuals(metric: MetricGeometry, harmonics: Any) -> tuple[jax.Array, jax.Array]:
    matrices = metric_continuum_matrices(metric, harmonics)
    eigensystem = generalized_hermitian_eigensystem(matrices.stiffness, matrices.mass)
    residuals = generalized_eigen_residual_norms(matrices.stiffness, matrices.mass, eigensystem)
    branches = jnp.sqrt(jnp.maximum(jnp.real(eigensystem.eigenvalues), 0.0)) / (2.0 * jnp.pi)
    return branches, residuals


def _minimum_gap_or_nan(branches_hz: Any, surfaces: Any) -> ContinuumGapSummary:
    branches = jnp.asarray(branches_hz)
    if branches.ndim != 2 or branches.shape[1] < 2:
        return ContinuumGapSummary(
            min_gap_hz=float("nan"),
            surface=float("nan"),
            lower_frequency_hz=float("nan"),
            upper_frequency_hz=float("nan"),
            radial_index=-1,
            sorted_branch_index=-1,
        )
    return minimum_gap_summary(branches, surfaces)


def _relative_l2_np(actual: np.ndarray, expected: np.ndarray, *, floor: float = 1.0e-30) -> float:
    denominator = max(float(np.linalg.norm(expected)), float(floor))
    return float(np.linalg.norm(actual - expected) / denominator)


def _interpolate_branch_matrix_np(matrix: np.ndarray, source_s: np.ndarray, target_s: np.ndarray) -> np.ndarray:
    return _interpolate_radial_array_np(matrix, source_s, target_s)


def _directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return int(total)


def _finite_min(values: Any) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.min(finite)) if finite.size else float("nan")


def _finite_max(values: Any) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.max(finite)) if finite.size else float("nan")


def _frequency_range_text(values: Any) -> str:
    return f"{_finite_min(values):.6g} Hz to {_finite_max(values):.6g} Hz"


def _percentile(values: Any, percentile: float) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.percentile(finite, percentile)) if finite.size else float("nan")
