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

from .config import FaxConfig
from .continuum import continuum_branch_matrix
from .equilibrium import EquilibriumGeometry, analytic_circular_tokamak, from_vmec_wout
from .gaps import ContinuumGapSummary, minimum_gap_summary
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
from .progress import PhaseRecord, ProgressReporter
from .validation import GateResult, registered_validation_gates, run_validation_suite


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
    results = run_validation_suite(tags=selected_tags or None, include_skipped=True)
    for result in results:
        reporter.log(_format_gate_line(result))
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


def run_continuum(
    config: FaxConfig,
    *,
    progress: bool = True,
    stream: TextIO | None = None,
    write_artifacts: bool = True,
) -> ContinuumRunSummary:
    """Compute the current local and metric continuum scaffold and write artifacts."""
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
    gap_summary = minimum_gap_summary(metric_branches, metric_geometry.s)
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
        message="Computed profile and metric-matrix continuum scaffold.",
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
    if config.equilibrium.wout:
        reporter.log(f"reading VMEC wout: {config.equilibrium.wout}")
        profile = from_vmec_wout(config.equilibrium.wout, name=config.run.case)
        metric = metric_from_vmec_wout(
            config.equilibrium.wout,
            theta_points=theta_points,
            zeta_points=zeta_points,
            name=config.run.case,
        )
        source = "vmec_wout"
    elif config.equilibrium.source == "analytic":
        reporter.log("using analytic circular-tokamak fixture")
        profile = analytic_circular_tokamak(radial_points=config.grid.radial_points)
        profile = _with_config_field_periods(profile, config)
        metric = metric_from_equilibrium(profile, theta_points=theta_points, zeta_points=zeta_points)
        source = "analytic_circular_tokamak"
    else:
        reporter.log("direct vmec_jax execution is planned; using analytic fixture until that adapter is implemented")
        profile = analytic_circular_tokamak(radial_points=config.grid.radial_points)
        profile = _with_config_field_periods(profile, config)
        metric = metric_from_equilibrium(profile, theta_points=theta_points, zeta_points=zeta_points)
        source = "analytic_placeholder_for_vmec_jax"

    if config.physics.density_scale != 1.0:
        profile = profile.with_density_scale(config.physics.density_scale)
        metric = metric.with_density_scale(config.physics.density_scale)
    return profile, metric, source


def _with_config_field_periods(geometry: EquilibriumGeometry, config: FaxConfig) -> EquilibriumGeometry:
    if config.grid.field_periods is None:
        return geometry
    return replace(geometry, field_periods=int(config.grid.field_periods))


def _apply_precision(config: FaxConfig) -> None:
    jax.config.update("jax_enable_x64", config.numerics.precision == "float64")


def _mode_family_from_config(config: FaxConfig, *, field_periods: int) -> ModeFamily:
    m_ceiling = max(1, int(config.grid.m_max))
    n_ceiling = max(0, int(config.grid.n_max))
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


def _output_dir(config: FaxConfig) -> Path:
    return Path(config.run.output_dir).expanduser()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


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
