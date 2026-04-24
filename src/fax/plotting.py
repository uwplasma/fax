"""Optional plotting helpers for FAX workflow artifacts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .gap_models import BasicAlfvenGapMap, basic_alfven_gap_map


@dataclass(frozen=True)
class ContinuumArtifacts:
    """Loaded files produced by ``fax continuum``."""

    artifact_dir: Path
    summary: dict[str, Any]
    metrics: dict[str, Any]
    branches: dict[str, np.ndarray]


@dataclass(frozen=True)
class ConvergenceArtifacts:
    """Loaded files produced by ``fax convergence``."""

    artifact_dir: Path
    summary: dict[str, Any]
    metrics: dict[str, np.ndarray]


def load_continuum_artifacts(artifact_dir: str | Path) -> ContinuumArtifacts:
    """Load ``fax continuum`` JSON/CSV artifacts from a directory."""
    root = Path(artifact_dir).expanduser().resolve()
    summary_path = root / "continuum_summary.json"
    metrics_path = root / "continuum_metrics.json"
    branches_path = root / "continuum_branches.csv"
    missing = [str(path) for path in (summary_path, metrics_path, branches_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing continuum artifact file(s): {', '.join(missing)}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    branches = _read_numeric_csv(branches_path)
    return ContinuumArtifacts(artifact_dir=root, summary=summary, metrics=metrics, branches=branches)


def load_convergence_artifacts(artifact_dir: str | Path) -> ConvergenceArtifacts:
    """Load ``fax convergence`` JSON/CSV artifacts from a directory."""
    root = Path(artifact_dir).expanduser().resolve()
    summary_path = root / "convergence_summary.json"
    metrics_path = root / "convergence_metrics.csv"
    missing = [str(path) for path in (summary_path, metrics_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing convergence artifact file(s): {', '.join(missing)}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = _read_numeric_csv(metrics_path)
    return ConvergenceArtifacts(artifact_dir=root, summary=summary, metrics=metrics)


def plot_continuum_artifacts(
    artifact_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    stem: str = "continuum_artifact_dashboard",
    formats: tuple[str, ...] = ("svg", "png"),
    max_branches: int = 10,
) -> tuple[Path, ...]:
    """Plot branch and gap diagnostics from ``fax continuum`` artifacts.

    Matplotlib is imported lazily so the core package can stay lightweight. Use
    the ``plots`` or ``dev`` extras when calling this function.
    """
    artifacts = load_continuum_artifacts(artifact_dir)
    output_path = Path(output_dir).expanduser().resolve() if output_dir is not None else artifacts.artifact_dir
    output_path.mkdir(parents=True, exist_ok=True)
    plt = _setup_matplotlib()

    s = artifacts.branches["s"]
    profile_columns = _branch_columns(artifacts.branches, "profile_")
    metric_columns = _branch_columns(artifacts.branches, "metric_eigen_")
    if not profile_columns or not metric_columns:
        raise ValueError("continuum_branches.csv must contain profile_* and metric_eigen_* columns")

    profile_matrix = _column_matrix(artifacts.branches, profile_columns)
    metric_matrix = _column_matrix(artifacts.branches, metric_columns)
    sorted_metric = np.sort(metric_matrix, axis=1)
    nearest_gap = np.min(np.diff(sorted_metric, axis=1), axis=1)
    min_gap_index = int(np.argmin(nearest_gap))

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), constrained_layout=True)
    _plot_branch_panel(
        axes[0],
        s=s,
        matrix=profile_matrix,
        columns=profile_columns,
        title="Profile continuum branches",
        max_branches=max_branches,
    )
    _plot_branch_panel(
        axes[1],
        s=s,
        matrix=metric_matrix,
        columns=metric_columns,
        title="Metric eigen-branches",
        max_branches=max_branches,
    )

    axes[2].plot(s, nearest_gap / 1.0e3, color="#2e7d32", linewidth=1.9)
    axes[2].scatter([s[min_gap_index]], [nearest_gap[min_gap_index] / 1.0e3], color="#111111", s=22, zorder=3)
    axes[2].set_title("Nearest metric branch gap")
    axes[2].set_xlabel("normalized toroidal flux s")
    axes[2].set_ylabel("gap [kHz]")
    axes[2].set_xlim(float(np.min(s)), float(np.max(s)))
    axes[2].set_ylim(bottom=0.0)

    case = artifacts.summary.get("case", "unknown case")
    residual = artifacts.summary.get("metric_residual_p95")
    residual_text = "" if residual is None else f" | residual p95={float(residual):.2e}"
    fig.suptitle(f"{case}: continuum artifact dashboard{residual_text}", fontsize=12, weight="bold")

    paths = _save_figure(fig, output_path, stem, formats=formats)
    plt.close(fig)
    return paths


def plot_convergence_artifacts(
    artifact_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    stem: str = "convergence_artifact_dashboard",
    formats: tuple[str, ...] = ("svg", "png"),
) -> tuple[Path, ...]:
    """Plot radial, angular, Fourier, and residual metrics from convergence artifacts."""
    artifacts = load_convergence_artifacts(artifact_dir)
    output_path = Path(output_dir).expanduser().resolve() if output_dir is not None else artifacts.artifact_dir
    output_path.mkdir(parents=True, exist_ok=True)
    plt = _setup_matplotlib()

    records = _summary_records(artifacts.summary)
    if not records:
        raise ValueError("convergence_summary.json does not contain any records")
    max_radial = max(int(record["radial_points"]) for record in records)
    max_angular = max(int(record["theta_points"]) for record in records)
    max_mode = max(int(record["m_max"]) for record in records)
    radial_records = _select_records(records, theta_points=max_angular, m_max=max_mode)
    angular_records = _select_records(records, radial_points=max_radial, m_max=max_mode)
    mode_records = _select_records(records, radial_points=max_radial, theta_points=max_angular)

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), constrained_layout=True)
    _plot_metric_curve(
        axes[0],
        records=radial_records,
        x_key="radial_points",
        y_key="frequency_l2_error_vs_reference",
        title="Radial refinement",
        xlabel="radial points",
        color="#2458a6",
        marker="o",
    )
    _plot_metric_curve(
        axes[1],
        records=angular_records,
        x_key="theta_points",
        y_key="frequency_l2_error_vs_reference",
        title="Angular refinement",
        xlabel="theta/zeta points",
        color="#b13f2a",
        marker="s",
    )
    _plot_metric_curve(
        axes[2],
        records=mode_records,
        x_key="branch_count",
        y_key="mode_truncation_l2_error_vs_global_reference",
        title="Mode and residual checks",
        xlabel="branch count",
        color="#2e7d32",
        marker="^",
        label="mode truncation",
    )
    if mode_records:
        axes[2].semilogy(
            [float(record["branch_count"]) for record in mode_records],
            [_positive_plot_value(record["residual_p95"]) for record in mode_records],
            marker="d",
            linewidth=1.4,
            color="#6d3a9c",
            label="residual p95",
        )
        axes[2].legend(loc="best")

    case = artifacts.summary.get("case", "unknown case")
    reference = artifacts.summary.get("reference", {})
    reference_text = ""
    if isinstance(reference, dict):
        reference_text = (
            f" | ref radial={reference.get('radial_points', '?')}, "
            f"theta={reference.get('theta_points', '?')}, m_max={reference.get('m_max', '?')}"
        )
    fig.suptitle(f"{case}: convergence artifact dashboard{reference_text}", fontsize=12, weight="bold")

    paths = _save_figure(fig, output_path, stem, formats=formats)
    plt.close(fig)
    return paths


def plot_basic_alfven_gap_map(
    output_dir: str | Path,
    *,
    demo: BasicAlfvenGapMap | None = None,
    stem: str = "basic_alfven_gap_map",
    formats: tuple[str, ...] = ("svg", "png"),
) -> tuple[Path, ...]:
    """Plot a fast analytic BAE/TAE/EAE continuum gap demonstration."""
    data = basic_alfven_gap_map() if demo is None else demo
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    plt = _setup_matplotlib()

    q = np.asarray(data.q)
    order = np.argsort(q)
    q_sorted = q[order]
    branches = np.asarray(data.continuum_branches_hz)[order, :] / 1.0e3
    tae_lower = np.asarray(data.tae.lower_frequency_hz)[order] / 1.0e3
    tae_upper = np.asarray(data.tae.upper_frequency_hz)[order] / 1.0e3
    eae_lower = np.asarray(data.eae.lower_frequency_hz)[order] / 1.0e3
    eae_upper = np.asarray(data.eae.upper_frequency_hz)[order] / 1.0e3
    bae = np.asarray(data.bae_frequency_hz)[order] / 1.0e3

    fig, (ax, ax_bae) = plt.subplots(
        2,
        1,
        figsize=(7.7, 6.1),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.25]},
    )
    labels = [r"uncoupled $(m,n)=(1,1)$", r"uncoupled $(2,1)$", r"uncoupled $(3,1)$"]
    for index, label in enumerate(labels):
        ax.plot(q_sorted, branches[:, index], color="#767676", linewidth=1.0, linestyle="--", alpha=0.72, label=label)

    ax.plot(q_sorted, tae_lower, color="#2458a6", linewidth=1.9, label="TAE coupled branches")
    ax.plot(q_sorted, tae_upper, color="#2458a6", linewidth=1.9)
    tae_q = float(1.0 / data.tae.estimate.iota)
    eae_q = float(1.0 / data.eae.estimate.iota)
    bae_q = float(data.bae_q)
    ax.fill_between(q_sorted, tae_lower, tae_upper, where=np.abs(q_sorted - tae_q) < 0.28, color="#2458a6", alpha=0.18)

    ax.plot(q_sorted, eae_lower, color="#b13f2a", linewidth=1.9, label="EAE coupled branches")
    ax.plot(q_sorted, eae_upper, color="#b13f2a", linewidth=1.9)
    ax.fill_between(q_sorted, eae_lower, eae_upper, where=np.abs(q_sorted - eae_q) < 0.35, color="#b13f2a", alpha=0.14)

    for gap_q, color, text, y_khz in [
        (tae_q, "#2458a6", "TAE gap\nm,m+1", float(data.tae.estimate.center_frequency_hz / 1.0e3)),
        (eae_q, "#b13f2a", "EAE gap\nm,m+2", float(data.eae.estimate.center_frequency_hz / 1.0e3)),
    ]:
        ax.axvline(gap_q, color=color, linewidth=1.1, alpha=0.45)
        ax.annotate(
            text,
            xy=(gap_q, y_khz),
            xytext=(gap_q + 0.08, y_khz + 42.0),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.0},
            fontsize=8,
            color=color,
            ha="left",
            va="bottom",
        )

    ax.set_title("Basic Alfvén continuum gaps in a large-aspect-ratio tokamak")
    ax.set_ylabel("frequency [kHz]")
    ax.set_xlim(float(np.min(q_sorted)), float(np.max(q_sorted)))
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=7, ncols=2, loc="upper left")
    ax.text(
        0.98,
        0.93,
        r"$\omega_A=v_A|n-m/q|/R$; shaded regions show minimal avoided crossings.",
        transform=ax.transAxes,
        fontsize=8,
        color="#333333",
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.0},
    )

    ax_bae.plot(q_sorted, bae, color="#2e7d32", linewidth=2.0, label="BAE accumulation")
    ax_bae.fill_between(q_sorted, 0.9 * bae, 1.1 * bae, color="#2e7d32", alpha=0.15)
    ax_bae.axvline(bae_q, color="#2e7d32", linewidth=1.1, alpha=0.55)
    ax_bae.annotate(
        "BAE gap\nfinite beta",
        xy=(bae_q, float(data.bae_frequency_at_surface_hz / 1.0e3)),
        xytext=(bae_q + 0.08, float(data.bae_frequency_at_surface_hz / 1.0e3) + 5.0),
        arrowprops={"arrowstyle": "->", "color": "#2e7d32", "lw": 1.0},
        fontsize=8,
        color="#2e7d32",
        ha="left",
        va="bottom",
    )
    ax_bae.set_xlabel("safety factor q")
    ax_bae.set_ylabel("frequency [kHz]")
    ax_bae.set_ylim(bottom=0.0)
    ax_bae.legend(fontsize=7, loc="upper right")

    paths = _save_figure(fig, output_path, stem, formats=formats)
    plt.close(fig)
    return paths


def _summary_records(summary: dict[str, Any]) -> list[dict[str, float]]:
    raw_records = summary.get("records", [])
    records: list[dict[str, float]] = []
    for raw in raw_records:
        if not isinstance(raw, dict):
            continue
        record: dict[str, float] = {}
        for key, value in raw.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and np.isfinite(value):
                record[key] = float(value)
        records.append(record)
    return records


def _select_records(records: list[dict[str, float]], **criteria: int) -> list[dict[str, float]]:
    selected = []
    for record in records:
        if all(int(record.get(key, -1)) == int(value) for key, value in criteria.items()):
            selected.append(record)
    return sorted(
        selected,
        key=lambda item: (
            int(item.get("radial_points", 0)),
            int(item.get("theta_points", 0)),
            int(item.get("m_max", 0)),
            int(item.get("branch_count", 0)),
        ),
    )


def _plot_metric_curve(
    ax: Any,
    *,
    records: list[dict[str, float]],
    x_key: str,
    y_key: str,
    title: str,
    xlabel: str,
    color: str,
    marker: str,
    label: str | None = None,
) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("relative metric")
    if not records:
        ax.text(0.5, 0.5, "not enough records", transform=ax.transAxes, ha="center", va="center")
        return
    x_values = [float(record[x_key]) for record in records if x_key in record and y_key in record]
    y_values = [_positive_plot_value(record[y_key]) for record in records if x_key in record and y_key in record]
    ax.semilogy(x_values, y_values, marker=marker, linewidth=1.8, color=color, label=label)


def _positive_plot_value(value: float, *, floor: float = 1.0e-16) -> float:
    value_float = float(value)
    return value_float if np.isfinite(value_float) and value_float > floor else floor


def _read_numeric_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        columns: dict[str, list[float]] = {field: [] for field in reader.fieldnames}
        for row in reader:
            for field in reader.fieldnames:
                columns[field].append(float(row[field]))
    return {field: np.asarray(values, dtype=float) for field, values in columns.items()}


def _branch_columns(branches: dict[str, np.ndarray], prefix: str) -> tuple[str, ...]:
    return tuple(column for column in branches if column.startswith(prefix) and column.endswith("_hz"))


def _column_matrix(branches: dict[str, np.ndarray], columns: tuple[str, ...]) -> np.ndarray:
    return np.column_stack([branches[column] for column in columns])


def _plot_branch_panel(
    ax: Any,
    *,
    s: np.ndarray,
    matrix: np.ndarray,
    columns: tuple[str, ...],
    title: str,
    max_branches: int,
) -> None:
    for index, column in enumerate(columns[:max_branches]):
        linewidth = 1.35 if index < 4 else 0.9
        alpha = 0.95 if index < 4 else 0.55
        ax.plot(s, matrix[:, index] / 1.0e3, linewidth=linewidth, alpha=alpha, label=_readable_label(column))
    ax.set_title(title)
    ax.set_xlabel("normalized toroidal flux s")
    ax.set_ylabel("frequency [kHz]")
    ax.set_xlim(float(np.min(s)), float(np.max(s)))
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=7, loc="best")


def _readable_label(column: str) -> str:
    label = column.removeprefix("profile_").removeprefix("metric_diagonal_").removeprefix("metric_eigen_")
    label = label.removesuffix("_hz")
    if label.isdigit():
        return f"eigen {int(label)}"
    return label.replace("minus", "-").replace("_", ", ")


def _setup_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "font.size": 9,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "svg.hashsalt": "fax-artifact-plots",
        }
    )
    return plt


def _save_figure(fig: Any, output_dir: Path, stem: str, *, formats: tuple[str, ...]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for suffix in formats:
        normalized = suffix.lower().removeprefix(".")
        if normalized not in {"png", "svg", "pdf"}:
            raise ValueError(f"unsupported figure format: {suffix!r}")
        path = output_dir / f"{stem}.{normalized}"
        if normalized == "svg":
            fig.savefig(path, metadata={"Date": None})
            _normalize_svg(path)
        elif normalized == "png":
            fig.savefig(path, facecolor="white")
        else:
            fig.savefig(path)
        paths.append(path)
    return tuple(paths)


def _normalize_svg(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
