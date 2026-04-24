"""Generate validation and reference figures for the Sphinx docs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from fax.config import EquilibriumConfig, FaxConfig, GridConfig, RunConfig
from fax.continuum import continuum_branch_matrix
from fax.equilibrium import analytic_circular_tokamak, from_vmec_wout
from fax.gaps import minimum_gap_summary, sorted_branch_gaps
from fax.metric import (
    diagonal_continuum_frequencies_hz,
    generalized_eigen_residual_norms,
    generalized_hermitian_eigensystem,
    metric_continuum_matrices,
    metric_from_equilibrium,
)
from fax.modes import stellarator_mode_family
from fax.plotting import plot_continuum_artifacts
from fax.reference_cases import landreman_paul_qa_lowres
from fax.workflows import run_continuum, run_convergence
from fax.validation import registered_validation_gates, run_validation_suite


def _setup_matplotlib():
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
            "svg.hashsalt": "fax-validation-figures",
        }
    )
    return plt


def _harmonic_label(m: int, n: int) -> str:
    return f"m={m}, n={n}"


def _normalize_svg(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def _save_figure(fig, output_dir: Path, stem: str) -> list[Path]:
    """Save SVG and PNG variants for docs and README rendering."""
    svg_path = output_dir / f"{stem}.svg"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(svg_path, metadata={"Date": None})
    _normalize_svg(svg_path)
    fig.savefig(png_path, facecolor="white")
    return [svg_path, png_path]


def generate_cylinder_scaling(output_dir: Path) -> list[Path]:
    """Generate an analytic continuum scaling figure."""
    plt = _setup_matplotlib()
    geometry = analytic_circular_tokamak(radial_points=220, iota_axis=0.22, iota_edge=0.82)
    harmonics = jnp.asarray([[1, 0], [2, 1], [3, 2], [4, 3]], dtype=float)
    branches_khz = np.asarray(continuum_branch_matrix(geometry, harmonics)) / 1.0e3

    fig, ax = plt.subplots(figsize=(6.3, 3.4), constrained_layout=True)
    for index, (m, n) in enumerate(np.asarray(harmonics, dtype=int)):
        ax.plot(np.asarray(geometry.s), branches_khz[:, index], linewidth=1.8, label=_harmonic_label(int(m), int(n)))

    ax.set_title("Analytic large-aspect-ratio continuum smoke test")
    ax.set_xlabel("normalized toroidal flux s")
    ax.set_ylabel("frequency [kHz]")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.legend(ncol=2, loc="upper center")

    paths = _save_figure(fig, output_dir, "analytic_continuum_scaling")
    plt.close(fig)
    return paths


def generate_landreman_reference(output_dir: Path) -> list[Path]:
    """Generate the first Landreman-Paul QA reference-profile figure."""
    plt = _setup_matplotlib()
    case = landreman_paul_qa_lowres()
    if case.wout is None or not case.wout.exists():
        raise FileNotFoundError(f"Landreman-Paul QA wout not found: {case.wout}")

    geometry = from_vmec_wout(case.wout, name=case.case_id)
    harmonics = jnp.asarray([[1, 0], [2, 1], [3, 2], [4, 3]], dtype=float)
    branches_khz = np.asarray(continuum_branch_matrix(geometry, harmonics)) / 1.0e3
    s = np.asarray(geometry.s)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)

    ax = axes[0]
    ax.plot(s, np.asarray(geometry.iota), color="#2458a6", linewidth=1.9, label="iota")
    ax.set_title("LandremanPaul2021 QA lowres profiles")
    ax.set_xlabel("normalized toroidal flux s")
    ax.set_ylabel("rotational transform")
    ax.set_xlim(0.0, 1.0)
    ax_b = ax.twinx()
    ax_b.plot(s, np.asarray(geometry.magnetic_field), color="#b13f2a", linewidth=1.7, label="B00")
    ax_b.set_ylabel("representative |B| [T]")
    lines = ax.get_lines() + ax_b.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="best")

    ax = axes[1]
    for index, (m, n) in enumerate(np.asarray(harmonics, dtype=int)):
        ax.plot(s, branches_khz[:, index], linewidth=1.7, label=_harmonic_label(int(m), int(n)))
    ax.set_title("Local continuum smoke-test branches")
    ax.set_xlabel("normalized toroidal flux s")
    ax.set_ylabel("frequency [kHz]")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=8)

    paths = _save_figure(fig, output_dir, "landreman_paul_qa_lowres_profiles")
    plt.close(fig)
    return paths


def generate_gap_sensitivity_dashboard(output_dir: Path) -> list[Path]:
    """Generate a mode-family/gap/sensitivity dashboard figure."""
    plt = _setup_matplotlib()
    case = landreman_paul_qa_lowres()
    if case.wout is None or not case.wout.exists():
        raise FileNotFoundError(f"Landreman-Paul QA wout not found: {case.wout}")

    geometry = from_vmec_wout(case.wout, name=case.case_id)
    family = stellarator_mode_family(
        poloidal_modes=range(1, 5),
        base_n=1,
        field_periods=geometry.field_periods,
        sidebands=(-1, 0, 1),
    )
    branches = continuum_branch_matrix(geometry, family.as_array())
    branches_khz = np.asarray(branches) / 1.0e3
    s = np.asarray(geometry.s)
    sorted_branches, gaps = sorted_branch_gaps(branches)
    min_gap = minimum_gap_summary(branches, geometry.s)

    def selected_branch_mean(density_scale, iota_shift, field_scale):
        adjusted = geometry.with_mass_density(geometry.mass_density * density_scale)
        adjusted = adjusted.__class__(
            name=adjusted.name,
            s=adjusted.s,
            iota=adjusted.iota + iota_shift,
            magnetic_field=adjusted.magnetic_field * field_scale,
            mass_density=adjusted.mass_density,
            major_radius=adjusted.major_radius,
            minor_radius=adjusted.minor_radius,
            field_periods=adjusted.field_periods,
            metadata=adjusted.metadata,
        )
        return jnp.mean(continuum_branch_matrix(adjusted, jnp.asarray([[2.0, 1.0]])))

    density_sensitivity = jax.grad(lambda x: selected_branch_mean(x, 0.0, 1.0))(1.0)
    iota_sensitivity = jax.grad(lambda x: selected_branch_mean(1.0, x, 1.0))(0.0)
    field_sensitivity = jax.grad(lambda x: selected_branch_mean(1.0, 0.0, x))(1.0)
    sensitivities = np.asarray([density_sensitivity, iota_sensitivity, field_sensitivity], dtype=float) / 1.0e3

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.1), constrained_layout=True)

    ax = axes[0]
    for index, label in enumerate(family.labels()):
        linewidth = 1.1 if index % 3 == 1 else 0.75
        alpha = 0.95 if index % 3 == 1 else 0.45
        ax.plot(s, branches_khz[:, index], linewidth=linewidth, alpha=alpha, label=label if index % 3 == 1 else None)
    ax.set_title("QA mode-family branches")
    ax.set_xlabel("s")
    ax.set_ylabel("frequency [kHz]")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[1]
    min_gap_khz = np.min(np.asarray(gaps), axis=1) / 1.0e3
    ax.plot(s, min_gap_khz, color="#2e7d32", linewidth=1.8)
    ax.scatter([min_gap.surface], [min_gap.min_gap_hz / 1.0e3], color="#111111", s=22, zorder=3)
    ax.set_title("Nearest-branch separation")
    ax.set_xlabel("s")
    ax.set_ylabel("minimum gap [kHz]")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)

    ax = axes[2]
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    labels = ["density scale", "iota shift", "field scale"]
    ax.bar(labels, sensitivities, color=colors)
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_title("Autodiff sensitivity")
    ax.set_ylabel("d mean f / d parameter [kHz]")
    ax.tick_params(axis="x", rotation=28)

    paths = _save_figure(fig, output_dir, "gap_sensitivity_dashboard")
    plt.close(fig)
    return paths


def generate_continuum_artifact_dashboard(output_dir: Path) -> list[Path]:
    """Generate a dashboard from files written by the continuum workflow."""
    case = landreman_paul_qa_lowres()
    if case.wout is None or not case.wout.exists():
        raise FileNotFoundError(f"Landreman-Paul QA wout not found: {case.wout}")

    with tempfile.TemporaryDirectory(prefix="fax-continuum-artifacts-") as tmp:
        config = FaxConfig(
            run=RunConfig(case=case.case_id, model="continuum", output_dir=tmp),
            equilibrium=EquilibriumConfig(
                source="wout",
                vmec_input=str(case.vmec_input) if case.vmec_input is not None else None,
                wout=str(case.wout),
            ),
            grid=GridConfig(radial_points=50, m_max=4, n_max=4, field_periods=2),
        )
        run_continuum(config, progress=False)
        return list(
            plot_continuum_artifacts(
                tmp,
                output_dir=output_dir,
                stem="continuum_artifact_dashboard",
                max_branches=9,
            )
        )


def generate_workflow_diagram(output_dir: Path) -> list[Path]:
    """Generate a newcomer-facing input/operator/output workflow diagram."""
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(9.4, 4.2), constrained_layout=True)
    ax.axis("off")

    boxes = [
        ("Inputs", "TOML case\nVMEC wout / vmec_jax\nprofiles\nmode family", 0.04, 0.55, "#e8f1fb"),
        ("Adapters", "EquilibriumGeometry\nMetricGeometry\nBoozer path planned", 0.28, 0.55, "#e9f7ef"),
        ("Operators", "continuum branches\nK(s), M(s)\nvalidation gates", 0.52, 0.55, "#fff4df"),
        ("Outputs", "frequencies\nmode labels\ngaps, sensitivities\nfigures, metrics", 0.76, 0.55, "#f4ecfb"),
    ]
    for title, body, x, y, color in boxes:
        patch = plt.Rectangle((x, y - 0.22), 0.19, 0.38, facecolor=color, edgecolor="#333333", linewidth=1.0)
        ax.add_patch(patch)
        ax.text(x + 0.095, y + 0.095, title, ha="center", va="center", weight="bold", fontsize=11)
        ax.text(x + 0.095, y - 0.045, body, ha="center", va="center", fontsize=9, linespacing=1.35)
    for x0 in (0.23, 0.47, 0.71):
        ax.annotate("", xy=(x0 + 0.035, 0.55), xytext=(x0, 0.55), arrowprops={"arrowstyle": "->", "lw": 1.7})

    ax.text(0.5, 0.18, "JAX transformations: jit, vmap, grad over operators and diagnostics", ha="center", fontsize=10)
    ax.text(0.5, 0.09, "Current scope: scaffold gates and metric-matrix checks; full 3-D STELLGAP/AE3D validation is planned.", ha="center", fontsize=9)

    paths = _save_figure(fig, output_dir, "workflow_io_diagram")
    plt.close(fig)
    return paths


def generate_equations_overview(output_dir: Path) -> list[Path]:
    """Generate a compact equation overview figure."""
    plt = _setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), constrained_layout=True)
    for ax in axes:
        ax.axis("off")

    axes[0].set_title("Local shear-Alfven continuum", fontsize=12, weight="bold")
    axes[0].text(
        0.02,
        0.72,
        r"$v_A(s)=B(s)/\sqrt{\mu_0\rho(s)}$" "\n"
        r"$k_{\parallel,mn}(s)=[n-m\iota(s)]/R$" "\n"
        r"$\omega_{A,mn}(s)=v_A(s)\,|k_{\parallel,mn}(s)|$",
        fontsize=13,
        va="top",
    )
    axes[0].text(
        0.02,
        0.21,
        "Used for analytic limits, profile smoke tests,\nand density/iota/field sensitivity checks.",
        fontsize=9,
    )

    axes[1].set_title("Metric continuum matrix", fontsize=12, weight="bold")
    axes[1].text(
        0.02,
        0.75,
        r"$K_{ij}=\left\langle J\,B^2 k_i k_j\phi_i^*\phi_j/\mu_0\right\rangle$" "\n"
        r"$M_{ij}=\left\langle J\,\rho\,\phi_i^*\phi_j\right\rangle$" "\n"
        r"$K(s)x=\omega^2\,M(s)x$",
        fontsize=12,
        va="top",
    )
    axes[1].text(
        0.02,
        0.21,
        "Used for Hermiticity, residual, angular-grid,\nand VMEC-adapter gates.",
        fontsize=9,
    )

    fig.suptitle("Equations currently implemented in FAX", fontsize=13, weight="bold")
    paths = _save_figure(fig, output_dir, "equations_overview")
    plt.close(fig)
    return paths


def generate_validation_gate_summary(output_dir: Path) -> list[Path]:
    """Generate a pass/fail summary for the registered validation gates."""
    plt = _setup_matplotlib()
    gates = registered_validation_gates()
    results_by_id = {result.gate_id: result for result in run_validation_suite()}
    labels = [gate.gate_id.replace("_", "\n") for gate in gates]
    ratios = []
    colors = []
    statuses = []
    for gate in gates:
        result = results_by_id[gate.gate_id]
        statuses.append(result.status)
        if result.status == "skipped" or not np.isfinite(result.metric):
            ratios.append(1.0)
            colors.append("#9e9e9e")
        else:
            ratios.append(result.metric / gate.tolerance if gate.tolerance > 0.0 else 0.0)
            colors.append("#2e7d32" if result.status == "passed" else "#b3261e")

    y = np.arange(len(gates))
    fig, ax = plt.subplots(figsize=(8.6, 4.4), constrained_layout=True)
    ax.barh(y, ratios, color=colors)
    ax.axvline(1.0, color="#222222", linewidth=1.0, linestyle="--", label="tolerance")
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel("metric / tolerance")
    ax.set_title("Registered validation gates")
    ax.set_xlim(0.0, max(1.2, float(np.nanmax(ratios)) * 1.15))
    for idx, (ratio, status) in enumerate(zip(ratios, statuses, strict=True)):
        ax.text(min(ratio + 0.03, ax.get_xlim()[1] * 0.96), idx, status, va="center", fontsize=8)
    ax.legend(loc="lower right")

    paths = _save_figure(fig, output_dir, "validation_gate_summary")
    plt.close(fig)
    return paths


def generate_metric_residual_convergence(output_dir: Path) -> list[Path]:
    """Generate a residual and angular-refinement smoke-test figure."""
    plt = _setup_matplotlib()
    geometry = analytic_circular_tokamak(radial_points=28, iota_axis=0.24, iota_edge=0.72)
    harmonics = jnp.asarray([[1, 0], [2, 2], [3, 3]], dtype=float)
    profile = continuum_branch_matrix(geometry, harmonics)
    angular_points = np.asarray([8, 12, 16, 24, 32, 48])
    diagonal_errors = []
    residuals = []
    for count in angular_points:
        metric_geometry = metric_from_equilibrium(geometry, theta_points=int(count), zeta_points=int(count))
        diagonal = diagonal_continuum_frequencies_hz(metric_geometry, harmonics)
        matrices = metric_continuum_matrices(metric_geometry, harmonics)
        eigensystem = generalized_hermitian_eigensystem(matrices.stiffness, matrices.mass)
        residual = generalized_eigen_residual_norms(matrices.stiffness, matrices.mass, eigensystem)
        diagonal_errors.append(float(jnp.linalg.norm(diagonal - profile) / jnp.maximum(jnp.linalg.norm(profile), 1.0e-30)))
        residuals.append(float(jnp.max(residual)))

    fig, ax = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
    ax.loglog(angular_points, diagonal_errors, marker="o", linewidth=1.8, label="profile-limit frequency error")
    ax.loglog(angular_points, residuals, marker="s", linewidth=1.8, label="max generalized residual")
    ax.axhline(1.0e-5, color="#555555", linestyle="--", linewidth=1.0, label="residual gate")
    ax.set_xlabel("theta/zeta grid points")
    ax.set_ylabel("relative metric")
    ax.set_title("Metric scaffold residual and angular-grid checks")
    ax.legend()

    paths = _save_figure(fig, output_dir, "metric_residual_convergence")
    plt.close(fig)
    return paths


def generate_landreman_convergence_dashboard(output_dir: Path) -> list[Path]:
    """Generate a Landreman-Paul adapter convergence dashboard."""
    plt = _setup_matplotlib()
    case = landreman_paul_qa_lowres()
    if case.wout is None or not case.wout.exists():
        raise FileNotFoundError(f"Landreman-Paul QA wout not found: {case.wout}")

    config = FaxConfig(
        run=RunConfig(case=case.case_id, model="continuum", output_dir=str(output_dir)),
        equilibrium=EquilibriumConfig(
            source="wout",
            vmec_input=str(case.vmec_input) if case.vmec_input is not None else None,
            wout=str(case.wout),
        ),
        grid=GridConfig(radial_points=50, m_max=2, n_max=2, field_periods=2),
    )
    summary = run_convergence(
        config,
        progress=False,
        radial_points=(24, 40, 50),
        angular_points=(8, 12),
        mode_max_values=(1, 2),
        write_artifacts=False,
    )
    records = summary.records
    radial_values = np.asarray(summary.radial_points)
    max_radial = int(np.max(radial_values))
    max_angular = int(np.max(summary.angular_points))
    max_mode = int(np.max(summary.mode_max_values))

    def _select(**criteria):
        selected = []
        for record in records:
            if all(int(record[key]) == int(value) for key, value in criteria.items()):
                selected.append(record)
        return sorted(selected, key=lambda item: (int(item["radial_points"]), int(item["theta_points"]), int(item["m_max"])))

    radial_records = _select(theta_points=max_angular, m_max=max_mode)
    angular_records = _select(radial_points=max_radial, m_max=max_mode)
    mode_records = _select(radial_points=max_radial, theta_points=max_angular)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4), constrained_layout=True)

    ax = axes[0]
    ax.semilogy(
        [record["radial_points"] for record in radial_records],
        [max(record["frequency_l2_error_vs_reference"], 1.0e-16) for record in radial_records],
        marker="o",
        linewidth=1.8,
        color="#2458a6",
    )
    ax.set_title("Radial refinement")
    ax.set_xlabel("radial points")
    ax.set_ylabel("relative L2 branch error")

    ax = axes[1]
    ax.semilogy(
        [record["theta_points"] for record in angular_records],
        [max(record["frequency_l2_error_vs_reference"], 1.0e-16) for record in angular_records],
        marker="s",
        linewidth=1.8,
        color="#b13f2a",
    )
    ax.set_title("Angular refinement")
    ax.set_xlabel("theta/zeta points")
    ax.set_ylabel("relative L2 branch error")

    ax = axes[2]
    ax.semilogy(
        [record["branch_count"] for record in mode_records],
        [max(record["mode_truncation_l2_error_vs_global_reference"], 1.0e-16) for record in mode_records],
        marker="^",
        linewidth=1.8,
        color="#2e7d32",
        label="mode truncation",
    )
    ax.semilogy(
        [record["branch_count"] for record in mode_records],
        [max(record["residual_p95"], 1.0e-16) for record in mode_records],
        marker="d",
        linewidth=1.4,
        color="#6d3a9c",
        label="residual p95",
    )
    ax.set_title("Mode/residual checks")
    ax.set_xlabel("branch count")
    ax.set_ylabel("relative metric")
    ax.legend(loc="best")

    fig.suptitle("LandremanPaul2021 QA adapter convergence smoke test", fontsize=12, weight="bold")
    paths = _save_figure(fig, output_dir, "landreman_paul_convergence_dashboard")
    plt.close(fig)
    return paths


def write_figure_manifest(output_dir: Path, paths: list[Path]) -> Path:
    """Write a JSON manifest describing generated figures and their claims."""
    manifest = {
        "generator": "tools/generate_validation_figures.py",
        "validation_gates": [
            {
                "id": gate.gate_id,
                "title": gate.title,
                "layer": gate.layer,
                "quantity": gate.quantity,
                "tolerance": gate.tolerance,
                "citation": gate.citation,
                "command": gate.command,
                "tags": list(gate.tags),
            }
            for gate in registered_validation_gates()
        ],
        "figures": [
            {
                "file": path.name,
                "format": path.suffix.removeprefix("."),
                "status": "smoke-test",
                "claim": (
                    "Development figure generated from analytic/local continuum formulas; "
                    "not a full 3-D stellarator Alfvén-continuum validation."
                ),
            }
            for path in paths
        ],
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="docs/_static", help="Directory for generated SVG figures.")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for generated in [
        generate_workflow_diagram(output_dir),
        generate_equations_overview(output_dir),
        generate_cylinder_scaling(output_dir),
        generate_landreman_reference(output_dir),
        generate_gap_sensitivity_dashboard(output_dir),
        generate_continuum_artifact_dashboard(output_dir),
        generate_metric_residual_convergence(output_dir),
        generate_landreman_convergence_dashboard(output_dir),
        generate_validation_gate_summary(output_dir),
    ]:
        paths.extend(generated)
    paths.append(write_figure_manifest(output_dir, paths))
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
