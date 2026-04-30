"""Generate validation and reference figures for the Sphinx docs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from fax.boozer import (
    BoozerSpectrum,
    boozer_3d_continuum_frequencies_hz,
    boozer_3d_continuum_matrices,
    boozer_3d_operator_angular_convergence,
    boozer_angle_grid,
    boozer_continuum_frequencies_hz,
    boozer_continuum_matrices,
    boozer_embedding_jacobian_shape_error,
    boozer_geometry_field_relative_error,
    boozer_geometry_grid,
    boozer_geometry_jacobian_identity_error,
    boozer_harmonic_orthogonality_error,
    boozer_jacobian_identity_error,
    boozer_jacobian_weight,
    boozer_metric_continuum_matrices,
    boozer_projection_error,
    normalized_nonaxisymmetric_boozer_energy,
    read_boozmn_geometry,
    reconstruct_boozer_field,
    run_booz_xform_jax_geometry_from_wout_data,
    run_vmec_jax_boozer_solve_convergence,
    run_vmec_jax_fixed_boundary_wout_data,
    straight_field_line_phase_error,
)
from fax.config import EquilibriumConfig, FaxConfig, GridConfig, RunConfig
from fax.continuum import continuum_branch_matrix
from fax.cross_code import (
    ae3d_dense_spectrum_parity,
    ae3d_eigenvalue_residual_summary,
    ae3d_global_mode_parity,
    ae3d_matrix_symmetry_summary,
    ae3d_omega2_eigenvalue_error,
    read_ae3d_eigenvalues,
    read_ae3d_omega2,
    read_stellgap_alfven_post,
    stellgap_postprocess_error,
    stellgap_spec_postprocessed_continuum,
)
from fax.equilibrium import analytic_circular_tokamak, from_vmec_wout
from fax.fast_ions import FastIonPopulation, resonance_scan
from fax.gaps import minimum_gap_summary, sorted_branch_gaps
from fax.gap_models import eae_gap_estimate, hae_gap_estimate, tae_gap_estimate
from fax.global_modes import constant_problem_discrete_omega2, constant_radial_problem, solve_radial_eigenmodes
from fax.global_modes import radial_block_problem_from_boozer_geometry, solve_radial_block_eigenmodes
from fax.metric import (
    diagonal_continuum_frequencies_hz,
    generalized_eigen_residual_norms,
    generalized_hermitian_eigensystem,
    generalized_hermitian_eigenvalues,
    metric_continuum_matrices,
    metric_from_equilibrium,
)
from fax.modes import stellarator_mode_family
from fax.optimization import radial_mode_avoidance_objective
from fax.plotting import (
    plot_basic_alfven_gap_map,
    plot_continuum_artifacts,
    plot_convergence_artifacts,
    plot_global_eigenmode_artifacts,
)
from fax.reference_cases import landreman_paul_qa_lowres
from fax.units import PROTON_MASS
from fax.workflows import run_benchmark, run_continuum, run_convergence, run_global_eigenmodes
from fax.validation import registered_validation_gates, run_validation_suite

jax.config.update("jax_enable_x64", True)


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


def generate_global_gap_fastion_dashboard(output_dir: Path) -> list[Path]:
    """Generate a dashboard for global modes, gap estimates, fast ions, and gradients."""
    plt = _setup_matplotlib()
    problem = constant_radial_problem(radial_points=64, continuum_frequency_hz=105_000.0, radial_coupling_omega2=3.0e9)
    result = solve_radial_eigenmodes(problem)
    exact_frequency_hz = np.sqrt(np.asarray(constant_problem_discrete_omega2(problem))) / (2.0 * np.pi)
    numerical_frequency_hz = np.asarray(result.frequencies_hz)
    interior_s = np.asarray(problem.s[1:-1])
    mode_shapes = np.asarray(result.eigenvectors[:, :3])
    mode_shapes = mode_shapes / np.maximum(np.max(np.abs(mode_shapes), axis=0, keepdims=True), 1.0e-30)

    geometry = analytic_circular_tokamak(radial_points=256, iota_axis=0.42, iota_edge=0.86, major_radius=6.2)
    gap_estimates = [
        tae_gap_estimate(geometry, m=1, n=1, inverse_aspect_ratio=0.08),
        eae_gap_estimate(geometry, m=1, n=1, ellipticity=0.055),
        hae_gap_estimate(geometry, m=2, n=2, helical_coupling=0.07, delta_m=1),
    ]

    population = FastIonPopulation(
        density_fraction=0.045,
        pressure_gradient=-2.5,
        transit_frequency_hz=32_000.0,
        precession_frequency_hz=4_500.0,
        resonance_width_hz=2_800.0,
    )
    frequency_grid = jnp.linspace(5_000.0, 105_000.0, 360)
    harmonics = (-1, 0, 1, 2)
    drive = resonance_scan(frequency_grid, toroidal_mode=2, transit_harmonics=harmonics, population=population)

    design_geometry = analytic_circular_tokamak(radial_points=72, iota_axis=0.28, iota_edge=0.70)
    parameters = jnp.asarray([0.0, 0.0, 0.0])

    def objective(params):
        return radial_mode_avoidance_objective(
            params,
            geometry=design_geometry,
            harmonic=(2, 1),
            avoid_window_hz=(115_000.0, 145_000.0),
            softness_hz=4_000.0,
        )

    gradient = np.asarray(jax.grad(objective)(parameters))

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.6), constrained_layout=True)

    ax = axes[0, 0]
    mode_index = np.arange(1, 9)
    ax.plot(mode_index, exact_frequency_hz[:8] / 1.0e3, marker="o", linewidth=1.7, label="exact FD spectrum")
    ax.plot(mode_index, numerical_frequency_hz[:8] / 1.0e3, marker="s", linewidth=1.2, linestyle="--", label="FAX eigensolve")
    ax.set_title("Global radial spectrum gate")
    ax.set_xlabel("radial mode index")
    ax.set_ylabel("frequency [kHz]")
    ax.legend(fontsize=8)
    ax.text(
        0.04,
        0.08,
        f"max residual={float(jnp.max(result.residuals[:8])):.1e}",
        transform=ax.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "boxstyle": "round,pad=0.25"},
    )

    ax = axes[0, 1]
    for index in range(3):
        ax.plot(interior_s, mode_shapes[:, index], linewidth=1.6, label=f"mode {index + 1}")
    ax.set_title("Dirichlet radial eigenfunctions")
    ax.set_xlabel("normalized toroidal flux s")
    ax.set_ylabel("normalized envelope")
    ax.set_xlim(0.0, 1.0)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    labels = [estimate.family for estimate in gap_estimates]
    centers = np.asarray([float(estimate.center_frequency_hz) for estimate in gap_estimates]) / 1.0e3
    half_widths = np.asarray([float(estimate.width_frequency_hz) for estimate in gap_estimates]) / 2.0e3
    x = np.arange(len(gap_estimates))
    ax.errorbar(x, centers, yerr=half_widths, fmt="o", capsize=4, color="#2458a6", linewidth=1.5)
    ax.set_xticks(x, labels)
    ax.set_title("Analytic gap-center gates")
    ax.set_ylabel("frequency [kHz]")
    ax.set_xlim(-0.35, len(gap_estimates) - 0.65)
    ax.set_ylim(bottom=max(0.0, float(np.min(centers - half_widths)) - 8.0), top=float(np.max(centers + half_widths)) + 12.0)
    for x_index, estimate in enumerate(gap_estimates):
        ax.annotate(
            f"iota={float(estimate.iota):.3f}",
            xy=(x_index, centers[x_index]),
            xytext=(0, 11),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    ax = axes[1, 1]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    for index, harmonic in enumerate(harmonics):
        ax.plot(np.asarray(frequency_grid) / 1.0e3, np.asarray(drive[index]), linewidth=1.5, color=colors[index], label=f"l={harmonic}")
    ax.set_title("Fast-ion resonance screen")
    ax.set_xlabel("mode frequency [kHz]")
    ax.set_ylabel("reduced drive")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    inset = ax.inset_axes([0.58, 0.12, 0.36, 0.34])
    inset_x = np.arange(3)
    inset.bar(inset_x, gradient, color=["#4c78a8", "#f58518", "#54a24b"])
    inset.set_xticks(inset_x, ["rho", "iota", "B"])
    inset.axhline(0.0, color="#222222", linewidth=0.7)
    inset.set_title("AD objective grad", fontsize=8)
    inset.tick_params(axis="both", labelsize=7)

    fig.suptitle("Reduced global-mode, gap, fast-ion, and optimization validation dashboard", fontsize=12, weight="bold")
    paths = _save_figure(fig, output_dir, "global_gap_fastion_dashboard")
    plt.close(fig)
    return paths


def generate_boozer_harmonic_validation(output_dir: Path) -> list[Path]:
    """Generate a Boozer harmonic reconstruction/projection validation figure."""
    plt = _setup_matplotlib()
    spectrum = BoozerSpectrum(
        source="synthetic_figure",
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
    theta, zeta = boozer_angle_grid(theta_points=96, zeta_points=96)
    field = reconstruct_boozer_field(spectrum, surface_index=0, theta=theta, zeta=zeta)
    harmonics = jnp.column_stack([spectrum.xm, spectrum.xn])
    mode_labels = [f"({int(m)},{int(n)})" for m, n in np.asarray(harmonics)]
    nonaxisymmetric_energy = float(normalized_nonaxisymmetric_boozer_energy(spectrum))
    metrics = {
        "projection": float(boozer_projection_error(spectrum, surface_index=0, theta_points=96, zeta_points=96)),
        "orthogonality": float(boozer_harmonic_orthogonality_error(spectrum.xm, spectrum.xn, theta_points=96, zeta_points=96)),
        "phase slope": float(straight_field_line_phase_error(harmonics, iota=0.73, zeta_points=128)),
    }

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), constrained_layout=True)

    ax = axes[0]
    image = ax.imshow(
        np.asarray(field),
        origin="lower",
        extent=(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi),
        aspect="auto",
        cmap="viridis",
    )
    ax.set_title(r"Reconstructed $|B|(\theta_B,\zeta_B)$")
    ax.set_xlabel(r"$\zeta_B$")
    ax.set_ylabel(r"$\theta_B$")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="T")

    ax = axes[1]
    x = np.arange(len(mode_labels))
    width = 0.38
    ax.bar(x - width / 2.0, np.asarray(spectrum.bmnc[0]), width=width, label="cos")
    ax.bar(x + width / 2.0, np.asarray(spectrum.bmns[0]), width=width, label="sin")
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.set_xticks(x, mode_labels)
    ax.set_title("Synthetic Boozer coefficients")
    ax.set_xlabel("(m,n)")
    ax.set_ylabel("coefficient [T]")
    ax.legend(fontsize=8)

    ax = axes[2]
    metric_labels = list(metrics)
    metric_values = np.asarray([metrics[label] for label in metric_labels])
    gate_values = np.asarray([2.0e-6, 2.0e-6, 1.0e-4])
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    x_metric = np.arange(len(metric_labels))
    ax.bar(x_metric, metric_values, color=colors)
    ax.scatter(x_metric, gate_values, color="#222222", marker="_", s=260, linewidths=2.0, label="gate")
    ax.set_xticks(x_metric, metric_labels)
    ax.set_yscale("log")
    ax.set_title("Boozer representation gates")
    ax.set_ylabel("metric")
    ax.tick_params(axis="x", rotation=25)
    ax.text(
        0.04,
        0.90,
        f"non-axisymmetric energy={nonaxisymmetric_energy:.2e}",
        transform=ax.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "boxstyle": "round,pad=0.25"},
    )
    ax.legend(fontsize=8)

    fig.suptitle("Boozer harmonic validation: reconstruction, projection, and straight-field-line phase", fontsize=12, weight="bold")
    paths = _save_figure(fig, output_dir, "boozer_harmonic_validation")
    plt.close(fig)
    return paths


def generate_boozer_continuum_coupling(output_dir: Path) -> list[Path]:
    """Generate a reduced Boozer continuum coupling validation figure."""
    plt = _setup_matplotlib()
    magnetic_field = 2.7
    mass_density = 3.4e-8
    major_radius = 5.5
    iota = 0.61
    constant_spectrum = BoozerSpectrum(
        source="constant_boozer_figure",
        s=jnp.asarray([0.5]),
        xm=jnp.asarray([0.0]),
        xn=jnp.asarray([0.0]),
        bmnc=jnp.asarray([[magnetic_field]]),
        bmns=None,
        field_periods=1,
    )
    harmonics = jnp.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    matrix_frequency = np.asarray(
        boozer_continuum_frequencies_hz(
            constant_spectrum,
            harmonics,
            iota=iota,
            major_radius=major_radius,
            mass_density=mass_density,
        )[0]
    )
    k_parallel = (np.asarray(harmonics)[:, 1] - np.asarray(harmonics)[:, 0] * iota) / major_radius
    analytic_frequency = np.sort(magnetic_field / np.sqrt(4.0e-7 * np.pi * mass_density) * np.abs(k_parallel) / (2.0 * np.pi))

    coupled_harmonics = jnp.asarray([[1.0, 0.0], [2.0, 1.0]])
    coupled_spectrum = BoozerSpectrum(
        source="coupled_boozer_figure",
        s=jnp.asarray([0.5]),
        xm=jnp.asarray([0.0, 1.0]),
        xn=jnp.asarray([0.0, 1.0]),
        bmnc=jnp.asarray([[2.8, 0.08]]),
        bmns=None,
        field_periods=1,
    )
    matrices = boozer_continuum_matrices(
        coupled_spectrum,
        coupled_harmonics,
        iota=1.0 / 3.0,
        major_radius=5.2,
        mass_density=4.1e-8,
        theta_points=96,
        zeta_points=96,
    )
    normalized_stiffness = np.abs(np.asarray(matrices.stiffness[0]))
    normalized_stiffness = normalized_stiffness / np.maximum(np.max(np.abs(np.diag(normalized_stiffness))), 1.0e-30)

    amplitudes = np.linspace(0.0, 0.16, 20)
    lower = []
    upper = []
    for amplitude in amplitudes:
        spectrum = BoozerSpectrum(
            source="amplitude_scan",
            s=jnp.asarray([0.5]),
            xm=jnp.asarray([0.0, 1.0]),
            xn=jnp.asarray([0.0, 1.0]),
            bmnc=jnp.asarray([[2.8, float(amplitude)]]),
            bmns=None,
            field_periods=1,
        )
        frequencies = np.asarray(
            boozer_continuum_frequencies_hz(
                spectrum,
                coupled_harmonics,
                iota=1.0 / 3.0,
                major_radius=5.2,
                mass_density=4.1e-8,
                theta_points=96,
                zeta_points=96,
            )[0]
        )
        lower.append(float(frequencies[0] / 1.0e3))
        upper.append(float(frequencies[1] / 1.0e3))

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), constrained_layout=True)

    ax = axes[0]
    x = np.arange(1, len(analytic_frequency) + 1)
    ax.plot(x, analytic_frequency / 1.0e3, marker="o", linewidth=1.6, label="analytic profile")
    ax.plot(x, matrix_frequency / 1.0e3, marker="s", linestyle="--", linewidth=1.3, label="Boozer matrix")
    ax.set_title("Constant-field Boozer limit")
    ax.set_xlabel("sorted branch")
    ax.set_ylabel("frequency [kHz]")
    ax.legend(fontsize=8)

    ax = axes[1]
    image = ax.imshow(normalized_stiffness, origin="lower", cmap="magma", vmin=0.0)
    ax.set_title(r"$|K_{ij}|$ with $(m,n)=(1,1)$ in $B$")
    ax.set_xticks([0, 1], ["(1,0)", "(2,1)"])
    ax.set_yticks([0, 1], ["(1,0)", "(2,1)"])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="normalized")

    ax = axes[2]
    ax.plot(amplitudes / 2.8, lower, marker="o", linewidth=1.4, label="lower")
    ax.plot(amplitudes / 2.8, upper, marker="s", linewidth=1.4, label="upper")
    ax.fill_between(amplitudes / 2.8, lower, upper, color="#4c78a8", alpha=0.14)
    ax.set_title("Coupling opens a reduced gap")
    ax.set_xlabel(r"$B_{1,1}/B_{0,0}$")
    ax.set_ylabel("frequency [kHz]")
    ax.legend(fontsize=8)

    fig.suptitle("Reduced Boozer continuum coupling validation", fontsize=12, weight="bold")
    paths = _save_figure(fig, output_dir, "boozer_continuum_coupling")
    plt.close(fig)
    return paths


def generate_boozer_jacobian_identity(output_dir: Path) -> list[Path]:
    """Generate a Boozer-Jacobian metric identity validation figure."""
    plt = _setup_matplotlib()
    spectrum = BoozerSpectrum(
        source="jacobian_identity_figure",
        s=jnp.asarray([0.5]),
        xm=jnp.asarray([0.0, 1.0, 2.0]),
        xn=jnp.asarray([0.0, 1.0, 2.0]),
        bmnc=jnp.asarray([[2.8, 0.12, -0.035]]),
        bmns=None,
        field_periods=1,
    )
    theta, zeta = boozer_angle_grid(theta_points=96, zeta_points=96)
    field = reconstruct_boozer_field(spectrum, surface_index=0, theta=theta, zeta=zeta)
    jacobian = boozer_jacobian_weight(
        spectrum,
        jacobian_numerator=1.6,
        theta_points=96,
        zeta_points=96,
    )[0]
    identity = jacobian * field**2
    identity_residual = identity - 1.6
    identity_error = float(boozer_jacobian_identity_error(spectrum, jacobian_numerator=1.6, theta_points=96, zeta_points=96))
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
    mass = np.abs(np.asarray(matrices.mass[0]))
    mass = mass / np.maximum(np.max(np.abs(np.diag(mass))), 1.0e-30)

    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.3), constrained_layout=True)

    panels = [
        (axes[0], np.asarray(field), r"$|B|$", "T", "viridis"),
        (axes[1], np.asarray(jacobian), r"$J_B=(G+\iota I)/B^2$", "normalized", "magma"),
        (axes[2], np.asarray(identity_residual), r"$J_B |B|^2-(G+\iota I)$", "residual", "coolwarm"),
    ]
    for ax, values, title, label, cmap in panels:
        image = ax.imshow(values, origin="lower", extent=(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi), aspect="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel(r"$\zeta_B$")
        ax.set_ylabel(r"$\theta_B$")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=label)

    ax = axes[3]
    image = ax.imshow(mass, origin="lower", cmap="magma", vmin=0.0)
    ax.set_title("Weighted mass coupling")
    ax.set_xticks([0, 1], ["(1,0)", "(2,1)"])
    ax.set_yticks([0, 1], ["(1,0)", "(2,1)"])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="normalized")
    ax.text(
        0.04,
        0.90,
        f"identity error={identity_error:.1e}",
        transform=ax.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "boxstyle": "round,pad=0.25"},
    )

    fig.suptitle("Boozer-Jacobian metric identity validation", fontsize=12, weight="bold")
    paths = _save_figure(fig, output_dir, "boozer_jacobian_identity")
    plt.close(fig)
    return paths


def generate_boozmn_3d_geometry_operator(output_dir: Path) -> list[Path]:
    """Generate a real-boozmn geometry and local 3-D continuum figure."""
    plt = _setup_matplotlib()
    path = Path("/Users/rogerio/local/booz_xform_jax/tests/test_files/boozmn_LandremanSenguptaPlunk_section5p3.nc")
    if not path.exists():
        raise FileNotFoundError(f"booz_xform_jax boozmn fixture not found: {path}")

    geometry = read_boozmn_geometry(path)
    grid = boozer_geometry_grid(geometry, theta_points=64, zeta_points=64)
    harmonics = jnp.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0]])
    matrices = boozer_3d_continuum_matrices(
        geometry,
        harmonics,
        mass_density=3.4e-8,
        theta_points=32,
        zeta_points=32,
    )
    frequencies = np.asarray(
        boozer_3d_continuum_frequencies_hz(
            geometry,
            harmonics,
            mass_density=3.4e-8,
            theta_points=32,
            zeta_points=32,
        )
    )
    surface_index = int(np.argmin(np.abs(np.asarray(geometry.s) - 0.5)))
    numerator = geometry.boozer_g + geometry.iota * geometry.boozer_i
    identity_residual = grid.signed_jacobian * grid.magnetic_field**2 - numerator[:, None, None]
    identity_error = float(boozer_geometry_jacobian_identity_error(geometry, theta_points=64, zeta_points=64))
    shape_error = float(boozer_embedding_jacobian_shape_error(geometry, theta_points=48, zeta_points=48))
    hermitian_error = float(
        np.linalg.norm(np.asarray(matrices.stiffness[0]) - np.conjugate(np.asarray(matrices.stiffness[0]).T))
        / max(np.linalg.norm(np.asarray(matrices.stiffness[0])), 1.0e-30)
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0), constrained_layout=True)

    panels = [
        (axes[0, 0], np.asarray(grid.magnetic_field[surface_index]), r"real $|B|(\theta_B,\zeta_B)$", "T", "viridis"),
        (
            axes[0, 1],
            np.asarray(identity_residual[surface_index]),
            r"$\sqrt{g}|B|^2-(G+\iota I)$",
            "identity residual",
            "coolwarm",
        ),
        (
            axes[1, 0],
            np.asarray(np.log10(np.maximum(np.asarray(grid.grad_s_squared[surface_index]), 1.0e-30))),
            r"embedding $\log_{10}|\nabla s|^2$",
            "normalized-s metric",
            "magma",
        ),
    ]
    for ax, values, title, label, cmap in panels:
        image = ax.imshow(values, origin="lower", extent=(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi), aspect="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel(r"$\zeta_B$")
        ax.set_ylabel(r"$\theta_B$")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=label)

    ax = axes[1, 1]
    for index in range(frequencies.shape[1]):
        ax.plot(np.asarray(geometry.s), frequencies[:, index] / 1.0e3, linewidth=1.4, label=f"branch {index + 1}")
    ax.set_title("local 3-D Boozer continuum")
    ax.set_xlabel("normalized toroidal flux s")
    ax.set_ylabel("frequency [kHz]")
    ax.set_xlim(float(np.min(np.asarray(geometry.s))), float(np.max(np.asarray(geometry.s))))
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=8, ncol=2)
    ax.text(
        0.03,
        0.95,
        f"identity={identity_error:.1e}\nshape={shape_error:.2e}\nHermitian={hermitian_error:.1e}",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "boxstyle": "round,pad=0.25"},
    )

    fig.suptitle("Real boozmn geometry: Boozer identity, embedding metric, and local 3-D continuum", fontsize=12, weight="bold")
    paths = _save_figure(fig, output_dir, "boozmn_3d_geometry_operator")
    plt.close(fig)
    return paths


def generate_boozmn_global_3d_eigenmodes(output_dir: Path) -> list[Path]:
    """Generate a real-boozmn global radial 3-D eigenmode figure."""
    plt = _setup_matplotlib()
    path = Path("/Users/rogerio/local/booz_xform_jax/tests/test_files/boozmn_LandremanSenguptaPlunk_section5p3.nc")
    if not path.exists():
        raise FileNotFoundError(f"booz_xform_jax boozmn fixture not found: {path}")

    geometry = read_boozmn_geometry(path)
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
    local_omega2 = generalized_hermitian_eigenvalues(problem.local_stiffness, problem.local_mass)
    local_frequencies_khz = np.asarray(jnp.sqrt(jnp.maximum(jnp.real(local_omega2), 0.0)) / (2.0 * jnp.pi)) / 1.0e3
    structures = np.asarray(result.mode_structures[:4])
    envelope = np.linalg.norm(structures, axis=2)
    harmonic_components = np.real(structures[0])
    harmonic_components = harmonic_components / max(float(np.max(np.abs(harmonic_components))), 1.0e-30)
    residuals = np.asarray(result.residuals[:8])

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.7), constrained_layout=True)

    ax = axes[0, 0]
    for branch in range(local_frequencies_khz.shape[1]):
        ax.plot(np.asarray(geometry.s), local_frequencies_khz[:, branch], color="#777777", alpha=0.35, linewidth=1.0)
    colors = ["#2458a6", "#b13f2a", "#2e7d32", "#6f4a8e"]
    for mode in range(4):
        ax.axhline(float(result.frequencies_hz[mode]) / 1.0e3, color=colors[mode], linewidth=1.4, label=f"global {mode + 1}")
    ax.set_title("global modes over local continuum")
    ax.set_xlabel("normalized toroidal flux s")
    ax.set_ylabel("frequency [kHz]")
    ax.set_xlim(float(np.min(np.asarray(geometry.s))), float(np.max(np.asarray(geometry.s))))
    ax.legend(fontsize=8, ncol=2)

    ax = axes[0, 1]
    for mode in range(4):
        ax.plot(np.asarray(geometry.s), envelope[mode] / max(float(np.max(envelope[mode])), 1.0e-30), linewidth=1.5, label=f"mode {mode + 1}")
    ax.set_title("radial envelope norms")
    ax.set_xlabel("s")
    ax.set_ylabel("normalized |xi|")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    image = ax.imshow(
        harmonic_components.T,
        origin="lower",
        aspect="auto",
        extent=(float(np.min(np.asarray(geometry.s))), float(np.max(np.asarray(geometry.s))), 0.5, harmonic_components.shape[1] + 0.5),
        cmap="coolwarm",
    )
    ax.set_title("mode 1 harmonic structure")
    ax.set_xlabel("s")
    ax.set_ylabel("harmonic index")
    ax.set_yticks(np.arange(1, harmonic_components.shape[1] + 1))
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="real amplitude")

    ax = axes[1, 1]
    ax.semilogy(np.arange(1, residuals.size + 1), residuals, marker="o", linewidth=1.4)
    ax.axhline(2.0e-5, color="#555555", linestyle="--", linewidth=1.0, label="gate")
    ax.set_title("generalized eigen residuals")
    ax.set_xlabel("sorted mode")
    ax.set_ylabel("relative residual")
    ax.legend(fontsize=8)

    fig.suptitle("Global radial 3-D Boozer eigenmode operator from real boozmn geometry", fontsize=12, weight="bold")
    paths = _save_figure(fig, output_dir, "boozmn_global_3d_eigenmodes")
    plt.close(fig)
    return paths


def generate_stellgap_ctok_comparison(output_dir: Path) -> list[Path]:
    """Generate the STELLGAP CTOK comparison dashboard when local artifacts exist."""
    boozmn = Path("/Users/rogerio/local/Stellgap_test_inputs/boozmn_ctok.nc")
    alfven_post = Path("/Users/rogerio/local/Stellgap_test_problem/alfven_post")
    if not boozmn.exists() or not alfven_post.exists():
        return []
    plt = _setup_matplotlib()
    geometry = read_boozmn_geometry(boozmn)
    reference = read_stellgap_alfven_post(alfven_post)
    harmonics = jnp.asarray([[float(m), 2.0] for m in range(13)])
    fax_khz = np.asarray(
        boozer_3d_continuum_frequencies_hz(
            geometry,
            harmonics,
            mass_density=2.0 * PROTON_MASS * 1.0e19,
            theta_points=32,
            zeta_points=32,
        )
        / 1.0e3
    )
    unique_s = np.unique(np.asarray(reference.s))
    surfaces = []
    errors = []
    for surface_index, surface in enumerate(np.asarray(geometry.s)):
        nearest_index = int(np.argmin(np.abs(unique_s - surface)))
        if abs(float(unique_s[nearest_index] - surface)) > 2.0e-3:
            continue
        mask = np.asarray(reference.s) == unique_s[nearest_index]
        ref = np.sort(np.asarray(reference.frequency_khz)[mask])
        if ref.shape[0] != 13:
            continue
        err = np.linalg.norm(np.sort(fax_khz[surface_index]) - ref) / max(np.linalg.norm(ref), 1.0e-30)
        surfaces.append(float(surface))
        errors.append(float(err))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
    axes[0].scatter(np.asarray(reference.s), np.asarray(reference.frequency_khz), s=3, alpha=0.28, label="STELLGAP")
    for branch in range(fax_khz.shape[1]):
        axes[0].plot(np.asarray(geometry.s), fax_khz[:, branch], color="black", alpha=0.38, linewidth=0.8)
    axes[0].set_title("CTOK continuum branches")
    axes[0].set_xlabel("s")
    axes[0].set_ylabel("frequency [kHz]")
    axes[0].set_ylim(0.0, 8_500.0)
    axes[0].legend(loc="upper right")
    axes[1].plot(surfaces, errors, marker=".", linewidth=1.0)
    axes[1].axhline(0.04, linestyle="--", color="black", linewidth=1.0, label="gate")
    axes[1].set_title("relative spectrum error")
    axes[1].set_xlabel("s")
    axes[1].set_ylabel("relative L2 error")
    axes[1].legend(loc="upper left")
    paths = _save_figure(fig, output_dir, "stellgap_ctok_comparison")
    plt.close(fig)
    return paths


def generate_external_code_artifact_dashboard(output_dir: Path) -> list[Path]:
    """Generate artifact-level STELLGAP/AE3D parity diagnostics."""
    stellgap_dir = Path("/Users/rogerio/local/Stellgap_test_problem")
    ae3d_dir = Path("/Users/rogerio/local/AE3D_test_problem")
    alfven_spec = stellgap_dir / "alfven_spec"
    alfven_post = stellgap_dir / "alfven_post"
    omega2_path = ae3d_dir / "omega2.dat"
    egn_values_path = ae3d_dir / "egn_values.dat"
    egn_mode_path = ae3d_dir / "egn_mode_asci.dat"
    a_matrix = ae3d_dir / "a_matrix.dat"
    b_matrix = ae3d_dir / "b_matrix.dat"
    if not all(
        path.exists()
        for path in (alfven_spec, alfven_post, omega2_path, egn_values_path, egn_mode_path, a_matrix, b_matrix)
    ):
        return []
    plt = _setup_matplotlib()
    stellgap_comparison = stellgap_postprocess_error(alfven_spec_path=alfven_spec, alfven_post_path=alfven_post)
    ae3d_comparison = ae3d_omega2_eigenvalue_error(omega2_path=omega2_path, egn_values_path=egn_values_path)
    ae3d_residual = ae3d_eigenvalue_residual_summary(egn_values_path)
    a_summary = ae3d_matrix_symmetry_summary(a_matrix)
    b_summary = ae3d_matrix_symmetry_summary(b_matrix)
    dense_parity = ae3d_dense_spectrum_parity(
        a_matrix_path=a_matrix,
        b_matrix_path=b_matrix,
        egn_mode_path=egn_mode_path,
        max_modes=16,
    )
    global_parity = ae3d_global_mode_parity(
        a_matrix_path=a_matrix,
        b_matrix_path=b_matrix,
        egn_mode_path=egn_mode_path,
        max_modes=8,
    )
    stellgap_processed = stellgap_spec_postprocessed_continuum(alfven_spec)
    stellgap_reference = read_stellgap_alfven_post(alfven_post)
    predicted = np.asarray(stellgap_processed.frequency_khz)
    reference = np.asarray(stellgap_reference.frequency_khz)
    count = min(predicted.shape[0], reference.shape[0])
    stellgap_error = np.abs(predicted[:count] - reference[:count]) / np.maximum(np.abs(reference[:count]), 1.0e-30)
    omega2 = read_ae3d_omega2(omega2_path)
    egn_values = read_ae3d_eigenvalues(egn_values_path)
    raw = np.asarray(omega2.eigenvalue)
    ae3d_predicted = np.where(raw > 0.0, np.sqrt(np.maximum(raw, 0.0)), raw)
    ae3d_reference = np.asarray(egn_values.frequency_khz)
    ae3d_count = min(ae3d_predicted.shape[0], ae3d_reference.shape[0])

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.4), constrained_layout=True)
    axes[0, 0].scatter(np.asarray(stellgap_reference.s), reference, s=3, alpha=0.32, label="STELLGAP alfven_post")
    axes[0, 0].scatter(np.asarray(stellgap_processed.s), predicted, s=2, alpha=0.22, label="FAX postprocess")
    axes[0, 0].set_title("STELLGAP postprocess parity")
    axes[0, 0].set_xlabel("s")
    axes[0, 0].set_ylabel("frequency [kHz]")
    axes[0, 0].legend(loc="upper right", fontsize=8)

    axes[0, 1].semilogy(stellgap_error, linewidth=1.1)
    axes[0, 1].axhline(1.0e-5, color="black", linestyle="--", linewidth=1.0, label="gate")
    axes[0, 1].set_title(f"STELLGAP row error, max={stellgap_comparison.max_relative_error:.2e}")
    axes[0, 1].set_xlabel("postprocessed row")
    axes[0, 1].set_ylabel("relative error")
    axes[0, 1].legend(loc="upper right", fontsize=8)

    axes[1, 0].scatter(ae3d_reference[:ae3d_count], ae3d_predicted[:ae3d_count], s=5, alpha=0.5)
    lo = min(float(np.min(ae3d_reference[:ae3d_count])), float(np.min(ae3d_predicted[:ae3d_count])))
    hi = max(float(np.max(ae3d_reference[:ae3d_count])), float(np.max(ae3d_predicted[:ae3d_count])))
    axes[1, 0].plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.0)
    axes[1, 0].set_title(f"AE3D omega2 convention, max={ae3d_comparison.max_relative_error:.2e}")
    axes[1, 0].set_xlabel("AE3D egn_values")
    axes[1, 0].set_ylabel("FAX reproduction")

    labels = ["A asym.", "B asym.", "residual", "omega2", "dense eig.", "global freq.", "mode shape"]
    values = [
        a_summary.relative_asymmetry,
        b_summary.relative_asymmetry,
        ae3d_residual.max_residual,
        ae3d_comparison.max_relative_error,
        dense_parity.max_eigenvalue_relative_error,
        global_parity.max_frequency_relative_error,
        global_parity.max_mass_weighted_mode_shape_error,
    ]
    axes[1, 1].bar(
        labels,
        values,
        color=["#4c78a8", "#72b7b2", "#f58518", "#54a24b", "#b279a2", "#e45756", "#499894"],
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].axhline(1.0e-5, color="black", linestyle="--", linewidth=1.0, label="matrix/eigenvalue gate")
    axes[1, 1].axhline(1.0e-2, color="#777777", linestyle=":", linewidth=1.0, label="residual gate")
    axes[1, 1].set_title(
        f"AE3D matrices: N={a_summary.dimension}, shape err={global_parity.max_mass_weighted_mode_shape_error:.2e}"
    )
    axes[1, 1].set_ylabel("metric")
    axes[1, 1].tick_params(axis="x", rotation=18)
    axes[1, 1].legend(loc="upper right", fontsize=8)

    fig.suptitle("Public external-code artifact parity checks", fontsize=12, weight="bold")
    paths = _save_figure(fig, output_dir, "external_code_artifact_dashboard")
    plt.close(fig)
    metrics = {
        "stellgap_postprocess": {
            "count": stellgap_comparison.count,
            "mean_relative_error": stellgap_comparison.mean_relative_error,
            "p95_relative_error": stellgap_comparison.p95_relative_error,
            "max_relative_error": stellgap_comparison.max_relative_error,
        },
        "ae3d_omega2_eigenvalue": {
            "count": ae3d_comparison.count,
            "mean_relative_error": ae3d_comparison.mean_relative_error,
            "p95_relative_error": ae3d_comparison.p95_relative_error,
            "max_relative_error": ae3d_comparison.max_relative_error,
        },
        "ae3d_residual": {
            "count": ae3d_residual.count,
            "p95_residual": ae3d_residual.p95_residual,
            "max_residual": ae3d_residual.max_residual,
        },
        "ae3d_a_matrix": {
            "dimension": a_summary.dimension,
            "nonzero_count": a_summary.nonzero_count,
            "relative_asymmetry": a_summary.relative_asymmetry,
        },
        "ae3d_b_matrix": {
            "dimension": b_summary.dimension,
            "nonzero_count": b_summary.nonzero_count,
            "relative_asymmetry": b_summary.relative_asymmetry,
        },
        "ae3d_dense_spectrum": {
            "count": dense_parity.count,
            "matrix_dimension": dense_parity.matrix_dimension,
            "real_eigenvalue_count": dense_parity.real_eigenvalue_count,
            "max_eigenvalue_relative_error": dense_parity.max_eigenvalue_relative_error,
            "p95_eigenvalue_relative_error": dense_parity.p95_eigenvalue_relative_error,
            "max_frequency_relative_error": dense_parity.max_frequency_relative_error,
            "p95_frequency_relative_error": dense_parity.p95_frequency_relative_error,
        },
        "ae3d_global_mode": {
            "count": global_parity.count,
            "matrix_dimension": global_parity.matrix_dimension,
            "real_eigenvalue_count": global_parity.real_eigenvalue_count,
            "max_frequency_relative_error": global_parity.max_frequency_relative_error,
            "p95_frequency_relative_error": global_parity.p95_frequency_relative_error,
            "max_euclidean_mode_shape_error": global_parity.max_euclidean_mode_shape_error,
            "p95_euclidean_mode_shape_error": global_parity.p95_euclidean_mode_shape_error,
            "max_mass_weighted_mode_shape_error": global_parity.max_mass_weighted_mode_shape_error,
            "p95_mass_weighted_mode_shape_error": global_parity.p95_mass_weighted_mode_shape_error,
        },
    }
    (output_dir / "external_code_artifact_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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


def generate_global_eigenmode_artifact_dashboard(output_dir: Path) -> list[Path]:
    """Generate a dashboard from files written by the global eigenmode workflow."""
    with tempfile.TemporaryDirectory(prefix="fax-global-eigenmode-artifacts-") as tmp:
        config = FaxConfig(
            run=RunConfig(case="global_eigenmodes", model="global_eigenmode", output_dir=tmp),
            equilibrium=EquilibriumConfig(source="analytic"),
            grid=GridConfig(radial_points=32, m_max=2, n_max=1, field_periods=1),
        )
        run_global_eigenmodes(config, progress=False, max_modes=6)
        return list(
            plot_global_eigenmode_artifacts(
                Path(tmp) / "eigenmodes",
                output_dir=output_dir,
                stem="global_eigenmode_artifact_dashboard",
            )
        )


def generate_boozmn_global_eigenmode_artifact_dashboard(output_dir: Path) -> list[Path]:
    """Generate a dashboard from the TOML/CLI real-boozmn global workflow path."""
    path = Path("/Users/rogerio/local/booz_xform_jax/tests/test_files/boozmn_LandremanSenguptaPlunk_section5p3.nc")
    if not path.exists():
        raise FileNotFoundError(f"booz_xform_jax boozmn fixture not found: {path}")

    with tempfile.TemporaryDirectory(prefix="fax-boozmn-global-eigenmode-artifacts-") as tmp:
        config = FaxConfig(
            run=RunConfig(case="boozmn_global_eigenmodes", model="global_eigenmode", output_dir=tmp),
            equilibrium=EquilibriumConfig(source="wout", use_boozer=True, boozmn=str(path)),
            grid=GridConfig(radial_points=5, m_max=2, n_max=1, field_periods=3),
        )
        run_global_eigenmodes(config, progress=False, max_modes=6)
        return list(
            plot_global_eigenmode_artifacts(
                Path(tmp) / "eigenmodes",
                output_dir=output_dir,
                stem="boozmn_global_eigenmode_artifact_dashboard",
            )
        )


def generate_workflow_diagram(output_dir: Path) -> list[Path]:
    """Generate a newcomer-facing input/operator/output workflow diagram."""
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(9.4, 4.2), constrained_layout=True)
    ax.axis("off")

    boxes = [
        ("Inputs", "TOML case\nVMEC wout / vmec_jax\nprofiles\nmode family", 0.04, 0.55, "#e8f1fb"),
        ("Adapters", "EquilibriumGeometry\nMetricGeometry\nBoozer spectra", 0.28, 0.55, "#e9f7ef"),
        ("Operators", "continuum branches\nK(s), M(s)\nradial modes", 0.52, 0.55, "#fff4df"),
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
    ax.text(0.5, 0.09, "Current scope: reduced/profile and metric gates; full 3-D STELLGAP/AE3D validation remains planned.", ha="center", fontsize=9)

    paths = _save_figure(fig, output_dir, "workflow_io_diagram")
    plt.close(fig)
    return paths


def generate_equations_overview(output_dir: Path) -> list[Path]:
    """Generate a compact equation overview figure."""
    plt = _setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.7), constrained_layout=True)
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

    axes[2].set_title("Global modes and fast ions", fontsize=12, weight="bold")
    axes[2].text(
        0.02,
        0.80,
        r"$K_r \xi=\omega^2 M_r \xi$" "\n"
        r"$K_g=\mathrm{blockdiag}(K^{3D})+D_s^\dagger C_sD_s$" "\n"
        r"$\iota_{\rm gap}=2n/(2m+\Delta m)$" "\n"
        r"$f_{\rm mode}=n f_\phi+\ell f_\theta$",
        fontsize=11,
        va="top",
    )
    axes[2].text(
        0.02,
        0.17,
        "Used for scalar and block radial spectra,\ngap centers, resonances, and gradients.",
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
    labels = [f"{gate.layer}: {gate.gate_id}".replace("_", " ") for gate in gates]
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
    fig_height = max(7.0, 0.28 * len(gates))
    fig, ax = plt.subplots(figsize=(12.0, fig_height))
    ax.barh(y, ratios, color=colors, height=0.68)
    ax.axvline(1.0, color="#222222", linewidth=1.0, linestyle="--", label="tolerance")
    for index in range(1, len(gates)):
        if gates[index].layer != gates[index - 1].layer:
            ax.axhline(index - 0.5, color="#bdbdbd", linewidth=0.7, alpha=0.75)
    ax.set_yticks(y, labels, fontsize=6.2)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlabel("metric / tolerance")
    ax.set_title("Registered validation gates")
    ax.set_xlim(0.0, max(1.2, float(np.nanmax(ratios)) * 1.15))
    for idx, (ratio, status) in enumerate(zip(ratios, statuses, strict=True)):
        ax.text(min(ratio + 0.025, ax.get_xlim()[1] * 0.96), idx, status, va="center", fontsize=6.4)
    ax.legend(loc="lower right")
    fig.subplots_adjust(left=0.42, right=0.98, top=0.965, bottom=0.05)

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
    """Generate a Landreman-Paul adapter convergence dashboard from public artifacts."""
    case = landreman_paul_qa_lowres()
    if case.wout is None or not case.wout.exists():
        raise FileNotFoundError(f"Landreman-Paul QA wout not found: {case.wout}")

    with tempfile.TemporaryDirectory(prefix="fax-convergence-artifacts-") as tmp:
        config = FaxConfig(
            run=RunConfig(case=case.case_id, model="continuum", output_dir=tmp),
            equilibrium=EquilibriumConfig(
                source="wout",
                vmec_input=str(case.vmec_input) if case.vmec_input is not None else None,
                wout=str(case.wout),
            ),
            grid=GridConfig(radial_points=50, m_max=2, n_max=2, field_periods=2),
        )
        run_convergence(
            config,
            progress=False,
            radial_points=(24, 40, 50),
            angular_points=(8, 12),
            mode_max_values=(1, 2),
        )
        return list(
            plot_convergence_artifacts(
                Path(tmp) / "convergence",
                output_dir=output_dir,
                stem="landreman_paul_convergence_dashboard",
            )
        )


def generate_vmec_boozer_bridge_convergence(output_dir: Path) -> list[Path]:
    """Generate convergence evidence for the live VMEC-JAX/Boozer bridge."""
    plt = _setup_matplotlib()
    case = landreman_paul_qa_lowres()
    if not case.vmec_input.exists():
        raise FileNotFoundError(f"Landreman-Paul QA VMEC input not found: {case.vmec_input}")

    wout = run_vmec_jax_fixed_boundary_wout_data(
        case.vmec_input,
        max_iter=1,
        theta_points=12,
        zeta_points=4,
    )
    truncations = [(2, 2), (3, 3), (4, 4), (5, 5)]
    geometries = [
        run_booz_xform_jax_geometry_from_wout_data(
            wout,
            mboz=mboz,
            nboz=nboz,
            surfaces=(0.25, 0.5),
            jit=False,
        )
        for mboz, nboz in truncations
    ]
    reference = geometries[-1]
    harmonics = jnp.asarray([[1.0, 0.0], [2.0, 1.0]])
    reference_frequencies = boozer_3d_continuum_frequencies_hz(
        reference,
        harmonics,
        mass_density=3.4e-8,
        theta_points=16,
        zeta_points=16,
    )

    mode_counts: list[int] = []
    field_errors: list[float] = []
    frequency_errors: list[float] = []
    identity_errors: list[float] = []
    frequencies_by_truncation = []
    for geometry in geometries:
        mode_counts.append(int(geometry.bmnc.shape[1]))
        field_errors.append(
            float(
                jnp.max(
                    boozer_geometry_field_relative_error(
                        geometry,
                        reference,
                        theta_points=24,
                        zeta_points=16,
                    )
                )
            )
        )
        frequencies = boozer_3d_continuum_frequencies_hz(
            geometry,
            harmonics,
            mass_density=3.4e-8,
            theta_points=16,
            zeta_points=16,
        )
        frequencies_by_truncation.append(np.asarray(frequencies) / 1.0e3)
        frequency_errors.append(
            float(
                jnp.linalg.norm(frequencies - reference_frequencies)
                / jnp.maximum(jnp.linalg.norm(reference_frequencies), 1.0e-30)
            )
        )
        identity_errors.append(float(boozer_geometry_jacobian_identity_error(geometry, theta_points=16, zeta_points=16)))

    grid = boozer_geometry_grid(reference, theta_points=64, zeta_points=64)
    surface_index = int(np.argmin(np.abs(np.asarray(reference.s) - 0.5)))
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.6), constrained_layout=True)

    ax = axes[0]
    image = ax.imshow(
        np.asarray(grid.magnetic_field[surface_index]),
        origin="lower",
        extent=(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi),
        aspect="auto",
        cmap="viridis",
    )
    ax.set_title(r"reference $(m_{boz},n_{boz})=(5,5)$")
    ax.set_xlabel(r"$\zeta_B$")
    ax.set_ylabel(r"$\theta_B$")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=r"$|B|$ [T]")

    ax = axes[1]
    positive_floor = 1.0e-8
    ax.semilogy(mode_counts, np.maximum(field_errors, positive_floor), marker="o", linewidth=1.5, label=r"sampled $|B|$")
    ax.semilogy(mode_counts, np.maximum(frequency_errors, positive_floor), marker="s", linewidth=1.5, label="local continuum")
    ax.semilogy(mode_counts, np.maximum(identity_errors, positive_floor), marker="^", linewidth=1.5, label="Jacobian identity")
    ax.axhline(1.0e-2, color="#222222", linestyle="--", linewidth=1.0, label="gate")
    ax.set_title("Boozer truncation convergence")
    ax.set_xlabel("Boozer harmonic count")
    ax.set_ylabel("relative error")
    ax.set_ylim(positive_floor, 3.0e-2)
    ax.legend(fontsize=7)

    ax = axes[2]
    s = np.asarray(reference.s)
    coarse_freq = frequencies_by_truncation[-2]
    reference_freq = frequencies_by_truncation[-1]
    for branch in range(reference_freq.shape[1]):
        ax.plot(s, reference_freq[:, branch], marker="o", linewidth=1.7, label=f"ref branch {branch + 1}")
        ax.plot(s, coarse_freq[:, branch], marker="s", linestyle="--", linewidth=1.2, label=f"4/4 branch {branch + 1}")
    ax.set_title("continuum stability under truncation")
    ax.set_xlabel("normalized toroidal flux s")
    ax.set_ylabel("frequency [kHz]")
    ax.set_xlim(float(np.min(s)), float(np.max(s)))
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Live Landreman-Paul VMEC-JAX to Boozer bridge convergence smoke test", fontsize=12, weight="bold")
    paths = _save_figure(fig, output_dir, "vmec_boozer_bridge_convergence")
    plt.close(fig)
    return paths


def generate_vmec_jax_solve_convergence(output_dir: Path) -> list[Path]:
    """Generate VMEC residual convergence propagated through Boozer/FAX."""
    plt = _setup_matplotlib()
    case = landreman_paul_qa_lowres()
    if not case.vmec_input.exists() or not case.wout.exists():
        raise FileNotFoundError(f"Landreman-Paul QA VMEC input/reference wout not found: {case.missing_paths()}")

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
    max_iters = np.asarray(result.max_iters)
    fsq = np.asarray(result.fsq_totals)
    field_errors = np.asarray(result.field_relative_errors)
    frequency_errors = np.asarray(result.frequency_relative_errors)
    jacobian_errors = np.asarray(result.jacobian_identity_errors)
    runtime = np.asarray(result.runtime_seconds)

    fig, axes = plt.subplots(1, 4, figsize=(14.8, 3.6), constrained_layout=True)
    axes[0].loglog(max_iters, fsq, marker="o", linewidth=1.6)
    axes[0].set_title("VMEC force residual")
    axes[0].set_xlabel("requested VMEC iterations")
    axes[0].set_ylabel(r"$f_{sq,r}+f_{sq,z}+f_{sq,\lambda}$")
    axes[0].grid(True, which="both", alpha=0.25)

    axes[1].semilogy(max_iters, frequency_errors, marker="s", linewidth=1.6, color="#2458a6")
    axes[1].axhline(4.0e-2, color="#222222", linestyle="--", linewidth=1.0, label="gate")
    axes[1].set_title("FAX continuum error")
    axes[1].set_xlabel("requested VMEC iterations")
    axes[1].set_ylabel("relative frequency error")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, which="both", alpha=0.25)

    surfaces = np.asarray(result.reference_geometry.s)
    for surface_index, surface in enumerate(surfaces):
        axes[2].semilogy(
            max_iters,
            field_errors[:, surface_index],
            marker="o",
            linewidth=1.4,
            label=f"s={surface:.2f}",
        )
    axes[2].semilogy(max_iters, jacobian_errors, marker="^", linewidth=1.4, label="Jacobian")
    axes[2].set_title(r"Boozer $|B|$ and Jacobian")
    axes[2].set_xlabel("requested VMEC iterations")
    axes[2].set_ylabel("relative error")
    axes[2].legend(fontsize=7)
    axes[2].grid(True, which="both", alpha=0.25)

    axes[3].bar([str(value) for value in max_iters], runtime, color="#4c78a8")
    axes[3].set_title("local runtime")
    axes[3].set_xlabel("requested VMEC iterations")
    axes[3].set_ylabel("seconds")
    axes[3].text(
        0.04,
        0.96,
        f"final metric={result.max_metric:.2e}\nresidual ratio={result.residual_reduction_ratio:.2e}",
        transform=axes[3].transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "boxstyle": "round,pad=0.25"},
    )

    fig.suptitle("Landreman-Paul VMEC-JAX solve convergence through Boozer/FAX", fontsize=12, weight="bold")
    paths = _save_figure(fig, output_dir, "vmec_jax_solve_convergence")
    plt.close(fig)
    return paths


def generate_vmec_boozer_operator_angular_convergence(output_dir: Path) -> list[Path]:
    """Generate angular-convergence evidence for the live VMEC-JAX/Boozer operator."""
    plt = _setup_matplotlib()
    case = landreman_paul_qa_lowres()
    if not case.vmec_input.exists():
        raise FileNotFoundError(f"Landreman-Paul QA VMEC input not found: {case.vmec_input}")

    geometry = run_booz_xform_jax_geometry_from_wout_data(
        run_vmec_jax_fixed_boundary_wout_data(
            case.vmec_input,
            max_iter=1,
            theta_points=12,
            zeta_points=4,
        ),
        mboz=5,
        nboz=5,
        surfaces=(0.25, 0.5),
        jit=False,
    )
    harmonics = jnp.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    result = boozer_3d_operator_angular_convergence(
        geometry,
        harmonics,
        mass_density=3.4e-8,
        grids=((8, 8), (12, 12), (16, 16), (20, 20), (24, 24)),
        reference_grid=(32, 32),
    )
    grid_points = np.asarray([theta * zeta for theta, zeta in result.grids])
    reference = np.asarray(result.reference_frequencies_hz) / 1.0e3
    selected = np.asarray(result.frequencies_hz[2]) / 1.0e3
    residual = selected - reference
    grid = boozer_geometry_grid(geometry, theta_points=64, zeta_points=64)
    surface_index = int(np.argmin(np.abs(np.asarray(geometry.s) - 0.5)))

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.6), constrained_layout=True)

    ax = axes[0]
    image = ax.imshow(
        np.asarray(grid.grad_s_squared[surface_index]),
        origin="lower",
        extent=(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi),
        aspect="auto",
        cmap="magma",
    )
    ax.set_title(r"embedding $|\nabla s|^2$")
    ax.set_xlabel(r"$\zeta_B$")
    ax.set_ylabel(r"$\theta_B$")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="normalized")

    ax = axes[1]
    ax.loglog(grid_points, np.asarray(result.relative_frequency_errors), marker="o", linewidth=1.6, label="frequency")
    ax.loglog(grid_points, np.maximum(np.asarray(result.hermitian_errors), 1.0e-16), marker="s", linewidth=1.4, label="Hermitian")
    ax.axhline(5.0e-3, color="#222222", linestyle="--", linewidth=1.0, label="gate")
    ax.set_title("operator angular convergence")
    ax.set_xlabel(r"$N_\theta N_\zeta$")
    ax.set_ylabel("relative metric")
    ax.set_ylim(1.0e-16, 1.0e-2)
    ax.legend(fontsize=8)

    ax = axes[2]
    s = np.asarray(geometry.s)
    for branch in range(reference.shape[1]):
        ax.plot(s, reference[:, branch], marker="o", linewidth=1.6, label=f"ref {branch + 1}")
        ax.plot(s, selected[:, branch], marker="s", linestyle="--", linewidth=1.2, label=f"16x16 {branch + 1}")
    max_residual_hz = float(np.max(np.abs(residual)) * 1.0e3)
    ax.text(
        0.03,
        0.95,
        f"max |df|={max_residual_hz:.1f} Hz",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "boxstyle": "round,pad=0.25"},
    )
    ax.set_title("frequency stability")
    ax.set_xlabel("normalized toroidal flux s")
    ax.set_ylabel("frequency [kHz]")
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Live Landreman-Paul Boozer local-operator angular convergence", fontsize=12, weight="bold")
    paths = _save_figure(fig, output_dir, "vmec_boozer_operator_angular_convergence")
    plt.close(fig)
    return paths


def generate_benchmark_family_packet(output_dir: Path) -> list[Path]:
    """Generate a benchmark-packet figure tied to physics gates."""
    plt = _setup_matplotlib()
    with tempfile.TemporaryDirectory() as tmp:
        config = FaxConfig(
            run=RunConfig(case="benchmark_family_packet", output_dir=tmp),
            equilibrium=EquilibriumConfig(source="analytic"),
            grid=GridConfig(radial_points=18, m_max=2, n_max=1, field_periods=1),
        )
        summary = run_benchmark(
            config,
            progress=False,
            repeats=1,
            warmups=0,
            families=("analytic-local", "global-block", "global-matrix-free", "global-krylov"),
        )
    results = summary.family_results
    label_map = {
        "continuum": "cont.\nworkflow",
        "analytic-local": "analytic\nlocal",
        "global-block": "dense\nglobal",
        "global-matrix-free": "matrix\nfree",
        "global-krylov": "sparse\nKrylov",
    }
    labels = [label_map.get(str(result["family"]), str(result["family"]).replace("-", "\n")) for result in results]
    cold = np.asarray([float(result["cold_seconds"]) for result in results], dtype=float)
    warm = np.asarray([float(result["warm_result"]["median_seconds"]) for result in results], dtype=float)
    ratios = np.asarray(
        [
            float(result["metric_value"]) / max(float(result["tolerance"]), 1.0e-30)
            if result["metric_value"] is not None and result["tolerance"] is not None
            else np.nan
            for result in results
        ],
        dtype=float,
    )

    summary_path = output_dir / "benchmark_family_packet_summary.json"
    summary_payload = {
        "case": summary.case,
        "status": summary.status,
        "message": summary.message,
        "families": summary.family_results,
        "note": "Temporary run directories are intentionally omitted from this docs summary.",
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.2), constrained_layout=True)
    x = np.arange(len(labels))

    ax = axes[0]
    width = 0.35
    ax.bar(x - width / 2.0, cold, width=width, color="#4c78a8", label="cold")
    ax.bar(x + width / 2.0, warm, width=width, color="#f58518", label="warm")
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylabel("runtime [s]")
    ax.set_title("cold vs warm")
    ax.legend(fontsize=8)

    ax = axes[1]
    colors = ["#2e7d32" if result["gate_passed"] else "#b13f2a" for result in results]
    ax.bar(x, ratios, color=colors)
    ax.axhline(1.0, color="#222222", linestyle="--", linewidth=1.0, label="gate")
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("metric / tolerance")
    ax.set_title("physics gate margin")
    ax.legend(fontsize=8)

    ax = axes[2]
    text_lines = [
        "Benchmark packet",
        f"JAX backend: {results[0]['environment']['jax_backend']}",
        "families:",
        *[
            f"{result['family']}: {result['physics_metric']}"
            for result in results
        ],
    ]
    ax.axis("off")
    ax.text(
        0.0,
        1.0,
        "\n".join(text_lines),
        transform=ax.transAxes,
        fontsize=8.5,
        va="top",
        family="monospace",
    )

    fig.suptitle("FAX benchmark families tie runtime to physics gates", fontsize=12, weight="bold")
    paths = _save_figure(fig, output_dir, "benchmark_family_packet")
    plt.close(fig)
    return paths + [summary_path]


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
        list(plot_basic_alfven_gap_map(output_dir)),
        generate_gap_sensitivity_dashboard(output_dir),
        generate_global_gap_fastion_dashboard(output_dir),
        generate_boozer_harmonic_validation(output_dir),
        generate_boozer_continuum_coupling(output_dir),
        generate_boozer_jacobian_identity(output_dir),
        generate_boozmn_3d_geometry_operator(output_dir),
        generate_boozmn_global_3d_eigenmodes(output_dir),
        generate_stellgap_ctok_comparison(output_dir),
        generate_external_code_artifact_dashboard(output_dir),
        generate_continuum_artifact_dashboard(output_dir),
        generate_global_eigenmode_artifact_dashboard(output_dir),
        generate_boozmn_global_eigenmode_artifact_dashboard(output_dir),
        generate_metric_residual_convergence(output_dir),
        generate_landreman_convergence_dashboard(output_dir),
        generate_vmec_boozer_bridge_convergence(output_dir),
        generate_vmec_jax_solve_convergence(output_dir),
        generate_vmec_boozer_operator_angular_convergence(output_dir),
        generate_benchmark_family_packet(output_dir),
        generate_validation_gate_summary(output_dir),
    ]:
        paths.extend(generated)
    paths.append(write_figure_manifest(output_dir, paths))
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
