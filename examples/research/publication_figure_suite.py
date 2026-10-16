"""Generate a ten-case FAX publication-figure suite.

The suite is intentionally explicit about validation status. Some figures are
accepted validation/parity artifacts, while others are research workflow
templates for the current reduced FAX operators.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from fax.boozer import (  # noqa: E402
    BoozerSpectrum,
    boozer_3d_continuum_frequencies_hz,
    boozer_angle_grid,
    boozer_geometry_grid,
    boozer_harmonic_orthogonality_error,
    boozer_projection_error,
    normalized_nonaxisymmetric_boozer_energy,
    read_boozmn_geometry,
    reconstruct_boozer_field,
)
from fax.config import EquilibriumConfig, FaxConfig, GridConfig, RunConfig  # noqa: E402
from fax.continuum import continuum_branch_matrix  # noqa: E402
from fax.cross_code import (  # noqa: E402
    ae3d_continuum_placement_parity,
    ae3d_dense_spectrum_parity,
    ae3d_global_mode_parity,
    ae3d_matrix_eigenmode_parity,
    read_ae3d_eigenmodes,
    read_stellgap_alfven_post,
    stellgap_ctok_local_continuum_error,
)
from fax.equilibrium import analytic_circular_tokamak, from_vmec_wout  # noqa: E402
from fax.fast_ions import FastIonPopulation, eigenfunction_fast_ion_drive, reduced_fast_ion_drive  # noqa: E402
from fax.gap_models import basic_alfven_gap_map  # noqa: E402
from fax.global_modes import (  # noqa: E402
    constant_radial_block_problem,
    radial_block_problem_from_boozer_geometry,
    radial_problem_from_equilibrium,
    solve_radial_block_eigenmodes,
    solve_radial_eigenmodes,
)
from fax.metric import diagonal_continuum_frequencies_hz, metric_from_vmec_wout  # noqa: E402
from fax.plotting import plot_convergence_artifacts  # noqa: E402
from fax.reference_cases import portable_paths, EXTERNAL_ROOT, landreman_paul_qa_lowres  # noqa: E402
from fax.units import PROTON_MASS  # noqa: E402
from fax.validation import finite_difference_derivative  # noqa: E402
from fax.workflows import run_convergence  # noqa: E402


DEFAULT_BOOZMN = EXTERNAL_ROOT / "booz_xform_jax/tests/test_files/boozmn_LandremanSenguptaPlunk_section5p3.nc"
DEFAULT_STELLGAP_BOOZMN = EXTERNAL_ROOT / "Stellgap_test_inputs/boozmn_ctok.nc"
DEFAULT_STELLGAP_POST = EXTERNAL_ROOT / "Stellgap_test_problem/alfven_post"
DEFAULT_AE3D_DIR = EXTERNAL_ROOT / "AE3D_test_problem"
DEFAULT_AE3D_BOOZMN = EXTERNAL_ROOT / "AE3D_test_inputs/boozmn_ctok.nc"


CaseRunner = Callable[[Path, tuple[str, ...]], dict[str, object]]


def _style() -> None:
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "font.size": 9,
            "legend.frameon": False,
            "svg.hashsalt": "fax-publication-suite",
        }
    )


def _save(fig, output_dir: Path, stem: str, formats: tuple[str, ...]) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    for suffix in formats:
        normalized = suffix.lower().removeprefix(".")
        if normalized not in {"png", "svg", "pdf"}:
            raise ValueError(f"unsupported figure format: {suffix}")
        path = output_dir / f"{stem}.{normalized}"
        if normalized == "png":
            fig.savefig(path, facecolor="white")
        elif normalized == "svg":
            fig.savefig(path, metadata={"Date": None})
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
        else:
            fig.savefig(path)
        files.append(str(path))
    plt.close(fig)
    return files


def _skip(case_id: str, reason: str) -> dict[str, object]:
    return {"case": case_id, "status": "skipped", "reason": reason, "figures": []}


def _case_gap_taxonomy(output_dir: Path, formats: tuple[str, ...]) -> dict[str, object]:
    case_id = "01_gap_taxonomy"
    data = basic_alfven_gap_map(radial_points=240)
    q = np.asarray(data.q)
    order = np.argsort(q)
    q = q[order]
    branches = np.asarray(data.continuum_branches_hz)[order] / 1.0e3
    bae = np.asarray(data.bae_frequency_hz)[order] / 1.0e3
    tae = data.tae.estimate
    eae = data.eae.estimate

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), constrained_layout=True)
    for index in range(branches.shape[1]):
        axes[0].plot(q, branches[:, index], color="#777777", lw=1.0, ls="--", alpha=0.75)
    axes[0].fill_between(q, np.asarray(data.tae.lower_frequency_hz)[order] / 1.0e3, np.asarray(data.tae.upper_frequency_hz)[order] / 1.0e3, color="#2458a6", alpha=0.18, label="TAE")
    axes[0].fill_between(q, np.asarray(data.eae.lower_frequency_hz)[order] / 1.0e3, np.asarray(data.eae.upper_frequency_hz)[order] / 1.0e3, color="#b13f2a", alpha=0.16, label="EAE")
    axes[0].plot(q, bae, color="#2e7d32", lw=1.8, label="BAE marker")
    axes[0].axvline(float(1.0 / tae.iota), color="#2458a6", lw=1.0)
    axes[0].axvline(float(1.0 / eae.iota), color="#b13f2a", lw=1.0)
    axes[0].set_title("Alfven continuum gap taxonomy")
    axes[0].set_xlabel("safety factor q")
    axes[0].set_ylabel("frequency [kHz]")
    axes[0].set_ylim(bottom=0.0)
    axes[0].legend(loc="upper left", ncols=3)

    labels = ["BAE", "TAE", "EAE"]
    centers = [
        float(data.bae_frequency_at_surface_hz / 1.0e3),
        float(tae.center_frequency_hz / 1.0e3),
        float(eae.center_frequency_hz / 1.0e3),
    ]
    widths = [0.0, float(tae.width_frequency_hz / 1.0e3), float(eae.width_frequency_hz / 1.0e3)]
    axes[1].bar(labels, centers, color=["#2e7d32", "#2458a6", "#b13f2a"], alpha=0.82)
    axes[1].errorbar(labels, centers, yerr=np.asarray(widths) / 2.0, color="black", lw=1.0, fmt="none")
    axes[1].set_title("reference gap centers and widths")
    axes[1].set_ylabel("frequency [kHz]")
    axes[1].set_ylim(bottom=0.0)

    files = _save(fig, output_dir, case_id, formats)
    return {"case": case_id, "status": "completed", "figures": files, "tae_q": float(1.0 / tae.iota), "eae_q": float(1.0 / eae.iota)}


def _case_convergence(output_dir: Path, formats: tuple[str, ...]) -> dict[str, object]:
    case_id = "02_convergence_validation"
    with tempfile.TemporaryDirectory(prefix="fax_publication_convergence_") as scratch:
        artifacts = Path(scratch) / case_id
        config = FaxConfig(
            run=RunConfig(case=case_id, model="continuum", output_dir=str(artifacts)),
            equilibrium=EquilibriumConfig(source="analytic"),
            grid=GridConfig(radial_points=12, m_max=2, n_max=2, field_periods=1),
        )
        summary = run_convergence(
            config,
            progress=False,
            radial_points=(8, 12, 16),
            angular_points=(8, 12),
            mode_max_values=(1, 2),
        )
        files = [
            str(path)
            for path in plot_convergence_artifacts(
                artifacts / "convergence",
                output_dir=output_dir,
                stem=case_id,
                formats=formats,
            )
        ]
    return {
        "case": case_id,
        "status": "completed",
        "figures": files,
        "max_frequency_l2_error": summary.max_frequency_l2_error,
        "max_residual_p95": summary.max_residual_p95,
    }


def _case_landreman_wout(output_dir: Path, formats: tuple[str, ...]) -> dict[str, object]:
    case_id = "03_landreman_paul_wout_continuum"
    ref = landreman_paul_qa_lowres()
    if ref.wout is None or not ref.wout.exists():
        return _skip(case_id, f"missing local wout fixture: {ref.wout}")
    geometry = from_vmec_wout(ref.wout, name=ref.case_id)
    metric = metric_from_vmec_wout(ref.wout, theta_points=16, zeta_points=16, name=ref.case_id)
    harmonics = jnp.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    profile = np.asarray(continuum_branch_matrix(geometry, harmonics)) / 1.0e3
    metric_branches = np.asarray(diagonal_continuum_frequencies_hz(metric, harmonics)) / 1.0e3
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), constrained_layout=True)
    axes[0].plot(np.asarray(geometry.s), np.asarray(geometry.iota), lw=1.8)
    axes[0].set_title("Landreman-Paul QA iota")
    axes[0].set_xlabel("s")
    axes[0].set_ylabel("iota")
    axes[1].plot(np.asarray(geometry.s), np.asarray(geometry.magnetic_field), lw=1.8, color="#b13f2a")
    axes[1].set_title("flux-surface |B| profile")
    axes[1].set_xlabel("s")
    axes[1].set_ylabel("|B| [T]")
    for branch in range(profile.shape[1]):
        axes[2].plot(np.asarray(geometry.s), profile[:, branch], color="#777777", ls="--", lw=1.0)
        axes[2].plot(np.asarray(metric.s), metric_branches[:, branch], lw=1.2)
    axes[2].set_title("profile vs metric diagonal branches")
    axes[2].set_xlabel("s")
    axes[2].set_ylabel("frequency [kHz]")
    axes[2].set_ylim(bottom=0.0)
    files = _save(fig, output_dir, case_id, formats)
    return {"case": case_id, "status": "completed", "figures": files, "radial_points": int(geometry.s.shape[0])}


def _case_boozer_representation(output_dir: Path, formats: tuple[str, ...]) -> dict[str, object]:
    case_id = "04_boozer_representation"
    spectrum = BoozerSpectrum(
        source="synthetic_publication",
        s=np.asarray([0.25, 0.75]),
        xm=np.asarray([0, 1, 2, 1], dtype=float),
        xn=np.asarray([0, 0, 1, -1], dtype=float),
        bmnc=np.asarray([[2.8, 0.10, -0.035, 0.022], [2.7, 0.08, -0.025, 0.018]]),
        bmns=np.asarray([[0.0, 0.025, 0.012, -0.008], [0.0, 0.020, 0.010, -0.006]]),
        field_periods=2,
    )
    theta, zeta = boozer_angle_grid(theta_points=96, zeta_points=96)
    field = np.asarray(reconstruct_boozer_field(spectrum, surface_index=0, theta=theta, zeta=zeta))
    amplitudes = np.max(np.abs(np.asarray(spectrum.bmnc)), axis=0)
    labels = [f"({int(m)},{int(n)})" for m, n in zip(np.asarray(spectrum.xm), np.asarray(spectrum.xn), strict=True)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    mesh = axes[0].pcolormesh(np.asarray(theta[:, 0]), np.asarray(zeta[0]), field.T, shading="auto")
    fig.colorbar(mesh, ax=axes[0], label="|B| [T]")
    axes[0].set_title("Boozer |B|(theta,zeta)")
    axes[0].set_xlabel("theta")
    axes[0].set_ylabel("zeta")
    axes[1].bar(labels, amplitudes, color="#2458a6")
    axes[1].set_title("dominant Boozer harmonics")
    axes[1].set_ylabel("max radial amplitude [T]")
    axes[1].tick_params(axis="x", rotation=25)
    files = _save(fig, output_dir, case_id, formats)
    return {
        "case": case_id,
        "status": "completed",
        "figures": files,
        "projection_error": float(boozer_projection_error(spectrum, surface_index=0)),
        "orthogonality_error": float(boozer_harmonic_orthogonality_error(spectrum.xm, spectrum.xn)),
        "nonaxisymmetric_energy": float(normalized_nonaxisymmetric_boozer_energy(spectrum)),
    }


def _case_boozer_local_3d(output_dir: Path, formats: tuple[str, ...]) -> dict[str, object]:
    case_id = "05_boozer_local_3d_continuum"
    if not DEFAULT_BOOZMN.exists():
        return _skip(case_id, f"missing boozmn fixture: {DEFAULT_BOOZMN}")
    geometry = read_boozmn_geometry(DEFAULT_BOOZMN)
    harmonics = jnp.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    frequencies = np.asarray(
        boozer_3d_continuum_frequencies_hz(geometry, harmonics, mass_density=3.4e-8, theta_points=24, zeta_points=24)
    ) / 1.0e3
    grid = boozer_geometry_grid(geometry, theta_points=32, zeta_points=32)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    mesh = axes[0].pcolormesh(np.asarray(grid.theta[:, 0]), np.asarray(grid.zeta[0]), np.asarray(grid.magnetic_field[0]).T, shading="auto")
    fig.colorbar(mesh, ax=axes[0], label="|B| [T]")
    axes[0].set_title("real boozmn |B| surface")
    axes[0].set_xlabel("theta_B")
    axes[0].set_ylabel("zeta_B")
    for branch in range(frequencies.shape[1]):
        axes[1].plot(np.asarray(geometry.s), frequencies[:, branch], lw=1.2)
    axes[1].set_title("local 3-D Boozer continuum")
    axes[1].set_xlabel("s")
    axes[1].set_ylabel("frequency [kHz]")
    axes[1].set_ylim(bottom=0.0)
    files = _save(fig, output_dir, case_id, formats)
    return {"case": case_id, "status": "completed", "figures": files, "surface_count": int(geometry.s.shape[0])}


def _case_global_modes(output_dir: Path, formats: tuple[str, ...]) -> dict[str, object]:
    case_id = "06_global_radial_eigenmodes"
    geometry = analytic_circular_tokamak(radial_points=48, iota_axis=0.25, iota_edge=0.72)
    problem = radial_problem_from_equilibrium(geometry, harmonic=(2, 1), radial_coupling_fraction=0.02)
    result = solve_radial_eigenmodes(problem)
    block = constant_radial_block_problem(radial_points=28, continuum_frequency_hz=(95_000.0, 145_000.0))
    block_result = solve_radial_block_eigenmodes(block)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
    axes[0].plot(np.asarray(problem.s), np.asarray(problem.continuum_omega) / (2.0 * np.pi * 1.0e3), lw=1.5)
    for freq in np.asarray(result.frequencies_hz[:5]) / 1.0e3:
        axes[0].axhline(freq, color="#2458a6", lw=0.8, alpha=0.55)
    axes[0].set_title("global modes over local continuum")
    axes[0].set_xlabel("s")
    axes[0].set_ylabel("frequency [kHz]")
    for mode in range(3):
        axes[1].plot(np.asarray(problem.s[1:-1]), np.asarray(result.eigenvectors[:, mode]), lw=1.2, label=f"mode {mode + 1}")
    axes[1].set_title("radial eigenfunctions")
    axes[1].set_xlabel("s")
    axes[1].set_ylabel("amplitude")
    axes[1].legend(loc="best")
    axes[2].semilogy(np.asarray(block_result.residuals[:12]), marker="o", lw=1.0)
    axes[2].set_title("block eigenpair residuals")
    axes[2].set_xlabel("mode")
    axes[2].set_ylabel("relative residual")
    files = _save(fig, output_dir, case_id, formats)
    return {"case": case_id, "status": "completed", "figures": files, "lowest_frequency_hz": float(result.frequencies_hz[0])}


def _case_fast_ion_drive(output_dir: Path, formats: tuple[str, ...]) -> dict[str, object]:
    case_id = "07_fast_ion_drive_damping"
    s = jnp.linspace(0.0, 1.0, 100)
    center = jnp.exp(-((s - 0.45) / 0.10) ** 2)
    edge = jnp.exp(-((s - 0.78) / 0.08) ** 2)
    structures = jnp.stack([center, edge], axis=0)[:, :, None]
    population = FastIonPopulation(
        density_fraction=0.035,
        pressure_gradient=-jnp.exp(-((s - 0.45) / 0.13) ** 2),
        transit_frequency_hz=jnp.full_like(s, 40_000.0),
        precession_frequency_hz=jnp.full_like(s, 5_000.0),
        resonance_width_hz=jnp.full_like(s, 2_500.0),
    )
    frequencies = jnp.linspace(20_000.0, 90_000.0, 160)
    drive_scan = jnp.asarray(
        [
            jnp.max(reduced_fast_ion_drive(freq, toroidal_mode=1, transit_harmonic=1, population=population))
            for freq in frequencies
        ]
    )
    result = eigenfunction_fast_ion_drive(
        jnp.asarray([45_000.0, 45_000.0]),
        structures,
        s,
        toroidal_mode=1,
        transit_harmonics=(1,),
        population=population,
        drive_scale_hz=1_000.0,
        damping_rate_hz=8.0,
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
    axes[0].plot(np.asarray(frequencies) / 1.0e3, np.asarray(drive_scan), lw=1.6)
    axes[0].set_title("resonance drive scan")
    axes[0].set_xlabel("mode frequency [kHz]")
    axes[0].set_ylabel("drive indicator")
    axes[1].plot(np.asarray(s), np.asarray(population.pressure_gradient), lw=1.4, label="EP pressure gradient")
    axes[1].plot(np.asarray(s), np.asarray(center / jnp.max(center)), lw=1.2, label="core mode")
    axes[1].plot(np.asarray(s), np.asarray(edge / jnp.max(edge)), lw=1.2, label="edge mode")
    axes[1].set_title("mode/drive overlap")
    axes[1].set_xlabel("s")
    axes[1].legend(loc="best")
    axes[2].bar(["core", "edge"], np.asarray(result.net_growth_rate_hz), color=["#2458a6", "#b13f2a"])
    axes[2].axhline(0.0, color="black", lw=1.0)
    axes[2].set_title("net growth-rate screen")
    axes[2].set_ylabel("net growth [Hz]")
    files = _save(fig, output_dir, case_id, formats)
    return {"case": case_id, "status": "completed", "figures": files, "core_net_growth_hz": float(result.net_growth_rate_hz[0])}


def _case_sensitivity(output_dir: Path, formats: tuple[str, ...]) -> dict[str, object]:
    case_id = "08_differentiable_sensitivity"
    base = analytic_circular_tokamak(radial_points=40, iota_axis=0.25, iota_edge=0.65)
    harmonics = jnp.asarray([[2.0, 1.0], [3.0, 2.0]])

    def mean_frequency(density_scale: float, field_scale: float, iota_shift: float) -> jax.Array:
        geometry = replace(
            base,
            iota=base.iota + iota_shift,
            magnetic_field=base.magnetic_field * field_scale,
        ).with_density_scale(density_scale)
        return jnp.mean(continuum_branch_matrix(geometry, harmonics))

    density_values = np.linspace(0.6, 1.8, 45)
    density_curve = np.asarray([mean_frequency(value, 1.0, 0.0) for value in density_values]) / 1.0e3
    reference_frequency = float(mean_frequency(1.0, 1.0, 0.0))
    gradient = jax.grad(lambda x: mean_frequency(x, 1.0, 0.0))(1.0)
    finite_difference = finite_difference_derivative(lambda x: float(mean_frequency(x, 1.0, 0.0)), 1.0, step=1.0e-3)
    field_gradient = jax.grad(lambda x: mean_frequency(1.0, x, 0.0))(1.0)
    iota_gradient = jax.grad(lambda x: mean_frequency(1.0, 1.0, x))(0.0)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), constrained_layout=True)
    axes[0].plot(density_values, density_curve, lw=1.8)
    axes[0].set_title("density sensitivity")
    axes[0].set_xlabel("density scale")
    axes[0].set_ylabel("mean branch frequency [kHz]")
    labels = ["rho AD", "rho FD", "B AD", "iota AD"]
    values = [
        float(gradient / reference_frequency),
        float(finite_difference / reference_frequency),
        float(field_gradient / reference_frequency),
        float(iota_gradient / reference_frequency),
    ]
    axes[1].bar(labels, values, color=["#2458a6", "#72b7b2", "#b13f2a", "#2e7d32"])
    axes[1].set_title("autodiff checks")
    axes[1].set_ylabel("relative sensitivity")
    axes[1].tick_params(axis="x", rotation=20)
    files = _save(fig, output_dir, case_id, formats)
    return {"case": case_id, "status": "completed", "figures": files, "density_gradient_hz": float(gradient), "finite_difference_hz": float(finite_difference)}


def _case_stellgap_global_screen(output_dir: Path, formats: tuple[str, ...]) -> dict[str, object]:
    case_id = "09_stellgap_ctok_gap_anchored_global_screen"
    if not DEFAULT_STELLGAP_BOOZMN.exists() or not DEFAULT_STELLGAP_POST.exists():
        return _skip(case_id, "missing local STELLGAP CTOK boozmn/alfven_post artifacts")
    geometry = read_boozmn_geometry(DEFAULT_STELLGAP_BOOZMN)
    reference = read_stellgap_alfven_post(DEFAULT_STELLGAP_POST)
    harmonics = jnp.asarray([[float(m), 2.0] for m in range(0, 7)])
    local = np.asarray(
        boozer_3d_continuum_frequencies_hz(
            geometry,
            harmonics,
            mass_density=2.0 * PROTON_MASS * 1.0e19,
            theta_points=16,
            zeta_points=16,
        )
    ) / 1.0e3
    problem = radial_block_problem_from_boozer_geometry(
        geometry,
        harmonics,
        mass_density=2.0 * PROTON_MASS * 1.0e19,
        radial_stiffness_method="mass",
        radial_coupling_fraction=0.002,
        theta_points=12,
        zeta_points=12,
    )
    result = solve_radial_block_eigenmodes(problem)
    comparison = stellgap_ctok_local_continuum_error(
        boozmn_path=DEFAULT_STELLGAP_BOOZMN,
        alfven_post_path=DEFAULT_STELLGAP_POST,
        mass_density=2.0 * PROTON_MASS * 1.0e19,
        theta_points=32,
        zeta_points=32,
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)
    axes[0].scatter(np.asarray(reference.s), np.asarray(reference.frequency_khz), s=3, alpha=0.22, label="STELLGAP CTOK")
    for branch in range(local.shape[1]):
        axes[0].plot(np.asarray(geometry.s), local[:, branch], color="black", lw=0.8, alpha=0.52)
    for freq in np.asarray(result.frequencies_hz[:6]) / 1.0e3:
        axes[0].axhline(freq, color="#b13f2a", lw=0.8, alpha=0.55)
    axes[0].set_title("CTOK continuum and FAX global modes")
    axes[0].set_xlabel("s")
    axes[0].set_ylabel("frequency [kHz]")
    axes[0].set_ylim(0.0, 8_500.0)
    axes[0].legend(loc="upper right")
    for mode in range(3):
        axes[1].plot(np.asarray(geometry.s), np.linalg.norm(np.asarray(result.mode_structures[mode]), axis=1), lw=1.2, label=f"mode {mode + 1}")
    axes[1].set_title("global radial envelopes")
    axes[1].set_xlabel("s")
    axes[1].set_ylabel("harmonic norm")
    axes[1].legend(loc="best")
    axes[2].semilogy(np.asarray(result.residuals[:12]), marker="o", lw=1.0, label="global residual")
    axes[2].axhline(comparison.p95_relative_error, color="#2458a6", ls="--", lw=1.0, label="STELLGAP p95 local error")
    axes[2].set_title("residual and local-continuum parity")
    axes[2].set_xlabel("mode/metric")
    axes[2].set_ylabel("relative metric")
    axes[2].legend(loc="best")
    files = _save(fig, output_dir, case_id, formats)
    return {
        "case": case_id,
        "status": "completed",
        "figures": files,
        "stellgap_p95_relative_error": comparison.p95_relative_error,
        "lowest_global_frequency_hz": float(result.frequencies_hz[0]),
        "max_residual": float(jnp.max(result.residuals[:12])),
    }


def _case_ae3d_matrix_parity(output_dir: Path, formats: tuple[str, ...]) -> dict[str, object]:
    case_id = "10_ae3d_matrix_eigenmode_parity"
    a_matrix = DEFAULT_AE3D_DIR / "a_matrix.dat"
    b_matrix = DEFAULT_AE3D_DIR / "b_matrix.dat"
    egn_mode = DEFAULT_AE3D_DIR / "egn_mode_asci.dat"
    if not all(path.exists() for path in (a_matrix, b_matrix, egn_mode)):
        return _skip(case_id, "missing local AE3D public matrix/eigenmode artifacts")
    data = read_ae3d_eigenmodes(egn_mode, max_modes=8)
    parity = ae3d_matrix_eigenmode_parity(
        a_matrix_path=a_matrix,
        b_matrix_path=b_matrix,
        egn_mode_path=egn_mode,
        max_modes=8,
    )
    dense_parity = ae3d_dense_spectrum_parity(
        a_matrix_path=a_matrix,
        b_matrix_path=b_matrix,
        egn_mode_path=egn_mode,
        max_modes=16,
    )
    global_parity = ae3d_global_mode_parity(
        a_matrix_path=a_matrix,
        b_matrix_path=b_matrix,
        egn_mode_path=egn_mode,
        max_modes=8,
    )
    placement = (
        ae3d_continuum_placement_parity(
            egn_mode_path=egn_mode,
            boozmn_path=DEFAULT_AE3D_BOOZMN,
            mass_density=2.0 * PROTON_MASS * 1.0e19,
            max_modes=128,
            theta_points=32,
            zeta_points=32,
        )
        if DEFAULT_AE3D_BOOZMN.exists()
        else None
    )
    modes = np.asarray(data.modes)
    first_mode = np.abs(modes[0])
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.9), constrained_layout=True)
    frequencies = np.sqrt(np.maximum(np.asarray(data.eigenvalue_omega2), 0.0)) / 1.0e3
    axes[0].plot(np.arange(1, len(frequencies) + 1), frequencies, marker="o", lw=1.2)
    axes[0].set_title("AE3D public eigenfrequencies")
    axes[0].set_xlabel("mode index")
    axes[0].set_ylabel("sqrt(lambda) [kHz]")
    mesh = axes[1].pcolormesh(np.asarray(data.rho), np.arange(first_mode.shape[1]), first_mode.T, shading="auto")
    fig.colorbar(mesh, ax=axes[1], label="|component|")
    axes[1].set_title("first AE3D mode structure")
    axes[1].set_xlabel("rho")
    axes[1].set_ylabel("harmonic index")
    metrics = [
        parity.max_residual,
        parity.p95_residual,
        parity.max_rayleigh_relative_error,
        parity.max_frequency_relative_error,
        dense_parity.max_eigenvalue_relative_error,
        global_parity.max_mass_weighted_mode_shape_error,
        placement.p95_closest_branch_relative_error if placement is not None else np.nan,
        placement.max_closest_branch_relative_error if placement is not None else np.nan,
    ]
    axes[2].bar(
        ["max res.", "p95 res.", "Rayleigh", "freq.", "dense eig.", "shape", "place p95", "place max"],
        metrics,
        color=["#2458a6", "#72b7b2", "#b13f2a", "#2e7d32", "#b279a2", "#499894", "#f28e2b", "#8cd17d"],
    )
    axes[2].set_yscale("log")
    axes[2].set_title("AE3D matrix/eigenmode parity")
    axes[2].set_ylabel("relative metric")
    axes[2].tick_params(axis="x", rotation=20)
    files = _save(fig, output_dir, case_id, formats)
    return {
        "case": case_id,
        "status": "completed",
        "figures": files,
        **parity.__dict__,
        "dense_spectrum_max_eigenvalue_relative_error": dense_parity.max_eigenvalue_relative_error,
        "dense_spectrum_max_frequency_relative_error": dense_parity.max_frequency_relative_error,
        "global_mode_max_frequency_relative_error": global_parity.max_frequency_relative_error,
        "global_mode_max_mass_weighted_mode_shape_error": global_parity.max_mass_weighted_mode_shape_error,
        "continuum_placement_p95_closest_branch_relative_error": None
        if placement is None
        else placement.p95_closest_branch_relative_error,
        "continuum_placement_max_closest_branch_relative_error": None
        if placement is None
        else placement.max_closest_branch_relative_error,
    }


CASES: dict[str, CaseRunner] = {
    "01": _case_gap_taxonomy,
    "02": _case_convergence,
    "03": _case_landreman_wout,
    "04": _case_boozer_representation,
    "05": _case_boozer_local_3d,
    "06": _case_global_modes,
    "07": _case_fast_ion_drive,
    "08": _case_sensitivity,
    "09": _case_stellgap_global_screen,
    "10": _case_ae3d_matrix_parity,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/publication_figure_suite", help="Directory for figures and summary JSON.")
    parser.add_argument("--format", action="append", default=[], choices=("svg", "png", "pdf"), help="Figure format; repeatable. Defaults to svg and png.")
    parser.add_argument("--case", action="append", default=[], choices=tuple(CASES), help="Run one case id; repeatable. Defaults to all ten cases.")
    args = parser.parse_args(argv)

    _style()
    output_dir = Path(args.output_dir).expanduser().resolve()
    formats = tuple(args.format) if args.format else ("svg", "png")
    selected = tuple(args.case) if args.case else tuple(CASES)
    results = [CASES[case_id](output_dir, formats) for case_id in selected]
    payload = {
        "case_count": len(results),
        "completed_count": sum(result["status"] == "completed" for result in results),
        "skipped_count": sum(result["status"] == "skipped" for result in results),
        "results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "publication_figure_suite_summary.json").write_text(json.dumps(portable_paths(payload), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(portable_paths(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
