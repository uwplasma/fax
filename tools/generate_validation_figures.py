"""Generate validation and reference figures for the Sphinx docs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from fax.continuum import continuum_branch_matrix
from fax.equilibrium import analytic_circular_tokamak, from_vmec_wout
from fax.gaps import minimum_gap_summary, sorted_branch_gaps
from fax.modes import stellarator_mode_family
from fax.reference_cases import landreman_paul_qa_lowres


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
        }
    )
    return plt


def _harmonic_label(m: int, n: int) -> str:
    return f"m={m}, n={n}"


def _save_svg(fig, path: Path) -> Path:
    """Save an SVG and normalize trailing whitespace for clean diffs."""
    fig.savefig(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    return path


def generate_cylinder_scaling(output_dir: Path) -> Path:
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

    path = output_dir / "analytic_continuum_scaling.svg"
    _save_svg(fig, path)
    plt.close(fig)
    return path


def generate_landreman_reference(output_dir: Path) -> Path:
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

    path = output_dir / "landreman_paul_qa_lowres_profiles.svg"
    _save_svg(fig, path)
    plt.close(fig)
    return path


def generate_gap_sensitivity_dashboard(output_dir: Path) -> Path:
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

    path = output_dir / "gap_sensitivity_dashboard.svg"
    _save_svg(fig, path)
    plt.close(fig)
    return path


def write_figure_manifest(output_dir: Path, paths: list[Path]) -> Path:
    """Write a JSON manifest describing generated figures and their claims."""
    manifest = {
        "generator": "tools/generate_validation_figures.py",
        "figures": [
            {
                "file": path.name,
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
    paths = [
        generate_cylinder_scaling(output_dir),
        generate_landreman_reference(output_dir),
        generate_gap_sensitivity_dashboard(output_dir),
    ]
    paths.append(write_figure_manifest(output_dir, paths))
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
