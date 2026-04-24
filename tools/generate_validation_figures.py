"""Generate validation and reference figures for the Sphinx docs."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from fax.continuum import continuum_branch_matrix
from fax.equilibrium import analytic_circular_tokamak, from_vmec_wout
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
    fig.savefig(path)
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
    fig.savefig(path)
    plt.close(fig)
    return path


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
    ]
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

