"""Solve the global radial 3-D Boozer eigenmode problem for a ``boozmn`` file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from fax.reference_cases import EXTERNAL_ROOT
from fax.boozer import read_boozmn_geometry
from fax.global_modes import radial_block_problem_from_boozer_geometry, solve_radial_block_eigenmodes

jax.config.update("jax_enable_x64", True)


DEFAULT_BOOZMN = EXTERNAL_ROOT / "booz_xform_jax/tests/test_files/boozmn_LandremanSenguptaPlunk_section5p3.nc"


def _write_figure(output_dir: Path, stem: str, geometry, result, mode_count: int) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    structures = np.asarray(result.mode_structures[:mode_count])
    envelope = np.linalg.norm(structures, axis=2)
    harmonic0 = np.real(structures[0])
    harmonic0 = harmonic0 / max(float(np.max(np.abs(harmonic0))), 1.0e-30)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), constrained_layout=True)

    ax = axes[0]
    for mode in range(mode_count):
        ax.axhline(result.frequencies_hz[mode] / 1.0e3, linewidth=1.3, label=f"mode {mode + 1}")
    ax.set_title("global radial frequencies")
    ax.set_xlabel("normalized toroidal flux s")
    ax.set_ylabel("frequency [kHz]")
    ax.set_xlim(float(np.min(np.asarray(geometry.s))), float(np.max(np.asarray(geometry.s))))
    ax.legend(fontsize=8)

    ax = axes[1]
    for mode in range(mode_count):
        ax.plot(np.asarray(geometry.s), envelope[mode] / max(np.max(envelope[mode]), 1.0e-30), linewidth=1.5, label=f"mode {mode + 1}")
    ax.set_title("radial envelopes")
    ax.set_xlabel("s")
    ax.set_ylabel("normalized |xi|")
    ax.legend(fontsize=8)

    ax = axes[2]
    image = ax.imshow(
        harmonic0.T,
        origin="lower",
        aspect="auto",
        extent=(float(np.min(np.asarray(geometry.s))), float(np.max(np.asarray(geometry.s))), 0.5, harmonic0.shape[1] + 0.5),
        cmap="coolwarm",
    )
    ax.set_title("mode 1 harmonic components")
    ax.set_xlabel("s")
    ax.set_ylabel("harmonic index")
    ax.set_yticks(np.arange(1, harmonic0.shape[1] + 1))
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="real amplitude")

    paths = []
    for suffix in ("svg", "png"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, facecolor="white")
        paths.append(str(path))
    plt.close(fig)
    return paths


def main(argv: list[str] | None = None) -> int:
    """Print a compact global radial 3-D Boozer eigenmode packet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boozmn", default=str(DEFAULT_BOOZMN), help="Path to a boozmn netCDF file.")
    parser.add_argument("--output-dir", default="outputs/boozmn_global_3d_eigenmodes", help="Directory for figure files.")
    parser.add_argument("--theta-points", type=int, default=16, help="Boozer theta grid points.")
    parser.add_argument("--zeta-points", type=int, default=16, help="Boozer zeta grid points.")
    parser.add_argument("--mass-density", type=float, default=3.4e-8, help="Mass density used in the local continuum matrix.")
    parser.add_argument("--radial-coupling-fraction", type=float, default=0.004, help="Radial stiffness as a fraction of the median local omega^2.")
    parser.add_argument("--modes", type=int, default=4, help="Number of modes to summarize and plot.")
    args = parser.parse_args(argv)

    path = Path(args.boozmn).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"boozmn file not found: {path}")

    geometry = read_boozmn_geometry(path)
    harmonics = jnp.asarray([[1.0, 0.0], [2.0, 1.0]])
    problem = radial_block_problem_from_boozer_geometry(
        geometry,
        harmonics,
        mass_density=args.mass_density,
        radial_coupling_fraction=args.radial_coupling_fraction,
        theta_points=args.theta_points,
        zeta_points=args.zeta_points,
    )
    result = solve_radial_block_eigenmodes(problem)
    mode_count = min(int(args.modes), int(result.frequencies_hz.shape[0]))
    figure_files = _write_figure(Path(args.output_dir), "boozmn_global_3d_eigenmodes", geometry, result, mode_count)
    payload = {
        "diagnostic": "global radial 3-D Boozer eigenmodes",
        "boozmn": str(path),
        "surface_count": int(geometry.s.shape[0]),
        "harmonics": [[int(m), int(n)] for m, n in np.asarray(harmonics, dtype=int)],
        "frequencies_hz": [float(value) for value in np.asarray(result.frequencies_hz[:mode_count])],
        "max_residual": float(np.max(np.asarray(result.residuals[:mode_count]))),
        "mode_structure_shape": list(np.asarray(result.mode_structures).shape),
        "figure_files": figure_files,
        "scope": "block radial shear-Alfven envelope operator from local 3-D Boozer matrices; not yet a STELLGAP/AE3D cross-code benchmark",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
