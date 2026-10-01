"""Run the local 3-D Boozer continuum operator on a real ``boozmn`` file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from fax.boozer import (
    boozer_3d_continuum_frequencies_hz,
    boozer_embedding_jacobian_shape_error,
    boozer_geometry_grid,
    boozer_geometry_jacobian_identity_error,
    read_boozmn_geometry,
)


DEFAULT_BOOZMN = Path("../booz_xform_jax/tests/test_files/boozmn_LandremanSenguptaPlunk_section5p3.nc")


def _write_figure(output_dir: Path, stem: str, geometry, grid, frequencies_hz: np.ndarray) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    surface_index = int(np.argmin(np.abs(np.asarray(geometry.s) - 0.5)))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)

    ax = axes[0]
    image = ax.imshow(
        np.asarray(grid.magnetic_field[surface_index]),
        origin="lower",
        extent=(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi),
        aspect="auto",
        cmap="viridis",
    )
    ax.set_title(r"real $|B|(\theta_B,\zeta_B)$")
    ax.set_xlabel(r"$\zeta_B$")
    ax.set_ylabel(r"$\theta_B$")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="T")

    ax = axes[1]
    for branch in range(frequencies_hz.shape[1]):
        ax.plot(np.asarray(geometry.s), frequencies_hz[:, branch] / 1.0e3, linewidth=1.5, label=f"branch {branch + 1}")
    ax.set_title("local 3-D Boozer continuum")
    ax.set_xlabel("normalized toroidal flux s")
    ax.set_ylabel("frequency [kHz]")
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=8)

    paths = []
    for suffix in ("svg", "png"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, facecolor="white")
        paths.append(str(path))
    plt.close(fig)
    return paths


def main(argv: list[str] | None = None) -> int:
    """Print a compact real-Boozer validation and continuum packet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boozmn", default=str(DEFAULT_BOOZMN), help="Path to a boozmn netCDF file.")
    parser.add_argument("--output-dir", default="outputs/boozmn_3d_continuum", help="Directory for figure files.")
    parser.add_argument("--theta-points", type=int, default=32, help="Boozer theta grid points.")
    parser.add_argument("--zeta-points", type=int, default=32, help="Boozer zeta grid points.")
    parser.add_argument("--mass-density", type=float, default=3.4e-8, help="Mass density used in the local continuum matrix.")
    args = parser.parse_args(argv)

    path = Path(args.boozmn).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"boozmn file not found: {path}")

    geometry = read_boozmn_geometry(path)
    grid = boozer_geometry_grid(geometry, theta_points=args.theta_points, zeta_points=args.zeta_points)
    harmonics = jnp.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0]])
    frequencies_hz = np.asarray(
        boozer_3d_continuum_frequencies_hz(
            geometry,
            harmonics,
            mass_density=args.mass_density,
            theta_points=args.theta_points,
            zeta_points=args.zeta_points,
        )
    )
    figure_files = _write_figure(Path(args.output_dir), "boozmn_3d_continuum", geometry, grid, frequencies_hz)
    payload = {
        "diagnostic": "real boozmn local 3-D Boozer continuum",
        "boozmn": str(path),
        "field_periods": geometry.field_periods,
        "surface_count": int(geometry.s.shape[0]),
        "harmonics": [[int(m), int(n)] for m, n in np.asarray(harmonics, dtype=int)],
        "jacobian_identity_error": float(
            boozer_geometry_jacobian_identity_error(
                geometry,
                theta_points=args.theta_points,
                zeta_points=args.zeta_points,
            )
        ),
        "embedding_jacobian_shape_error": float(
            boozer_embedding_jacobian_shape_error(
                geometry,
                theta_points=args.theta_points,
                zeta_points=args.zeta_points,
            )
        ),
        "min_frequency_hz": float(np.min(frequencies_hz)),
        "max_frequency_hz": float(np.max(frequencies_hz)),
        "figure_files": figure_files,
        "scope": "local surface-by-surface operator; not a global radial eigenmode or cross-code STELLGAP validation",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
