"""Compare FAX local Boozer continuum branches with the ORNL STELLGAP CTOK test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from fax.boozer import boozer_3d_continuum_frequencies_hz, read_boozmn_geometry
from fax.cross_code import read_stellgap_alfven_post, stellgap_ctok_local_continuum_error
from fax.units import PROTON_MASS

jax.config.update("jax_enable_x64", True)


DEFAULT_BOOZMN = Path("/Users/rogerio/local/Stellgap_test_inputs/boozmn_ctok.nc")
DEFAULT_ALFVEN_POST = Path("/Users/rogerio/local/Stellgap_test_problem/alfven_post")


def _write_figure(output_dir: Path, boozmn: Path, alfven_post: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
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

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
    axes[0].scatter(np.asarray(reference.s), np.asarray(reference.frequency_khz), s=3, alpha=0.28, label="STELLGAP")
    for branch in range(fax_khz.shape[1]):
        axes[0].plot(np.asarray(geometry.s), fax_khz[:, branch], color="black", alpha=0.38, linewidth=0.8)
    axes[0].set_title("CTOK continuum branches")
    axes[0].set_xlabel("s")
    axes[0].set_ylabel("frequency [kHz]")
    axes[0].set_ylim(0.0, 8_500.0)
    axes[0].legend(loc="upper right")

    unique_s = np.unique(np.asarray(reference.s))
    errors = []
    surfaces = []
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
    axes[1].plot(surfaces, errors, marker=".", linewidth=1.0)
    axes[1].axhline(0.04, linestyle="--", color="black", linewidth=1.0, label="gate")
    axes[1].set_title("relative spectrum error")
    axes[1].set_xlabel("s")
    axes[1].set_ylabel("relative L2 error")
    axes[1].legend(loc="upper left")

    files = []
    for suffix in ("svg", "png"):
        path = output_dir / f"stellgap_ctok_comparison.{suffix}"
        fig.savefig(path, facecolor="white")
        files.append(str(path))
    plt.close(fig)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boozmn", default=str(DEFAULT_BOOZMN), help="Path to STELLGAP CTOK boozmn_ctok.nc.")
    parser.add_argument("--alfven-post", default=str(DEFAULT_ALFVEN_POST), help="Path to STELLGAP alfven_post.")
    parser.add_argument("--output-dir", default="outputs/stellgap_ctok_comparison", help="Directory for figures.")
    args = parser.parse_args(argv)

    boozmn = Path(args.boozmn).expanduser().resolve()
    alfven_post = Path(args.alfven_post).expanduser().resolve()
    if not boozmn.exists():
        raise FileNotFoundError(f"missing boozmn file: {boozmn}")
    if not alfven_post.exists():
        raise FileNotFoundError(f"missing STELLGAP alfven_post file: {alfven_post}")
    comparison = stellgap_ctok_local_continuum_error(
        boozmn_path=boozmn,
        alfven_post_path=alfven_post,
        mass_density=2.0 * PROTON_MASS * 1.0e19,
    )
    figure_files = _write_figure(Path(args.output_dir), boozmn, alfven_post)
    print(
        json.dumps(
            {
                "diagnostic": "STELLGAP CTOK local continuum comparison",
                "matched_surfaces": comparison.count,
                "mean_relative_error": comparison.mean_relative_error,
                "p95_relative_error": comparison.p95_relative_error,
                "max_relative_error": comparison.max_relative_error,
                "figure_files": figure_files,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
