from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

from fax.boozer import (
    boozer_3d_operator_angular_convergence,
    run_vmex_boozer_truncation_convergence,
)
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
            "savefig.dpi": 180,
        }
    )
    return plt


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live VMEX/Boozer convergence smoke checks.")
    parser.add_argument("--vmec-input", default=None, help="VMEC input path. Defaults to the local Landreman-Paul QA case.")
    parser.add_argument("--output-dir", default="outputs/vmec_boozer_convergence", help="Directory for figure artifacts.")
    parser.add_argument("--format", default="png", choices=("png", "svg"), help="Figure format.")
    args = parser.parse_args()

    case = landreman_paul_qa_lowres()
    vmec_input = Path(args.vmec_input).expanduser() if args.vmec_input else case.vmec_input
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    with redirect_stdout(sys.stderr):
        truncation = run_vmex_boozer_truncation_convergence(
            vmec_input,
            coarse_mboz=4,
            coarse_nboz=4,
            reference_mboz=5,
            reference_nboz=5,
            surfaces=(0.25, 0.5),
            max_iter=1,
            jit=False,
        )
        angular = boozer_3d_operator_angular_convergence(
            truncation.reference_geometry,
            jnp.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]]),
            mass_density=3.4e-8,
            grids=((12, 12), (16, 16), (20, 20)),
            reference_grid=(32, 32),
        )

    plt = _setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2), constrained_layout=True)

    ax = axes[0]
    metrics = [
        float(jnp.max(truncation.field_relative_error)),
        float(truncation.frequency_relative_error),
        float(truncation.coarse_jacobian_identity_error),
        float(truncation.reference_jacobian_identity_error),
    ]
    ax.bar(["B", "freq", "J coarse", "J ref"], metrics, color="#2e7d32")
    ax.axhline(1.0e-2, color="#222222", linestyle="--", linewidth=1.0, label="gate")
    ax.set_yscale("log")
    ax.set_ylim(1.0e-4, 3.0e-2)
    ax.set_title("Boozer truncation")
    ax.set_ylabel("relative metric")
    ax.legend(fontsize=8)

    ax = axes[1]
    grid_points = np.asarray([theta * zeta for theta, zeta in angular.grids])
    ax.loglog(grid_points, np.asarray(angular.relative_frequency_errors), marker="o", linewidth=1.6, label="frequency")
    ax.loglog(grid_points, np.maximum(np.asarray(angular.hermitian_errors), 1.0e-16), marker="s", linewidth=1.4, label="Hermitian")
    ax.axhline(5.0e-3, color="#222222", linestyle="--", linewidth=1.0, label="gate")
    ax.set_title("Operator angular grid")
    ax.set_xlabel("angular grid points")
    ax.set_ylabel("relative metric")
    ax.set_ylim(1.0e-16, 1.0e-2)
    ax.legend(fontsize=8)

    fig.suptitle("Live VMEX/Boozer convergence smoke checks", fontsize=12, weight="bold")
    figure_path = output_dir / f"vmec_boozer_convergence.{args.format}"
    fig.savefig(figure_path, facecolor="white")
    plt.close(fig)

    payload = {
        "vmec_input": str(vmec_input),
        "truncation_max_metric": truncation.max_metric,
        "truncation_field_relative_error": [float(value) for value in np.asarray(truncation.field_relative_error)],
        "truncation_frequency_relative_error": float(truncation.frequency_relative_error),
        "truncation_coarse_jacobian_identity_error": truncation.coarse_jacobian_identity_error,
        "truncation_reference_jacobian_identity_error": truncation.reference_jacobian_identity_error,
        "angular_max_metric": angular.max_metric,
        "angular_relative_frequency_errors": [float(value) for value in np.asarray(angular.relative_frequency_errors)],
        "angular_hermitian_errors": [float(value) for value in np.asarray(angular.hermitian_errors)],
        "figure_files": [str(figure_path)],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
