"""Generate a dashboard for public STELLGAP and AE3D answer artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from fax.cross_code import (  # noqa: E402
    ae3d_dense_spectrum_parity,
    ae3d_eigenvalue_residual_summary,
    ae3d_matrix_symmetry_summary,
    ae3d_omega2_eigenvalue_error,
    read_ae3d_eigenvalues,
    read_ae3d_omega2,
    read_stellgap_alfven_post,
    stellgap_postprocess_error,
    stellgap_spec_postprocessed_continuum,
)

DEFAULT_STELLGAP_DIR = Path("/Users/rogerio/local/Stellgap_test_problem")
DEFAULT_AE3D_DIR = Path("/Users/rogerio/local/AE3D_test_problem")


def build_dashboard(stellgap_dir: Path, ae3d_dir: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    alfven_spec = stellgap_dir / "alfven_spec"
    alfven_post = stellgap_dir / "alfven_post"
    omega2_path = ae3d_dir / "omega2.dat"
    egn_values_path = ae3d_dir / "egn_values.dat"
    egn_mode_path = ae3d_dir / "egn_mode_asci.dat"
    a_matrix = ae3d_dir / "a_matrix.dat"
    b_matrix = ae3d_dir / "b_matrix.dat"
    for path in (alfven_spec, alfven_post, omega2_path, egn_values_path, egn_mode_path, a_matrix, b_matrix):
        if not path.exists():
            raise FileNotFoundError(path)

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

    stellgap_processed = stellgap_spec_postprocessed_continuum(alfven_spec)
    stellgap_reference = read_stellgap_alfven_post(alfven_post)
    predicted = np.asarray(stellgap_processed.frequency_khz)
    reference = np.asarray(stellgap_reference.frequency_khz)
    count = min(predicted.shape[0], reference.shape[0])
    stellgap_error = np.abs(predicted[:count] - reference[:count]) / np.maximum(np.abs(reference[:count]), 1.0e-30)

    omega2 = read_ae3d_omega2(omega2_path)
    egn_values = read_ae3d_eigenvalues(egn_values_path)
    raw_eigenvalue = np.asarray(omega2.eigenvalue)
    ae3d_predicted = np.where(raw_eigenvalue > 0.0, np.sqrt(np.maximum(raw_eigenvalue, 0.0)), raw_eigenvalue)
    ae3d_reference = np.asarray(egn_values.frequency_khz)
    ae3d_count = min(ae3d_predicted.shape[0], ae3d_reference.shape[0])

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.4), constrained_layout=True)
    axes[0, 0].scatter(np.asarray(stellgap_reference.s), reference, s=3, alpha=0.35, label="alfven_post")
    axes[0, 0].scatter(
        np.asarray(stellgap_processed.s),
        predicted,
        s=2,
        alpha=0.25,
        label="FAX postprocess",
    )
    axes[0, 0].set_title("STELLGAP postprocess parity")
    axes[0, 0].set_xlabel("s")
    axes[0, 0].set_ylabel("frequency [kHz]")
    axes[0, 0].legend(loc="upper right")

    axes[0, 1].semilogy(stellgap_error, lw=1.1)
    axes[0, 1].axhline(1.0e-5, color="k", ls="--", lw=1.0, label="gate")
    axes[0, 1].set_title("STELLGAP row-wise postprocess error")
    axes[0, 1].set_xlabel("postprocessed row")
    axes[0, 1].set_ylabel("relative error")
    axes[0, 1].legend(loc="upper right")

    axes[1, 0].scatter(ae3d_reference[:ae3d_count], ae3d_predicted[:ae3d_count], s=5, alpha=0.5)
    lo = min(float(np.min(ae3d_reference[:ae3d_count])), float(np.min(ae3d_predicted[:ae3d_count])))
    hi = max(float(np.max(ae3d_reference[:ae3d_count])), float(np.max(ae3d_predicted[:ae3d_count])))
    axes[1, 0].plot([lo, hi], [lo, hi], "k--", lw=1.0)
    axes[1, 0].set_title("AE3D omega2 -> egn_values")
    axes[1, 0].set_xlabel("egn_values table")
    axes[1, 0].set_ylabel("FAX convention reproduction")

    labels = ["A asym.", "B asym.", "AE3D residual", "omega2 error", "dense eig."]
    values = [
        a_summary.relative_asymmetry,
        b_summary.relative_asymmetry,
        ae3d_residual.max_residual,
        ae3d_comparison.max_relative_error,
        dense_parity.max_eigenvalue_relative_error,
    ]
    axes[1, 1].bar(labels, values, color=["#4c78a8", "#72b7b2", "#f58518", "#54a24b", "#b279a2"])
    axes[1, 1].set_yscale("log")
    axes[1, 1].axhline(1.0e-5, color="k", ls="--", lw=1.0, label="matrix/eigenvalue gate")
    axes[1, 1].axhline(1.0e-2, color="#777777", ls=":", lw=1.0, label="residual gate")
    axes[1, 1].set_title("AE3D matrix and spectrum parity")
    axes[1, 1].set_ylabel("metric")
    axes[1, 1].tick_params(axis="x", rotation=18)
    axes[1, 1].legend(loc="upper right")

    paths = []
    for suffix in ("svg", "png"):
        path = output_dir / f"external_code_artifact_dashboard.{suffix}"
        fig.savefig(path, dpi=180)
        paths.append(str(path))
    plt.close(fig)

    metrics = {
        "stellgap_postprocess": stellgap_comparison.__dict__,
        "ae3d_omega2_eigenvalue": ae3d_comparison.__dict__,
        "ae3d_residual": ae3d_residual.__dict__,
        "ae3d_a_matrix": a_summary.__dict__,
        "ae3d_b_matrix": b_summary.__dict__,
        "ae3d_dense_spectrum": {
            "count": dense_parity.count,
            "matrix_dimension": dense_parity.matrix_dimension,
            "real_eigenvalue_count": dense_parity.real_eigenvalue_count,
            "max_eigenvalue_relative_error": dense_parity.max_eigenvalue_relative_error,
            "p95_eigenvalue_relative_error": dense_parity.p95_eigenvalue_relative_error,
            "max_frequency_relative_error": dense_parity.max_frequency_relative_error,
            "p95_frequency_relative_error": dense_parity.p95_frequency_relative_error,
        },
        "figures": paths,
    }
    (output_dir / "external_code_artifact_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stellgap-dir", default=str(DEFAULT_STELLGAP_DIR), help="Directory with alfven_spec/alfven_post.")
    parser.add_argument("--ae3d-dir", default=str(DEFAULT_AE3D_DIR), help="Directory with AE3D answer artifacts.")
    parser.add_argument("--output-dir", default="outputs/external_code_artifacts", help="Output directory.")
    args = parser.parse_args()
    metrics = build_dashboard(Path(args.stellgap_dir), Path(args.ae3d_dir), Path(args.output_dir))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
