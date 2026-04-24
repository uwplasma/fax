"""Generate a compact validation packet from public FAX workflows.

This script is the recommended first research workflow for new users. It runs
validation, continuum, convergence, and plotting commands through the Python
API and writes artifacts under ``outputs/research_validation_packet``.
"""

from __future__ import annotations

from pathlib import Path

from fax.config import EquilibriumConfig, FaxConfig, GridConfig, RunConfig
from fax.plotting import plot_continuum_artifacts, plot_convergence_artifacts
from fax.workflows import run_continuum, run_convergence, run_validation


def main() -> int:
    """Run the current reviewer-facing validation packet."""
    root = Path("outputs/research_validation_packet")
    config = FaxConfig(
        run=RunConfig(case="research_validation_packet", model="continuum", output_dir=str(root)),
        equilibrium=EquilibriumConfig(source="analytic"),
        grid=GridConfig(radial_points=32, m_max=3, n_max=3, field_periods=1),
    )

    validation = run_validation(config, tags=("analytic", "metric"), progress=True)
    continuum = run_continuum(config, progress=True)
    convergence = run_convergence(
        config,
        progress=True,
        radial_points=(16, 32, 48),
        angular_points=(8, 12, 16),
        mode_max_values=(1, 2, 3),
    )
    plot_continuum_artifacts(root, output_dir=root / "figures")
    plot_convergence_artifacts(root / "convergence", output_dir=root / "figures")

    print("Validation status:", validation.status)
    print("Continuum branch count:", continuum.harmonic_count)
    print("Max convergence L2 error:", f"{convergence.max_frequency_l2_error:.3e}")
    print("Artifacts:", root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
