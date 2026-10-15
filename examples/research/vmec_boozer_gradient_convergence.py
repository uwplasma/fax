#!/usr/bin/env python
"""Run a VMEC-boundary/Boozer/FAX gradient convergence ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fax.vmec_boozer_design import (
    VmecBoozerFaxDesignSettings,
    run_vmec_boundary_alfven_gradient_convergence,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check convergence of an end-to-end VMEC-boundary Alfvén design gradient."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "input.LandremanPaul2021_QA_lowres",
        help="VMEC input file for the fixed-boundary design-gradient ladder.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vmec_boozer_gradient_convergence"))
    parser.add_argument("--parameter-count", type=int, default=1)
    parser.add_argument("--fd-step", type=float, default=1.0e-4)
    parser.add_argument("--descent-step", type=float, default=1.0e-3)
    parser.add_argument(
        "--skip-boozer-refinement",
        action="store_true",
        help="Only compare VMEX force tolerances; omit the final Boozer/angular refinement level.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base_settings = VmecBoozerFaxDesignSettings(
        input_path=args.input,
        parameter_count=int(args.parameter_count),
        surfaces=(0.25, 0.40, 0.55, 0.70),
        mboz=3,
        nboz=3,
        theta_points=8,
        zeta_points=8,
    )
    levels: tuple[tuple[str, dict[str, int]], ...] = (
        ("ftol1e-10", {"vmec_ftol": 1.0e-10}),
        ("ftol1e-12", {"vmec_ftol": 1.0e-12}),
    )
    if not args.skip_boozer_refinement:
        levels = (
            *levels,
            (
                "ftol1e-12_boozer4_grid10",
                {"vmec_ftol": 1.0e-12, "mboz": 4, "nboz": 4, "theta_points": 10, "zeta_points": 10},
            ),
        )

    print("FAX VMEC-boundary gradient convergence ladder")
    print("=============================================")
    print(f"input:       {Path(base_settings.input_path).expanduser().resolve()}")
    print(f"parameters:  {base_settings.parameter_count} VMEC boundary coefficient(s)")
    print("levels:")
    for label, overrides in levels:
        merged = {**base_settings.__dict__, **overrides}
        print(
            f"  - {label}: vmec_ftol={merged['vmec_ftol']:.0e}, "
            f"mboz={merged['mboz']}, nboz={merged['nboz']}, "
            f"grid={merged['theta_points']}x{merged['zeta_points']}"
        )
    print("running finite-difference gradient checks at each level ...")

    result = run_vmec_boundary_alfven_gradient_convergence(
        base_settings,
        levels=levels,
        finite_difference_step=float(args.fd_step),
        descent_step=float(args.descent_step),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "vmec_boozer_gradient_convergence.json"
    summary_path.write_text(json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8")
    figure_path = args.output_dir / "vmec_boozer_gradient_convergence.png"
    figure_svg_path = args.output_dir / "vmec_boozer_gradient_convergence.svg"
    _plot_convergence(result, figure_path, figure_svg_path)

    print(f"reference:       {result.reference_label}")
    print(f"max AD/FD error: {result.max_relative_gradient_error:.3e}")
    print(f"max freq drift:  {result.max_frequency_delta_to_reference_hz:.3e} Hz")
    print(f"max grad drift:  {result.max_gradient_delta_to_reference:.3e}")
    print(f"wrote:           {summary_path}")
    print(f"figure:          {figure_path}")
    print(f"figure svg:      {figure_svg_path}")


def _plot_convergence(result, path: Path, svg_path: Path) -> None:
    import matplotlib.pyplot as plt

    points = result.points
    labels = [point.label.replace("_", "\n") for point in points]
    frequencies = [np.asarray(point.frequencies_hz, dtype=float) / 1.0e3 for point in points]
    gradient_error = np.asarray([point.relative_gradient_error for point in points], dtype=float)
    reference_frequency_norm = max(float(np.linalg.norm(frequencies[-1])), 1.0e-30)
    reference_gradient_norm = max(float(np.linalg.norm(np.asarray(points[-1].gradient, dtype=float))), 1.0e-30)
    reference_objective = max(abs(float(points[-1].objective)), 1.0e-30)
    frequency_drift = np.asarray(
        [np.linalg.norm(spectrum - frequencies[-1]) / reference_frequency_norm for spectrum in frequencies],
        dtype=float,
    )
    gradient_drift = np.asarray(
        [point.gradient_delta_to_reference / reference_gradient_norm for point in points],
        dtype=float,
    )
    objective_drift = np.asarray(
        [point.objective_delta_to_reference / reference_objective for point in points],
        dtype=float,
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4), constrained_layout=True)

    mode_index = np.arange(1, len(frequencies[0]) + 1)
    colors = ("#235789", "#008060", "#d95f02", "#6f4aa0")
    for index, spectrum in enumerate(frequencies):
        axes[0].plot(mode_index, spectrum, "o-", color=colors[index % len(colors)], label=points[index].label)
    axes[0].set_xlabel("global mode index")
    axes[0].set_ylabel("frequency (kHz)")
    axes[0].set_title("Frequency refinement")
    axes[0].legend(frameon=False, fontsize=7)

    x = np.arange(len(points))
    axes[1].bar(x, gradient_error, color="#235789")
    axes[1].set_xticks(x, labels)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("relative AD/FD error")
    axes[1].set_title("Gradient correctness")

    width = 0.26
    axes[2].bar(x - width, np.maximum(frequency_drift, 1.0e-16), width, color="#235789", label="frequency")
    axes[2].bar(x, np.maximum(gradient_drift, 1.0e-16), width, color="#008060", label="gradient")
    axes[2].bar(x + width, np.maximum(objective_drift, 1.0e-16), width, color="#d95f02", label="objective")
    axes[2].set_xticks(x, labels)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("relative drift")
    axes[2].set_title("Drift to reference")
    axes[2].legend(frameon=False, fontsize=7)

    fig.suptitle("VMEC boundary -> Boozer -> FAX design-gradient convergence", fontsize=12)
    fig.savefig(path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
