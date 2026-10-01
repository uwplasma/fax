#!/usr/bin/env python
"""Validate a VMEC-boundary Alfvén objective gradient and make a summary plot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fax.vmec_boozer_design import VmecBoozerFaxDesignSettings, validate_vmec_boundary_alfven_gradient


def _parse_surfaces(text: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in text.split(",") if item.strip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a differentiable vmec_jax -> booz_xform_jax -> FAX Alfvén design-gradient check."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../vmec_jax/examples/data/input.LandremanPaul2021_QA_lowres"),
        help="VMEC input file used for the fixed-boundary design check.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vmec_boozer_alfven_design_gradient"))
    parser.add_argument("--max-mode", type=int, default=1)
    parser.add_argument("--parameter-count", type=int, default=1)
    parser.add_argument("--surfaces", type=str, default="0.25,0.40,0.55,0.70")
    parser.add_argument("--mboz", type=int, default=3)
    parser.add_argument("--nboz", type=int, default=3)
    parser.add_argument("--vmec-max-iter", type=int, default=4)
    parser.add_argument("--fd-step", type=float, default=1.0e-4)
    parser.add_argument("--descent-step", type=float, default=1.0e-3)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = VmecBoozerFaxDesignSettings(
        input_path=args.input,
        max_mode=int(args.max_mode),
        parameter_count=int(args.parameter_count),
        surfaces=_parse_surfaces(args.surfaces),
        mboz=int(args.mboz),
        nboz=int(args.nboz),
        vmec_max_iter=int(args.vmec_max_iter),
    )

    print("FAX VMEC-boundary Alfvén design-gradient check")
    print("================================================")
    print(f"input:          {Path(settings.input_path).expanduser().resolve()}")
    print(f"boundary DOFs:  max_mode={settings.max_mode}, count={settings.parameter_count}")
    print(f"Boozer grid:    mboz={settings.mboz}, nboz={settings.nboz}, surfaces={settings.surfaces}")
    print(f"VMEC solver:    differentiable fixed-boundary steps={settings.vmec_max_iter}")
    print("running AD gradient, centered finite differences, and one descent line search ...")

    result = validate_vmec_boundary_alfven_gradient(
        settings,
        finite_difference_step=float(args.fd_step),
        descent_step=float(args.descent_step),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "parameter_names": list(result.parameter_names),
        "objective": float(result.objective),
        "descent_objective": float(result.descent_objective),
        "relative_gradient_error": float(result.relative_gradient_error),
        "gradient": np.asarray(result.gradient, dtype=float).tolist(),
        "finite_difference_gradient": np.asarray(result.finite_difference_gradient, dtype=float).tolist(),
        "initial_frequencies_hz": np.asarray(result.initial_frequencies_hz, dtype=float).tolist(),
        "descent_frequencies_hz": np.asarray(result.descent_frequencies_hz, dtype=float).tolist(),
        "descent_step_norm": float(result.descent_step_norm),
        "metadata": list(result.metadata),
    }
    summary_path = args.output_dir / "vmec_boozer_alfven_design_gradient.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    figure_path = args.output_dir / "vmec_boozer_alfven_design_gradient.png"
    figure_svg_path = args.output_dir / "vmec_boozer_alfven_design_gradient.svg"
    _plot_result(result, settings, figure_path, figure_svg_path)

    print(f"parameters:     {', '.join(result.parameter_names)}")
    print(f"objective:      {float(result.objective):.8e} -> {float(result.descent_objective):.8e}")
    print(f"gradient error: {result.relative_gradient_error:.3e}")
    print(f"wrote:          {summary_path}")
    print(f"figure:         {figure_path}")
    print(f"figure svg:     {figure_svg_path}")


def _plot_result(result, settings: VmecBoozerFaxDesignSettings, path: Path, svg_path: Path) -> None:
    import matplotlib.pyplot as plt

    names = list(result.parameter_names)
    x = np.arange(len(names))
    initial = np.asarray(result.initial_frequencies_hz, dtype=float) / 1.0e3
    final = np.asarray(result.descent_frequencies_hz, dtype=float) / 1.0e3
    gradient = np.asarray(result.gradient, dtype=float)
    finite_difference = np.asarray(result.finite_difference_gradient, dtype=float)
    lower, upper = np.asarray(settings.avoid_window_hz, dtype=float) / 1.0e3

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), constrained_layout=True)

    modes = np.arange(1, len(initial) + 1)
    axes[0].axhspan(lower, upper, color="#f2c94c", alpha=0.28, label="risk window")
    axes[0].plot(modes, initial, "o-", color="#235789", label="initial")
    axes[0].plot(modes, final, "s-", color="#008060", label="after one step")
    axes[0].set_xlabel("global mode index")
    axes[0].set_ylabel("frequency (kHz)")
    axes[0].set_title("Alfvén modes")
    axes[0].legend(frameon=False, fontsize=8)

    width = 0.35
    axes[1].bar(x - width / 2, gradient, width, color="#235789", label="AD")
    axes[1].bar(x + width / 2, finite_difference, width, color="#d95f02", label="finite difference")
    axes[1].set_xticks(x, names, rotation=25, ha="right")
    axes[1].set_ylabel("d objective / d coefficient")
    axes[1].set_title("Gradient parity")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].bar([0, 1], [float(result.objective), float(result.descent_objective)], color=["#235789", "#008060"])
    axes[2].set_xticks([0, 1], ["initial", "descent"])
    axes[2].set_ylabel("smooth risk objective")
    axes[2].set_title("Alfvén-stable step")
    axes[2].set_yscale("log")

    fig.suptitle("VMEC boundary -> Boozer harmonics -> FAX Alfvén design gradient", fontsize=12)
    fig.savefig(path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
