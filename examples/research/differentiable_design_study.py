"""Differentiable FAX design study for sensitivity, UQ, and inverse design."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from fax.reference_cases import portable_paths  # noqa: E402
from fax.equilibrium import analytic_circular_tokamak  # noqa: E402
from fax.fast_ions import FastIonPopulation  # noqa: E402
from fax.optimization import (  # noqa: E402
    gradient_descent_minimize,
    linearized_radial_mode_uncertainty,
    radial_fast_ion_growth_screen,
    radial_fast_ion_risk_objective,
    radial_mode_frequency_jacobian,
    radial_mode_frequency_vector,
)


def _save(fig, output_dir: Path, stem: str, formats: tuple[str, ...]) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    for suffix in formats:
        normalized = suffix.lower().removeprefix(".")
        if normalized not in {"png", "svg", "pdf"}:
            raise ValueError(f"unsupported figure format: {suffix}")
        path = output_dir / f"{stem}.{normalized}"
        if normalized == "png":
            fig.savefig(path, facecolor="white", dpi=200)
        elif normalized == "svg":
            fig.savefig(path, metadata={"Date": None})
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
                encoding="utf-8",
            )
        else:
            fig.savefig(path)
        files.append(str(path))
    plt.close(fig)
    return files


def run_design_study(output_dir: Path, formats: tuple[str, ...], *, steps: int = 16) -> dict[str, object]:
    """Run a compact differentiable AE risk-reduction study."""
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "font.size": 9,
            "legend.frameon": False,
            "svg.hashsalt": "fax-differentiable-design-study",
        }
    )

    geometry = analytic_circular_tokamak(radial_points=40, iota_axis=0.26, iota_edge=0.72)
    harmonic = (2, 1)
    mode_count = 4
    radial_coupling_fraction = 0.02
    frequency_parameters = jnp.zeros((3,))
    design_parameters = jnp.zeros((4,))
    avoid_window_hz = (80_000.0, 100_000.0)
    s = jnp.asarray(geometry.s)
    population = FastIonPopulation(
        density_fraction=0.03,
        pressure_gradient=-jnp.exp(-((s - 0.45) / 0.12) ** 2),
        transit_frequency_hz=jnp.full_like(s, 83_000.0),
        precession_frequency_hz=jnp.full_like(s, 5_000.0),
        resonance_width_hz=jnp.full_like(s, 4_000.0),
    )

    covariance = jnp.diag(jnp.asarray([0.04, 0.005, 0.02]) ** 2)
    jacobian = radial_mode_frequency_jacobian(
        frequency_parameters,
        geometry=geometry,
        harmonic=harmonic,
        mode_count=mode_count,
        radial_coupling_fraction=radial_coupling_fraction,
    )
    uncertainty = linearized_radial_mode_uncertainty(
        frequency_parameters,
        covariance=covariance,
        geometry=geometry,
        harmonic=harmonic,
        mode_count=mode_count,
        radial_coupling_fraction=radial_coupling_fraction,
    )

    def objective(params: jax.Array) -> jax.Array:
        return radial_fast_ion_risk_objective(
            params,
            geometry=geometry,
            harmonic=harmonic,
            population=population,
            toroidal_mode=1,
            transit_harmonics=(1,),
            avoid_window_hz=avoid_window_hz,
            mode_count=mode_count,
            drive_scale_hz=1_000.0,
            damping_rate_hz=5.0,
            regularization=0.2,
            radial_coupling_fraction=radial_coupling_fraction,
        )

    optimization = gradient_descent_minimize(
        objective,
        design_parameters,
        step_size=0.05,
        steps=steps,
        gradient_clip_norm=1.0,
    )
    initial_frequencies = radial_mode_frequency_vector(
        frequency_parameters,
        geometry=geometry,
        harmonic=harmonic,
        mode_count=mode_count,
        radial_coupling_fraction=radial_coupling_fraction,
    )
    optimized_frequencies = radial_mode_frequency_vector(
        optimization.parameters[:3],
        geometry=geometry,
        harmonic=harmonic,
        mode_count=mode_count,
        radial_coupling_fraction=radial_coupling_fraction,
    )
    initial_drive = radial_fast_ion_growth_screen(
        design_parameters,
        geometry=geometry,
        harmonic=harmonic,
        population=population,
        toroidal_mode=1,
        transit_harmonics=(1,),
        mode_count=mode_count,
        drive_scale_hz=1_000.0,
        damping_rate_hz=5.0,
        radial_coupling_fraction=radial_coupling_fraction,
    )
    optimized_drive = radial_fast_ion_growth_screen(
        optimization.parameters,
        geometry=geometry,
        harmonic=harmonic,
        population=population,
        toroidal_mode=1,
        transit_harmonics=(1,),
        mode_count=mode_count,
        drive_scale_hz=1_000.0,
        damping_rate_hz=5.0,
        radial_coupling_fraction=radial_coupling_fraction,
    )

    modes = np.arange(1, mode_count + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.2), constrained_layout=True)
    sensitivity = np.asarray(jacobian).T / 1.0e3
    im = axes[0, 0].imshow(sensitivity, aspect="auto", cmap="coolwarm")
    fig.colorbar(im, ax=axes[0, 0], label="df/dp [kHz / parameter]")
    axes[0, 0].set_title("autodiff sensitivity matrix")
    axes[0, 0].set_xticks(np.arange(mode_count), modes)
    axes[0, 0].set_yticks(np.arange(3), ["log rho", "iota shift", "log B"])
    axes[0, 0].set_xlabel("radial eigenmode")

    axes[0, 1].axhspan(avoid_window_hz[0] / 1.0e3, avoid_window_hz[1] / 1.0e3, color="#b13f2a", alpha=0.14, label="risk window")
    axes[0, 1].errorbar(
        modes - 0.08,
        np.asarray(uncertainty.mean_hz) / 1.0e3,
        yerr=2.0 * np.asarray(uncertainty.sigma_hz) / 1.0e3,
        fmt="o",
        color="#2458a6",
        label="initial +/- 2 sigma",
    )
    axes[0, 1].plot(modes + 0.08, np.asarray(optimized_frequencies) / 1.0e3, "s", color="#2e7d32", label="optimized")
    axes[0, 1].set_title("frequency UQ and inverse-design move")
    axes[0, 1].set_xlabel("mode")
    axes[0, 1].set_ylabel("frequency [kHz]")
    axes[0, 1].legend(loc="best")

    axes[1, 0].semilogy(np.asarray(optimization.objective_history), marker="o", lw=1.2)
    axes[1, 0].set_title("gradient optimization history")
    axes[1, 0].set_xlabel("iteration")
    axes[1, 0].set_ylabel("risk objective")

    width = 0.36
    axes[1, 1].bar(modes - width / 2, np.asarray(initial_drive.net_growth_rate_hz), width=width, label="initial", color="#2458a6")
    axes[1, 1].bar(modes + width / 2, np.asarray(optimized_drive.net_growth_rate_hz), width=width, label="optimized", color="#2e7d32")
    axes[1, 1].axhline(0.0, color="black", lw=1.0)
    axes[1, 1].set_title("fast-ion growth screen")
    axes[1, 1].set_xlabel("mode")
    axes[1, 1].set_ylabel("net growth [Hz]")
    axes[1, 1].legend(loc="best")

    files = _save(fig, output_dir, "differentiable_design_study", formats)
    payload = {
        "figure_files": files,
        "initial_objective": float(optimization.objective_history[0]),
        "final_objective": float(optimization.objective_history[-1]),
        "initial_frequencies_hz": np.asarray(initial_frequencies).tolist(),
        "optimized_frequencies_hz": np.asarray(optimized_frequencies).tolist(),
        "frequency_jacobian_hz_per_parameter": np.asarray(jacobian).tolist(),
        "linearized_sigma_hz": np.asarray(uncertainty.sigma_hz).tolist(),
        "initial_net_growth_hz": np.asarray(initial_drive.net_growth_rate_hz).tolist(),
        "optimized_net_growth_hz": np.asarray(optimized_drive.net_growth_rate_hz).tolist(),
        "optimized_parameters": np.asarray(optimization.parameters).tolist(),
    }
    (output_dir / "differentiable_design_study_summary.json").write_text(
        json.dumps(portable_paths(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/differentiable_design_study")
    parser.add_argument("--format", action="append", default=[], choices=("svg", "png", "pdf"))
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).expanduser().resolve()
    formats = tuple(args.format) if args.format else ("svg", "png")
    payload = run_design_study(output_dir, formats, steps=args.steps)
    print(json.dumps(portable_paths(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
