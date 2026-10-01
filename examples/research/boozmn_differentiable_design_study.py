"""Differentiable design study on a real ``boozmn`` global-mode operator."""

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

from fax.boozer import read_boozmn_geometry  # noqa: E402
from fax.fast_ions import FastIonPopulation  # noqa: E402
from fax.optimization import (  # noqa: E402
    boozer_fast_ion_growth_screen,
    boozer_fast_ion_risk_objective,
    boozer_global_mode_frequency_jacobian,
    boozer_global_mode_frequency_vector,
    gradient_descent_minimize,
    linearized_boozer_global_mode_uncertainty,
)
from fax.units import PROTON_MASS  # noqa: E402


DEFAULT_BOOZMN = Path("../booz_xform_jax/tests/test_files/boozmn_LandremanSenguptaPlunk_section5p3.nc")


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


def run_design_study(
    boozmn: Path,
    output_dir: Path,
    formats: tuple[str, ...],
    *,
    steps: int = 8,
    theta_points: int = 8,
    zeta_points: int = 8,
) -> dict[str, object]:
    """Run a real-Boozer differentiable risk-reduction study."""
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
            "svg.hashsalt": "fax-boozmn-differentiable-design-study",
        }
    )
    geometry = read_boozmn_geometry(boozmn)
    harmonics = jnp.asarray([[1.0, 0.0], [2.0, 1.0]])
    mass_density = 2.0 * PROTON_MASS * 1.0e19
    radial_stiffness_scale = 1.0e-4
    mode_count = 4
    frequency_parameters = jnp.zeros((3,))
    design_parameters = jnp.zeros((4,))
    avoid_window_hz = (200_000.0, 300_000.0)
    common = {
        "geometry": geometry,
        "harmonics": harmonics,
        "mass_density": mass_density,
        "mode_count": mode_count,
        "theta_points": theta_points,
        "zeta_points": zeta_points,
        "radial_stiffness_scale": radial_stiffness_scale,
    }
    covariance = jnp.diag(jnp.asarray([0.04, 0.01, 0.10]) ** 2)
    frequencies = boozer_global_mode_frequency_vector(frequency_parameters, **common)
    jacobian = boozer_global_mode_frequency_jacobian(frequency_parameters, **common)
    uncertainty = linearized_boozer_global_mode_uncertainty(
        frequency_parameters,
        covariance=covariance,
        **common,
    )

    s = jnp.asarray(geometry.s)
    population = FastIonPopulation(
        density_fraction=0.03,
        pressure_gradient=-jnp.exp(-((s - 0.5) / 0.2) ** 2),
        transit_frequency_hz=jnp.full_like(s, 250_000.0),
        precession_frequency_hz=jnp.full_like(s, 10_000.0),
        resonance_width_hz=jnp.full_like(s, 15_000.0),
    )

    def objective(params: jax.Array) -> jax.Array:
        return boozer_fast_ion_risk_objective(
            params,
            **common,
            population=population,
            toroidal_mode=1,
            transit_harmonics=(1,),
            avoid_window_hz=avoid_window_hz,
            drive_scale_hz=1_000.0,
            damping_rate_hz=5.0,
            frequency_softness_hz=5_000.0,
            regularization=0.1,
        )

    optimization = gradient_descent_minimize(
        objective,
        design_parameters,
        step_size=0.05,
        steps=steps,
        gradient_clip_norm=1.0,
    )
    optimized_frequencies = boozer_global_mode_frequency_vector(optimization.parameters[:3], **common)
    initial_drive = boozer_fast_ion_growth_screen(
        design_parameters,
        **common,
        population=population,
        toroidal_mode=1,
        transit_harmonics=(1,),
        drive_scale_hz=1_000.0,
        damping_rate_hz=5.0,
    )
    optimized_drive = boozer_fast_ion_growth_screen(
        optimization.parameters,
        **common,
        population=population,
        toroidal_mode=1,
        transit_harmonics=(1,),
        drive_scale_hz=1_000.0,
        damping_rate_hz=5.0,
    )

    modes = np.arange(1, mode_count + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.2), constrained_layout=True)
    im = axes[0, 0].imshow(np.asarray(jacobian).T / 1.0e3, aspect="auto", cmap="coolwarm")
    fig.colorbar(im, ax=axes[0, 0], label="df/dp [kHz / parameter]")
    axes[0, 0].set_title("real-boozmn autodiff sensitivity")
    axes[0, 0].set_xticks(np.arange(mode_count), modes)
    axes[0, 0].set_yticks(np.arange(3), ["log rho", "iota shift", "log C_s"])
    axes[0, 0].set_xlabel("global eigenmode")

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
    axes[0, 1].set_title("frequency UQ and design move")
    axes[0, 1].set_xlabel("mode")
    axes[0, 1].set_ylabel("frequency [kHz]")
    axes[0, 1].legend(loc="best")

    axes[1, 0].semilogy(np.asarray(optimization.objective_history), marker="o", lw=1.2)
    axes[1, 0].set_title("real-Boozer risk optimization")
    axes[1, 0].set_xlabel("iteration")
    axes[1, 0].set_ylabel("risk objective")

    width = 0.36
    axes[1, 1].bar(modes - width / 2, np.asarray(initial_drive.net_growth_rate_hz), width=width, label="initial", color="#2458a6")
    axes[1, 1].bar(modes + width / 2, np.asarray(optimized_drive.net_growth_rate_hz), width=width, label="optimized", color="#2e7d32")
    axes[1, 1].axhline(0.0, color="black", lw=1.0)
    axes[1, 1].set_title("fast-ion growth on eigenfunctions")
    axes[1, 1].set_xlabel("mode")
    axes[1, 1].set_ylabel("net growth [Hz]")
    axes[1, 1].legend(loc="best")

    files = _save(fig, output_dir, "boozmn_differentiable_design_study", formats)
    payload = {
        "boozmn": str(boozmn),
        "figure_files": files,
        "initial_objective": float(optimization.objective_history[0]),
        "final_objective": float(optimization.objective_history[-1]),
        "initial_frequencies_hz": np.asarray(frequencies).tolist(),
        "optimized_frequencies_hz": np.asarray(optimized_frequencies).tolist(),
        "frequency_jacobian_hz_per_parameter": np.asarray(jacobian).tolist(),
        "linearized_sigma_hz": np.asarray(uncertainty.sigma_hz).tolist(),
        "initial_net_growth_hz": np.asarray(initial_drive.net_growth_rate_hz).tolist(),
        "optimized_net_growth_hz": np.asarray(optimized_drive.net_growth_rate_hz).tolist(),
        "optimized_parameters": np.asarray(optimization.parameters).tolist(),
    }
    (output_dir / "boozmn_differentiable_design_study_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boozmn", default=str(DEFAULT_BOOZMN), help="Path to a real boozmn netCDF file.")
    parser.add_argument("--output-dir", default="outputs/boozmn_differentiable_design_study")
    parser.add_argument("--format", action="append", default=[], choices=("svg", "png", "pdf"))
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--theta-points", type=int, default=8)
    parser.add_argument("--zeta-points", type=int, default=8)
    args = parser.parse_args(argv)

    boozmn = Path(args.boozmn).expanduser().resolve()
    if not boozmn.exists():
        raise FileNotFoundError(f"boozmn file not found: {boozmn}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    formats = tuple(args.format) if args.format else ("svg", "png")
    payload = run_design_study(
        boozmn,
        output_dir,
        formats,
        steps=args.steps,
        theta_points=args.theta_points,
        zeta_points=args.zeta_points,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
