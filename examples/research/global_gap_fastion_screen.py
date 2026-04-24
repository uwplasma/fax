"""Screen reduced global modes, analytic gaps, fast-ion resonance, and gradients."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp

from fax.equilibrium import analytic_circular_tokamak
from fax.fast_ions import FastIonPopulation, resonance_scan
from fax.gap_models import eae_gap_estimate, hae_gap_estimate, tae_gap_estimate
from fax.global_modes import (
    constant_problem_discrete_omega2,
    constant_radial_problem,
    radial_problem_from_equilibrium,
    solve_radial_eigenmodes,
)
from fax.optimization import radial_mode_avoidance_objective


def main() -> int:
    """Print a compact reduced-physics screening packet."""
    fixture = constant_radial_problem(radial_points=48, continuum_frequency_hz=100_000.0, radial_coupling_omega2=3.0e9)
    fixture_result = solve_radial_eigenmodes(fixture)
    exact_omega2 = constant_problem_discrete_omega2(fixture)

    geometry = analytic_circular_tokamak(radial_points=72, iota_axis=0.28, iota_edge=0.70)
    radial_problem = radial_problem_from_equilibrium(geometry, harmonic=(2, 1))
    radial_result = solve_radial_eigenmodes(radial_problem)

    population = FastIonPopulation(
        density_fraction=0.04,
        pressure_gradient=-2.0,
        transit_frequency_hz=35_000.0,
        precession_frequency_hz=4_000.0,
        resonance_width_hz=2_500.0,
    )
    frequency_grid = jnp.linspace(10_000.0, 90_000.0, 240)
    transit_harmonics = (-1, 0, 1, 2)
    drive = resonance_scan(
        frequency_grid,
        toroidal_mode=2,
        transit_harmonics=transit_harmonics,
        population=population,
    )
    peak_index = jnp.argmax(jnp.max(drive, axis=0))

    def objective(parameters):
        return radial_mode_avoidance_objective(
            parameters,
            geometry=geometry,
            harmonic=(2, 1),
            avoid_window_hz=(115_000.0, 145_000.0),
            softness_hz=4_000.0,
        )

    parameters = jnp.asarray([0.0, 0.0, 0.0])
    gradient = jax.grad(objective)(parameters)
    gaps = [
        tae_gap_estimate(geometry, m=1, n=1, inverse_aspect_ratio=0.08),
        eae_gap_estimate(geometry, m=1, n=1, ellipticity=0.05),
        hae_gap_estimate(geometry, m=2, n=2, helical_coupling=0.07),
    ]

    payload = {
        "diagnostic": "reduced global/gap/fast-ion screening packet",
        "constant_fixture_relative_omega2_error": float(
            jnp.linalg.norm(fixture_result.omega2[:8] - exact_omega2[:8]) / jnp.linalg.norm(exact_omega2[:8])
        ),
        "constant_fixture_max_residual": float(jnp.max(fixture_result.residuals[:8])),
        "lowest_radial_mode_frequency_hz": float(radial_result.frequencies_hz[0]),
        "gap_estimates": [
            {
                "family": estimate.family,
                "surface": float(estimate.surface),
                "iota": float(estimate.iota),
                "center_frequency_hz": float(estimate.center_frequency_hz),
                "width_frequency_hz": float(estimate.width_frequency_hz),
            }
            for estimate in gaps
        ],
        "fast_ion_peak": {
            "frequency_hz": float(frequency_grid[peak_index]),
            "drive": float(jnp.max(drive[:, peak_index])),
        },
        "avoidance_objective": float(objective(parameters)),
        "avoidance_gradient": {
            "log_density_scale": float(gradient[0]),
            "iota_shift": float(gradient[1]),
            "log_field_scale": float(gradient[2]),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
