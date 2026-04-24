"""Toy inverse-design example for moving an analytic continuum branch target."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp

from fax.continuum import continuum_branch_matrix
from fax.equilibrium import EquilibriumGeometry, analytic_circular_tokamak


def shifted_geometry(base: EquilibriumGeometry, iota_shift: jax.Array) -> EquilibriumGeometry:
    """Return the analytic fixture with a scalar rotational-transform shift."""
    return EquilibriumGeometry(
        name=base.name,
        s=base.s,
        iota=base.iota + iota_shift,
        magnetic_field=base.magnetic_field,
        mass_density=base.mass_density,
        major_radius=base.major_radius,
        minor_radius=base.minor_radius,
        field_periods=base.field_periods,
        metadata=base.metadata,
    )


def main() -> int:
    """Optimize a one-parameter iota shift against a target branch frequency."""
    base = analytic_circular_tokamak(radial_points=96, iota_axis=0.25, iota_edge=0.72)
    harmonic = jnp.asarray([[2.0, 1.0]])
    initial_mean = jnp.mean(continuum_branch_matrix(base, harmonic))
    target_iota_shift = jnp.asarray(-0.01)
    target = jnp.mean(continuum_branch_matrix(shifted_geometry(base, target_iota_shift), harmonic))

    def objective(raw_parameter: jax.Array) -> jax.Array:
        iota_shift = 0.05 * jnp.tanh(raw_parameter)
        geometry = shifted_geometry(base, iota_shift)
        mean_frequency = jnp.mean(continuum_branch_matrix(geometry, harmonic))
        regularization = 1.0e-6 * (iota_shift / 0.05) ** 2
        return ((mean_frequency - target) / target) ** 2 + regularization

    grad_objective = jax.grad(objective)
    raw_parameter = jnp.asarray(0.0)
    history = []
    for iteration in range(40):
        value = objective(raw_parameter)
        gradient = grad_objective(raw_parameter)
        raw_parameter = raw_parameter - 20.0 * gradient
        iota_shift = 0.05 * jnp.tanh(raw_parameter)
        history.append(
            {
                "iteration": iteration,
                "objective": float(value),
                "iota_shift": float(iota_shift),
                "gradient": float(gradient),
            }
        )

    iota_shift = 0.05 * jnp.tanh(raw_parameter)
    final_mean = jnp.mean(continuum_branch_matrix(shifted_geometry(base, iota_shift), harmonic))
    payload = {
        "diagnostic": "one-parameter inverse design on analytic continuum mean frequency",
        "initial_mean_frequency_hz": float(initial_mean),
        "target_mean_frequency_hz": float(target),
        "target_iota_shift": float(target_iota_shift),
        "final_mean_frequency_hz": float(final_mean),
        "final_iota_shift": float(iota_shift),
        "history": history,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
