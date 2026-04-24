"""Differentiate a continuum diagnostic with respect to density, iota, and field scale."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp

from fax.continuum import continuum_branch_matrix
from fax.equilibrium import EquilibriumGeometry, analytic_circular_tokamak


def scaled_geometry(base: EquilibriumGeometry, *, density_scale: float, iota_shift: float, field_scale: float) -> EquilibriumGeometry:
    """Return a perturbed analytic geometry for sensitivity demonstrations."""
    return EquilibriumGeometry(
        name=base.name,
        s=base.s,
        iota=base.iota + iota_shift,
        magnetic_field=base.magnetic_field * field_scale,
        mass_density=base.mass_density * density_scale,
        major_radius=base.major_radius,
        minor_radius=base.minor_radius,
        field_periods=base.field_periods,
        metadata=base.metadata,
    )


def main() -> int:
    """Print JAX sensitivities of a mean branch frequency diagnostic."""
    base = analytic_circular_tokamak(radial_points=96, iota_axis=0.25, iota_edge=0.72)
    harmonic = jnp.asarray([[2.0, 1.0]])

    def mean_frequency(parameters: jax.Array) -> jax.Array:
        density_scale, iota_shift, field_scale = parameters
        geometry = scaled_geometry(
            base,
            density_scale=density_scale,
            iota_shift=iota_shift,
            field_scale=field_scale,
        )
        return jnp.mean(continuum_branch_matrix(geometry, harmonic))

    parameters = jnp.asarray([1.0, 0.0, 1.0])
    value = mean_frequency(parameters)
    gradient = jax.grad(mean_frequency)(parameters)
    payload = {
        "diagnostic": "mean local continuum frequency for harmonic m=2,n=1",
        "mean_frequency_hz": float(value),
        "gradient": {
            "density_scale": float(gradient[0]),
            "iota_shift": float(gradient[1]),
            "field_scale": float(gradient[2]),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
