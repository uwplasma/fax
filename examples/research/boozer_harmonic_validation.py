"""Validate Boozer harmonic reconstruction, projection, and phase convention."""

from __future__ import annotations

import json

import jax.numpy as jnp

from fax.boozer import (
    BoozerSpectrum,
    boozer_harmonic_orthogonality_error,
    boozer_projection_error,
    normalized_nonaxisymmetric_boozer_energy,
    straight_field_line_phase_error,
)


def main() -> int:
    """Print a compact Boozer harmonic validation packet."""
    spectrum = BoozerSpectrum(
        source="synthetic_research_example",
        s=jnp.asarray([0.25, 0.75]),
        xm=jnp.asarray([0.0, 1.0, 2.0, 1.0]),
        xn=jnp.asarray([0.0, 0.0, 1.0, -1.0]),
        bmnc=jnp.asarray(
            [
                [2.8, 0.10, -0.035, 0.022],
                [2.7, 0.08, -0.025, 0.018],
            ]
        ),
        bmns=jnp.asarray(
            [
                [0.0, 0.025, 0.012, -0.008],
                [0.0, 0.020, 0.010, -0.006],
            ]
        ),
        field_periods=2,
    )
    harmonics = jnp.column_stack([spectrum.xm, spectrum.xn])
    payload = {
        "diagnostic": "Boozer harmonic reconstruction and straight-field-line validation",
        "projection_error_surface_0": float(boozer_projection_error(spectrum, surface_index=0)),
        "projection_error_surface_1": float(boozer_projection_error(spectrum, surface_index=1)),
        "orthogonality_error": float(boozer_harmonic_orthogonality_error(spectrum.xm, spectrum.xn)),
        "straight_field_line_phase_error": float(straight_field_line_phase_error(harmonics, iota=0.73)),
        "normalized_nonaxisymmetric_energy": float(normalized_nonaxisymmetric_boozer_energy(spectrum)),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
