from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fax.boozer import (
    BoozerSpectrum,
    boozer_angle_grid,
    boozer_harmonic_orthogonality_error,
    boozer_projection_error,
    dominant_boozer_modes,
    normalized_nonaxisymmetric_boozer_energy,
    project_boozer_field_to_modes,
    read_boozmn,
    reconstruct_boozer_field,
    run_booz_xform_jax_from_wout,
    straight_field_line_phase_error,
)


BOOZ_XFORM_TEST_FILES = Path("/Users/rogerio/local/booz_xform_jax/tests/test_files")


def _synthetic_spectrum() -> BoozerSpectrum:
    return BoozerSpectrum(
        source="synthetic",
        s=np.asarray([0.25, 0.75]),
        xm=np.asarray([0, 1, 2, 1], dtype=float),
        xn=np.asarray([0, 0, 1, -1], dtype=float),
        bmnc=np.asarray(
            [
                [2.8, 0.10, -0.035, 0.022],
                [2.7, 0.08, -0.025, 0.018],
            ]
        ),
        bmns=np.asarray(
            [
                [0.0, 0.025, 0.012, -0.008],
                [0.0, 0.020, 0.010, -0.006],
            ]
        ),
        field_periods=2,
    )


def test_synthetic_boozer_field_projects_back_to_coefficients():
    spectrum = _synthetic_spectrum()
    theta, zeta = boozer_angle_grid(theta_points=64, zeta_points=64)
    field = reconstruct_boozer_field(spectrum, surface_index=0, theta=theta, zeta=zeta)
    bmnc, bmns = project_boozer_field_to_modes(field, theta, zeta, spectrum.xm, spectrum.xn)

    np.testing.assert_allclose(np.asarray(bmnc), np.asarray(spectrum.bmnc[0]), rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_allclose(np.asarray(bmns), np.asarray(spectrum.bmns[0]), rtol=2.0e-6, atol=2.0e-6)
    assert float(boozer_projection_error(spectrum, surface_index=1, theta_points=64, zeta_points=64)) < 2.0e-6


def test_boozer_harmonic_orthogonality_and_straight_field_line_phase():
    spectrum = _synthetic_spectrum()
    harmonics = np.column_stack([np.asarray(spectrum.xm), np.asarray(spectrum.xn)])

    assert float(boozer_harmonic_orthogonality_error(spectrum.xm, spectrum.xn, theta_points=64, zeta_points=64)) < 2.0e-6
    assert float(straight_field_line_phase_error(harmonics, iota=0.73, zeta_points=128)) < 1.0e-4


def test_normalized_nonaxisymmetric_boozer_energy_is_jittable():
    import jax

    spectrum = _synthetic_spectrum()
    energy = jax.jit(normalized_nonaxisymmetric_boozer_energy)(spectrum)

    assert 0.0 < float(energy) < 1.0


@pytest.mark.requires_booz_xform_jax
def test_read_boozmn_spectrum_and_dominant_modes():
    path = BOOZ_XFORM_TEST_FILES / "boozmn_circular_tokamak.nc"
    if not path.exists():
        pytest.skip(f"missing local booz_xform_jax fixture: {path}")

    spectrum = read_boozmn(path)

    assert spectrum.bmnc.ndim == 2
    assert spectrum.bmnc.shape[0] == spectrum.s.shape[0]
    assert 0.0 <= float(normalized_nonaxisymmetric_boozer_energy(spectrum)) <= 1.0
    modes = dominant_boozer_modes(spectrum, count=3)
    assert len(modes) == 3
    assert modes[0][2] > 0.0


@pytest.mark.requires_booz_xform_jax
def test_run_booz_xform_jax_from_wout_smoke():
    path = BOOZ_XFORM_TEST_FILES / "wout_circular_tokamak.nc"
    if not path.exists():
        pytest.skip(f"missing local booz_xform_jax fixture: {path}")

    spectrum = run_booz_xform_jax_from_wout(path, mboz=4, nboz=4, surfaces=(0.25, 0.5), jit=False)

    assert spectrum.bmnc.shape[0] == 2
    assert spectrum.field_periods >= 1
    assert np.all(np.isfinite(np.asarray(spectrum.bmnc)))
