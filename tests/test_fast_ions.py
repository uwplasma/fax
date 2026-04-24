from __future__ import annotations

import jax
import numpy as np

from fax.fast_ions import (
    FastIonPopulation,
    lorentzian_resonance_weight,
    reduced_fast_ion_drive,
    resonance_mismatch_hz,
    resonance_scan,
)


def _population() -> FastIonPopulation:
    return FastIonPopulation(
        density_fraction=0.03,
        pressure_gradient=-1.5,
        transit_frequency_hz=40_000.0,
        precession_frequency_hz=5_000.0,
        resonance_width_hz=2_000.0,
    )


def test_resonance_mismatch_is_zero_on_resonance():
    population = _population()
    mode_frequency = 2.0 * population.precession_frequency_hz + population.transit_frequency_hz

    mismatch = resonance_mismatch_hz(mode_frequency, toroidal_mode=2, transit_harmonic=1, population=population)

    np.testing.assert_allclose(float(mismatch), 0.0, atol=1.0e-12)
    np.testing.assert_allclose(float(lorentzian_resonance_weight(mismatch, population.resonance_width_hz)), 1.0)


def test_reduced_drive_peaks_on_resonance_and_is_differentiable():
    population = _population()
    resonant_frequency = 2.0 * population.precession_frequency_hz + population.transit_frequency_hz
    off_resonant_frequency = resonant_frequency + 10.0 * population.resonance_width_hz

    resonant = reduced_fast_ion_drive(resonant_frequency, toroidal_mode=2, transit_harmonic=1, population=population)
    off_resonant = reduced_fast_ion_drive(off_resonant_frequency, toroidal_mode=2, transit_harmonic=1, population=population)
    gradient = jax.grad(
        lambda frequency: reduced_fast_ion_drive(frequency, toroidal_mode=2, transit_harmonic=1, population=population)
    )(resonant_frequency)

    assert float(resonant) > float(off_resonant) > 0.0
    np.testing.assert_allclose(float(gradient), 0.0, atol=1.0e-12)


def test_resonance_scan_returns_one_value_per_harmonic():
    values = resonance_scan(45_000.0, toroidal_mode=1, transit_harmonics=(-1, 0, 1), population=_population())

    assert values.shape == (3,)
    assert np.all(np.asarray(values) >= 0.0)
