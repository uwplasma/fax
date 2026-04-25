from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from fax.fast_ions import (
    FastIonPopulation,
    eigenfunction_fast_ion_drive,
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


def test_eigenfunction_fast_ion_drive_tracks_mode_localization_and_gradients():
    s = jnp.linspace(0.0, 1.0, 80)
    center_mode = jnp.exp(-((s - 0.45) / 0.10) ** 2)
    edge_mode = jnp.exp(-((s - 0.82) / 0.08) ** 2)
    structures = jnp.stack([center_mode, edge_mode], axis=0)[:, :, None]

    def center_drive(density_fraction):
        population = FastIonPopulation(
            density_fraction=density_fraction,
            pressure_gradient=-jnp.exp(-((s - 0.45) / 0.12) ** 2),
            transit_frequency_hz=jnp.full_like(s, 40_000.0),
            precession_frequency_hz=jnp.full_like(s, 5_000.0),
            resonance_width_hz=jnp.full_like(s, 2_000.0),
        )
        return eigenfunction_fast_ion_drive(
            jnp.asarray([45_000.0, 45_000.0]),
            structures,
            s,
            toroidal_mode=1,
            transit_harmonics=(1,),
            population=population,
            drive_scale_hz=1_000.0,
            damping_rate_hz=1.0,
        ).net_growth_rate_hz[0]

    population = FastIonPopulation(
        density_fraction=0.03,
        pressure_gradient=-jnp.exp(-((s - 0.45) / 0.12) ** 2),
        transit_frequency_hz=jnp.full_like(s, 40_000.0),
        precession_frequency_hz=jnp.full_like(s, 5_000.0),
        resonance_width_hz=jnp.full_like(s, 2_000.0),
    )
    result = eigenfunction_fast_ion_drive(
        jnp.asarray([45_000.0, 45_000.0]),
        structures,
        s,
        toroidal_mode=1,
        transit_harmonics=(1,),
        population=population,
        drive_scale_hz=1_000.0,
        damping_rate_hz=1.0,
    )

    assert float(result.drive_indicator[0]) > 3.0 * float(result.drive_indicator[1])
    assert float(result.net_growth_rate_hz[0]) > float(result.net_growth_rate_hz[1])
    assert float(jax.grad(center_drive)(0.03)) > 0.0
