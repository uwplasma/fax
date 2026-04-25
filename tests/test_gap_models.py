from __future__ import annotations

import jax
import numpy as np

from fax.equilibrium import analytic_circular_tokamak
from fax.gap_models import (
    bae_accumulation_frequency_hz,
    basic_alfven_gap_map,
    coupled_gap_estimate,
    eae_gap_branch_map,
    eae_gap_estimate,
    hae_gap_estimate,
    tae_gap_branch_map,
    tae_gap_estimate,
)


def test_tae_gap_center_matches_crossing_condition():
    geometry = analytic_circular_tokamak(radial_points=256, iota_axis=0.45, iota_edge=0.85, major_radius=6.2)

    gap = tae_gap_estimate(geometry, m=1, n=1, inverse_aspect_ratio=0.1)

    np.testing.assert_allclose(float(gap.iota), 2.0 / 3.0, rtol=2.0e-2)
    np.testing.assert_allclose(float(gap.width_frequency_hz), 0.1 * float(gap.center_frequency_hz), rtol=1.0e-6)
    assert float(gap.lower_frequency_hz) < float(gap.center_frequency_hz) < float(gap.upper_frequency_hz)


def test_coupled_gap_estimate_pytree_roundtrip_preserves_frequency_window():
    geometry = analytic_circular_tokamak(radial_points=128, iota_axis=0.45, iota_edge=0.85, major_radius=6.2)
    gap = tae_gap_estimate(geometry, m=1, n=1, inverse_aspect_ratio=0.08)
    leaves, treedef = jax.tree_util.tree_flatten(gap)
    reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

    assert reconstructed.family == "TAE"
    np.testing.assert_allclose(
        np.asarray([reconstructed.lower_frequency_hz, reconstructed.center_frequency_hz, reconstructed.upper_frequency_hz]),
        np.asarray([gap.lower_frequency_hz, gap.center_frequency_hz, gap.upper_frequency_hz]),
        rtol=1.0e-12,
    )


def test_eae_and_hae_gap_estimates_have_expected_family_labels():
    geometry = analytic_circular_tokamak(radial_points=128, iota_axis=0.25, iota_edge=0.85)

    eae = eae_gap_estimate(geometry, m=1, n=1, ellipticity=0.04)
    hae = hae_gap_estimate(geometry, m=1, n=1, helical_coupling=0.03)

    assert eae.family == "EAE"
    assert hae.family == "HAE"
    assert float(eae.center_frequency_hz) > 0.0
    assert float(hae.center_frequency_hz) > 0.0


def test_coupled_gap_rejects_invalid_inputs():
    geometry = analytic_circular_tokamak(radial_points=32)

    try:
        coupled_gap_estimate(geometry, m=1, n=1, delta_m=0, coupling_strength=0.1, family="bad")
    except ValueError as exc:
        assert "delta_m" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_coupled_branch_maps_open_tae_and_eae_gaps():
    geometry = analytic_circular_tokamak(radial_points=256, iota_axis=1.0 / 1.05, iota_edge=1.0 / 3.2, major_radius=3.0)

    tae = tae_gap_branch_map(geometry, m=1, n=1, inverse_aspect_ratio=0.10)
    eae = eae_gap_branch_map(geometry, m=1, n=1, ellipticity=0.06)

    for gap in [tae, eae]:
        index = int(np.argmin(np.abs(np.asarray(geometry.s) - float(gap.estimate.surface))))
        coupled_width = float(gap.upper_frequency_hz[index] - gap.lower_frequency_hz[index])
        uncoupled_width = abs(float(gap.uncoupled_a_frequency_hz[index] - gap.uncoupled_b_frequency_hz[index]))
        assert coupled_width > uncoupled_width
        assert coupled_width > 0.0


def test_bae_accumulation_frequency_scaling():
    geometry = analytic_circular_tokamak(radial_points=64, iota_axis=0.8, iota_edge=0.4, major_radius=3.0)

    cold = bae_accumulation_frequency_hz(geometry, ion_temperature_ev=1_000.0, electron_temperature_ev=1_000.0)
    hot = bae_accumulation_frequency_hz(geometry, ion_temperature_ev=4_000.0, electron_temperature_ev=4_000.0)

    np.testing.assert_allclose(np.asarray(hot / cold), 2.0, rtol=1.0e-6)
    assert np.all(np.asarray(cold) > 0.0)


def test_bae_accumulation_frequency_rejects_unphysical_temperatures():
    geometry = analytic_circular_tokamak(radial_points=16)

    for kwargs, message in [
        ({"ion_temperature_ev": 0.0, "electron_temperature_ev": 1_000.0}, "ion_temperature_ev"),
        ({"ion_temperature_ev": 1_000.0, "electron_temperature_ev": -1.0}, "electron_temperature_ev"),
        ({"ion_temperature_ev": 1_000.0, "electron_temperature_ev": 1_000.0, "ion_mass": 0.0}, "ion_mass"),
    ]:
        try:
            bae_accumulation_frequency_hz(geometry, **kwargs)
        except ValueError as exc:
            assert message in str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"expected ValueError for {message}")


def test_basic_alfven_gap_map_contains_ordered_literature_gaps():
    demo = basic_alfven_gap_map(radial_points=192)

    tae_frequency = float(demo.tae.estimate.center_frequency_hz)
    eae_frequency = float(demo.eae.estimate.center_frequency_hz)
    bae_frequency = float(demo.bae_frequency_at_surface_hz)

    np.testing.assert_allclose(float(1.0 / demo.tae.estimate.iota), 1.5, rtol=2.0e-2)
    np.testing.assert_allclose(float(1.0 / demo.eae.estimate.iota), 2.0, rtol=2.0e-2)
    assert bae_frequency < tae_frequency < eae_frequency
