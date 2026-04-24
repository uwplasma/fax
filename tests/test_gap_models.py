from __future__ import annotations

import numpy as np

from fax.equilibrium import analytic_circular_tokamak
from fax.gap_models import coupled_gap_estimate, eae_gap_estimate, hae_gap_estimate, tae_gap_estimate


def test_tae_gap_center_matches_crossing_condition():
    geometry = analytic_circular_tokamak(radial_points=256, iota_axis=0.45, iota_edge=0.85, major_radius=6.2)

    gap = tae_gap_estimate(geometry, m=1, n=1, inverse_aspect_ratio=0.1)

    np.testing.assert_allclose(float(gap.iota), 2.0 / 3.0, rtol=2.0e-2)
    np.testing.assert_allclose(float(gap.width_frequency_hz), 0.1 * float(gap.center_frequency_hz), rtol=1.0e-6)
    assert float(gap.lower_frequency_hz) < float(gap.center_frequency_hz) < float(gap.upper_frequency_hz)


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
