from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fax.continuum import continuum_branch_matrix
from fax.equilibrium import analytic_circular_tokamak
from fax.gaps import branch_separation_objective, minimum_gap_summary, sorted_branch_gaps
from fax.modes import stellarator_mode_family


def test_sorted_branch_gaps_known_matrix():
    branches = jnp.asarray([[3.0, 1.0, 2.0], [10.0, 2.0, 6.0]])

    sorted_branches, gaps = sorted_branch_gaps(branches)

    np.testing.assert_allclose(np.asarray(sorted_branches), np.asarray([[1.0, 2.0, 3.0], [2.0, 6.0, 10.0]]))
    np.testing.assert_allclose(np.asarray(gaps), np.asarray([[1.0, 1.0], [4.0, 4.0]]))


def test_minimum_gap_summary_reports_surface_and_bounds():
    branches = jnp.asarray([[3.0, 1.0, 2.0], [10.0, 2.0, 6.0], [4.0, 4.25, 7.0]])
    surfaces = jnp.asarray([0.0, 0.5, 1.0])

    summary = minimum_gap_summary(branches, surfaces)

    assert summary.min_gap_hz == pytest.approx(0.25)
    assert summary.surface == pytest.approx(1.0)
    assert summary.lower_frequency_hz == pytest.approx(4.0)
    assert summary.upper_frequency_hz == pytest.approx(4.25)
    assert summary.radial_index == 2


def test_branch_separation_objective_is_differentiable():
    def objective(scale):
        branches = jnp.asarray([[scale, 2.0 * scale, 4.0 * scale], [2.0 * scale, 3.0 * scale, 8.0 * scale]])
        return branch_separation_objective(branches, floor_hz=0.1)

    value = float(objective(2.0))
    grad = float(jax.grad(objective)(2.0))

    assert value > 0.0
    assert grad < 0.0


def test_gap_summary_for_generated_mode_family_smoke():
    geometry = analytic_circular_tokamak(radial_points=64, iota_axis=0.25, iota_edge=0.85)
    family = stellarator_mode_family(poloidal_modes=range(1, 4), base_n=1, field_periods=2, sidebands=(-1, 0, 1))
    branches = continuum_branch_matrix(geometry, family.as_array())

    summary = minimum_gap_summary(branches, geometry.s)

    assert summary.min_gap_hz >= 0.0
    assert 0.0 <= summary.surface <= 1.0


def test_gap_helpers_reject_invalid_shapes():
    with pytest.raises(ValueError, match="two-dimensional"):
        sorted_branch_gaps(jnp.ones(3))
    with pytest.raises(ValueError, match="at least two"):
        sorted_branch_gaps(jnp.ones((3, 1)))
    with pytest.raises(ValueError, match="surfaces"):
        minimum_gap_summary(jnp.ones((3, 2)), jnp.ones(2))

