"""Gap and branch-separation diagnostics for continuum smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp


@dataclass(frozen=True)
class ContinuumGapSummary:
    """Summary of the smallest branch separation in a continuum matrix."""

    min_gap_hz: float
    surface: float
    lower_frequency_hz: float
    upper_frequency_hz: float
    radial_index: int
    sorted_branch_index: int


def sorted_branch_gaps(branches_hz: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return sorted branches and nearest-neighbor gaps.

    Parameters
    ----------
    branches_hz:
        Array with shape ``(radial_points, branch_count)``.
    """
    branches = jnp.asarray(branches_hz)
    if branches.ndim != 2:
        raise ValueError("branches_hz must be two-dimensional")
    if branches.shape[1] < 2:
        raise ValueError("at least two branches are required to define gaps")
    sorted_branches = jnp.sort(branches, axis=1)
    return sorted_branches, jnp.diff(sorted_branches, axis=1)


def minimum_gap_summary(branches_hz: Any, surfaces: Any) -> ContinuumGapSummary:
    """Return the smallest non-negative nearest-neighbor branch separation."""
    sorted_branches, gaps = sorted_branch_gaps(branches_hz)
    surfaces_array = jnp.asarray(surfaces)
    if surfaces_array.ndim != 1 or surfaces_array.shape[0] != sorted_branches.shape[0]:
        raise ValueError("surfaces must be one-dimensional with one value per radial point")

    flat_index = int(jnp.argmin(gaps))
    radial_count, gap_count = gaps.shape
    radial_index = flat_index // gap_count
    branch_index = flat_index % gap_count
    if radial_index >= radial_count:
        raise ValueError("internal gap index out of range")

    lower = sorted_branches[radial_index, branch_index]
    upper = sorted_branches[radial_index, branch_index + 1]
    return ContinuumGapSummary(
        min_gap_hz=float(gaps[radial_index, branch_index]),
        surface=float(surfaces_array[radial_index]),
        lower_frequency_hz=float(lower),
        upper_frequency_hz=float(upper),
        radial_index=int(radial_index),
        sorted_branch_index=int(branch_index),
    )


def branch_separation_objective(branches_hz: Any, *, floor_hz: float = 1.0) -> jnp.ndarray:
    """Smooth penalty that grows when branches get close.

    This is a smoke-test objective for inverse-design demonstrations. It is not
    a physical stability metric by itself.
    """
    _sorted, gaps = sorted_branch_gaps(branches_hz)
    return jnp.mean(1.0 / (gaps + float(floor_hz)))

