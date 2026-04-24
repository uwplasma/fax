"""Mode-family helpers for stellarator continuum calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ModeFamily:
    """A finite set of coupled ``(m, n)`` harmonics.

    Stellarator symmetry couples toroidal harmonics separated by the number of
    field periods. This object keeps the harmonic array and the generating
    metadata together so figures, tests, and future operators use the same
    labels.
    """

    harmonics: tuple[tuple[int, int], ...]
    base_n: int
    field_periods: int
    sidebands: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.field_periods <= 0:
            raise ValueError("field_periods must be positive")
        if not self.harmonics:
            raise ValueError("harmonics must be non-empty")

    def as_array(self) -> jnp.ndarray:
        """Return harmonics as an array with columns ``(m, n)``."""
        return jnp.asarray(self.harmonics, dtype=float)

    def labels(self) -> tuple[str, ...]:
        """Return compact labels for plotting and reports."""
        return tuple(f"m={m}, n={n}" for m, n in self.harmonics)


def stellarator_mode_family(
    *,
    poloidal_modes: Iterable[int],
    base_n: int,
    field_periods: int,
    sidebands: Iterable[int] = (-1, 0, 1),
) -> ModeFamily:
    """Return harmonics with ``n = base_n + k * field_periods``.

    Parameters
    ----------
    poloidal_modes:
        Poloidal mode numbers to include.
    base_n:
        Base toroidal mode number.
    field_periods:
        Number of field periods ``Nfp``.
    sidebands:
        Integer sideband offsets ``k``.
    """
    m_values = tuple(int(m) for m in poloidal_modes)
    k_values = tuple(int(k) for k in sidebands)
    if not m_values:
        raise ValueError("poloidal_modes must be non-empty")
    if not k_values:
        raise ValueError("sidebands must be non-empty")
    if int(field_periods) <= 0:
        raise ValueError("field_periods must be positive")

    harmonics = tuple(
        (m, int(base_n) + k * int(field_periods))
        for m in m_values
        for k in k_values
    )
    return ModeFamily(
        harmonics=harmonics,
        base_n=int(base_n),
        field_periods=int(field_periods),
        sidebands=k_values,
    )


def unique_harmonics(harmonics: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Return harmonics sorted and de-duplicated."""
    array = np.asarray(tuple(harmonics), dtype=int)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("harmonics must have shape (count, 2)")
    unique = np.unique(array, axis=0)
    return tuple((int(m), int(n)) for m, n in unique)

