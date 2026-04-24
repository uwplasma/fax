"""Physical constants and unit helpers used by ``fax``."""

from __future__ import annotations

import math


MU0 = 4.0e-7 * math.pi
"""Vacuum permeability in SI units, H/m."""

PROTON_MASS = 1.67262192369e-27
"""Proton mass in kg."""


def ion_mass_density(number_density: float, *, ion_mass: float = PROTON_MASS) -> float:
    """Return ion mass density in kg/m^3 from number density in m^-3."""
    return float(number_density) * float(ion_mass)

