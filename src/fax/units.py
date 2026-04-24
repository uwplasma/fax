"""Physical constants and unit helpers used by ``fax``."""

from __future__ import annotations

import math


MU0 = 4.0e-7 * math.pi
"""Vacuum permeability in SI units, H/m."""

PROTON_MASS = 1.67262192369e-27
"""Proton mass in kg."""

ELEMENTARY_CHARGE = 1.602176634e-19
"""Elementary charge in coulombs, equal to one electron-volt per volt."""


def ion_mass_density(number_density: float, *, ion_mass: float = PROTON_MASS) -> float:
    """Return ion mass density in kg/m^3 from number density in m^-3."""
    return float(number_density) * float(ion_mass)


def ev_to_joule(energy_ev: float) -> float:
    """Return an energy in joules from electron-volts."""
    return float(energy_ev) * ELEMENTARY_CHARGE
