from __future__ import annotations

import numpy as np
import pytest

from fax.modes import stellarator_mode_family, unique_harmonics


def test_stellarator_mode_family_uses_field_period_sidebands():
    family = stellarator_mode_family(poloidal_modes=[1, 2], base_n=1, field_periods=2, sidebands=[-1, 0, 1])

    assert family.harmonics == ((1, -1), (1, 1), (1, 3), (2, -1), (2, 1), (2, 3))
    np.testing.assert_allclose(np.asarray(family.as_array()), np.asarray(family.harmonics, dtype=float))
    assert family.labels()[0] == "m=1, n=-1"


def test_unique_harmonics_sorts_and_deduplicates():
    assert unique_harmonics([(2, 1), (1, 0), (2, 1)]) == ((1, 0), (2, 1))


def test_mode_family_rejects_empty_inputs():
    with pytest.raises(ValueError, match="poloidal_modes"):
        stellarator_mode_family(poloidal_modes=[], base_n=0, field_periods=2)
    with pytest.raises(ValueError, match="sidebands"):
        stellarator_mode_family(poloidal_modes=[1], base_n=0, field_periods=2, sidebands=[])
    with pytest.raises(ValueError, match="field_periods"):
        stellarator_mode_family(poloidal_modes=[1], base_n=0, field_periods=0)

