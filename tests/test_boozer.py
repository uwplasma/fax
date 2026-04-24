from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fax.boozer import dominant_boozer_modes, normalized_nonaxisymmetric_boozer_energy, read_boozmn, run_booz_xform_jax_from_wout


BOOZ_XFORM_TEST_FILES = Path("/Users/rogerio/local/booz_xform_jax/tests/test_files")


@pytest.mark.requires_booz_xform_jax
def test_read_boozmn_spectrum_and_dominant_modes():
    path = BOOZ_XFORM_TEST_FILES / "boozmn_circular_tokamak.nc"
    if not path.exists():
        pytest.skip(f"missing local booz_xform_jax fixture: {path}")

    spectrum = read_boozmn(path)

    assert spectrum.bmnc.ndim == 2
    assert spectrum.bmnc.shape[0] == spectrum.s.shape[0]
    assert 0.0 <= float(normalized_nonaxisymmetric_boozer_energy(spectrum)) <= 1.0
    modes = dominant_boozer_modes(spectrum, count=3)
    assert len(modes) == 3
    assert modes[0][2] > 0.0


@pytest.mark.requires_booz_xform_jax
def test_run_booz_xform_jax_from_wout_smoke():
    path = BOOZ_XFORM_TEST_FILES / "wout_circular_tokamak.nc"
    if not path.exists():
        pytest.skip(f"missing local booz_xform_jax fixture: {path}")

    spectrum = run_booz_xform_jax_from_wout(path, mboz=4, nboz=4, surfaces=(0.25, 0.5), jit=False)

    assert spectrum.bmnc.shape[0] == 2
    assert spectrum.field_periods >= 1
    assert np.all(np.isfinite(np.asarray(spectrum.bmnc)))
