from __future__ import annotations

import fax


def test_version_exists():
    assert fax.__version__


def test_recent_public_exports_exist():
    assert fax.GyroLandauClosure
    assert fax.gyro_landau_closure_drive
    assert fax.stellgap_ctok_gap_parity
    assert fax.run_stellgap_ctok_executable
