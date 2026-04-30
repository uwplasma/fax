from __future__ import annotations

import fax


def test_version_exists():
    assert fax.__version__


def test_recent_public_exports_exist():
    assert fax.GyroLandauClosure
    assert fax.GyroLandauCalibrationResult
    assert fax.gyro_landau_closure_drive
    assert fax.fit_gyro_landau_eigenfunction_closure
    assert fax.stellgap_ctok_gap_parity
    assert fax.run_stellgap_ctok_executable
    assert fax.Ae3dGlobalModeParity
    assert fax.ae3d_global_mode_parity
    assert fax.KineticGrowthRateDataset
    assert fax.kinetic_growth_rate_parity
