from __future__ import annotations

from pathlib import Path

import pytest

from fax.cross_code import (
    ae3d_eigenvalue_residual_summary,
    ae3d_matrix_eigenmode_parity,
    ae3d_matrix_symmetry_summary,
    ae3d_omega2_eigenvalue_error,
    read_ae3d_eigenmodes,
    read_ae3d_omega2,
    read_ae3d_eigenvalues,
    read_ae3d_sparse_matrix,
    read_stellgap_alfven_spec,
    read_stellgap_alfven_post,
    stellgap_postprocess_error,
    stellgap_spec_postprocessed_continuum,
    stellgap_ctok_local_continuum_error,
)
from fax.units import PROTON_MASS


def test_cross_code_text_readers_parse_minimal_tables(tmp_path: Path):
    alfven_post = tmp_path / "alfven_post"
    alfven_post.write_text("2\n0.1 100.0 1 2\n0.2 200.0 2 2\n")
    alfven_spec = tmp_path / "alfven_spec"
    alfven_spec.write_text("0.1 10000.0 0.0 1.0 1 2\n0.2 40000.0 0.0 1.0 2 2\n")
    egn_values = tmp_path / "egn_values.dat"
    egn_values.write_text("100.0 1e-6 -1e-6\n-4.0 2e-6 -2e-6\n")
    omega2 = tmp_path / "omega2.dat"
    omega2.write_text("1 10000.0 0.0 1.0 10000.0\n2 -4.0 0.0 1.0 -4.0\n")
    matrix = tmp_path / "a_matrix.dat"
    matrix.write_text("1 1 2.0\n1 2 0.5\n2 1 0.5\n2 2 3.0\n")

    continuum = read_stellgap_alfven_post(alfven_post)
    spectrum = read_stellgap_alfven_spec(alfven_spec)
    processed = stellgap_spec_postprocessed_continuum(alfven_spec)
    eigenvalues = read_ae3d_eigenvalues(egn_values)
    omega = read_ae3d_omega2(omega2)

    assert continuum.s.shape == (2,)
    assert continuum.frequency_khz[1] == 200.0
    assert spectrum.alpha_real.shape == (2,)
    assert processed.frequency_khz[0] == 100.0
    assert eigenvalues.frequency_khz.shape == (2,)
    assert eigenvalues.residual_real[0] == 1.0e-6
    assert omega.eigenvalue[1] == -4.0
    summary = ae3d_eigenvalue_residual_summary(egn_values)
    assert summary.count == 2
    assert summary.max_residual > 0.0
    assert ae3d_omega2_eigenvalue_error(omega2_path=omega2, egn_values_path=egn_values).max_relative_error == 0.0
    assert ae3d_matrix_symmetry_summary(matrix).relative_asymmetry == 0.0


@pytest.mark.requires_booz_xform_jax
def test_stellgap_ctok_local_continuum_comparison_passes_or_skips():
    boozmn = Path("/Users/rogerio/local/Stellgap_test_inputs/boozmn_ctok.nc")
    alfven_post = Path("/Users/rogerio/local/Stellgap_test_problem/alfven_post")
    if not boozmn.exists() or not alfven_post.exists():
        pytest.skip("missing local ORNL STELLGAP CTOK test artifacts")

    comparison = stellgap_ctok_local_continuum_error(
        boozmn_path=boozmn,
        alfven_post_path=alfven_post,
        mass_density=2.0 * PROTON_MASS * 1.0e19,
        theta_points=32,
        zeta_points=32,
    )

    assert comparison.count >= 90
    assert comparison.p95_relative_error < 4.0e-2


def test_stellgap_and_ae3d_public_artifact_consistency_passes_or_skips():
    stellgap_spec = Path("/Users/rogerio/local/Stellgap_test_problem/alfven_spec")
    stellgap_post = Path("/Users/rogerio/local/Stellgap_test_problem/alfven_post")
    omega2 = Path("/Users/rogerio/local/AE3D_test_problem/omega2.dat")
    egn_values = Path("/Users/rogerio/local/AE3D_test_problem/egn_values.dat")
    a_matrix = Path("/Users/rogerio/local/AE3D_test_problem/a_matrix.dat")
    b_matrix = Path("/Users/rogerio/local/AE3D_test_problem/b_matrix.dat")
    if not all(path.exists() for path in (stellgap_spec, stellgap_post, omega2, egn_values, a_matrix, b_matrix)):
        pytest.skip("missing local ORNL STELLGAP/AE3D public answer artifacts")

    stellgap = stellgap_postprocess_error(alfven_spec_path=stellgap_spec, alfven_post_path=stellgap_post)
    ae3d = ae3d_omega2_eigenvalue_error(omega2_path=omega2, egn_values_path=egn_values)
    a_summary = ae3d_matrix_symmetry_summary(a_matrix)
    b_summary = ae3d_matrix_symmetry_summary(b_matrix)

    assert stellgap.count >= 3800
    assert stellgap.max_relative_error < 1.0e-5
    assert ae3d.count == 1287
    assert ae3d.max_relative_error < 1.0e-5
    assert a_summary.dimension == b_summary.dimension == 1287
    assert a_summary.relative_asymmetry < 1.0e-5
    assert b_summary.relative_asymmetry < 1.0e-12


def test_ae3d_public_matrix_eigenmode_parity_passes_or_skips():
    root = Path("/Users/rogerio/local/AE3D_test_problem")
    a_matrix = root / "a_matrix.dat"
    b_matrix = root / "b_matrix.dat"
    egn_mode = root / "egn_mode_asci.dat"
    if not all(path.exists() for path in (a_matrix, b_matrix, egn_mode)):
        pytest.skip("missing local ORNL AE3D public matrix/eigenmode artifacts")

    a = read_ae3d_sparse_matrix(a_matrix)
    modes = read_ae3d_eigenmodes(egn_mode, max_modes=3)
    parity = ae3d_matrix_eigenmode_parity(
        a_matrix_path=a_matrix,
        b_matrix_path=b_matrix,
        egn_mode_path=egn_mode,
        max_modes=8,
    )

    assert a.shape == (1287, 1287)
    assert modes.modes.shape == (3, 99, 13)
    assert parity.count == 8
    assert parity.matrix_dimension == 1287
    assert parity.max_residual < 5.0e-6
    assert parity.max_rayleigh_relative_error < 2.0e-6
    assert parity.max_frequency_relative_error < 2.0e-6
