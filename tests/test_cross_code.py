from __future__ import annotations

from pathlib import Path

import pytest

from fax.cross_code import (
    ae3d_eigenvalue_residual_summary,
    read_ae3d_eigenvalues,
    read_stellgap_alfven_post,
    stellgap_ctok_local_continuum_error,
)
from fax.units import PROTON_MASS


def test_cross_code_text_readers_parse_minimal_tables(tmp_path: Path):
    alfven_post = tmp_path / "alfven_post"
    alfven_post.write_text("2\n0.1 100.0 1 2\n0.2 200.0 2 2\n")
    egn_values = tmp_path / "egn_values.dat"
    egn_values.write_text("100.0 1e-6 -1e-6\n200.0 2e-6 -2e-6\n")

    continuum = read_stellgap_alfven_post(alfven_post)
    eigenvalues = read_ae3d_eigenvalues(egn_values)

    assert continuum.s.shape == (2,)
    assert continuum.frequency_khz[1] == 200.0
    assert eigenvalues.frequency_khz.shape == (2,)
    assert eigenvalues.residual_real[0] == 1.0e-6
    summary = ae3d_eigenvalue_residual_summary(egn_values)
    assert summary.count == 2
    assert summary.max_residual > 0.0


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
