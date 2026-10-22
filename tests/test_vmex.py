"""End-to-end ``vmex -> booz_xform_jax -> FAX`` gates (skipped without ``vmex``)."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fax

pytest.importorskip("vmex", exc_type=ImportError)
INPUT = Path(__file__).parents[1] / "examples" / "data" / "input.LandremanPaul2021_QA_lowres"


def test_vmec_input_to_continuum():
    geo = fax.from_vmec_input(INPUT, ns=16, mboz=6, nboz=6, surfaces=(0.3, 0.5, 0.7))
    assert geo.nfp == 2 and geo.s.shape == (3,)
    f = fax.continuum(geo, fax.mode_family(1, 4), fax.mass_density(1e19), 16, 16)
    assert jnp.all(jnp.isfinite(f)) and f.shape == (3, 5)


@pytest.mark.xfail(
    strict=True,
    reason="upstream: vmex/booz_xform iota derivative wrt zs01 is off by 1.1e-4; the AE3D operator amplifies it to 2.3e-4 (plan P3)",
)
def test_boundary_gradient_matches_finite_differences():
    design = fax.prepare_design(INPUT, harmonics=[[1.0, 0.0], [2.0, 1.0]], density=3.4e-8, parameters=("rc01", "zs01"), ntheta=8, nzeta=8)
    objective = lambda p: jnp.sum(fax.design_frequencies(p, design, mode_count=2))  # noqa: E731
    p = jnp.zeros(2)
    grad = jax.grad(objective)(p)
    fd = np.array([(objective(p + e) - objective(p - e)) / 2e-4 for e in 1e-4 * jnp.eye(2)])
    np.testing.assert_allclose(grad, fd, rtol=1e-4)
