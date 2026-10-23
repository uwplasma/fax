"""End-to-end ``vmex -> booz_xform_jax -> FAX`` gates (skipped without ``vmex``)."""

from dataclasses import replace
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


def test_boundary_gradient_and_radial_refinement():
    """Lowest n = 0 global frequency of LP-QA: AD vs re-solve FD in ``zs01`` while only the FE grid is refined.

    ``ns``, ``ftol`` and ``mboz`` stay fixed; the FE nodes are uniform on ``[0, 1]``.
    At every level the FAX chain (interpolation, metric, Hellmann-Feynman) matches
    FD along the re-solved Boozer geometry to ~1e-8, and the full adjoint
    gradient matches the re-solve FD to 3e-5: the remainder is vmex's released
    m = 1 gauge, which the adjoint holds fixed and each re-solve lets drift
    (closure to 3e-9 on iota, plan.md P3). The lowest eigenvalue is the
    continuum minimum, so both it and its gradient converge at first order.
    """
    h = 1e-4
    harmonics = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    design = fax.prepare_design(INPUT, harmonics=harmonics, density=3.4e-8, parameters=("zs01",), ntheta=8, nzeta=8)
    g0, gp, gm = (fax.design_geometry(jnp.array([x]), design) for x in (0.0, h, -h))
    dg = jax.tree_util.tree_map(lambda a, b: (a - b) / (2 * h), gp, gm)
    values, grads = [], []
    for nodes in (16, 32, 64):
        d = replace(design, nodes=nodes)
        f = lambda g: fax.geometry_frequencies(g, d, 1)[0]  # noqa: B023, E731
        value, ad = jax.jvp(f, (g0,), (dg,))
        np.testing.assert_allclose(ad, (f(gp) - f(gm)) / (2 * h), rtol=1e-4)
        values, grads = values + [value], grads + [ad]
    for x in (values, grads):
        ratio = (x[0] - x[1]) / (x[1] - x[2])
        assert 1.7 < ratio < 2.3  # first order
    d = replace(design, nodes=32)
    np.testing.assert_allclose(jax.grad(lambda p: fax.design_frequencies(p, d, 1)[0])(jnp.zeros(1))[0], grads[1], rtol=1e-4)
    targeted = replace(d, options={**d.options, "target_hz": 0.98 * float(values[1]), "count": 2})
    v, t = jax.jvp(lambda g: fax.geometry_frequencies(g, targeted, 1)[0], (g0,), (dg,))
    np.testing.assert_allclose([v, t], [values[1], grads[1]], rtol=1e-8)
