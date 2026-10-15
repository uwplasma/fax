from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fax.continuum import alfven_speed
from fax.equilibrium import analytic_circular_tokamak, analytic_cylinder, from_vmec_wout
from fax.reference_cases import landreman_paul_qa_lowres
from fax.units import MU0, ion_mass_density


def test_analytic_cylinder_profiles_and_units():
    geometry = analytic_cylinder(radial_points=8, iota=0.5, magnetic_field=2.0, number_density=1.0e20)

    assert geometry.name == "analytic_cylinder"
    assert geometry.field_periods == 1
    assert geometry.s.shape == (8,)
    np.testing.assert_allclose(np.asarray(geometry.iota), 0.5)
    np.testing.assert_allclose(np.asarray(geometry.mass_density), ion_mass_density(1.0e20))

    expected_speed = 2.0 / np.sqrt(MU0 * ion_mass_density(1.0e20))
    np.testing.assert_allclose(np.asarray(alfven_speed(geometry.magnetic_field, geometry.mass_density)), expected_speed)


def test_equilibrium_geometry_is_jit_compatible_pytree():
    geometry = analytic_circular_tokamak(radial_points=16)

    @jax.jit
    def mean_alfven_speed(geom):
        return jnp.mean(alfven_speed(geom.magnetic_field, geom.mass_density))

    value = mean_alfven_speed(geometry)

    assert np.isfinite(float(value))
    assert float(value) > 0.0


def test_density_scaling_changes_alfven_speed_by_sqrt():
    geometry = analytic_cylinder(radial_points=8)
    scaled = geometry.with_density_scale(4.0)

    base_speed = jnp.mean(alfven_speed(geometry.magnetic_field, geometry.mass_density))
    scaled_speed = jnp.mean(alfven_speed(scaled.magnetic_field, scaled.mass_density))

    np.testing.assert_allclose(float(scaled_speed / base_speed), 0.5, rtol=2.0e-6)


def test_rejects_inconsistent_profile_shapes():
    with pytest.raises(ValueError, match="mass_density"):
        analytic_cylinder(radial_points=8).with_mass_density(jnp.ones(7))


@pytest.mark.requires_vmex
def test_landreman_paul_wout_adapter_smoke():
    case = landreman_paul_qa_lowres()
    if not case.is_available():
        pytest.skip(f"missing local Landreman-Paul QA paths: {case.missing_paths()}")

    geometry = from_vmec_wout(case.wout, name=case.case_id)  # type: ignore[arg-type]

    assert geometry.name == case.case_id
    assert geometry.field_periods == 2
    assert geometry.s.ndim == 1
    assert geometry.s.size == 50
    assert geometry.iota.shape == geometry.s.shape
    assert geometry.magnetic_field.shape == geometry.s.shape
    assert float(jnp.min(geometry.magnetic_field)) > 0.0
    assert float(geometry.major_radius) > 0.0
    assert float(geometry.minor_radius) > 0.0
    assert dict(geometry.metadata)["mpol"] == 8
    assert dict(geometry.metadata)["ntor"] == 8
