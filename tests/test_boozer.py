from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import jax

from fax.boozer import (
    BoozerGeometry,
    BoozerSpectrum,
    boozer_3d_continuum_frequencies_hz,
    boozer_3d_continuum_matrices,
    boozer_angle_grid,
    boozer_continuum_frequencies_hz,
    boozer_continuum_matrices,
    boozer_embedding_jacobian_shape_error,
    boozer_geometry_field_and_jacobian,
    boozer_geometry_jacobian_identity_error,
    boozer_harmonic_orthogonality_error,
    boozer_jacobian_identity_error,
    boozer_jacobian_weight,
    boozer_metric_continuum_frequencies_hz,
    boozer_metric_continuum_matrices,
    boozer_parallel_wavenumber,
    boozer_projection_error,
    dominant_boozer_modes,
    normalized_nonaxisymmetric_boozer_energy,
    project_boozer_field_to_modes,
    read_boozmn,
    read_boozmn_geometry,
    reconstruct_boozer_field,
    run_booz_xform_jax_geometry_from_wout,
    run_booz_xform_jax_from_wout,
    straight_field_line_phase_error,
)
from fax.global_modes import radial_block_problem_from_boozer_geometry, solve_radial_block_eigenmodes
from fax.units import MU0


BOOZ_XFORM_TEST_FILES = Path("/Users/rogerio/local/booz_xform_jax/tests/test_files")


def _synthetic_spectrum() -> BoozerSpectrum:
    return BoozerSpectrum(
        source="synthetic",
        s=np.asarray([0.25, 0.75]),
        xm=np.asarray([0, 1, 2, 1], dtype=float),
        xn=np.asarray([0, 0, 1, -1], dtype=float),
        bmnc=np.asarray(
            [
                [2.8, 0.10, -0.035, 0.022],
                [2.7, 0.08, -0.025, 0.018],
            ]
        ),
        bmns=np.asarray(
            [
                [0.0, 0.025, 0.012, -0.008],
                [0.0, 0.020, 0.010, -0.006],
            ]
        ),
        field_periods=2,
    )


def _constant_boozer_geometry() -> BoozerGeometry:
    magnetic_field = 2.7
    major_radius = 5.5
    iota = 0.61
    numerator = magnetic_field * major_radius
    return BoozerGeometry(
        source="constant_boozer_geometry",
        s=np.asarray([0.2, 0.8]),
        xm=np.asarray([0.0]),
        xn=np.asarray([0.0]),
        bmnc=np.asarray([[magnetic_field], [magnetic_field]]),
        bmns=None,
        rmnc=None,
        rmns=None,
        zmns=None,
        zmnc=None,
        pmns=None,
        pmnc=None,
        gmnc=np.asarray([[numerator / magnetic_field**2], [numerator / magnetic_field**2]]),
        gmns=None,
        iota=np.asarray([iota, iota]),
        boozer_i=np.asarray([0.0, 0.0]),
        boozer_g=np.asarray([numerator, numerator]),
        field_periods=1,
    )


def test_synthetic_boozer_field_projects_back_to_coefficients():
    spectrum = _synthetic_spectrum()
    theta, zeta = boozer_angle_grid(theta_points=64, zeta_points=64)
    field = reconstruct_boozer_field(spectrum, surface_index=0, theta=theta, zeta=zeta)
    bmnc, bmns = project_boozer_field_to_modes(field, theta, zeta, spectrum.xm, spectrum.xn)

    np.testing.assert_allclose(np.asarray(bmnc), np.asarray(spectrum.bmnc[0]), rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_allclose(np.asarray(bmns), np.asarray(spectrum.bmns[0]), rtol=2.0e-6, atol=2.0e-6)
    assert float(boozer_projection_error(spectrum, surface_index=1, theta_points=64, zeta_points=64)) < 2.0e-6


def test_boozer_harmonic_orthogonality_and_straight_field_line_phase():
    spectrum = _synthetic_spectrum()
    harmonics = np.column_stack([np.asarray(spectrum.xm), np.asarray(spectrum.xn)])

    assert float(boozer_harmonic_orthogonality_error(spectrum.xm, spectrum.xn, theta_points=64, zeta_points=64)) < 2.0e-6
    assert float(straight_field_line_phase_error(harmonics, iota=0.73, zeta_points=128)) < 1.0e-4


def test_boozer_geometry_pytree_roundtrip_preserves_field_and_jacobian_identity():
    geometry = _constant_boozer_geometry()
    leaves, treedef = jax.tree_util.tree_flatten(geometry)
    reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
    spectrum = reconstructed.as_spectrum()

    theta, zeta, field, jacobian = boozer_geometry_field_and_jacobian(reconstructed, theta_points=16, zeta_points=16)
    reconstructed_field = reconstruct_boozer_field(spectrum, surface_index=0, theta=theta, zeta=zeta)

    np.testing.assert_allclose(np.asarray(reconstructed_field), np.asarray(field[0]), rtol=1.0e-12)
    np.testing.assert_allclose(
        np.asarray(jacobian[0] * field[0] ** 2),
        np.asarray(reconstructed.boozer_g[0] + reconstructed.iota[0] * reconstructed.boozer_i[0]),
        rtol=1.0e-12,
    )


def test_normalized_nonaxisymmetric_boozer_energy_is_jittable():
    import jax

    spectrum = _synthetic_spectrum()
    energy = jax.jit(normalized_nonaxisymmetric_boozer_energy)(spectrum)

    assert 0.0 < float(energy) < 1.0


def test_constant_boozer_continuum_matches_profile_limit():
    magnetic_field = 2.7
    mass_density = 3.4e-8
    major_radius = 5.5
    iota = 0.61
    spectrum = BoozerSpectrum(
        source="constant_boozer",
        s=np.asarray([0.2, 0.8]),
        xm=np.asarray([0.0]),
        xn=np.asarray([0.0]),
        bmnc=np.asarray([[magnetic_field], [magnetic_field]]),
        bmns=None,
        field_periods=1,
    )
    harmonics = np.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])

    matrices = boozer_continuum_matrices(
        spectrum,
        harmonics,
        iota=iota,
        major_radius=major_radius,
        mass_density=mass_density,
    )
    frequencies = boozer_continuum_frequencies_hz(
        spectrum,
        harmonics,
        iota=iota,
        major_radius=major_radius,
        mass_density=mass_density,
    )
    k_parallel = (harmonics[:, 1] - harmonics[:, 0] * iota) / major_radius
    expected_frequency = magnetic_field / np.sqrt(MU0 * mass_density) * np.abs(k_parallel) / (2.0 * np.pi)

    np.testing.assert_allclose(np.asarray(frequencies[0]), np.sort(expected_frequency), rtol=1.0e-5)
    np.testing.assert_allclose(np.asarray(matrices.mass[0]), np.eye(3) * mass_density, rtol=3.0e-6, atol=1.0e-12)
    stiffness = np.asarray(matrices.stiffness[0])
    off_diagonal = stiffness - np.diag(np.diag(stiffness))
    assert np.max(np.abs(off_diagonal)) / np.max(np.abs(np.diag(stiffness))) < 1.0e-8


def test_boozer_continuum_sideband_coupling_is_hermitian_and_jittable():
    spectrum = BoozerSpectrum(
        source="coupled_boozer",
        s=np.asarray([0.5]),
        xm=np.asarray([0.0, 1.0]),
        xn=np.asarray([0.0, 1.0]),
        bmnc=np.asarray([[2.8, 0.08]]),
        bmns=None,
        field_periods=1,
    )
    harmonics = np.asarray([[1.0, 0.0], [2.0, 1.0]])

    matrices = boozer_continuum_matrices(
        spectrum,
        harmonics,
        iota=0.43,
        major_radius=5.2,
        mass_density=4.1e-8,
        theta_points=96,
        zeta_points=96,
    )
    stiffness = np.asarray(matrices.stiffness[0])

    np.testing.assert_allclose(stiffness, np.conjugate(stiffness.T), rtol=1.0e-6, atol=1.0e-6)
    assert abs(stiffness[0, 1]) / max(abs(stiffness[0, 0]), abs(stiffness[1, 1])) > 1.0e-3

    import jax

    jitted = jax.jit(
        lambda spec: boozer_continuum_matrices(
            spec,
            harmonics,
            iota=0.43,
            major_radius=5.2,
            mass_density=4.1e-8,
            theta_points=32,
            zeta_points=32,
        ).stiffness
    )(spectrum)
    assert jitted.shape == (1, 2, 2)


def test_boozer_sine_harmonics_enter_reduced_and_metric_continuum_matrices():
    spectrum = _synthetic_spectrum()
    no_sine = BoozerSpectrum(
        source="synthetic_no_sine",
        s=spectrum.s,
        xm=spectrum.xm,
        xn=spectrum.xn,
        bmnc=spectrum.bmnc,
        bmns=None,
        field_periods=spectrum.field_periods,
    )
    harmonics = np.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])

    reduced = boozer_continuum_matrices(
        spectrum,
        harmonics,
        iota=np.asarray([0.42, 0.56]),
        major_radius=5.1,
        mass_density=np.asarray([3.0e-8, 4.0e-8]),
        theta_points=64,
        zeta_points=64,
    )
    reduced_without_sine = boozer_continuum_matrices(
        no_sine,
        harmonics,
        iota=np.asarray([0.42, 0.56]),
        major_radius=5.1,
        mass_density=np.asarray([3.0e-8, 4.0e-8]),
        theta_points=64,
        zeta_points=64,
    )
    metric_weighted = boozer_metric_continuum_matrices(
        spectrum,
        harmonics,
        iota=np.asarray([0.42, 0.56]),
        major_radius=5.1,
        mass_density=np.asarray([3.0e-8, 4.0e-8]),
        jacobian_numerator=np.asarray([1.2, 1.4]),
        theta_points=64,
        zeta_points=64,
    )

    assert reduced.stiffness.shape == (2, 3, 3)
    assert metric_weighted.mass.shape == (2, 3, 3)
    assert float(np.linalg.norm(np.asarray(reduced.stiffness - reduced_without_sine.stiffness))) > 0.0


def test_boozer_parallel_wavenumber_shape():
    harmonics = np.asarray([[1.0, 0.0], [2.0, 1.0]])
    values = boozer_parallel_wavenumber(harmonics, iota=np.asarray([0.4, 0.5]), major_radius=5.0)

    assert values.shape == (2, 2)


def test_boozer_jacobian_weight_satisfies_flux_function_identity():
    spectrum = _synthetic_spectrum()
    numerator = np.asarray([1.3, 1.7])
    jacobian = boozer_jacobian_weight(spectrum, jacobian_numerator=numerator, theta_points=64, zeta_points=64)

    assert jacobian.shape == (2, 64, 64)
    assert np.min(np.asarray(jacobian)) > 0.0
    assert float(boozer_jacobian_identity_error(spectrum, jacobian_numerator=numerator)) < 2.0e-6


def test_boozer_metric_continuum_constant_field_matches_profile_limit():
    magnetic_field = 2.7
    mass_density = 3.4e-8
    major_radius = 5.5
    iota = 0.61
    spectrum = BoozerSpectrum(
        source="constant_boozer_metric",
        s=np.asarray([0.2]),
        xm=np.asarray([0.0]),
        xn=np.asarray([0.0]),
        bmnc=np.asarray([[magnetic_field]]),
        bmns=None,
        field_periods=1,
    )
    harmonics = np.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    frequencies = boozer_metric_continuum_frequencies_hz(
        spectrum,
        harmonics,
        iota=iota,
        major_radius=major_radius,
        mass_density=mass_density,
        jacobian_numerator=1.4,
    )
    k_parallel = (harmonics[:, 1] - harmonics[:, 0] * iota) / major_radius
    expected_frequency = magnetic_field / np.sqrt(MU0 * mass_density) * np.abs(k_parallel) / (2.0 * np.pi)

    np.testing.assert_allclose(np.asarray(frequencies[0]), np.sort(expected_frequency), rtol=1.0e-5)


def test_boozer_metric_continuum_uses_jacobian_weighting_and_is_jittable():
    spectrum = BoozerSpectrum(
        source="jacobian_weighted_boozer",
        s=np.asarray([0.5]),
        xm=np.asarray([0.0, 1.0]),
        xn=np.asarray([0.0, 1.0]),
        bmnc=np.asarray([[2.8, 0.10]]),
        bmns=None,
        field_periods=1,
    )
    harmonics = np.asarray([[1.0, 0.0], [2.0, 1.0]])

    matrices = boozer_metric_continuum_matrices(
        spectrum,
        harmonics,
        iota=0.43,
        major_radius=5.2,
        mass_density=4.1e-8,
        jacobian_numerator=1.6,
        theta_points=96,
        zeta_points=96,
    )
    mass = np.asarray(matrices.mass[0])
    stiffness = np.asarray(matrices.stiffness[0])

    np.testing.assert_allclose(mass, np.conjugate(mass.T), rtol=1.0e-6, atol=1.0e-12)
    np.testing.assert_allclose(stiffness, np.conjugate(stiffness.T), rtol=1.0e-6, atol=1.0e-6)
    assert abs(mass[0, 1]) / max(abs(mass[0, 0]), abs(mass[1, 1])) > 1.0e-3
    assert abs(stiffness[0, 1]) / max(abs(stiffness[0, 0]), abs(stiffness[1, 1])) < 1.0e-5

    import jax

    jitted = jax.jit(
        lambda spec: boozer_metric_continuum_matrices(
            spec,
            harmonics,
            iota=0.43,
            major_radius=5.2,
            mass_density=4.1e-8,
            jacobian_numerator=1.6,
            theta_points=32,
            zeta_points=32,
        ).mass
    )(spectrum)
    assert jitted.shape == (1, 2, 2)


def test_boozer_3d_continuum_constant_field_matches_profile_limit():
    geometry = _constant_boozer_geometry()
    harmonics = np.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    mass_density = 3.4e-8

    matrices = boozer_3d_continuum_matrices(
        geometry,
        harmonics,
        mass_density=mass_density,
        radial_metric=1.0,
        theta_points=32,
        zeta_points=32,
    )
    frequencies = boozer_3d_continuum_frequencies_hz(
        geometry,
        harmonics,
        mass_density=mass_density,
        radial_metric=1.0,
        theta_points=32,
        zeta_points=32,
    )
    k_parallel = (harmonics[:, 1] - harmonics[:, 0] * float(geometry.iota[0])) / 5.5
    expected_frequency = 2.7 / np.sqrt(MU0 * mass_density) * np.abs(k_parallel) / (2.0 * np.pi)

    np.testing.assert_allclose(np.asarray(frequencies[0]), np.sort(expected_frequency), rtol=1.0e-5)
    np.testing.assert_allclose(np.asarray(matrices.mass[0]), np.eye(3) * mass_density * 5.5 / 2.7, rtol=3.0e-6, atol=1.0e-12)
    stiffness = np.asarray(matrices.stiffness[0])
    off_diagonal = stiffness - np.diag(np.diag(stiffness))
    assert np.max(np.abs(off_diagonal)) / np.max(np.abs(np.diag(stiffness))) < 3.0e-9


def test_boozer_3d_radial_metric_profiles_and_grids_are_accepted():
    geometry = _constant_boozer_geometry()
    harmonics = np.asarray([[1.0, 0.0], [2.0, 1.0]])
    theta, zeta = boozer_angle_grid(theta_points=12, zeta_points=10)

    profile_metric = boozer_3d_continuum_matrices(
        geometry,
        harmonics,
        mass_density=3.4e-8,
        radial_metric=np.asarray([1.0, 1.2]),
        theta_points=12,
        zeta_points=10,
    )
    grid_metric = boozer_3d_continuum_matrices(
        geometry,
        harmonics,
        mass_density=3.4e-8,
        radial_metric=np.ones((2, theta.shape[0], zeta.shape[1])),
        theta_points=12,
        zeta_points=10,
    )

    assert profile_metric.stiffness.shape == (2, 2, 2)
    assert grid_metric.mass.shape == (2, 2, 2)
    with pytest.raises(ValueError, match="radial_metric"):
        boozer_3d_continuum_matrices(
            geometry,
            harmonics,
            mass_density=3.4e-8,
            radial_metric=np.ones((3,)),
            theta_points=12,
            zeta_points=10,
        )


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
def test_read_boozmn_geometry_and_file_jacobian_identity():
    path = BOOZ_XFORM_TEST_FILES / "boozmn_circular_tokamak.nc"
    if not path.exists():
        pytest.skip(f"missing local booz_xform_jax fixture: {path}")

    geometry = read_boozmn_geometry(path)

    assert geometry.bmnc.shape == geometry.gmnc.shape
    assert geometry.rmnc is not None and geometry.zmns is not None
    assert geometry.field_periods >= 1
    assert float(boozer_geometry_jacobian_identity_error(geometry, theta_points=48, zeta_points=48)) < 2.0e-5


@pytest.mark.requires_booz_xform_jax
def test_boozer_embedding_jacobian_matches_file_shape():
    path = BOOZ_XFORM_TEST_FILES / "boozmn_LandremanSenguptaPlunk_section5p3.nc"
    if not path.exists():
        pytest.skip(f"missing local booz_xform_jax fixture: {path}")

    geometry = read_boozmn_geometry(path)
    shape_error = boozer_embedding_jacobian_shape_error(geometry, theta_points=32, zeta_points=32)

    assert float(shape_error) < 4.0e-2


@pytest.mark.requires_booz_xform_jax
def test_boozer_3d_real_geometry_operator_is_hermitian_and_finite():
    path = BOOZ_XFORM_TEST_FILES / "boozmn_LandremanSenguptaPlunk_section5p3.nc"
    if not path.exists():
        pytest.skip(f"missing local booz_xform_jax fixture: {path}")

    geometry = read_boozmn_geometry(path)
    harmonics = np.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    matrices = boozer_3d_continuum_matrices(
        geometry,
        harmonics,
        mass_density=3.4e-8,
        theta_points=24,
        zeta_points=24,
    )
    frequencies = boozer_3d_continuum_frequencies_hz(
        geometry,
        harmonics,
        mass_density=3.4e-8,
        theta_points=24,
        zeta_points=24,
    )

    np.testing.assert_allclose(
        np.asarray(matrices.stiffness[0]),
        np.conjugate(np.asarray(matrices.stiffness[0]).T),
        rtol=2.0e-6,
        atol=1.0e-5,
    )
    np.testing.assert_allclose(
        np.asarray(matrices.mass[0]),
        np.conjugate(np.asarray(matrices.mass[0]).T),
        rtol=2.0e-6,
        atol=1.0e-12,
    )
    assert np.all(np.isfinite(np.asarray(frequencies)))
    assert np.all(np.asarray(frequencies) >= 0.0)


@pytest.mark.requires_booz_xform_jax
def test_boozer_global_3d_radial_operator_is_finite_with_modes():
    path = BOOZ_XFORM_TEST_FILES / "boozmn_LandremanSenguptaPlunk_section5p3.nc"
    if not path.exists():
        pytest.skip(f"missing local booz_xform_jax fixture: {path}")

    geometry = read_boozmn_geometry(path)
    harmonics = np.asarray([[1.0, 0.0], [2.0, 1.0]])
    problem = radial_block_problem_from_boozer_geometry(
        geometry,
        harmonics,
        mass_density=3.4e-8,
        radial_coupling_fraction=0.004,
        theta_points=16,
        zeta_points=16,
    )
    result = solve_radial_block_eigenmodes(problem)

    assert result.mode_structures.shape[1:] == (int(geometry.s.shape[0]), 2)
    assert np.all(np.isfinite(np.asarray(result.frequencies_hz[:6])))
    assert np.all(np.asarray(result.frequencies_hz[:6]) >= 0.0)
    assert float(np.max(np.asarray(result.residuals[:6]))) < 2.0e-5


@pytest.mark.requires_booz_xform_jax
def test_run_booz_xform_jax_from_wout_smoke():
    path = BOOZ_XFORM_TEST_FILES / "wout_circular_tokamak.nc"
    if not path.exists():
        pytest.skip(f"missing local booz_xform_jax fixture: {path}")

    spectrum = run_booz_xform_jax_from_wout(path, mboz=4, nboz=4, surfaces=(0.25, 0.5), jit=False)

    assert spectrum.bmnc.shape[0] == 2
    assert spectrum.field_periods >= 1
    assert np.all(np.isfinite(np.asarray(spectrum.bmnc)))


@pytest.mark.requires_booz_xform_jax
def test_run_booz_xform_jax_geometry_from_wout_matches_asymmetric_reference():
    wout_path = BOOZ_XFORM_TEST_FILES / "wout_up_down_asymmetric_tokamak.nc"
    boozmn_path = BOOZ_XFORM_TEST_FILES / "boozmn_up_down_asymmetric_tokamak.nc"
    if not wout_path.exists() or not boozmn_path.exists():
        pytest.skip("missing local asymmetric booz_xform_jax fixtures")

    reference = read_boozmn_geometry(boozmn_path)
    geometry = run_booz_xform_jax_geometry_from_wout(
        wout_path,
        mboz=48,
        nboz=0,
        surfaces=tuple(float(value) for value in np.asarray(reference.s)),
        jit=False,
    )

    for name in ("bmnc", "bmns", "rmnc", "rmns", "zmns", "zmnc", "pmns", "pmnc", "gmnc", "gmns"):
        actual = getattr(geometry, name)
        expected = getattr(reference, name)
        assert actual is not None
        assert expected is not None
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=5.0e-10, atol=5.0e-11)

    assert float(boozer_geometry_jacobian_identity_error(geometry, theta_points=32, zeta_points=8)) < 2.0e-5
