from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from fax.equilibrium import analytic_circular_tokamak
from fax.global_modes import (
    assemble_radial_block_matrices,
    boozer_radial_stiffness_matrices,
    constant_block_problem_discrete_omega2,
    constant_radial_block_problem,
    constant_problem_discrete_omega2,
    constant_radial_problem,
    lowest_radial_block_frequency_hz,
    lowest_radial_frequency_hz,
    radial_block_mode_structures,
    radial_problem_from_equilibrium,
    solve_radial_block_eigenmodes,
    solve_radial_eigenmodes,
)
from fax.boozer import BoozerGeometry


def test_constant_radial_problem_matches_exact_discrete_spectrum():
    problem = constant_radial_problem(radial_points=48, continuum_frequency_hz=100_000.0, radial_coupling_omega2=3.0e9)

    result = solve_radial_eigenmodes(problem)
    expected = constant_problem_discrete_omega2(problem)

    np.testing.assert_allclose(np.asarray(result.omega2[:10]), np.asarray(expected[:10]), rtol=2.0e-6)
    assert float(jnp.max(result.residuals[:10])) < 1.0e-5


def test_radial_problem_from_equilibrium_is_jittable():
    geometry = analytic_circular_tokamak(radial_points=24, iota_axis=0.28, iota_edge=0.68)
    problem = radial_problem_from_equilibrium(geometry, harmonic=(2, 1))

    frequencies = jax.jit(lambda p: solve_radial_eigenmodes(p).frequencies_hz[:3])(problem)

    assert frequencies.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(frequencies)))
    assert bool(jnp.all(frequencies >= 0.0))


def test_lowest_radial_frequency_density_gradient_matches_finite_difference():
    problem = constant_radial_problem(radial_points=36, continuum_frequency_hz=110_000.0, radial_coupling_omega2=1.0e10)

    def frequency(scale):
        return lowest_radial_frequency_hz(scale, base_problem=problem)

    scale0 = 1.3
    autodiff = jax.grad(frequency)(scale0)
    step = 1.0e-4
    finite_difference = (frequency(scale0 + step) - frequency(scale0 - step)) / (2.0 * step)

    np.testing.assert_allclose(float(autodiff), float(finite_difference), rtol=3.0e-3)


def test_lowest_radial_frequency_density_gradient_matches_analytic_scaling():
    problem = constant_radial_problem(radial_points=36, continuum_frequency_hz=110_000.0, radial_coupling_omega2=1.0e10)

    def frequency(scale):
        return lowest_radial_frequency_hz(scale, base_problem=problem)

    scale0 = 1.3
    autodiff = jax.grad(frequency)(scale0)
    expected = -0.5 * frequency(scale0) / scale0

    np.testing.assert_allclose(float(autodiff), float(expected), rtol=5.0e-5)


def test_constant_radial_block_problem_matches_exact_discrete_spectrum():
    problem = constant_radial_block_problem(
        radial_points=22,
        continuum_frequency_hz=(90_000.0, 135_000.0),
        radial_coupling_omega2=(2.0e9, 3.5e9),
    )

    result = solve_radial_block_eigenmodes(problem)
    expected = constant_block_problem_discrete_omega2(problem)

    np.testing.assert_allclose(np.asarray(result.omega2[:12]), np.asarray(expected[:12]), rtol=4.0e-6)
    assert float(jnp.max(result.residuals[:12])) < 1.0e-5
    assert result.mode_structures.shape == (40, 22, 2)


def test_constant_radial_block_neumann_problem_matches_exact_discrete_spectrum():
    problem = constant_radial_block_problem(
        radial_points=22,
        continuum_frequency_hz=(90_000.0, 135_000.0),
        radial_coupling_omega2=(2.0e9, 3.5e9),
        boundary_condition="neumann",
    )

    result = solve_radial_block_eigenmodes(problem)
    expected = constant_block_problem_discrete_omega2(problem)

    np.testing.assert_allclose(np.asarray(result.omega2[:12]), np.asarray(expected[:12]), rtol=4.0e-6)
    assert float(jnp.max(result.residuals[:12])) < 1.0e-5
    assert result.mode_structures.shape == (44, 22, 2)


def test_radial_block_problem_is_jittable_and_hermitian():
    problem = constant_radial_block_problem(radial_points=14, continuum_frequency_hz=(80_000.0, 120_000.0))

    stiffness, mass = assemble_radial_block_matrices(problem)
    frequencies = jax.jit(lambda p: solve_radial_block_eigenmodes(p).frequencies_hz[:4])(problem)
    structures = radial_block_mode_structures(problem, solve_radial_block_eigenmodes(problem).eigenvectors)

    np.testing.assert_allclose(np.asarray(stiffness), np.conjugate(np.asarray(stiffness).T), rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(np.asarray(mass), np.conjugate(np.asarray(mass).T), rtol=1.0e-6, atol=1.0e-12)
    assert frequencies.shape == (4,)
    assert structures.shape[1:] == (14, 2)


def test_lowest_radial_block_frequency_density_gradient_matches_analytic_scaling():
    problem = constant_radial_block_problem(
        radial_points=16,
        continuum_frequency_hz=(95_000.0, 145_000.0),
        radial_coupling_omega2=(2.4e9, 3.1e9),
    )

    def frequency(scale):
        return lowest_radial_block_frequency_hz(scale, base_problem=problem)

    scale0 = 1.25
    autodiff = jax.grad(frequency)(scale0)
    expected = -0.5 * frequency(scale0) / scale0

    np.testing.assert_allclose(float(autodiff), float(expected), rtol=8.0e-5)


def test_boozer_radial_stiffness_matrices_are_hermitian():
    s = np.linspace(0.2, 0.8, 4)
    minor = 0.4 * s
    magnetic_field = 2.7
    major_radius = 5.5
    numerator = magnetic_field * major_radius
    zeros = np.zeros((4, 2))
    geometry = BoozerGeometry(
        source="synthetic_radial_metric",
        s=s,
        xm=np.asarray([0.0, 1.0]),
        xn=np.asarray([0.0, 0.0]),
        bmnc=np.column_stack([np.full_like(s, magnetic_field), np.zeros_like(s)]),
        bmns=zeros,
        rmnc=np.column_stack([np.full_like(s, major_radius), minor]),
        rmns=zeros,
        zmns=np.column_stack([np.zeros_like(s), minor]),
        zmnc=zeros,
        pmns=zeros,
        pmnc=zeros,
        gmnc=np.column_stack([np.full_like(s, numerator / magnetic_field**2), np.zeros_like(s)]),
        gmns=zeros,
        iota=np.full_like(s, 0.61),
        boozer_i=np.zeros_like(s),
        boozer_g=np.full_like(s, numerator),
        field_periods=1,
    )
    harmonics = np.asarray([[1.0, 0.0], [2.0, 1.0]])

    radial = boozer_radial_stiffness_matrices(geometry, harmonics, theta_points=24, zeta_points=24)

    np.testing.assert_allclose(np.asarray(radial[0]), np.conjugate(np.asarray(radial[0]).T), rtol=1.0e-6, atol=1.0e-8)
    assert np.all(np.linalg.eigvalsh(np.asarray(radial[0])) > 0.0)
