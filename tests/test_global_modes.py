from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fax.equilibrium import analytic_circular_tokamak
from fax.global_modes import (
    RadialBlockEigenmodeProblem,
    assemble_radial_block_matrices,
    assemble_radial_block_sparse_matrices,
    boozer_radial_stiffness_matrices,
    constant_block_problem_discrete_omega2,
    constant_radial_block_problem,
    constant_problem_discrete_omega2,
    constant_radial_problem,
    eigenpair_residual_norm,
    lowest_radial_block_frequency_hz,
    lowest_radial_frequency_hz,
    radial_block_cluster_mean_frequency_hz,
    radial_block_cluster_mean_omega2,
    radial_block_eigenvalue_sensitivity,
    radial_block_isolated_frequency_hz,
    radial_block_isolated_omega2,
    radial_block_mass_matvec,
    radial_block_matrix_free_residual_norm,
    radial_block_mode_structures,
    radial_block_problem_from_boozer_geometry,
    radial_block_rayleigh_omega2,
    radial_block_stiffness_matvec,
    radial_problem_from_equilibrium,
    solve_radial_block_eigenmodes,
    solve_radial_block_eigenmodes_sparse,
    solve_radial_eigenmodes,
)
from fax.boozer import BoozerGeometry


def _embedding_boozer_geometry(surface_count: int = 4) -> BoozerGeometry:
    s = np.linspace(0.2, 0.8, surface_count)
    minor = 0.4 * s
    magnetic_field = 2.7
    major_radius = 5.5
    numerator = magnetic_field * major_radius
    zeros = np.zeros((surface_count, 2))
    return BoozerGeometry(
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


def test_radial_block_residual_and_input_contracts_are_enforced():
    problem = constant_radial_block_problem(radial_points=14, continuum_frequency_hz=(90_000.0, 130_000.0))
    result = solve_radial_block_eigenmodes(problem)

    residual = eigenpair_residual_norm(result.stiffness, result.mass, result.omega2[0], result.eigenvectors[:, 0])

    assert float(residual) < 1.0e-6
    with pytest.raises(ValueError, match="continuum_frequency_hz"):
        constant_radial_block_problem(
            radial_points=8,
            harmonic_count=3,
            continuum_frequency_hz=(90_000.0, 130_000.0),
            radial_coupling_omega2=(2.0e9, 3.0e9, 4.0e9),
        )
    with pytest.raises(ValueError, match="boundary_condition"):
        RadialBlockEigenmodeProblem(
            s=problem.s,
            local_stiffness=problem.local_stiffness,
            local_mass=problem.local_mass,
            radial_stiffness=problem.radial_stiffness,
            harmonics=problem.harmonics,
            boundary_condition="periodic",
        )
    with pytest.raises(ValueError, match="harmonics"):
        RadialBlockEigenmodeProblem(
            s=problem.s,
            local_stiffness=problem.local_stiffness,
            local_mass=problem.local_mass,
            radial_stiffness=problem.radial_stiffness,
            harmonics=jnp.asarray([1.0, 0.0]),
        )
    with pytest.raises(ValueError, match="local_stiffness"):
        RadialBlockEigenmodeProblem(
            s=problem.s,
            local_stiffness=problem.local_stiffness[:, :1, :1],
            local_mass=problem.local_mass,
            radial_stiffness=problem.radial_stiffness,
            harmonics=problem.harmonics,
        )


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


@pytest.mark.parametrize("boundary_condition", ["dirichlet", "neumann"])
def test_radial_block_matrix_free_matvec_matches_dense_assembly(boundary_condition: str):
    problem = constant_radial_block_problem(
        radial_points=16,
        continuum_frequency_hz=(80_000.0, 120_000.0, 165_000.0),
        radial_coupling_omega2=(1.8e9, 2.4e9, 3.1e9),
        boundary_condition=boundary_condition,
    )
    stiffness, mass = assemble_radial_block_matrices(problem)
    matrix_size = int(stiffness.shape[0])
    vector = jnp.linspace(0.1, 1.0, matrix_size) + 0.02j * jnp.linspace(1.0, 0.2, matrix_size)

    stiffness_vector = radial_block_stiffness_matvec(problem, vector)
    mass_vector = radial_block_mass_matvec(problem, vector)
    jit_stiffness_vector, jit_mass_vector = jax.jit(
        lambda p, v: (radial_block_stiffness_matvec(p, v), radial_block_mass_matvec(p, v))
    )(problem, vector)

    np.testing.assert_allclose(np.asarray(stiffness_vector), np.asarray(stiffness @ vector), rtol=1.0e-12, atol=1.0e-7)
    np.testing.assert_allclose(np.asarray(mass_vector), np.asarray(mass @ vector), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(jit_stiffness_vector), np.asarray(stiffness_vector), rtol=1.0e-12, atol=1.0e-7)
    np.testing.assert_allclose(np.asarray(jit_mass_vector), np.asarray(mass_vector), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        np.asarray(radial_block_stiffness_matvec(problem, jnp.reshape(vector, (-1, 3)))),
        np.asarray(jnp.reshape(stiffness @ vector, (-1, 3))),
        rtol=1.0e-12,
        atol=1.0e-7,
    )


def test_radial_block_matrix_free_residual_and_rayleigh_match_eigenpair():
    problem = constant_radial_block_problem(
        radial_points=18,
        continuum_frequency_hz=(95_000.0, 145_000.0),
        radial_coupling_omega2=(2.4e9, 3.1e9),
    )
    result = solve_radial_block_eigenmodes(problem)
    vector = result.eigenvectors[:, 0]

    dense_residual = eigenpair_residual_norm(result.stiffness, result.mass, result.omega2[0], vector)
    matrix_free_residual = radial_block_matrix_free_residual_norm(problem, result.omega2[0], vector)
    rayleigh = radial_block_rayleigh_omega2(problem, vector)

    np.testing.assert_allclose(float(matrix_free_residual), float(dense_residual), rtol=1.0e-9, atol=1.0e-14)
    np.testing.assert_allclose(float(rayleigh), float(result.omega2[0]), rtol=1.0e-12)
    with pytest.raises(ValueError, match="active radial-block"):
        radial_block_stiffness_matvec(problem, jnp.ones((3,)))


def test_radial_block_matrix_free_rayleigh_density_gradient_matches_scaling():
    problem = constant_radial_block_problem(
        radial_points=18,
        continuum_frequency_hz=(95_000.0, 145_000.0),
        radial_coupling_omega2=(2.4e9, 3.1e9),
    )
    vector = solve_radial_block_eigenmodes(problem).eigenvectors[:, 0]

    def frequency(scale):
        scaled = RadialBlockEigenmodeProblem(
            s=problem.s,
            local_stiffness=problem.local_stiffness,
            local_mass=problem.local_mass * scale,
            radial_stiffness=problem.radial_stiffness,
            harmonics=problem.harmonics,
            boundary_condition=problem.boundary_condition,
        )
        return jnp.sqrt(radial_block_rayleigh_omega2(scaled, vector)) / (2.0 * jnp.pi)

    scale0 = 1.2
    autodiff = jax.grad(frequency)(scale0)
    expected = -0.5 * frequency(scale0) / scale0

    np.testing.assert_allclose(float(autodiff), float(expected), rtol=8.0e-6)


def test_radial_block_eigenvalue_sensitivity_matches_autodiff_tangents():
    problem = constant_radial_block_problem(
        radial_points=18,
        continuum_frequency_hz=(92_000.0, 150_000.0),
        radial_coupling_omega2=(2.2e9, 3.4e9),
    )
    result = solve_radial_block_eigenmodes(problem)
    zero_stiffness = jnp.zeros_like(problem.local_stiffness)
    zero_mass = jnp.zeros_like(problem.local_mass)
    zero_radial = jnp.zeros_like(problem.radial_stiffness)
    mass_tangent_problem = RadialBlockEigenmodeProblem(
        s=problem.s,
        local_stiffness=zero_stiffness,
        local_mass=problem.local_mass,
        radial_stiffness=zero_radial,
        harmonics=problem.harmonics,
        boundary_condition=problem.boundary_condition,
    )
    radial_tangent_problem = RadialBlockEigenmodeProblem(
        s=problem.s,
        local_stiffness=zero_stiffness,
        local_mass=zero_mass,
        radial_stiffness=problem.radial_stiffness,
        harmonics=problem.harmonics,
        boundary_condition=problem.boundary_condition,
    )

    def mass_scaled_omega2(alpha):
        scaled = RadialBlockEigenmodeProblem(
            s=problem.s,
            local_stiffness=problem.local_stiffness,
            local_mass=(1.0 + alpha) * problem.local_mass,
            radial_stiffness=problem.radial_stiffness,
            harmonics=problem.harmonics,
            boundary_condition=problem.boundary_condition,
        )
        return solve_radial_block_eigenmodes(scaled).omega2[0]

    def radial_scaled_omega2(alpha):
        scaled = RadialBlockEigenmodeProblem(
            s=problem.s,
            local_stiffness=problem.local_stiffness,
            local_mass=problem.local_mass,
            radial_stiffness=(1.0 + alpha) * problem.radial_stiffness,
            harmonics=problem.harmonics,
            boundary_condition=problem.boundary_condition,
        )
        return solve_radial_block_eigenmodes(scaled).omega2[0]

    mass_sensitivity = radial_block_eigenvalue_sensitivity(
        problem,
        mass_tangent_problem,
        result.omega2[0],
        2.3 * result.eigenvectors[:, 0],
    )
    radial_sensitivity = radial_block_eigenvalue_sensitivity(
        problem,
        radial_tangent_problem,
        result.omega2[0],
        result.eigenvectors[:, 0],
    )
    mass_autodiff = jax.grad(mass_scaled_omega2)(0.0)
    radial_autodiff = jax.grad(radial_scaled_omega2)(0.0)
    step = 2.0e-5
    radial_finite_difference = (radial_scaled_omega2(step) - radial_scaled_omega2(-step)) / (2.0 * step)

    np.testing.assert_allclose(float(mass_sensitivity), float(mass_autodiff), rtol=3.0e-10)
    np.testing.assert_allclose(float(mass_sensitivity), float(-result.omega2[0]), rtol=3.0e-10)
    np.testing.assert_allclose(float(radial_sensitivity), float(radial_autodiff), rtol=3.0e-7)
    np.testing.assert_allclose(float(radial_sensitivity), float(radial_finite_difference), rtol=3.0e-7)


def test_radial_block_eigenvalue_sensitivity_rejects_mismatched_tangent_problem():
    problem = constant_radial_block_problem(radial_points=8, continuum_frequency_hz=(90_000.0, 140_000.0))
    result = solve_radial_block_eigenmodes(problem)
    mismatched = constant_radial_block_problem(
        radial_points=8,
        continuum_frequency_hz=(90_000.0, 140_000.0),
        boundary_condition="neumann",
    )

    with pytest.raises(ValueError, match="boundary_condition"):
        radial_block_eigenvalue_sensitivity(problem, mismatched, result.omega2[0], result.eigenvectors[:, 0])


def test_radial_block_isolated_omega2_custom_jvp_matches_explicit_sensitivity():
    problem = constant_radial_block_problem(
        radial_points=18,
        continuum_frequency_hz=(92_000.0, 150_000.0),
        radial_coupling_omega2=(2.2e9, 3.4e9),
    )
    result = solve_radial_block_eigenmodes(problem)
    zero_stiffness = jnp.zeros_like(problem.local_stiffness)
    zero_mass = jnp.zeros_like(problem.local_mass)
    tangent_problem = RadialBlockEigenmodeProblem(
        s=problem.s,
        local_stiffness=zero_stiffness,
        local_mass=zero_mass,
        radial_stiffness=problem.radial_stiffness,
        harmonics=problem.harmonics,
        boundary_condition=problem.boundary_condition,
    )
    explicit = radial_block_eigenvalue_sensitivity(problem, tangent_problem, result.omega2[0], result.eigenvectors[:, 0])

    def omega2(alpha):
        scaled = RadialBlockEigenmodeProblem(
            s=problem.s,
            local_stiffness=problem.local_stiffness,
            local_mass=problem.local_mass,
            radial_stiffness=(1.0 + alpha) * problem.radial_stiffness,
            harmonics=problem.harmonics,
            boundary_condition=problem.boundary_condition,
        )
        return radial_block_isolated_omega2(scaled, mode_index=0)

    custom_jvp = jax.grad(omega2)(0.0)
    step = 2.0e-5
    finite_difference = (omega2(step) - omega2(-step)) / (2.0 * step)
    frequency = radial_block_isolated_frequency_hz(problem, mode_index=0)
    jit_frequency = jax.jit(lambda p: radial_block_isolated_frequency_hz(p, mode_index=0))(problem)

    np.testing.assert_allclose(float(radial_block_isolated_omega2(problem, mode_index=0)), float(result.omega2[0]), rtol=1.0e-12)
    np.testing.assert_allclose(float(frequency), float(result.frequencies_hz[0]), rtol=1.0e-12)
    np.testing.assert_allclose(float(jit_frequency), float(frequency), rtol=1.0e-12)
    np.testing.assert_allclose(float(custom_jvp), float(explicit), rtol=3.0e-7)
    np.testing.assert_allclose(float(custom_jvp), float(finite_difference), rtol=3.0e-7)


def test_radial_block_cluster_mean_omega2_is_stable_through_degeneracy():
    problem = constant_radial_block_problem(
        radial_points=14,
        continuum_frequency_hz=(92_000.0, 92_000.0),
        radial_coupling_omega2=(2.2e9, 2.2e9),
    )
    result = solve_radial_block_eigenmodes(problem)

    def omega2(alpha):
        scaled = RadialBlockEigenmodeProblem(
            s=problem.s,
            local_stiffness=problem.local_stiffness,
            local_mass=problem.local_mass,
            radial_stiffness=(1.0 + alpha) * problem.radial_stiffness,
            harmonics=problem.harmonics,
            boundary_condition=problem.boundary_condition,
        )
        return radial_block_cluster_mean_omega2(scaled, start_index=0, stop_index=2)

    value = radial_block_cluster_mean_omega2(problem, start_index=0, stop_index=2)
    frequency = radial_block_cluster_mean_frequency_hz(problem, start_index=0, stop_index=2)
    jit_value = jax.jit(lambda p: radial_block_cluster_mean_omega2(p, start_index=0, stop_index=2))(problem)
    custom_jvp = jax.grad(omega2)(0.0)
    step = 2.0e-5
    finite_difference = (omega2(step) - omega2(-step)) / (2.0 * step)

    np.testing.assert_allclose(float(result.omega2[0]), float(result.omega2[1]), rtol=1.0e-12)
    np.testing.assert_allclose(float(value), float(jnp.mean(result.omega2[:2])), rtol=1.0e-12)
    np.testing.assert_allclose(float(frequency), float(jnp.sqrt(jnp.maximum(value, 0.0)) / (2.0 * jnp.pi)), rtol=1.0e-12)
    np.testing.assert_allclose(float(jit_value), float(value), rtol=1.0e-12)
    np.testing.assert_allclose(float(custom_jvp), float(finite_difference), rtol=5.0e-7)


def test_sparse_radial_block_assembly_matches_dense_and_exact_spectrum():
    problem = constant_radial_block_problem(
        radial_points=28,
        continuum_frequency_hz=(85_000.0, 125_000.0, 170_000.0),
        radial_coupling_omega2=(1.8e9, 2.6e9, 3.4e9),
    )

    dense_stiffness, dense_mass = assemble_radial_block_matrices(problem)
    sparse_stiffness, sparse_mass = assemble_radial_block_sparse_matrices(problem)
    result = solve_radial_block_eigenmodes_sparse(problem, mode_count=10)
    expected = constant_block_problem_discrete_omega2(problem)

    np.testing.assert_allclose(sparse_stiffness.toarray(), np.asarray(dense_stiffness), rtol=1.0e-12, atol=1.0e-8)
    np.testing.assert_allclose(sparse_mass.toarray(), np.asarray(dense_mass), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(result.omega2), np.asarray(expected[:10]), rtol=8.0e-6)
    assert result.mode_structures.shape == (10, 28, 3)
    assert float(jnp.max(result.residuals)) < 2.0e-8
    assert sparse_stiffness.nnz < sparse_stiffness.shape[0] ** 2


def test_sparse_radial_block_solver_contracts():
    problem = constant_radial_block_problem(radial_points=6, continuum_frequency_hz=(90_000.0, 140_000.0))

    with pytest.raises(ValueError, match="mode_count"):
        solve_radial_block_eigenmodes_sparse(problem, mode_count=0)
    with pytest.raises(ValueError, match="mode_count"):
        solve_radial_block_eigenmodes_sparse(problem, mode_count=problem.local_stiffness.shape[0] * problem.harmonics.shape[0])


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
    geometry = _embedding_boozer_geometry()
    harmonics = np.asarray([[1.0, 0.0], [2.0, 1.0]])

    radial = boozer_radial_stiffness_matrices(geometry, harmonics, theta_points=24, zeta_points=24)

    np.testing.assert_allclose(np.asarray(radial[0]), np.conjugate(np.asarray(radial[0]).T), rtol=1.0e-6, atol=1.0e-8)
    assert np.all(np.linalg.eigvalsh(np.asarray(radial[0])) > 0.0)


def test_boozer_global_builder_supports_mass_and_explicit_radial_stiffness_profiles():
    geometry = _embedding_boozer_geometry()
    harmonics = np.asarray([[1.0, 0.0], [2.0, 1.0]])

    mass_scaled = radial_block_problem_from_boozer_geometry(
        geometry,
        harmonics,
        mass_density=3.4e-8,
        radial_stiffness_method="mass",
        theta_points=12,
        zeta_points=12,
    )
    profile_scaled = radial_block_problem_from_boozer_geometry(
        geometry,
        harmonics,
        mass_density=3.4e-8,
        radial_coupling_omega2=np.linspace(2.0e9, 3.0e9, int(geometry.s.shape[0])),
        theta_points=12,
        zeta_points=12,
    )
    full_block_scaled = radial_block_problem_from_boozer_geometry(
        geometry,
        harmonics,
        mass_density=3.4e-8,
        radial_coupling_omega2=np.asarray(mass_scaled.radial_stiffness),
        theta_points=12,
        zeta_points=12,
    )
    result = solve_radial_block_eigenmodes(mass_scaled)

    assert dict(mass_scaled.metadata)["radial_stiffness_method"] == "mass"
    assert profile_scaled.radial_stiffness.shape == mass_scaled.radial_stiffness.shape
    assert full_block_scaled.radial_stiffness.shape == mass_scaled.radial_stiffness.shape
    assert float(jnp.max(result.residuals[:4])) < 2.0e-5
    with pytest.raises(ValueError, match="radial_stiffness_method"):
        radial_block_problem_from_boozer_geometry(
            geometry,
            harmonics,
            mass_density=3.4e-8,
            radial_stiffness_method="unsupported",
            theta_points=12,
            zeta_points=12,
        )
    with pytest.raises(ValueError, match="radial_coupling_omega2"):
        radial_block_problem_from_boozer_geometry(
            geometry,
            harmonics,
            mass_density=3.4e-8,
            radial_coupling_omega2=np.ones((3,)),
            theta_points=12,
            zeta_points=12,
        )
    with pytest.raises(ValueError, match="harmonics"):
        boozer_radial_stiffness_matrices(geometry, np.ones((2, 3)), theta_points=12, zeta_points=12)
