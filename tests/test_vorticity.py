"""Independent dispersion and weak-form checks for physical potential modes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dataclasses import replace
from numpy.polynomial.legendre import leggauss

from fax.boozer import BoozerGeometry
from fax.metric import generalized_hermitian_isolated_eigenvalue
from fax.units import MU0
from fax.vorticity import (
    _sample_radial,
    physical_potential_problem_from_boozer_geometry,
    physical_potential_problem_from_grid,
    solve_physical_potential_eigenmodes,
)


def slab(s, harmonics=((1, 2),), density=1.7e-6, radius=None):
    theta, zeta = np.meshgrid(2*np.pi*np.arange(16)/16, 2*np.pi*np.arange(16)/16, indexing='ij')
    points = theta.size
    radius = np.full(len(s), 4.) if radius is None else np.asarray(radius)
    basis = np.broadcast_to(np.eye(3), (len(s), points, 3, 3)).copy()
    basis[..., 2, 2] = radius[:, None]
    return physical_potential_problem_from_grid(s, harmonics,
        theta=theta, zeta=zeta, magnetic_field=np.full((len(s), points), 2.),
        magnetic_field_angle_derivatives=np.zeros((len(s), points, 2)),
        covariant_basis=basis, iota=0., boozer_i=0., boozer_g=2*radius, mass_density=density)


@pytest.mark.parametrize('nodes', [5, 9])
def test_uniform_slab_frequency_independent_of_radial_wavelength_and_nonuniform_mesh(nodes):
    # All radial eigenfunctions have omega=k_parallel v_A in a uniform slab.
    # An added radial stiffness with local-only inertia fails this exact limit.
    s = np.linspace(.01, 1., nodes)**1.7
    density = 1.7e-6
    problem = slab(s, density=density)
    expected_omega2 = (2/4)**2 * 2**2/(MU0*density)
    np.testing.assert_allclose(problem.stiffness, expected_omega2*problem.mass, rtol=2e-13, atol=1e-13)
    result = solve_physical_potential_eigenmodes(problem)
    np.testing.assert_allclose(result.omega2, expected_omega2, rtol=2e-13)
    assert np.max(result.residuals) < 1e-12


def test_independent_angular_harmonics_obey_their_own_slab_dispersion():
    problem = slab(np.linspace(.05, .95, 6), harmonics=((1, 2), (3, 1)))
    result = solve_physical_potential_eigenmodes(problem)
    expected = np.repeat(np.array([1., 4.])*2**2/(MU0*1.7e-6*4**2), 4)
    np.testing.assert_allclose(result.omega2, expected, rtol=2e-13)
    np.testing.assert_allclose(problem.mass, problem.mass.conj().T, atol=1e-20)


def test_parallel_wavenumber_gradient_matches_independent_rayleigh_integral():
    s = np.array([.05, .2, .5, .7, .95])
    radius = 4+2*s
    values = np.array([0., .8, 1., .4, 0.])
    problem = slab(s, harmonics=((1, 2),), radius=radius)
    trial = values[1:-1]
    k = float(np.vdot(trial, np.asarray(problem.stiffness)@trial).real)
    mass = float(np.vdot(trial, np.asarray(problem.mass)@trial).real)
    # Independent high-order quadrature of |grad(ik_parallel Phi)|^2 and
    # mu0*rho/B^2 |grad Phi|^2 in a warped straight-field-line metric.
    expected_k, expected_mass = 0., 0.
    xi, weights = leggauss(40)
    for index, ds in enumerate(np.diff(s)):
        x = s[index]+ds*(xi+1)/2
        r = 4+2*x
        phi = values[index]+(values[index+1]-values[index])*(xi+1)/2
        dphi = (values[index+1]-values[index])/ds
        h = -2/r
        dh = 4/r**2
        expected_k += np.sum(ds/2*weights*r*((h*dphi+dh*phi)**2 + (h*phi)**2*(1+4/r**2)))
        expected_mass += np.sum(ds/2*weights*r*MU0*1.7e-6/4*(dphi**2+phi**2*(1+4/r**2)))
    # Two-point element quadrature, rather than an algebraic identity test.
    assert k == pytest.approx(expected_k, rel=3e-5)
    assert mass == pytest.approx(expected_mass, rel=3e-5)
    # A test of h*dPhi/ds alone would miss the physically required dh/ds term.


def test_density_sensitivity_has_the_physical_inverse_density_scaling():
    s = np.linspace(.05, .95, 5)
    def quotient(scale):
        problem = slab(s, density=scale*1.7e-6)
        vector = jnp.ones(3)
        return jnp.real(vector @ problem.stiffness @ vector/(vector @ problem.mass @ vector))
    value, derivative = jax.jvp(quotient, (jnp.array(1.),), (jnp.array(1.),))
    assert derivative == pytest.approx(-value, rel=2e-13)


def test_full_metric_and_angular_parallel_gradient_match_independent_quadrature():
    # Manufactured 3D coefficients, not a claimed MHD/QI equilibrium.
    # Nonorthogonal metric, nonzero I, magnetic shear, angular dB and radial rho.
    s = np.array([.05, .25, .6, .95])
    theta, zeta = np.meshgrid(2*np.pi*np.arange(24)/24, 2*np.pi*np.arange(24)/24, indexing='ij')
    angular = 1+.12*np.cos(theta)+.08*np.sin(2*zeta)
    b = np.broadcast_to(2*angular, (len(s), *theta.shape))
    db = np.broadcast_to(np.stack([-.24*np.sin(theta), .32*np.cos(2*zeta)], -1), (*b.shape, 2))
    embedding = np.array([[1., .2, -.15], [0., 1.1, .25], [.05, -.1, 4.]])
    basis = np.broadcast_to(embedding, (*b.shape, 3, 3))
    rho = 1.7e-6*(1-.2*s)
    rot, ci, cg = .2+.15*s, .08+.03*s, 8+.1*s
    values = np.array([0., .8+.2j, .4-.3j, 0.])
    m, n = 2, 1
    problem = physical_potential_problem_from_grid(s, [[m, n]], theta=theta, zeta=zeta,
        magnetic_field=b, magnetic_field_angle_derivatives=db, covariant_basis=basis,
        iota=rot, boozer_i=ci, boozer_g=cg, mass_density=rho)
    actual_k = np.vdot(values[1:-1], np.asarray(problem.stiffness)@values[1:-1]).real
    actual_mass = np.vdot(values[1:-1], np.asarray(problem.mass)@values[1:-1]).real
    expected_k, expected_mass = 0., 0.
    xi, weights = leggauss(30)
    for index, ds in enumerate(np.diff(s)):
        for x, weight in zip((xi+1)/2, weights*ds/2):
            phi = (1-x)*values[index]+x*values[index+1]
            dphi = (values[index+1]-values[index])/ds
            r, i, g = [(1-x)*p[index]+x*p[index+1] for p in (rot, ci, cg)]
            dr, di, dg = [(p[index+1]-p[index])/ds for p in (rot, ci, cg)]
            den = g+r*i
            h = 2*angular*(m*r-n)/den
            dhs = 2*angular*m*dr/den-h*(dg+dr*i+r*di)/den
            dht, dhz = db[0, ..., 0]*(m*r-n)/den, db[0, ..., 1]*(m*r-n)/den
            grad = np.broadcast_to([dphi, 1j*m*phi, -1j*n*phi], (*theta.shape, 3))
            gradp = 1j*np.stack([h*dphi+dhs*phi, (dht+1j*m*h)*phi, (dhz-1j*n*h)*phi], -1)
            # Solve E grad_cart = grad_coordinates independently of tensor assembly.
            cart = np.linalg.solve(embedding, grad.reshape(-1, 3).T).T
            cartp = np.linalg.solve(embedding, gradp.reshape(-1, 3).T).T
            jac = abs(np.linalg.det(embedding))
            density = (1-x)*rho[index]+x*rho[index+1]
            expected_k += weight*jac*np.mean(np.sum(abs(cartp)**2, axis=-1))
            expected_mass += weight*jac*MU0*density*np.mean(np.sum(abs(cart)**2, axis=-1)/(2*angular.ravel())**2)
    # Rational h(s) gives a small two-point quadrature error on coarse elements.
    # Require convergence toward the independent 30-point integral as well.
    assert actual_k == pytest.approx(expected_k, rel=1e-4)
    assert actual_mass == pytest.approx(expected_mass, rel=2e-7)
    fine_s = np.unique(np.concatenate([np.linspace(a, b, 9) for a, b in zip(s[:-1], s[1:])]))
    fine_b = np.broadcast_to(b[0], (len(fine_s), *theta.shape))
    fine = physical_potential_problem_from_grid(fine_s, [[m, n]], theta=theta, zeta=zeta,
        magnetic_field=fine_b,
        magnetic_field_angle_derivatives=np.broadcast_to(db[0], (*fine_b.shape, 2)),
        covariant_basis=np.broadcast_to(embedding, (*fine_b.shape, 3, 3)),
        iota=.2+.15*fine_s, boozer_i=.08+.03*fine_s, boozer_g=8+.1*fine_s,
        mass_density=1.7e-6*(1-.2*fine_s))
    fine_values = np.interp(fine_s, s, values)[1:-1]
    fine_k = np.vdot(fine_values, np.asarray(fine.stiffness)@fine_values).real
    assert fine_k == pytest.approx(expected_k, rel=2e-8)
    assert abs(fine_k-expected_k) < abs(actual_k-expected_k)/500


def synthetic_geometry():
    """Smooth 3D Fourier embedding for adapter/AD tests, not a QI benchmark."""
    s = jnp.linspace(.1, .9, 5)
    zero = jnp.zeros_like(s)
    zeros = jnp.zeros((len(s), 4))
    return BoozerGeometry(source='manufactured', s=s, xm=jnp.array([0., 1., 1., 2.]),
        xn=jnp.array([0., 0., 2., 2.]),
        bmnc=jnp.column_stack([3+zero, -.15*jnp.sqrt(s), zero, .12*s]),
        bmns=jnp.column_stack([zero, zero, .06*s, zero]),
        rmnc=jnp.column_stack([4+zero, .5*jnp.sqrt(s), .03*jnp.sqrt(s), zero]), rmns=zeros,
        zmns=jnp.column_stack([zero, .5*jnp.sqrt(s), .03*jnp.sqrt(s), zero]), zmnc=zeros,
        pmns=jnp.column_stack([zero, zero, .02*s, zero]), pmnc=zeros,
        gmnc=jnp.column_stack([1+zero, zero, zero, zero]), gmns=zeros,
        iota=.3+.2*s, boozer_i=.05*s, boozer_g=12+.1*s, field_periods=2)


def test_jit_reverse_gradient_through_geometry_and_isolated_eigenvalue():
    geometry = synthetic_geometry()
    modes = jnp.array([[2, 1], [3, 1]])

    def build(parameter):
        altered = replace(geometry, bmnc=geometry.bmnc.at[:, 3].multiply(1+parameter),
                          rmnc=geometry.rmnc.at[:, 2].multiply(1+.1*parameter))
        return physical_potential_problem_from_boozer_geometry(altered, modes,
            mass_density=1.7e-6*(1-.3*geometry.s), theta_points=12, zeta_points=12)

    compiled = jax.jit(build)
    eager, traced = build(jnp.array(.1)), compiled(jnp.array(.1))
    np.testing.assert_allclose(traced.stiffness, eager.stiffness, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(traced.mass, eager.mass, rtol=1e-12, atol=1e-20)
    spectrum = solve_physical_potential_eigenmodes(traced)
    assert np.max(spectrum.residuals) < 1e-11
    assert np.all(np.asarray(spectrum.omega2) > 0)

    def objective(parameter):
        problem = build(parameter)
        return generalized_hermitian_isolated_eigenvalue(problem.stiffness, problem.mass, 0)

    derivative = jax.jit(jax.grad(objective))(jnp.array(.1))
    step = 1e-4
    finite_difference = (objective(.1+step)-objective(.1-step))/(2*step)
    assert np.isfinite(derivative) and abs(derivative) > 1.
    assert derivative == pytest.approx(finite_difference, rel=2e-5)


def test_toroidal_offset_conventions_give_identical_operator():
    geometry = synthetic_geometry()
    modes = [[2, 1], [3, 1]]
    original = physical_potential_problem_from_boozer_geometry(geometry, modes,
        mass_density=1.7e-6, theta_points=12, zeta_points=12)
    reversed_offset = replace(geometry, pmns=-geometry.pmns, pmnc=-geometry.pmnc)
    converted = physical_potential_problem_from_boozer_geometry(reversed_offset, modes,
        mass_density=1.7e-6, theta_points=12, zeta_points=12, toroidal_offset_sign=-1)
    np.testing.assert_allclose(original.stiffness, converted.stiffness, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(original.mass, converted.mass, rtol=1e-13, atol=1e-20)


def test_zero_edge_density_and_angular_refinement_for_general_3d_geometry():
    geometry = synthetic_geometry()
    # The QI paper uses rho/rho_axis=1-s^5; its zero edge value is legitimate.
    geometry = replace(geometry, s=jnp.linspace(.1, 1., 5))
    density = 1.7e-6*(1-geometry.s**5)
    def build(angles):
        return physical_potential_problem_from_boozer_geometry(geometry, [[2, 1], [3, 1]],
            mass_density=density, theta_points=angles, zeta_points=angles)
    coarse, fine = build(12), build(24)
    np.testing.assert_allclose(coarse.stiffness, fine.stiffness, rtol=3e-6, atol=1e-8)
    np.testing.assert_allclose(coarse.mass, fine.mass, rtol=3e-6, atol=1e-18)
    assert np.linalg.eigvalsh(np.asarray(fine.mass))[0] > 0
    with pytest.raises(ValueError, match='Density'):
        slab(np.linspace(.1, 1., 5), density=-1.)


def test_cosine_sector_equals_explicit_positive_negative_complex_harmonics():
    geometry = synthetic_geometry()
    labels = np.array([[2, 1], [3, 1]])
    kwargs = dict(mass_density=1.7e-6, theta_points=16, zeta_points=16)
    real_problem = physical_potential_problem_from_boozer_geometry(geometry, labels,
        angular_basis='cosine', **kwargs)
    complex_problem = physical_potential_problem_from_boozer_geometry(geometry,
        np.vstack([labels, -labels]), **kwargs)
    # cos(phase)=(exp(i phase)+exp(-i phase))/2, independently of geometry.
    transform = np.kron(np.eye(len(geometry.s)-2), .5*np.vstack([np.eye(2), np.eye(2)]))
    for actual, full in ((real_problem.stiffness, complex_problem.stiffness),
                         (real_problem.mass, complex_problem.mass)):
        np.testing.assert_allclose(actual, transform.T@np.asarray(full)@transform,
                                   rtol=1e-12, atol=1e-18)
    assert not np.iscomplexobj(np.asarray(real_problem.stiffness))
    result = solve_physical_potential_eigenmodes(real_problem)
    assert np.max(result.residuals) < 1e-11


def circular_geometry():
    """Circular torus embedding with constant manufactured wave coefficients."""
    s = jnp.linspace(.4, .9, 5)
    zero = jnp.zeros_like(s)
    zeros = jnp.zeros((len(s), 4))
    return replace(synthetic_geometry(), s=s,
        rmnc=jnp.column_stack([4+zero, .5*jnp.sqrt(s), zero, zero]), rmns=zeros,
        zmns=jnp.column_stack([zero, .5*jnp.sqrt(s), zero, zero]), zmnc=zeros,
        pmnc=zeros, pmns=zeros,
        bmnc=jnp.column_stack([3+zero, zero, zero, zero]), bmns=zeros,
        iota=zero, boozer_i=zero, boozer_g=12+zero)


def test_full_domain_axis_integral_and_harmonic_boundary_conditions():
    geometry = circular_geometry()
    nodes = np.linspace(0., 1., 5)
    density = 1.7e-6
    kwargs = dict(radial_nodes=nodes, boundary_condition='axis', mass_density=density,
                  theta_points=16, zeta_points=16, angular_basis='cosine')
    # For the m=n=0 axis hat in a circular torus, the independently integrated
    # angular-average inertia is mu0*rho/B^2 * R_major. The polar singularity
    # cancels: J*g^ss=2*R*s; integrate s over the first cell / ds^2.
    radial = physical_potential_problem_from_boozer_geometry(geometry, [[0, 0]], **kwargs)
    assert radial.mass[0, 0] == pytest.approx(MU0*density/9*4, rel=2e-13)
    assert radial.active_dofs.tolist() == [0, 1, 2, 3]
    assert np.all(np.isfinite(radial.mass))

    # Constant k_parallel gives slab dispersion even in the polar metric.
    # m=0 retains its axis DOF; m=1 does not. No axis metric is sampled.
    problem = physical_potential_problem_from_boozer_geometry(geometry, [[0, 1], [1, 2]], **kwargs)
    assert problem.active_dofs.tolist() == [0, 2, 3, 4, 5, 6, 7]
    result = solve_physical_potential_eigenmodes(problem)
    expected = np.r_[np.full(4, 1.), np.full(3, 4.)]*9/(16*MU0*density)
    np.testing.assert_allclose(result.omega2, expected, rtol=3e-12)
    np.testing.assert_array_equal(result.mode_structures[:, 0, 1], 0.)
    np.testing.assert_array_equal(result.mode_structures[:, -1], 0.)
    assert np.any(abs(np.asarray(result.mode_structures[:, 0, 0])) > 0)
    assert np.max(result.residuals) < 1e-11
    fixed = physical_potential_problem_from_boozer_geometry(geometry, [[0, 1], [1, 2]],
        **dict(kwargs, boundary_condition='dirichlet'))
    for free_matrix, fixed_matrix in ((problem.stiffness, fixed.stiffness), (problem.mass, fixed.mass)):
        np.testing.assert_allclose(np.asarray(free_matrix)[1:, 1:], fixed_matrix, rtol=2e-13, atol=1e-20)


def test_half_mesh_interpolation_and_regular_axis_extension_derivatives():
    s = jnp.array([.1, .3, .5, .7])
    values = jnp.column_stack([2+3*s, .4*jnp.sqrt(s), .2*s])
    target = jnp.array([.025, .2, .4, .8])
    sampled, derivative = _sample_radial(s, values, target, jnp.array([0, 1, 2]))
    np.testing.assert_allclose(sampled[1:3], (np.asarray(values[:-2])+values[1:-1])/2, rtol=2e-14)
    np.testing.assert_allclose(sampled[0], [2+.075, .4*np.sqrt(.025), .2*.025], rtol=2e-14)
    np.testing.assert_allclose(derivative[0], [3., .2/np.sqrt(.025), .2], rtol=2e-14)
    tangent = jax.jvp(lambda t: _sample_radial(s, values, t, jnp.array([0, 1, 2]))[0],
                      (target,), (jnp.ones_like(target),))[1]
    np.testing.assert_allclose(tangent, derivative, rtol=2e-14)


def test_full_domain_operator_has_finite_jit_reverse_geometry_gradient():
    geometry = synthetic_geometry()
    labels = np.array([[0, 1], [2, 1]])

    def objective(parameter):
        altered = replace(geometry, rmnc=geometry.rmnc.at[:, 2].multiply(1+parameter))
        problem = physical_potential_problem_from_boozer_geometry(altered, labels,
            mass_density=1.7e-6*(1-.3*geometry.s), theta_points=12, zeta_points=12,
            radial_nodes=np.linspace(0., 1., 5), boundary_condition='axis')
        return generalized_hermitian_isolated_eigenvalue(problem.stiffness, problem.mass, 0)

    derivative = jax.jit(jax.grad(objective))(.1)
    step = 1e-4
    finite_difference = (objective(.1+step)-objective(.1-step))/(2*step)
    assert np.isfinite(derivative) and abs(derivative) > 1.
    assert derivative == pytest.approx(finite_difference, rel=2e-6)
    with pytest.raises(ValueError, match='span 0..1'):
        physical_potential_problem_from_boozer_geometry(geometry, labels, mass_density=1.,
            radial_nodes=np.linspace(.01, .99, 5), boundary_condition='axis')
