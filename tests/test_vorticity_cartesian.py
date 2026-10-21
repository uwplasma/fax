"""Check the weak form by finite differences in physical Cartesian space.

This manufactured toroidal embedding and directional field are not an MHD
equilibrium. The check tests the stated differential operator independently
of its Boozer-coordinate expansion, including magnetic shear and coupling
between different angular harmonics.
"""
import numpy as np
import pytest

from fax.boozer import BoozerGeometry
from fax.units import MU0
from fax.vorticity import physical_potential_problem_from_boozer_geometry


def test_cartesian_nested_differences_reproduce_coupled_sheared_bending_energy():
    nodes = np.linspace(.1, .9, 5)
    ds = np.diff(nodes)
    gauss = nodes[:-1, None]+ds[:, None]*(1+np.array([-1., 1.])/np.sqrt(3.))/2
    # Store analytic embedding coefficients symmetrically around the Gauss
    # samples. Their linear interpolation and slope approach the analytic
    # embedding without a separate interpolating derivative in the reference.
    source_s = np.sort(np.r_[gauss.ravel()-1e-5, gauss.ravel()+1e-5])
    r_major, a, elongation, q, field, density = 4., .5, 1.7, .25, 2., 1.7e-6
    zeros = np.zeros((len(source_s), 2))
    geometry = BoozerGeometry(source='manufactured toroidal mapping', s=source_s,
        xm=np.array([0., 1.]), xn=np.array([0., 0.]),
        rmnc=np.column_stack([r_major+0*source_s, a*np.sqrt(source_s)]), rmns=zeros,
        zmns=np.column_stack([0*source_s, elongation*a*np.sqrt(source_s)]), zmnc=zeros,
        pmnc=zeros, pmns=zeros, gmnc=zeros, gmns=zeros,
        bmnc=np.column_stack([field+0*source_s, 0*source_s]), bmns=zeros,
        iota=.3+.4*source_s, boozer_i=0*source_s,
        boozer_g=field/q+0*source_s, field_periods=1)
    labels = np.array([[2, 1], [4, 1]])
    values = np.array([[0., 0.], [.8, .2], [.5, -.4], [.3, .7], [0., 0.]])
    problem = physical_potential_problem_from_boozer_geometry(geometry, labels,
        mass_density=density, radial_nodes=nodes, angular_basis='cosine',
        theta_points=24, zeta_points=24)
    vector = values[1:-1].ravel()
    actual_k = np.vdot(vector, np.asarray(problem.stiffness)@vector).real
    actual_m = np.vdot(vector, np.asarray(problem.mass)@vector).real
    theta, zeta = np.meshgrid(2*np.pi*np.arange(24)/24, 2*np.pi*np.arange(24)/24, indexing='ij')
    target_s = np.repeat(gauss.ravel(), theta.size)
    angle, toroidal = np.tile(theta.ravel(), gauss.size), np.tile(zeta.ravel(), gauss.size)
    radius = r_major+a*np.sqrt(target_s)*np.cos(angle)
    xyz = np.column_stack([radius*np.cos(toroidal), radius*np.sin(toroidal),
                           elongation*a*np.sqrt(target_s)*np.sin(angle)])
    weights = np.repeat(np.repeat(ds/2, 2), theta.size)/theta.size
    jac = elongation*a*a*radius/2

    def inverse(points):
        radius = np.hypot(points[:, 0], points[:, 1])
        s = ((radius-r_major)/a)**2+(points[:, 2]/(elongation*a))**2
        theta = np.arctan2(points[:, 2]/elongation, radius-r_major)
        zeta = np.arctan2(points[:, 1], points[:, 0])
        return radius, s, theta, zeta

    def potential(points):
        _, s, theta, zeta = inverse(points)
        amplitudes = np.column_stack([np.interp(s, nodes, values[:, j]) for j in range(2)])
        return np.sum(amplitudes*np.cos(theta[:, None]*labels[:, 0]-zeta[:, None]*labels[:, 1]), axis=1)

    def directional_field(points):
        radius, s, _, _ = inverse(points)
        # b=q*(iota*r_theta+r_zeta), written directly in Cartesian components.
        et = np.column_stack([-points[:, 2]/elongation*points[:, 0]/radius,
                              -points[:, 2]/elongation*points[:, 1]/radius,
                              elongation*(radius-r_major)])
        ez = np.column_stack([-points[:, 1], points[:, 0], np.zeros(len(points))])
        return q*((.3+.4*s)[:, None]*et+ez)

    def gradient(function, points, step):
        return np.column_stack([(function(points+step*np.eye(3)[j])
                                -function(points-step*np.eye(3)[j]))/(2*step) for j in range(3)])

    estimates = []
    for step in (2e-4, 1e-4):
        def parallel(points):
            return np.sum(directional_field(points)*gradient(potential, points, step), axis=1)
        grad_phi = gradient(potential, xyz, step)
        grad_parallel = gradient(parallel, xyz, step)
        expected_k = np.sum(weights*jac*np.sum(grad_parallel**2, axis=1))
        expected_m = np.sum(weights*jac*MU0*density/field**2*np.sum(grad_phi**2, axis=1))
        estimates.append((expected_k, expected_m))
    coarse, fine = np.array(estimates)
    assert abs(fine[0]-coarse[0])/fine[0] < 3e-5
    assert actual_k == pytest.approx(fine[0], rel=2e-5)
    assert actual_m == pytest.approx(fine[1], rel=2e-6)
