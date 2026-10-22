"""Independent Cartesian finite differences on a manufactured helical map."""
import numpy as np

from fax.boozer import BoozerGeometry
from fax.units import MU0
from fax.vorticity import physical_potential_problem_from_boozer_geometry


def test_helical_cartesian_energy_with_nonzero_shear():
    shear = .4
    nodes = np.linspace(.1, .9, 5)
    widths = np.diff(nodes)
    samples = nodes[:-1, None]+widths[:, None]*(1+np.array([-1., 1.])/np.sqrt(3.))/2
    source = np.sort(np.r_[samples.ravel()-1e-5, samples.ravel()+1e-5])
    major, minor, elongation, helicity, q = 4., .5, 1.7, .04, .25
    field, density = 2., 1.7e-6
    zeros = np.zeros((len(source), 3))
    geometry = BoozerGeometry(source='manufactured helical embedding', s=source,
        xm=np.array([0., 1., 2.]), xn=np.array([0., 0., 2.]),
        rmnc=np.column_stack([major+0*source, minor*np.sqrt(source),
                              helicity*minor*np.sqrt(source)]), rmns=zeros,
        zmns=np.column_stack([0*source, elongation*minor*np.sqrt(source),
                              helicity*elongation*minor*np.sqrt(source)]), zmnc=zeros,
        pmnc=zeros, pmns=zeros, gmnc=zeros, gmns=zeros,
        bmnc=np.column_stack([field+0*source, 0*source, 0*source]), bmns=zeros,
        iota=.3+shear*source, boozer_i=0*source,
        boozer_g=field/q+0*source, field_periods=2)
    # Different toroidal numbers in the same Nfp=2 family. Their coupling
    # requires the helical geometry. Independent radial shapes expose the
    # antisymmetric mixed derivative coefficient.
    labels = np.array([[2, 1], [3, 3]])
    values = np.array([[0., 0.], [.8, .2], [.5, -.4], [.3, .7], [0., 0.]])
    problem = physical_potential_problem_from_boozer_geometry(geometry, labels,
        mass_density=density, radial_nodes=nodes, angular_basis='cosine',
        theta_points=32, zeta_points=32)
    stiffness, mass = np.asarray(problem.stiffness), np.asarray(problem.mass)
    vector = values[1:-1].ravel()
    native_k, native_m = vector@stiffness@vector, vector@mass@vector
    coupling = np.linalg.norm(stiffness[0::2, 1::2])/np.linalg.norm(stiffness)
    assert coupling > 1e-4

    theta, zeta = np.meshgrid(2*np.pi*np.arange(32)/32,
                              2*np.pi*np.arange(32)/32, indexing='ij')
    flux = np.repeat(samples.ravel(), theta.size)
    angle, toroidal = np.tile(theta.ravel(), samples.size), np.tile(zeta.ravel(), samples.size)
    phase = 2*angle-2*toroidal
    radius = major+minor*np.sqrt(flux)*(np.cos(angle)+helicity*np.cos(phase))
    height = elongation*minor*np.sqrt(flux)*(np.sin(angle)+helicity*np.sin(phase))
    xyz = np.column_stack([radius*np.cos(toroidal), radius*np.sin(toroidal), height])
    jac = elongation*minor**2*radius/2*(1+3*helicity*np.cos(angle-2*toroidal)+2*helicity**2)
    weights = np.repeat(np.repeat(widths/2, 2), theta.size)/theta.size

    def inverse(points):
        radius = np.hypot(points[:, 0], points[:, 1])
        toroidal = np.arctan2(points[:, 1], points[:, 0])
        x, y = (radius-major)/minor, points[:, 2]/(elongation*minor)
        radial, angle = np.hypot(x, y), np.arctan2(y, x)
        for _ in range(7):
            phase = 2*angle-2*toroidal
            cx, cy = np.cos(angle)+helicity*np.cos(phase), np.sin(angle)+helicity*np.sin(phase)
            dx, dy = -np.sin(angle)-2*helicity*np.sin(phase), np.cos(angle)+2*helicity*np.cos(phase)
            rx, ry = radial*cx-x, radial*cy-y
            determinant = radial*(cx*dy-dx*cy)
            radial, angle = (radial-(rx*radial*dy-ry*radial*dx)/determinant,
                              angle-(cx*ry-cy*rx)/determinant)
        return radius, radial**2, angle, toroidal

    def potential(points):
        _, flux, angle, toroidal = inverse(points)
        amplitudes = np.column_stack([np.interp(flux, nodes, values[:, j]) for j in range(2)])
        return np.sum(amplitudes*np.cos(angle[:, None]*labels[:, 0]
                                       -toroidal[:, None]*labels[:, 1]), axis=1)

    def directional_field(points):
        radius, flux, angle, toroidal = inverse(points)
        radial = minor*np.sqrt(flux)
        phase = 2*angle-2*toroidal
        r_theta = radial*(-np.sin(angle)-2*helicity*np.sin(phase))
        z_theta = elongation*radial*(np.cos(angle)+2*helicity*np.cos(phase))
        r_zeta = 2*helicity*radial*np.sin(phase)
        z_zeta = -2*elongation*helicity*radial*np.cos(phase)
        r_component = (.3+shear*flux)*r_theta+r_zeta
        z_component = (.3+shear*flux)*z_theta+z_zeta
        return q*np.column_stack([r_component*np.cos(toroidal)-radius*np.sin(toroidal),
                                   r_component*np.sin(toroidal)+radius*np.cos(toroidal),
                                   z_component])

    def gradient(function, points, step):
        return np.column_stack([(function(points+step*np.eye(3)[j])
                                -function(points-step*np.eye(3)[j]))/(2*step) for j in range(3)])

    estimates = []
    for step in (2e-4, 1e-4):
        def parallel(points):
            return np.sum(directional_field(points)*gradient(potential, points, step), axis=1)
        grad, grad_parallel = gradient(potential, xyz, step), gradient(parallel, xyz, step)
        estimates.append((np.sum(weights*jac*np.sum(grad_parallel**2, axis=1)),
                          np.sum(weights*jac*MU0*density/field**2*np.sum(grad**2, axis=1))))
    coarse, fine = np.asarray(estimates)
    k_error, m_error = abs(np.array([native_k, native_m])-fine)/abs(fine)
    assert abs(fine[0]-coarse[0])/fine[0] < 3e-5
    assert k_error < 2e-5
    assert m_error < 2e-6

