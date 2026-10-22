"""Verify full FAX assembly against an exact, sheared 3-D manufactured solution.

The continuous weak load uses elementary Cartesian derivatives, not FAX's
expanded coefficients or a matrix applied to exact nodal values. The field
is a manufactured direction, not a unit-vector MHD equilibrium. The index
controls reproduce only the two audited AE3D-like substitutions.
"""
import argparse
import csv
import os
from pathlib import Path
import threading

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss

from fax.units import MU0
from fax.vorticity import _angular_phase, _assemble_problem, _element_matrices

jax.config.update('jax_enable_x64', True)
LABELS = np.array([[2, 1], [3, 3]])
VARIANTS = ('native', 'radial_swap', 'mixed_swap', 'both_swaps')
Q, FIELD, DENSITY = .25, 2., 1.7e-6
SHIFT = Q**2*FIELD**2/(MU0*DENSITY)
REFERENCE = Path(__file__).with_name('references')/'vorticity_benchmark.csv'


def jacobian(points):
    """Elementary derivatives of (R cos zeta, R sin zeta, Z)."""
    s, t, z = np.asarray(points).T
    root, phase = np.sqrt(s), 2*t-2*z
    cs, sn, ct, st, cz, sz = np.cos(phase), np.sin(phase), np.cos(t), np.sin(t), np.cos(z), np.sin(z)
    radius = 4+.5*root*(ct+.16*cs)
    rs, rt, rz = .25/root*(ct+.16*cs), .5*root*(-st-.32*sn), .16*root*sn
    zs, zt, zz = .425/root*(st+.16*sn), .85*root*(ct+.32*cs), -.272*root*cs
    return np.stack((np.stack((rs*cz, rt*cz, rz*cz-radius*sz), -1),
                     np.stack((rs*sz, rt*sz, rz*sz+radius*cz), -1),
                     np.stack((zs, zt, zz), -1)), -2)


def derivatives(points, amplitudes, slopes, labels, shear):
    """Analytical phi, grad_q(phi), grad_q(D phi); D=Q(iota*d_theta+d_zeta)."""
    s, theta, zeta = points.T
    m, n = labels.T
    cs, sn = np.cos(theta[:, None]*m-zeta[:, None]*n), np.sin(theta[:, None]*m-zeta[:, None]*n)
    h = (.3+shear*s[:, None])*m-n
    grad = np.stack((slopes*cs, -amplitudes*m*sn, amplitudes*n*sn), -1)
    grad_d = np.stack((-Q*(h*slopes+shear*m*amplitudes)*sn,
                       -Q*h*m*amplitudes*cs, Q*h*n*amplitudes*cs), -1)
    return amplitudes*cs, grad, grad_d


def reference_element(left, right, shear, theta, zeta):
    gauss, weights = leggauss(8)
    radius = left+(right-left)*(gauss+1)/2
    points = np.column_stack((np.repeat(radius, theta.size), np.tile(theta.ravel(), 8), np.tile(zeta.ravel(), 8)))
    s, width = points[:, 0], right-left
    coordinate_map = jacobian(points)
    inverse = np.linalg.inv(coordinate_map)
    weight = np.repeat(weights*width/2, theta.size)/theta.size*abs(np.linalg.det(coordinate_map))
    x = (s-.1)/.8
    amplitudes = np.column_stack((np.sin(np.pi*x), .7*np.sin(2*np.pi*x)+.2*np.sin(3*np.pi*x)))
    slopes = np.column_stack((np.pi*np.cos(np.pi*x), 1.4*np.pi*np.cos(2*np.pi*x)+.6*np.pi*np.cos(3*np.pi*x)))/.8
    phi, gp, gdp = derivatives(points, amplitudes, slopes, LABELS, shear)
    phi = phi.sum(axis=1)
    gp = np.einsum('pa,pab->pb', gp.sum(axis=1), inverse)
    gdp = np.einsum('pa,pab->pb', gdp.sum(axis=1), inverse)
    shape = np.column_stack(((right-s)/width, (s-left)/width))
    v, gv, gdv = derivatives(points, np.repeat(shape, 2, axis=1),
        np.broadcast_to(np.repeat([-1/width, 1/width], 2), (len(s), 4)), np.tile(LABELS, (2, 1)), shear)
    gv, gdv = (np.einsum('pka,pab->pkb', a, inverse) for a in (gv, gdv))
    matrix = np.einsum('pki,pli->pkl', gdv, gdv)+Q**2*np.einsum('pki,pli->pkl', gv, gv)
    force = np.einsum('pki,pi->pk', gdv, gdp)+Q**2*np.einsum('pki,pi->pk', gv, gp)
    norms = np.column_stack((np.sum(gdp*gdp, axis=1)+Q**2*np.sum(gp*gp, axis=1), phi**2))
    return tuple(np.einsum('p,p...->...', weight, a) for a in
                 (matrix, force, v[:, :, None]*v[:, None, :], v*phi[:, None], norms))


def native_problem(nodes, shear, theta, zeta):
    """Unchanged production kernel and scatter; exact geometry at its Gauss points."""
    widths = np.diff(nodes)
    radius = nodes[:-1, None]+widths[:, None]*(1+np.array([-1., 1.])/np.sqrt(3))/2
    points = np.column_stack((np.repeat(radius.ravel(), theta.size),
                             np.tile(theta.ravel(), radius.size), np.tile(zeta.ravel(), radius.size)))
    basis = jacobian(points).swapaxes(-1, -2).reshape(len(widths), 2, theta.size, 3, 3)
    b = np.full((len(widths), 2, theta.size), FIELD)
    profile = np.zeros_like(radius)
    samples = (b, np.zeros_like(b), np.zeros((*b.shape, 2)), basis, .3+shear*radius,
               profile+shear, profile, profile, profile+FIELD/Q, profile, profile+DENSITY)
    phase, dphase = _angular_phase(theta, zeta, jnp.asarray(LABELS), 'cosine')
    quadrature = tuple(jnp.asarray(a) for a in samples)

    def element(index):
        return _element_matrices(jnp.asarray(widths)[index], None, None, None, None, None, None, None,
                                 jnp.asarray(LABELS), phase, dphase, tuple(a[index] for a in quadrature))
    return _assemble_problem(jnp.asarray(nodes), jnp.asarray(LABELS), element, 'cosine'), samples


def index_deltas(nodes, shear, theta, zeta, samples, active):
    """Two isolated source-minus-native coefficient changes; same quadrature and mass."""
    radial, mixed = (np.zeros((2*len(nodes), 2*len(nodes))) for _ in range(2))
    m, n = LABELS.T
    angle = theta.ravel()[:, None]*m-zeta.ravel()[:, None]*n
    sn, cs = np.sin(angle), np.cos(angle)
    for e, width in enumerate(np.diff(nodes)):
        dofs = np.arange(2*e, 2*e+4)
        for g, xi in enumerate((-1/np.sqrt(3), 1/np.sqrt(3))):
            shape, slope = np.array([(1-xi)/2, (1+xi)/2]), np.array([-1., 1.])/width
            basis = samples[3][e, g]
            metric, jac = np.linalg.inv(basis@basis.swapaxes(-1, -2)), abs(np.linalg.det(basis))
            kappa = n-m*samples[4][e, g]
            ss = np.einsum('p,pi,pj->ij', jac*metric[:, 0, 0]*Q**2, sn, sn)/theta.size
            sc = np.einsum('p,pi,pj->ij', jac*metric[:, 0, 1]*Q**2, sn, cs)/theta.size
            db = -shear*(kappa[:, None]*m-m[:, None]*kappa)*ss
            dd = -shear*(m[:, None]*m)*(kappa[:, None]-kappa)*(sc-sc.T)
            radial[np.ix_(dofs, dofs)] += width/2*(np.einsum('a,b,ij->aibj', shape, slope, db)
                +np.einsum('a,b,ij->aibj', slope, shape, db.T)).reshape(4, 4)
            mixed[np.ix_(dofs, dofs)] += width/2*np.einsum('a,b,ij->aibj', shape, shape, dd).reshape(4, 4)
    return tuple(a[np.ix_(active, active)] for a in (radial, mixed))


def benchmark_case(count, shear=.8, angles=32):
    nodes = np.linspace(.1, .9, count)
    theta, zeta = np.meshgrid(2*np.pi*np.arange(angles)/angles, 2*np.pi*np.arange(angles)/angles, indexing='ij')
    problem, samples = native_problem(nodes, shear, theta, zeta)
    size = 2*count
    matrix, mass = np.zeros((size, size)), np.zeros((size, size))
    force, projection, norms = np.zeros(size), np.zeros(size), np.zeros(2)
    for e, (left, right) in enumerate(zip(nodes[:-1], nodes[1:])):
        a, f, m, p, norm = reference_element(left, right, shear, theta, zeta)
        dofs = np.arange(2*e, 2*e+4)
        matrix[np.ix_(dofs, dofs)] += a
        mass[np.ix_(dofs, dofs)] += m
        force[dofs] += f
        projection[dofs] += p
        norms += norm
    active = np.asarray(problem.active_dofs)
    force, projection = force[active], projection[active]
    matrix, mass = (a[np.ix_(active, active)] for a in (matrix, mass))
    native = np.asarray(problem.stiffness)+SHIFT*np.asarray(problem.mass)
    dr, dm = index_deltas(nodes, shear, theta, zeta, samples, active)
    rows = []
    for name, delta in zip(VARIANTS, (0., dr, dm, dr+dm)):
        # Use NumPy's existing dense solve; no new solver implementation.
        values = np.linalg.solve(native+delta, force)
        l2 = (values@mass@values-2*values@projection+norms[1])/norms[1]
        energy = (values@matrix@values-2*values@force+norms[0])/norms[0]
        if min(l2, energy) < -1e-10:
            raise ArithmeticError('Negative independently integrated squared error')
        rows.append(dict(nodes=count, shear=shear, variant=name, l2_error=np.sqrt(max(l2, 0)),
                         energy_error=np.sqrt(max(energy, 0)), weak_residual=np.linalg.norm(matrix@values-force)/np.linalg.norm(force)))
    return rows


def plot(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    labels = ('Native FAX', 'Radial index swap', 'Mixed-metric index swap', 'Both swaps (AE3D-like)')
    for name, label in zip(VARIANTS, labels):
        selected = [r for r in rows if r['variant'] == name and float(r['shear']) != 0]
        for ax, metric in zip(axes, ('l2_error', 'weak_residual')):
            ax.loglog([int(r['nodes'])-1 for r in selected], [float(r[metric]) for r in selected], 'o-', label=label)
    for ax, title in zip(axes, ('Error against exact potential', 'Independent weak residual')):
        ax.set(xlabel='Radial elements', ylabel='Relative error', title=title)
        ax.grid(which='both', alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Exact manufactured solution: nonzero shear, 3-D geometry, fixed analytical forcing')
    fig.savefig(output/'vorticity_benchmark.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nodes', type=int, nargs='+', default=[9, 17, 33, 65, 129])
    parser.add_argument('--output', type=Path, default=Path('outputs/vorticity_benchmark'))
    parser.add_argument('--plot-only', action='store_true', help='Plot the checked-in CSV without running a solve')
    args = parser.parse_args()
    if len(args.nodes) < 3 or min(args.nodes) < 4 or any(b <= a for a, b in zip(args.nodes[:-1], args.nodes[1:])):
        parser.error('Use at least three increasing meshes with >=4 nodes')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        with REFERENCE.open() as stream:
            plot(list(csv.DictReader(stream)), args.output)
        return
    timer = threading.Timer(50, lambda: os._exit(124))
    timer.daemon = True
    timer.start()
    rows = benchmark_case(args.nodes[0], shear=0)
    for count in args.nodes:
        result = benchmark_case(count)
        rows.extend(result)
        print(f"nodes={count}: " + ', '.join(f"{r['variant']}={r['l2_error']:.6g}" for r in result), flush=True)
    with (args.output/'vorticity_benchmark.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, args.output)
    timer.cancel()


if __name__ == '__main__':
    main()
