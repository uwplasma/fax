"""Physical-potential vorticity (Knyazev et al., arXiv:2603.03118, eq. 4).

K_ij=int grad(b.grad f_i)*.grad(b.grad f_j); M_ij=int mu0*rho/B**2 grad(f_i)*.grad(f_j).
P1 radial elements, two-point Gauss and periodic angular quadrature; distinct
from the reduced radial envelope model in ``global_modes``.
"""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from .boozer import (
    BoozerGeometry, _radial_derivative, boozer_angle_grid,
    reconstruct_boozer_modes, reconstruct_boozer_modes_with_derivatives,
)
from .global_modes import RadialBlockEigenmodeResult
from .metric import generalized_eigen_residual_norms, generalized_hermitian_eigensystem
from .units import MU0


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PhysicalPotentialProblem:
    s: jax.Array
    harmonics: jax.Array
    stiffness: jax.Array
    mass: jax.Array
    active_dofs: jax.Array
    metadata: tuple = ()

    def tree_flatten(self):
        return (self.s, self.harmonics, self.stiffness, self.mass, self.active_dofs), self.metadata

    @classmethod
    def tree_unflatten(cls, metadata, children):
        return cls(*children, metadata=metadata)


@jax.jit
def _element_matrices(ds, field, field_angles, basis, iota, current_i, current_g,
                      density, harmonics, phase, phase_derivative, quadrature=None):
    """One two-node radial element, including derivatives of k_parallel."""
    nh = harmonics.shape[0]
    size = 2*nh
    k = jnp.zeros((size, size), dtype=jnp.result_type(phase, field if quadrature is None else quadrature[0]))
    mass = jnp.zeros_like(k)
    m, n = harmonics[:, 0], harmonics[:, 1]
    dshape = jnp.array([-1., 1.])/ds
    if quadrature is None:
        db = (field[1]-field[0])/ds
        diota = (iota[1]-iota[0])/ds
        di = (current_i[1]-current_i[0])/ds
        dg = (current_g[1]-current_g[0])/ds
    # The angular integral's common domain-size factor cancels in K/M.
    for q, xi in enumerate((-1/jnp.sqrt(3.), 1/jnp.sqrt(3.))):
        shape = jnp.array([(1-xi)/2, (1+xi)/2])
        if quadrature is None:
            b = jnp.einsum('a,ap->p', shape, field)
            b_angles = jnp.einsum('a,apd->pd', shape, field_angles)
            covariant = jnp.einsum('a,apij->pij', shape, basis)
            rot = shape @ iota
            ci, cg, rho = shape @ current_i, shape @ current_g, shape @ density
        else:
            b, db, b_angles, covariant, rot, diota, ci, di, cg, dg, rho = (p[q] for p in quadrature)
        jacobian = jnp.abs(jnp.linalg.det(covariant))
        metric = jnp.linalg.inv(covariant @ jnp.swapaxes(covariant, -1, -2))
        denominator = cg+rot*ci
        ddenominator = dg+diota*ci+rot*di
        l = rot*m-n
        h = b[:, None]*l/denominator
        dh_s = (db[:, None]*l+b[:, None]*diota*m)/denominator-h*ddenominator/denominator
        dh_theta = b_angles[:, 0, None]*l/denominator
        dh_zeta = b_angles[:, 1, None]*l/denominator
        # Flatten in node-major, then harmonic order, matching global DOFs.
        grad = jnp.stack([
            jnp.broadcast_to(phase[:, None, :]*dshape[None, :, None], (len(b), 2, nh)),
            phase_derivative[:, None, :]*m[None, None, :]*shape[None, :, None],
            phase_derivative[:, None, :]*(-n)[None, None, :]*shape[None, :, None],
        ], axis=1).reshape(len(b), 3, size)
        grad_parallel = jnp.stack([
            phase_derivative[:, None, :]*(h[:, None, :]*dshape[None, :, None]
                               + dh_s[:, None, :]*shape[None, :, None]),
            (phase_derivative*dh_theta-phase*m*h)[:, None, :]*shape[None, :, None],
            (phase_derivative*dh_zeta+phase*n*h)[:, None, :]*shape[None, :, None],
        ], axis=1).reshape(len(b), 3, size)
        weight = jacobian*ds/(2*len(b))
        inertia_weight = weight*MU0*rho/b**2
        k += jnp.einsum('pai,pab,pbj,p->ij', grad_parallel.conj(), metric, grad_parallel, weight)
        mass += jnp.einsum('pai,pab,pbj,p->ij', grad.conj(), metric, grad, inertia_weight)
    return k, mass


def _angular_phase(theta, zeta, labels, angular_basis):
    angle = (jnp.asarray(theta).ravel()[:, None]*labels[:, 0]
             - jnp.asarray(zeta).ravel()[:, None]*labels[:, 1])
    if angular_basis == 'complex':
        phase = jnp.exp(1j*angle)
        return phase, 1j*phase
    if angular_basis == 'cosine':
        return jnp.cos(angle), -jnp.sin(angle)
    raise ValueError('angular_basis must be complex or cosine')


def _assemble_problem(radial, labels, element, angular_basis, axis_indices=()):
    """Scatter element matrices using the same DOF map as mode restoration."""
    ns, nh = len(radial), len(labels)
    ke, me = jax.lax.map(element, jnp.arange(ns-1))
    active = jnp.concatenate([jnp.asarray(axis_indices, dtype=int), jnp.arange(nh, (ns-1)*nh)])
    size = len(active)
    mapping = jnp.full(ns*nh, size).at[active].set(jnp.arange(size))
    dofs = mapping[jnp.arange(ns-1)[:, None]*nh+jnp.arange(2*nh)[None, :]]
    rows, cols = dofs[:, :, None], dofs[:, None, :]
    k = jnp.zeros((size, size), dtype=ke.dtype).at[rows, cols].add(ke, mode='drop')
    mass = jnp.zeros_like(k).at[rows, cols].add(me, mode='drop')
    return PhysicalPotentialProblem(radial, labels, (k+k.conj().T)/2, (mass+mass.conj().T)/2,
        active, metadata=(
            ('operator', 'physical_potential_vorticity'), ('angular_basis', angular_basis),
            ('weak_form', 'K=int|grad(b.grad Phi)|^2; M=int(mu0*rho/B^2)|grad Phi|^2'),
            ('radial_elements', 'linear; two-point Gauss'),
        ))


def physical_potential_problem_from_grid(
    s, harmonics, *, theta, zeta, magnetic_field, magnetic_field_angle_derivatives,
    covariant_basis, iota, boozer_i, boozer_g, mass_density, angular_basis='complex',
):
    """Assemble on a straight-field-line grid with linearly interpolated data.

    ``covariant_basis[..., a, :]`` is dr/dq^a in an orthonormal frame,
    q=(s,theta,zeta); angular B derivatives end in (dB/dtheta,dB/dzeta).
    B and G+iota*I must use the same convention. ``angular_basis='cosine'``
    restricts the spatial parity; complex coefficients are unrestricted.
    """
    radial = jnp.asarray(s)
    labels = jnp.asarray(harmonics)
    ns, nh = len(radial), len(labels)
    if ns < 4 or labels.ndim != 2 or labels.shape[1] != 2:
        raise ValueError('Need at least four radial nodes and (m,n) harmonic pairs')
    if radial.ndim != 1:
        raise ValueError('Radial nodes must be one-dimensional')
    if not isinstance(radial, jax.core.Tracer):
        if not np.isfinite(np.asarray(radial)).all() or np.any(np.diff(np.asarray(radial)) <= 0):
            raise ValueError('Radial nodes must be finite and strictly increasing')
    b = jnp.asarray(magnetic_field).reshape(ns, -1)
    b_angles = jnp.asarray(magnetic_field_angle_derivatives).reshape(ns, -1, 2)
    basis = jnp.asarray(covariant_basis).reshape(ns, -1, 3, 3)
    phase, phase_derivative = _angular_phase(theta, zeta, labels, angular_basis)
    if basis.shape[1] != phase.shape[0] or b.shape != basis.shape[:2] or b_angles.shape[:2] != b.shape:
        raise ValueError('Angular sample shapes disagree')
    profiles = [jnp.broadcast_to(jnp.asarray(p), (ns,))
                for p in (iota, boozer_i, boozer_g, mass_density)]
    if not isinstance(b, jax.core.Tracer) and np.any(np.asarray(b) <= 0):
        raise ValueError('B must be positive')
    if not isinstance(profiles[-1], jax.core.Tracer):
        rho = np.asarray(profiles[-1])
        if np.any(rho < 0) or np.any(rho[:-1]+rho[1:] <= 0):
            raise ValueError('Density must be nonnegative and nonzero on each element')
    # Compiled sequential element loop: bounded quadrature memory, no Python
    # unrolling with radial resolution, and no full angular pair-phase tensor.
    def assemble_element(index):
        pair = jnp.array([index, index+1])
        return _element_matrices(radial[index+1]-radial[index], b[pair], b_angles[pair],
                                 basis[pair], *(p[pair] for p in profiles), labels, phase, phase_derivative)

    problem = _assemble_problem(radial, labels, assemble_element, angular_basis)
    return replace(problem, metadata=problem.metadata+(
        ('boundary_condition', 'dirichlet at supplied endpoints'),
    ))


def _sample_radial(s, values, target, poloidal_modes=None):
    """Linear sampling/extrapolation and its derivative; r**|m| toward axis."""
    index = jnp.clip(jnp.searchsorted(s, target, side='right')-1, 0, len(s)-2)
    extra = (1,)*(values.ndim-1)
    width = (s[index+1]-s[index]).reshape((-1,)+extra)
    weight = ((target-s[index])/(s[index+1]-s[index])).reshape((-1,)+extra)
    derivative = (values[index+1]-values[index])/width
    sampled = values[index]+weight*(values[index+1]-values[index])
    if poloidal_modes is not None:
        exponent = jnp.abs(poloidal_modes)/2
        # An already-full source mesh needs no lower extrapolation. Avoid an
        # unused divide by zero there, including during reverse-mode AD.
        first = jnp.where(s[0] > 0, s[0], 1.)
        power = (target[:, None]/first)**exponent
        regular = values[0]*power
        use_axis = (target[:, None] < s[0]) & (exponent > 0)
        sampled = jnp.where(use_axis, regular, sampled)
        derivative = jnp.where(use_axis, regular*exponent/target[:, None], derivative)
    return sampled, derivative


def _sample_boozer_geometry(geometry, target, theta, zeta, density, offset_sign):
    def sample(values, modes=None):
        if target is None:
            return values, None if modes is None else _radial_derivative(values, geometry.s)
        return _sample_radial(geometry.s, values, target, modes)

    def scalar(cosine, sine):
        if cosine is None and sine is None:
            raise ValueError('Full embedding and B arrays are required')
        cosine = jnp.zeros_like(sine) if cosine is None else cosine
        sine = jnp.zeros_like(cosine) if sine is None else sine
        c, dc = sample(cosine, geometry.xm)
        a, da = sample(sine, geometry.xm)
        value, dt, dz = reconstruct_boozer_modes_with_derivatives(c, a,
            geometry.xm, geometry.xn, theta, zeta)
        ds = reconstruct_boozer_modes(dc, da, geometry.xm, geometry.xn, theta, zeta)
        return value, jnp.stack([ds, dt, dz], axis=-1)

    r, dr = scalar(geometry.rmnc, geometry.rmns)
    _, dz = scalar(geometry.zmnc, geometry.zmns)
    _, dp = scalar(geometry.pmnc, geometry.pmns)
    dp = (offset_sign*dp).at[..., 2].add(1.)
    b, db = scalar(geometry.bmnc, geometry.bmns)
    basis = jnp.stack([dr, r[..., None]*dp, dz], axis=-1)
    ns = len(geometry.s) if target is None else len(target)
    b, db = b.reshape(ns, -1), db.reshape(ns, -1, 3)
    basis = basis.reshape(ns, -1, 3, 3)
    rot, drot = sample(geometry.iota)
    ci, di = sample(geometry.boozer_i)
    cg, dg = sample(geometry.boozer_g)
    rho, _ = sample(density)
    return b, db[..., 0], db[..., 1:], basis, rot, drot, ci, di, cg, dg, rho


def physical_potential_problem_from_boozer_geometry(
    geometry: BoozerGeometry, harmonics, *, mass_density, theta_points=64, zeta_points=64,
    toroidal_offset_sign=1, angular_basis='complex', radial_nodes=None, boundary_condition='dirichlet',
):
    """Build from Boozer B and embedding; density is sampled on ``geometry.s``.

    Default phi=zeta+p (Fortran boozmn); toroidal_offset_sign=-1 accepts nu=zeta-phi.
    With ``radial_nodes``, reconstruct geometry at interior Gauss points.
    ``boundary_condition='axis'`` requires s=0..1 and static harmonics: m=0
    axis values are free (P1 in s gives dPhi/dr=0); other m vanish there.
    All modes vanish at the edge; no singular axis metric is evaluated.
    Geometry surface coordinates must include any half-mesh offsets.
    """
    if toroidal_offset_sign not in (-1, 1):
        raise ValueError('toroidal_offset_sign must be +1 or -1')
    if boundary_condition not in ('dirichlet', 'axis'):
        raise ValueError('boundary_condition must be dirichlet or axis')
    if boundary_condition == 'axis' and radial_nodes is None:
        raise ValueError('Axis treatment requires full-domain radial_nodes')
    theta, zeta = boozer_angle_grid(theta_points=theta_points, zeta_points=zeta_points)
    density = jnp.broadcast_to(jnp.asarray(mass_density), geometry.s.shape)

    if radial_nodes is not None:
        radial, labels = jnp.asarray(radial_nodes), jnp.asarray(harmonics)
        if radial.ndim != 1 or len(radial) < 4 or labels.ndim != 2 or labels.shape[1] != 2:
            raise ValueError('Need at least four radial nodes and (m,n) harmonic pairs')
        if not isinstance(radial_nodes, jax.core.Tracer):
            nodes = np.asarray(radial_nodes)
            if not np.isfinite(nodes).all() or np.any(np.diff(nodes) <= 0) or nodes[0] < 0 or nodes[-1] > 1:
                raise ValueError('Radial nodes must increase within 0..1')
            if boundary_condition == 'axis' and (nodes[0] != 0 or nodes[-1] != 1):
                raise ValueError('Axis radial nodes must span 0..1')
        if not isinstance(density, jax.core.Tracer) and (np.any(np.asarray(density) < 0) or not np.any(np.asarray(density) > 0)):
            raise ValueError('Density must be nonnegative and nonzero')
        axis_indices = tuple(np.flatnonzero(np.asarray(harmonics)[:, 0] == 0)) if boundary_condition == 'axis' else ()
        phase, dphase = _angular_phase(theta, zeta, labels, angular_basis)
        # Batch Fourier sampling; retain bounded harmonic contractions in lax.map.
        targets = radial[:-1, None]+jnp.diff(radial)[:, None]*(1+jnp.array([-1., 1.])/jnp.sqrt(3.))/2
        samples = _sample_boozer_geometry(geometry, targets.ravel(), theta, zeta, density, toroidal_offset_sign)
        samples = tuple(p.reshape((len(radial)-1, 2)+p.shape[1:]) for p in samples)

        def element(index):
            ds = radial[index+1]-radial[index]
            quadrature = tuple(p[index] for p in samples)
            return _element_matrices(ds, None, None, None, None, None, None, None,
                                     labels, phase, dphase, quadrature)

        problem = _assemble_problem(radial, labels, element, angular_basis, axis_indices)
        return replace(problem, metadata=problem.metadata+(
            ('geometry', 'full Boozer embedding at element Gauss points'),
            ('toroidal_offset_sign', toroidal_offset_sign),
            ('geometry_interpolation', 'linear coefficients in s; r**abs(m) below first half-mesh surface'),
            ('boundary_condition', boundary_condition),
        ))

    b, _, db_angles, basis, rot, _, ci, _, cg, _, rho = _sample_boozer_geometry(
        geometry, None, theta, zeta, density, toroidal_offset_sign)
    problem = physical_potential_problem_from_grid(geometry.s, harmonics,
        theta=theta, zeta=zeta, magnetic_field=b,
        magnetic_field_angle_derivatives=db_angles, covariant_basis=basis,
        iota=rot, boozer_i=ci, boozer_g=cg, mass_density=rho, angular_basis=angular_basis)
    return replace(problem, metadata=problem.metadata+(
        ('geometry', 'full Boozer embedding metric'),
        ('toroidal_offset_sign', toroidal_offset_sign),
    ))


def solve_physical_potential_eigenmodes(problem):
    """Solve K Phi=omega^2 M Phi and restore constrained boundary values."""
    spectrum = generalized_hermitian_eigensystem(problem.stiffness, problem.mass)
    omega2 = spectrum.eigenvalues
    vectors = spectrum.eigenvectors
    residuals = generalized_eigen_residual_norms(problem.stiffness, problem.mass, spectrum)
    modes = jnp.zeros((len(omega2), len(problem.s)*len(problem.harmonics)), dtype=vectors.dtype)
    modes = modes.at[:, problem.active_dofs].set(vectors.T)
    modes = modes.reshape(len(omega2), len(problem.s), len(problem.harmonics))
    return RadialBlockEigenmodeResult(omega2, jnp.sqrt(jnp.maximum(omega2, 0))/(2*jnp.pi),
        vectors, modes, residuals, problem.stiffness, problem.mass)
