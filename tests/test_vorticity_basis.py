"""Check the disputed shear coefficients without assuming a cosine expansion.

Directional automatic derivatives construct b.grad(f), its radial derivative
and its poloidal derivative. They provide an independent reference for the
expanded harmonic coefficients and their complex-to-real basis change.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope='module')
def shear_coefficients_in_both_bases():
    labels = np.array([[10, 4], [6, 2], [3, 0]])
    signed = np.vstack([labels, -labels])
    theta, zeta = np.meshgrid(2*np.pi*np.arange(64)/64,
                              2*np.pi*np.arange(32)/32, indexing='ij')
    theta, zeta = jnp.asarray(theta.ravel()), jnp.asarray(zeta.ravel())
    phase = theta[:, None]*labels[:, 0]-zeta[:, None]*labels[:, 1]
    # Manufactured metric weights provide nonzero off-diagonal angular
    # couplings. These are algebra checks, not an MHD equilibrium benchmark.
    weight_ss = 1+.3*jnp.cos(phase[:, 0]-phase[:, 1])
    weight_st = .2*jnp.sin(phase[:, 0]-phase[:, 1])
    position, q, shear = jnp.array(.4), .25, .625

    def derivatives(modes, angular_basis):
        modes = jnp.asarray(modes)
        def parallel(s, poloidal_shift):
            iota = .15+shear*s
            def shifted_potential(distance):
                angle = ((theta+poloidal_shift+iota*distance)[:, None]*modes[:, 0]
                         -(zeta+distance)[:, None]*modes[:, 1])
                return jnp.cos(angle) if angular_basis == 'cosine' else jnp.exp(1j*angle)
            # Differentiate along the straight-field-line direction, rather
            # than writing an expanded kappa or magnetic-shear formula.
            return jax.jvp(shifted_potential, (jnp.array(0.),), (jnp.array(q),))[1]
        value, radial = jax.jvp(lambda s: parallel(s, jnp.array(0.)),
                                (position,), (jnp.array(1.),))
        poloidal = jax.jvp(lambda shift: parallel(position, shift),
                           (jnp.array(0.),), (jnp.array(1.),))[1]
        return value, radial, poloidal

    def gram(left, right, weight):
        return np.asarray(left.conj().T@(weight[:, None]*right))/len(theta)

    result = {}
    for name, modes in (('cosine', labels), ('complex', signed)):
        value, radial, poloidal = derivatives(modes, name)
        result[name] = dict(
            mixed_radial=gram(radial, value, weight_ss),
            radial_poloidal=(gram(radial, poloidal, weight_st)
                              +gram(poloidal, radial, weight_st)))
    m, n = labels.T
    kappa = n-m*float(.15+shear*position)
    sn, cs = jnp.sin(phase), jnp.cos(phase)
    ss = gram(sn, sn, weight_ss)
    sc = gram(sn, cs, weight_st)
    result['derived'] = dict(
        mixed_radial=-q*q*shear*m[:, None]*kappa[None, :]*ss,
        radial_poloidal=-q*q*shear*m[:, None]*m[None, :]*(
            kappa[None, :]*sc+kappa[:, None]*sc.T))
    result['interchanged'] = dict(
        mixed_radial=-q*q*shear*kappa[:, None]*m[None, :]*ss,
        radial_poloidal=-q*q*shear*m[:, None]*m[None, :]*(
            kappa[None, :]*sc.T+kappa[:, None]*sc))
    result['transform'] = .5*np.vstack([np.eye(len(labels)), np.eye(len(labels))])
    return result


@pytest.mark.parametrize('coefficient', ['mixed_radial', 'radial_poloidal'])
def test_shear_indices_are_preserved_by_complex_to_cosine_projection(
        shear_coefficients_in_both_bases, coefficient, record_property):
    data = shear_coefficients_in_both_bases
    cosine, full = data['cosine'][coefficient], data['complex'][coefficient]
    transform = data['transform']
    projected = transform.conj().T@full@transform
    scale = np.linalg.norm(cosine)
    projection_error = np.linalg.norm(cosine-projected)/scale
    derived_error = np.linalg.norm(cosine-data['derived'][coefficient])/scale
    interchanged_error = np.linalg.norm(cosine-data['interchanged'][coefficient])/scale
    record_property('projection_relative_error', float(projection_error))
    record_property('derived_index_relative_error', float(derived_error))
    record_property('interchanged_index_relative_error', float(interchanged_error))
    assert projection_error < 1e-12
    assert derived_error < 1e-12
    assert interchanged_error > .1
