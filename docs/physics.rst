Physics and numerics
====================

Boozer coordinates
------------------

On a flux surface labelled by the normalized toroidal flux :math:`s`, the
field in Boozer coordinates :math:`(\theta, \zeta)` is

.. math::

   \mathbf B = G\nabla\zeta + I\nabla\theta + K\nabla\psi,
   \qquad
   \sqrt g = \frac{G + \iota I}{B^2},
   \qquad
   \mathbf B\cdot\nabla = \frac{B^2}{G + \iota I}\left(\partial_\zeta + \iota\partial_\theta\right).

FAX evaluates :math:`|B|`, :math:`\sqrt g` and the embedding
:math:`(R, \phi = \zeta + p, Z)` from their Fourier harmonics
:math:`\cos(m\theta - n\zeta)`, :math:`\sin(m\theta - n\zeta)`, and obtains

.. math::

   |\nabla s|^2 = \frac{|\mathbf r_\theta\times\mathbf r_\zeta|^2}
                       {\left(\mathbf r_s\cdot\mathbf r_\theta\times\mathbf r_\zeta\right)^2}.

Shear-Alfvén continuum
----------------------

For low :math:`\beta` and incompressible perturbations, the radial electric
field component :math:`E` of a continuum mode satisfies on each surface
(Spong, Sanchez & Weller 2003)

.. math::

   \mu_0\rho\,\omega^2\,\frac{|\nabla s|^2}{B^2}\,E
   + \mathbf B\cdot\nabla\left(\frac{|\nabla s|^2}{B^2}\,\mathbf B\cdot\nabla E\right) = 0 .

Multiplying by :math:`\sqrt g\,e_i^*`, with :math:`e_j = e^{i(m_j\theta - n_j\zeta)}`,
averaging over the surface and integrating the second term by parts gives the
generalized Hermitian eigenproblem :math:`K x = \omega^2 M x` with

.. math::

   K_{ij} = \left\langle \frac{|\sqrt g|\,|\nabla s|^2}{\mu_0}\,
            k_{\parallel i}\,k_{\parallel j}\,e_i^* e_j \right\rangle,
   \qquad
   M_{ij} = \left\langle \frac{\rho\,|\sqrt g|\,|\nabla s|^2}{B^2}\, e_i^* e_j \right\rangle,
   \qquad
   k_{\parallel j} = \frac{(n_j - m_j\iota)\,|B|}{G + \iota I}.

The surface average :math:`\langle\cdot\rangle` is evaluated on a uniform
:math:`(\theta, \zeta)` grid, which is exact for band-limited integrands.
For a uniform field the matrices are diagonal and
:math:`\omega = |n - m\iota|\,v_A/R` with :math:`v_A = B/\sqrt{\mu_0\rho}`.

Gaps open where branches :math:`(m, n)` and
:math:`(m + \Delta m, n + \Delta n N_{fp})` cross, at

.. math::

   \iota = \frac{2n + \Delta n N_{fp}}{2m + \Delta m},\qquad
   \frac{\omega R}{v_A} = \frac{|\Delta m\, n - \Delta n\, m N_{fp}|}{2m + \Delta m},

which gives TAE (:math:`\Delta m = 1`), EAE (:math:`\Delta m = 2`), MAE
(:math:`\Delta m = 0`) and HAE (:math:`\Delta m, \Delta n \neq 0`) gaps.

Global operator (interim)
-------------------------

The present global model couples neighbouring surfaces with a radial envelope
term,

.. math::

   \left[K(s) - \partial_s C(s)\,\partial_s\right] x = \omega^2 M(s)\,x,
   \qquad
   C_{ij} = c\,\left\langle \frac{|\sqrt g|\,|\nabla s|^2 B^2}{\mu_0}\, e_i^* e_j\right\rangle,

discretized with second-order finite differences. It has no continuum
resonance, so its frequencies depend on the scale :math:`c`. It is kept so that
the differentiable pipeline can be exercised end to end, and will be replaced
by the AE3D weak form, in which :math:`\omega^2` multiplies the radial inertia
term (Spong, D'Azevedo & Todo 2010).

Eigenvalue derivatives
----------------------

For :math:`K x = \lambda M x` with Hermitian :math:`K`, positive-definite
:math:`M` and :math:`x^\dagger M x = 1`, a perturbation gives

.. math::

   \delta\lambda = x^\dagger\left(\delta K - \lambda\,\delta M\right)x .

FAX implements this as a custom JVP. For a cluster of eigenvalues the mean
derivative is the trace of the projected perturbation, which remains smooth
when eigenvalues inside the cluster cross. The VMEC boundary enters through
``vmex``'s implicit-function-theorem derivative of the converged
equilibrium and through ``booz_xform_jax``.
