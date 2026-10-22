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

Global eigenmodes (AE3D weak form)
----------------------------------

Global modes follow the reduced ideal-MHD shear-Alfvén model of AE3D (Spong,
D'Azevedo & Todo 2010). For the electrostatic potential :math:`\phi` and
:math:`u = \mathbf B\cdot\nabla\phi / B`,

.. math::

   \omega^2 \int \frac{\mu_0\rho}{B^2}\,\nabla\tilde\phi^*\cdot\nabla\phi\,\mathrm dV
   = \int \nabla\tilde u^*\cdot\nabla u\,\mathrm dV .

The pressure-gradient and parallel-current (kink) terms of AE3D are omitted:
on CTOK they move AE3D's global modes by less than 0.1%. The gradients use the
full contravariant metric :math:`g^{ij} = \nabla u^i\cdot\nabla u^j` of
:math:`(s, \theta, \zeta)`, obtained by inverting :math:`g_{ij} = \mathbf r_i\cdot\mathbf r_j`.
With :math:`\phi = \sum_{k,j} y_{kj} f_k(s) e_j` and linear hat functions
:math:`f_k` on the geometry surfaces, :math:`\nabla\phi = f'\,\mathbf X_0 + f\,\mathbf X_1`
and :math:`\nabla u = f'\,\mathbf W_0 + f\,\mathbf W_1`, where, with
:math:`c_j = i(\iota m_j - n_j) B/(G + \iota I)`,

.. math::

   \mathbf X_0 = e_j\nabla s,\quad
   \mathbf X_1 = i e_j (m_j\nabla\theta - n_j\nabla\zeta),\quad
   \mathbf W_0 = c_j e_j\nabla s,\quad
   \mathbf W_1 = \nabla(c_j e_j).

The surface averages of :math:`\mathbf X_a^*\cdot\mathbf X_b` (and of the
:math:`\mathbf W`) give :math:`H\times H` coefficient blocks for the
:math:`f'f'`, :math:`f'f` and :math:`ff` products; averaged over each radial
element and integrated exactly against the hat functions, they give the
Hermitian block-tridiagonal problem :math:`F y = \omega^2 G y`. Boundary
conditions are :math:`\phi = 0` at :math:`s = 1` and, for :math:`m \neq 0`, on
the axis; :math:`m = 0` harmonics are flat between the axis and the first
surface. The singular :math:`f'f'` blocks reproduce the local continuum, so the
spectrum has continuum accumulation points and discrete gap modes.

Eigenpairs near a target :math:`\omega^2` come from shift-invert subspace
iteration: :math:`F - \sigma G` is factored by block LU (the Thomas algorithm
with :math:`H\times H` blocks, :math:`O(N H^3)`), and Rayleigh–Ritz on
:math:`(F, G)` extracts the pairs. Small problems can use the dense solver.

Eigenvalue derivatives
----------------------

For :math:`K x = \lambda M x` with Hermitian :math:`K`, positive-definite
:math:`M` and :math:`x^\dagger M x = 1`, a perturbation gives

.. math::

   \delta\lambda = x^\dagger\left(\delta K - \lambda\,\delta M\right)x ,

and the same identity with :math:`(F, G)` and the shift-invert eigenvectors
differentiates targeted global eigenvalues.

FAX implements this as a custom JVP. For a cluster of eigenvalues the mean
derivative is the trace of the projected perturbation, which remains smooth
when eigenvalues inside the cluster cross. The VMEC boundary enters through
``vmex``'s implicit-function-theorem derivative of the converged
equilibrium and through ``booz_xform_jax``.
