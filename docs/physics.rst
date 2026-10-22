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

:math:`\partial_s c_j` contains the magnetic-shear term :math:`i\iota' m_j B/(G + \iota I)`.
A test checks :math:`F` and :math:`G` against real-space finite differences of
:math:`\phi` on the CTOK embedding to 0.3%; without the shear term :math:`F` is
off by 9–20%. AE3D's inter-surface shear couplings have the opposite sign, so
its CTOK TAE (337.9 kHz) sits 2.5% above the quadrature value of its own
eigenvector (329.7 kHz); FAX's best-overlap mode is at 272.8 kHz.

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

Energetic-particle drive
------------------------

The growth rate of a fixed eigenmode is the power the fast ions give to it
over twice its energy (perturbative, NOVA-K / CAS3D-K style),

.. math::

   \gamma_h = -\frac{\operatorname{Im}\delta W_h}{2\omega\,\delta K}
            = -\frac{P}{2W}, \qquad
   W = \int \frac{\rho\,|\nabla_\perp\phi|^2}{2B^2}\,d^3x .

The model is a large-aspect-ratio circular torus with
:math:`B = B_0(1-\epsilon\cos\theta)`, zero orbit width and zero Larmor radius,
and passing guiding centres with drift
:math:`v_d = (v_\parallel^2 + v_\perp^2/2)/(\Omega R)`. With
:math:`\phi = \sum_h \phi_h(s)\,e^{i(n\zeta - m_h\theta - \omega t)}`, the work
:math:`Z e\,v_d\cdot\delta E` along an orbit of pitch
:math:`\lambda = \mu B_0/E` is Fourier-decomposed in the uniform transit angle
:math:`\vartheta`. Harmonic :math:`p` resonates at
:math:`\omega = (nq - p)\,\omega_\theta(v, \lambda)`; for a TAE this gives
:math:`v_\parallel \approx v_A, v_A/3, v_A/5`. Since
:math:`\dot P_\zeta = (n/\omega)\dot E` for a single toroidal harmonic, linearizing
along unperturbed orbits and using the Landau prescription gives

.. math::

   P = -\frac{\pi}{2}\int d^3x\,d^3v \sum_p |A_p|^2\,
       \delta\!\left(\omega - (nq-p)\omega_\theta\right)\frac{QF}{\omega},
   \qquad
   \frac{QF}{\omega} = \frac{\partial F}{\partial E}
       + \frac{n}{\omega}\frac{\partial F}{\partial P_\zeta},

with :math:`P_\zeta = -Ze\psi_p` at zero orbit width. The delta function is
integrated exactly over speed at each pitch. The remaining pitch integral uses
Gauss–Legendre nodes and the orbit integral a uniform :math:`\theta` grid, so
:math:`\gamma` is smooth in every parameter without artificial broadening.
Maxwellian and slowing-down distributions are supported; the birth-energy
edge of the slowing-down distribution is smoothed over 2% of :math:`v_b`.
Bulk-ion Landau damping is the same integral for a flat thermal Maxwellian
(Fu & Van Dam 1989; Betti & Freidberg 1992). Continuum damping and electron
Landau damping are not computed; subtract them from
:func:`fax.growth_rate` as an external :math:`\gamma_d`.

Checks: the flat-Maxwellian rate scales as
:math:`x(1+2x^2+2x^4)e^{-x^2}`, :math:`x = v_A/v_{th}` (Fu & Van Dam). For the
Könies et al. (2018) ITPA TAE with a model two-harmonic eigenfunction, FAX gives
4.4, 16.6 and 33.1 :math:`\times 10^3\,\mathrm{s^{-1}}` at 100, 200 and 300 keV.
The published zero-Larmor-radius values are about 8–10, 25–33 and
38–43 :math:`\times 10^3\,\mathrm{s^{-1}}`, so the 300 keV point is within 20%
and the lower energies are about 40% low. Trapped particles and finite orbit
width are not included.
