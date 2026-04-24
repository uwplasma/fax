Equations
=========

The equation plan is staged so every implemented model has a simpler limiting
case that can be solved analytically or reproduced by a reference code.

Continuum
---------

The first target is the incompressible shear-Alfven continuum in straight-field
line coordinates. In its simplest local limit, each harmonic has

.. math::

   \omega_{A,mn}(s) = v_A(s) |k_{\parallel,mn}(s)|,
   \qquad
   v_A = \frac{B}{\sqrt{\mu_0 \rho}},

with the field-line wavenumber reducing in large-aspect-ratio notation to a
form proportional to ``n - m iota``. The production operator must use the full
metric/Jacobian data from the selected coordinate representation rather than
hard-coding this approximation.

The first metric-matrix scaffold uses the harmonic convention
``exp(i * (m * theta - n * zeta))`` and forms, on each flux surface,

.. math::

   K_{ij} =
   \left\langle J \frac{B^2}{\mu_0}
   k_{\parallel i} k_{\parallel j}
   \phi_i^* \phi_j \right\rangle_{\theta,\zeta},
   \qquad
   M_{ij} =
   \left\langle J \rho
   \phi_i^* \phi_j \right\rangle_{\theta,\zeta},

with ``K x = omega^2 M x``. In the uniform straight-field-line fixture this
reduces to the profile formula above. In VMEC ``wout`` smoke tests, ``J``,
``B``, and contravariant field components are reconstructed from Nyquist
Fourier coefficients. This is an operator-plumbing and symmetry milestone; the
reviewer-facing stellarator validation still requires comparison with
STELLGAP/AE3D-style reference calculations.

Literature anchors for these conventions include Chen & Zonca's review of
Alfven waves and energetic particles, STELLGAP-style stellarator continuum/gap
calculations, the Landreman-Paul precise-quasisymmetry configuration used as
the first VMEC-backed target, and recent quasisymmetric-stellarator continuum
analysis:

* Chen & Zonca, Rev. Mod. Phys. 88, 015008 (2016),
  https://doi.org/10.1103/RevModPhys.88.015008
* STELLGAP software record, https://www.osti.gov/biblio/code-4633
* Landreman & Paul, Phys. Rev. Lett. 128, 035001 (2022),
  https://doi.org/10.1103/PhysRevLett.128.035001
* Paul et al., J. Plasma Phys. 91, E101 (2025),
  https://doi.org/10.1017/S0022377825100524

Global Modes
------------

The first implemented global layer is a one-dimensional radial envelope
problem. It is deliberately narrower than AE3D/CAS3D/FAR3D-class 3-D
calculations, but it gives FAX a validated radial eigenvalue path above
surface-by-surface continuum matrices:

.. math::

   K_r(q) \xi = \omega^2 M_r(q) \xi.

where ``q`` denotes equilibrium, profile, and numerical parameters. ``M`` must
be positive definite for accepted Hermitian validation cases, and ``K`` must be
Hermitian to numerical tolerance unless the selected model explicitly includes
non-Hermitian physics. The current radial fixture uses Dirichlet boundary
conditions at the inner and outer radial endpoints. Its constant-coefficient
finite-difference limit has an exact discrete spectrum,

.. math::

   \omega_j^2 = \omega_A^2 +
   \frac{4 c_r}{\Delta s^2}
   \sin^2\left(\frac{j\pi}{2(N+1)}\right),

which is used as a reviewer-facing eigenvalue and residual gate.

Gap Estimates
-------------

Analytic gap screens use the crossing of coupled sidebands ``m`` and
``m + Delta m``:

.. math::

   n - m\iota = -\left[n - (m+\Delta m)\iota\right],
   \qquad
   \iota_{\rm gap} = \frac{2n}{2m+\Delta m}.

The implemented TAE/EAE/HAE helpers report the nearest radial crossing,
center frequency, and a configurable fractional width. These are analytic
screening gates and should be replaced by Boozer/metric coupling integrals
before making cross-code stellarator gap-width claims.

Fast-Ion Drive
--------------

The first fast-ion implementation is a reduced resonance diagnostic. It uses
the transit/precession resonance condition

.. math::

   f_{\rm mode} - n f_\phi - \ell f_\theta = 0

and a Lorentzian weight to rank candidate drive windows. Negative pressure
gradient gives positive drive in the current convention:

.. math::

   D_f =
   -\alpha_f \frac{\partial p_f}{\partial s}
   \frac{1}{1 + \left[(f_{\rm mode}-n f_\phi-\ell f_\theta)/\Delta f\right]^2}.

This is not yet a perturbative kinetic growth rate. The planned publication
path is still

.. math::

   \gamma_f \approx -\frac{\operatorname{Im}\delta W_f}{2 \omega W_{\rm MHD}},

validated against FAR3D/CAS3D-K-style limits after eigenfunctions and
energetic-particle moments are added.

Optimization Objective
----------------------

The current differentiable optimization screen penalizes radial eigenmode
frequencies inside a user-selected exclusion window:

.. math::

   J(q) =
   \frac{1}{N}\sum_j
   \sigma\left(\frac{f_j-f_{\min}}{\Delta f}\right)
   \sigma\left(\frac{f_{\max}-f_j}{\Delta f}\right)
   + \lambda \|p\|^2.

Here ``p = [log density scale, iota shift, log field scale]`` in the initial
profile fixture. This tests the end-to-end differentiable pattern that will be
connected to VMEC/vmec_jax design variables.

Boozer Path
-----------

``fax.boozer`` can read ``boozmn`` files and can call local
``booz_xform_jax`` from a VMEC ``wout`` when available. The implemented
diagnostic computes normalized non-axisymmetric Boozer ``|B|`` spectral energy
and dominant modes. It also validates the scalar harmonic representation

.. math::

   |B|(s,\theta_B,\zeta_B) =
   \sum_{m,n} B^c_{mn}(s)\cos(m\theta_B-n\zeta_B)
   + B^s_{mn}(s)\sin(m\theta_B-n\zeta_B),

with uniform-grid projection back to ``B^c_{mn}`` and ``B^s_{mn}``. The current
straight-field-line convention is

.. math::

   \theta_B(\zeta_B) = \theta_{B0} + \iota \zeta_B,
   \qquad
   \frac{d}{d\zeta_B}\left(m\theta_B-n\zeta_B\right)=m\iota-n.

A dedicated Boozer metric/operator assembly is still an open validation lane
before Boozer spectra become the default stellarator continuum path.

Equation Acceptance
-------------------

Each equation entering the code must have:

* a documentation page with conventions, units, and assumptions,
* a dense or analytic reference implementation,
* a unit test for dimensions and signs,
* a convergence test,
* a finite-difference or adjoint consistency test if it is differentiable,
* at least one validation case before it can support a publication claim.
