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

Global Modes
------------

The reduced global eigenmode problem is planned as a generalized eigenproblem,

.. math::

   K(q) x = \omega^2 M(q) x,

where ``q`` denotes equilibrium, profile, and numerical parameters. ``M`` must
be positive definite for accepted Hermitian validation cases, and ``K`` must be
Hermitian to numerical tolerance unless the selected model explicitly includes
non-Hermitian physics.

Fast-Ion Drive
--------------

The first fast-ion implementation should be perturbative with respect to a
validated eigenfunction:

.. math::

   \gamma_f \approx -\frac{\operatorname{Im}\delta W_f}{2 \omega W_{\rm MHD}}.

Resonance denominators and signs must be verified first in analytic transit,
bounce, and precession limits before using full stellarator geometry.

Equation Acceptance
-------------------

Each equation entering the code must have:

* a documentation page with conventions, units, and assumptions,
* a dense or analytic reference implementation,
* a unit test for dimensions and signs,
* a convergence test,
* a finite-difference or adjoint consistency test if it is differentiable,
* at least one validation case before it can support a publication claim.

