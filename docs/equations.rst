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

The metric-matrix local operator uses the harmonic convention
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
reduces to the profile formula above. In VMEC ``wout`` adapter checks, ``J``,
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

The global 3-D Boozer layer generalizes this scalar problem to a radial block
operator. On each surface, the unknown is the Boozer harmonic vector
``u_r = [u_{r1}, ..., u_{rH}]``. The local matrices are supplied by the 3-D
Boozer continuum operator, which uses the STELLGAP weighting (Spong, Sanchez,
Weller 2003) with ``B.grad = (B^2/(G + iota I)) (d_zeta + iota d_theta)``:

.. math::

   K^{3D}_{ij} =
   \left< \frac{|\sqrt{g}|\,|\nabla s|^2}{\mu_0}
   k_{\parallel i} k_{\parallel j} \phi_i^* \phi_j \right>,
   \qquad
   M^{3D}_{ij} =
   \left< \frac{|\sqrt{g}|\,|\nabla s|^2 \rho}{|B|^2}
   \phi_i^* \phi_j \right>.

The radial envelope term is assembled as a Hermitian finite-difference energy:

.. math::

   K_{\rm global} =
   \operatorname{blockdiag}(K_1^{3D},\ldots,K_N^{3D})
   + D_s^\dagger C_s D_s,
   \qquad
   M_{\rm global} =
   \operatorname{blockdiag}(M_1^{3D},\ldots,M_N^{3D}),

with either Dirichlet radial boundaries or zero-gradient Neumann boundaries.
The resulting generalized eigenproblem is

.. math::

   K_{\rm global} x = \omega^2 M_{\rm global} x.

The coefficient ``C_s`` is a Hermitian radial-stiffness block. For real Boozer
geometry runs, the default coefficient is the metric-derived radial magnetic
energy

.. math::

   (C_s)_{ij} =
   \left< \frac{|\sqrt{g}|\,|\nabla s|^2 |B|^2}{\mu_0}
   \phi_i^* \phi_j \right>_{\theta_B,\zeta_B}.

For analytic sensitivity tests FAX can also use the older local-mass scaled
coefficient. This is a validated global radial shear-Alfven envelope operator,
not yet a complete ideal-MHD CAS3D/AE3D replacement. Its acceptance gates are
exact diagonal constant-coefficient spectra for Dirichlet and Neumann
boundaries, generalized-eigen residuals, JAX density-scaling gradients,
metric-radial-stiffness Hermiticity, finite real-``boozmn`` mode structures,
and the first STELLGAP CTOK local-continuum comparison.

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

For the fast user-facing BAE/TAE/EAE example, FAX also solves a minimal
two-branch avoided-crossing model,

.. math::

   \begin{bmatrix}
   \omega_m^2 & c\,\omega_m\omega_{m+\Delta m}\\
   c\,\omega_m\omega_{m+\Delta m} & \omega_{m+\Delta m}^2
   \end{bmatrix}
   \phi = \omega^2 \phi,

where ``Delta m=1`` reproduces the toroidicity-induced TAE crossing and
``Delta m=2`` reproduces the ellipticity-induced EAE crossing in the
large-aspect-ratio picture. The corresponding reference surfaces are

.. math::

   q_{\rm TAE}=\frac{2m+1}{2n}, \qquad
   q_{\rm EAE}=\frac{m+1}{n}.

The same example marks the finite-beta/acoustic BAE accumulation frequency with
the fluid-order estimate

.. math::

   \omega_{\rm BAE} =
   \sqrt{\frac{7}{4}+\frac{T_e}{T_i}}\,
   \frac{v_{ti}}{qR}, \qquad
   v_{ti} = \sqrt{\frac{T_i}{m_i}}.

This BAE marker is intentionally a low-frequency reference scale; kinetic,
diamagnetic, isotope, and shaped-equilibrium corrections require later
validation layers.

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

The reduced gyro-Landau closure keeps the same resonance coordinate but exposes
calibrated coefficients for a FAR3D/CAS3D-K-facing parity path:

.. math::

   x =
   \frac{f_{\rm mode}-n f_\phi-\ell f_\theta}{\Delta f},
   \qquad
   D_{\rm GL} =
   -\alpha_f \frac{\partial p_f}{\partial s}
   \left(1+\frac{f_*}{\max(|f_{\rm mode}|,\Delta f)}\right)
   \left[
   \frac{C_0}{1+x^2}
   + \frac{C_1 x}{1+x^2}
   \right].

``C_0`` is the even resonant drive/damping coefficient, ``C_1`` is the odd
dispersive coefficient, ``f_*`` is a diamagnetic frequency scale, and
``gamma_L`` is a thermal Landau damping offset. This is a reduced closure, not
yet a first-principles perturbative kinetic growth rate. The publication path
still needs

.. math::

   \gamma_f \approx -\frac{\operatorname{Im}\delta W_f}{2 \omega W_{\rm MHD}},

validated against FAR3D/CAS3D-K-style limits after eigenfunctions and
energetic-particle moments are added.

The current global-mode diagnostic also projects this reduced drive onto
eigenfunctions:

.. math::

   \Gamma_k =
   S_f \frac{\int ds\, |\xi_k(s)|^2 \sum_\ell D_{\rm GL}(s,\ell)}
   {\int ds\,|\xi_k(s)|^2}
   - \gamma_{\rm damp} - \gamma_L.

This gives a differentiable screening metric for resonance overlap and radial
localization. It is a sign and sensitivity gate, not a substitute for a
drift-kinetic closure.

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
connected to VMEC/vmex design variables.

Boozer Path
-----------

``fax.boozer`` can read ``boozmn`` files, call local ``booz_xform_jax`` from a
VMEC ``wout`` file, or accept an in-memory VMEC ``wout``-like object produced by
``vmex``. All three paths populate the same ``BoozerGeometry`` contract when
the required harmonics are available. The implemented diagnostic computes
normalized non-axisymmetric Boozer ``|B|`` spectral energy and dominant modes.
It also validates the scalar harmonic representation

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

The next Boozer metric bridge uses the standard Boozer-Jacobian identity

.. math::

   J_B \equiv \sqrt{g}_B = \frac{G(\psi)+\iota(\psi)I(\psi)}{|B|^2},
   \qquad
   J_B |B|^2 = G+\iota I,

where ``G`` and ``I`` are flux functions in the covariant Boozer
representation. FAX currently accepts the numerator ``G+iota I`` as a
surface-wise input or normalization constant and validates the flux-function
identity before using the weight in reduced matrices.

The first reduced Boozer continuum matrix now uses this scalar representation
in the field-line-bending weight:

.. math::

   K_{ij}^{B} =
   \left\langle
   \frac{|B|^2}{\mu_0} k_{\parallel i} k_{\parallel j}
   \phi_i^* \phi_j
   \right\rangle_{\theta_B,\zeta_B},
   \qquad
   M_{ij}^{B} =
   \rho \left\langle \phi_i^* \phi_j\right\rangle_{\theta_B,\zeta_B}.

It recovers the analytic profile frequency in the constant-field limit and
opens controlled sideband coupling when a known Boozer harmonic is added to
``|B|``.

FAX now also reads real ``boozmn`` geometry harmonics ``R``, ``Z``, ``p``, and
``sqrt(g)_B`` and assembles a local 3-D Boozer continuum matrix. The
straight-field-line derivative is

.. math::

   k_{\parallel,mn}(s,\theta_B,\zeta_B) =
   \frac{\left[n-m\iota(s)\right]|B|(s,\theta_B,\zeta_B)}
        {G(s)+\iota(s)I(s)}.

Using the normalized-radial embedding metric
``A = |\nabla s|^2`` reconstructed from ``R(theta_B,zeta_B,s)``,
``Z(theta_B,zeta_B,s)``, and ``phi=zeta_B+p``, FAX forms

.. math::

   K_{ij}^{3D} =
   \left\langle
   |\sqrt{g}_B| A \frac{|B|^2}{\mu_0}
   k_{\parallel i} k_{\parallel j}\phi_i^*\phi_j
   \right\rangle_{\theta_B,\zeta_B},
   \qquad
   M_{ij}^{3D} =
   \left\langle
   |\sqrt{g}_B| A \rho
   \phi_i^*\phi_j
   \right\rangle_{\theta_B,\zeta_B}.

This is the first true ``boozmn`` geometry operator. It has analytic
constant-field, Hermiticity, real-Jacobian-identity, and embedding-Jacobian
shape gates. It is also the local block used by the global radial Boozer
operator. Full ideal-MHD STELLGAP/AE3D/CAS3D parity and kinetic/fast-ion drive
validation remain separate publication milestones.

With the Boozer-Jacobian bridge enabled, the reduced weights become

.. math::

   K_{ij}^{J_B} =
   \left\langle
   J_B \frac{|B|^2}{\mu_0} k_{\parallel i} k_{\parallel j}
   \phi_i^* \phi_j
   \right\rangle_{\theta_B,\zeta_B},
   \qquad
   M_{ij}^{J_B} =
   \rho \left\langle J_B \phi_i^* \phi_j\right\rangle_{\theta_B,\zeta_B}.

This separates a Boozer-coordinate identity gate from the later full-operator
work: ``J_B |B|^2`` must be a flux function, while nonuniform ``J_B`` can
couple the mass matrix. The missing publication-grade terms are the full
perpendicular metric tensor, radial derivatives, boundary conditions, and
cross-code comparisons.

Equation Acceptance
-------------------

Each equation entering the code must have:

* a documentation page with conventions, units, and assumptions,
* a dense or analytic reference implementation,
* a unit test for dimensions and signs,
* a convergence test,
* a finite-difference or adjoint consistency test if it is differentiable,
* at least one validation case before it can support a publication claim.
