Model Inventory
===============

This page records what FAX currently computes, what each layer accepts as
input, what it returns, which validation gates protect it, and which claims are
still out of scope. It is meant to be the first page a reviewer reads before
looking at equations or examples.

Status Labels
-------------

``validated``
   The implementation has analytic, residual, convergence, or cross-code gates
   with explicit tolerances in ``fax.validation``.

``smoke-tested``
   The plumbing is exercised on real artifacts or live adapters, but the result
   is not yet a publication-grade physics comparison.

``screening``
   The formula is useful for ranking or design triage and is differentiable,
   but it is intentionally lower fidelity than the target production model.

``planned``
   The interface, documentation, or benchmark target exists, but the physics
   layer should not yet be used for research conclusions.

Equilibrium and Geometry
------------------------

Analytic profile equilibria
   ``fax.equilibrium.analytic_cylinder`` and
   ``fax.equilibrium.analytic_circular_tokamak`` provide compact fixtures for
   local continuum formulas, radial global modes, density scaling, and gap
   markers. These are ``validated`` for the large-aspect-ratio limits they are
   designed to represent.

VMEC ``wout`` profiles
   ``fax.equilibrium.from_vmec_wout`` reads VMEC profiles and enough Fourier
   geometry to provide first adapter checks. The Landreman-Paul low-resolution
   QA case is the local stellarator reference target. This path is
   ``smoke-tested`` and protected by finite-output and radial-sampling gates.

Metric geometry
   ``fax.metric.MetricGeometry`` stores angular grids, Jacobian weights,
   magnetic field strength, density, and field-line derivative coefficients.
   ``metric_from_equilibrium`` is a validated analytic lift. ``metric_from_vmec_wout``
   is a VMEC Fourier reconstruction path used for adapter and residual checks.

Boozer geometry
   ``fax.boozer.BoozerGeometry`` is the common contract for file-backed
   ``boozmn`` files, live ``booz_xform_jax`` transforms, and in-memory
   ``vmec_jax`` handoff objects. Its current gates check harmonic parity,
   straight-field-line phase, the Boozer-Jacobian identity, asymmetric
   harmonics, and Landreman-Paul bridge convergence.

Continuum Operators
-------------------

Profile continuum
   ``fax.continuum.local_continuum_frequency_hz`` and
   ``continuum_branch_matrix`` compute the large-aspect-ratio shear-Alfven
   limit. Inputs are ``EquilibriumGeometry`` and harmonic pairs ``(m,n)``.
   Outputs are local branch frequencies in hertz. Validation covers uniform
   cylinder constancy and density scaling.

Metric continuum
   ``fax.metric.metric_continuum_matrices`` assembles Hermitian surface
   matrices ``K`` and ``M`` from angular-grid metric data. The generalized
   problem is ``K x = omega^2 M x``. Outputs include coupled branch frequencies,
   eigenvalues, eigenvectors, and residual diagnostics. Gates cover reduction to
   the profile limit, Hermiticity, residuals, angular convergence, radial
   convergence, and density gradients.

Reduced Boozer continuum
   ``fax.boozer.boozer_continuum_matrices`` keeps the validated Boozer harmonic
   representation but uses a large-aspect-ratio parallel wavenumber. It is a
   bridge used to test sideband coupling and sign conventions.

Local 3-D Boozer continuum
   ``fax.boozer.boozer_3d_continuum_matrices`` uses real Boozer geometry,
   Boozer-Jacobian weights, straight-field-line derivative coefficients, and
   sampled angular quadrature. It is validated in the constant-field limit and
   on real ``boozmn`` fixtures, and it is the local operator used by the global
   radial block model.

Global Eigenmode Operators
--------------------------

Scalar radial envelope model
   ``fax.global_modes.RadialEigenmodeProblem`` turns a local continuum branch
   into a one-dimensional generalized eigenproblem. It is validated against an
   exact finite-difference spectrum, residual checks, and density-gradient
   scaling. It is useful for examples, inverse design, and fast tests.

Block radial Boozer model
   ``fax.global_modes.RadialBlockEigenmodeProblem`` carries one Boozer harmonic
   vector per radial surface. Local ``K``/``M`` blocks come from the 3-D Boozer
   operator; radial stiffness comes from either metric-derived magnetic energy
   or a mass-scaled analytic fixture. Dirichlet and Neumann boundary variants
   are validated. Dense, sparse, and matrix-free application paths share the
   same finite-difference energy.

Sparse and matrix-free paths
   ``assemble_radial_block_sparse_matrices`` and
   ``solve_radial_block_eigenmodes_sparse`` are non-differentiable scale-up
   references. ``radial_block_stiffness_matvec`` and
   ``radial_block_mass_matvec`` are JAX-native matrix-free kernels used for
   future implicit/Krylov derivatives.

Differentiable Eigenvalue Models
--------------------------------

Isolated eigenvalues
   ``generalized_hermitian_eigenvalue_sensitivity`` implements
   ``x.H(dK-lambda dM)x/(x.H M x)``. ``generalized_hermitian_isolated_eigenvalue``
   wraps the dense solver with a custom JVP. Radial and Boozer helpers expose
   this as design objectives for one tracked, isolated mode.

Eigenvalue clusters
   ``generalized_hermitian_eigencluster_sensitivity`` computes a mass-orthonormal
   subspace and projected perturbation matrix for repeated or near-crossing
   eigenvalues. ``generalized_hermitian_cluster_mean_eigenvalue`` and the
   radial/Boozer cluster objectives differentiate an invariant cluster mean
   instead of an ill-defined single-mode label.

Gap and Fast-Ion Screens
------------------------

Analytic gap markers
   ``fax.gap_models`` reports BAE, TAE, EAE, and HAE reference locations and
   frequencies for large-aspect-ratio screening. These gates check ordering and
   analytic crossing conditions, not full stellarator gap widths.

Reduced fast-ion drive
   ``fax.fast_ions`` implements a transit/precession resonance mismatch,
   Lorentzian overlap, pressure-gradient sign convention, and eigenfunction
   weighting. It is a differentiable screening model, not a kinetic growth-rate
   replacement for FAR3D or CAS3D-K.

Optimization and UQ
-------------------

Profile objectives
   ``radial_mode_frequency_vector``, ``radial_mode_frequency_jacobian``,
   ``linearized_radial_mode_uncertainty``, and ``radial_mode_inverse_design``
   form the compact profile-design example stack.

Real-Boozer objectives
   ``boozer_global_mode_frequency_vector`` and related helpers apply the same
   sensitivity, UQ, inverse-design, fast-ion risk, isolated-mode, and
   cluster-mean patterns to real Boozer global radial operators.

Pipeline objectives
   ``boozer_pipeline_mode_avoidance_objective``,
   ``boozer_pipeline_isolated_mode_avoidance_objective``, and
   ``boozer_pipeline_cluster_mean_avoidance_objective`` accept a callable
   Boozer-geometry builder. This is the intended API boundary for future
   differentiable ``vmec_jax -> booz_xform_jax -> FAX`` design loops.

Cross-Code Comparisons
----------------------

Implemented artifact gates
   STELLGAP CTOK local-continuum comparison, STELLGAP post-processing parity,
   AE3D public matrix symmetry, AE3D matrix/eigenmode residuals, AE3D omega2
   convention parity, and independent dense-spectrum parity.

Still planned
   Broader STELLGAP geometries, FAX-vs-AE3D global radial mode comparisons,
   CAS3D/CAS3D-K benchmarks, FAR3D growth-rate comparisons, and GPU/runtime
   comparisons against executable reference codes.
