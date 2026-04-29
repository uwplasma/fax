Differentiability
=================

``fax`` will use several differentiation modes, chosen by problem size and
mathematical structure.

Dense reference path
--------------------

Small continuum and global-mode problems should differentiate directly through
JAX operations. This path is the correctness reference for custom adjoints.
The current gates differentiate profile continuum diagnostics and the reduced
radial generalized eigenproblem with respect to density, rotational-transform,
and magnetic-field profile parameters.
``boozer_pipeline_mode_avoidance_objective`` extends this pattern to a
callable Boozer-geometry builder, which is the intended API boundary for
``vmec_jax``/``booz_xform_jax`` pipelines. The file-backed and in-memory
``run_booz_xform_jax_geometry_from_wout*`` adapters now share the same
``BoozerGeometry`` contract and are validated against asymmetric Boozer
harmonics. ``run_vmec_jax_fixed_boundary_boozer_geometry`` adds the first live
``vmec_jax`` fixed-boundary object handoff into the same path and is covered by
a Landreman-Paul QA integration smoke gate. The companion
``run_vmec_jax_boozer_truncation_convergence`` diagnostic reuses one VMEC state
to compare nearby Boozer truncations before downstream operators are used for
gradient claims. ``run_vmec_jax_boozer_solve_convergence`` adds the next
pre-gradient gate by requiring VMEC residual reduction to propagate through
Boozer geometry and FAX continuum errors against a converged reference
``wout``. ``boozer_3d_operator_angular_convergence`` then checks the local 3-D
operator's angular quadrature on that fixed geometry. The dense
objective test still differentiates through a synthetic builder; the next
full-equilibrium gradient gate should replace that builder with a stable
differentiable ``vmec_jax`` solve and ``booz_xform_jax`` transform artifact.

Generalized eigenvalue path
---------------------------

For isolated Hermitian eigenpairs normalized by ``x.H @ M @ x = 1``, the planned
sensitivity is

.. math::

   d\lambda = x^\dagger (dK - \lambda dM) x.

The implemented helper ``generalized_hermitian_eigenvalue_sensitivity`` keeps
the normalization denominator explicitly,
``x.H @ (dK - lambda*dM) @ x / (x.H @ M @ x)``, so it works with eigenvectors
from FAX, SciPy, or cross-code artifacts. ``generalized_hermitian_isolated_eigenvalue``
wraps the dense generalized eigensolver with a custom JVP using that formula.
``radial_block_eigenvalue_sensitivity`` and ``radial_block_isolated_omega2``
apply the same pattern to the global radial block operator. Current tests and
validation gates compare the formula against direct autodiff and centered
finite differences for isolated eigenvalues. Repeated/near-crossing eigenvalues
remain a subspace problem and should not be reported as single-mode
sensitivities.

Implicit and matrix-free path
-----------------------------

Large problems should avoid differentiating through every Krylov iteration.
Instead, use implicit differentiation or custom VJP/JVP rules that solve the
adjoint linear system directly. The first production-facing pieces are now in
``fax.global_modes``: ``radial_block_stiffness_matvec``,
``radial_block_mass_matvec``, ``radial_block_matrix_free_residual_norm``, and
``radial_block_rayleigh_omega2``. These functions apply the same finite-
difference radial energy as the dense and sparse assemblies, but keep the
operator as JAX transformations over arrays. The validation tests compare
matrix-free application to dense assembly for Dirichlet and Neumann boundaries,
check eigenpair residuals and Rayleigh quotients, and differentiate a
density-scaling Rayleigh frequency gate.

JAX ecosystem trials
--------------------

``lineax``
   Candidate for structured linear solves, transposed adjoint solves, and
   matrix-free shift-invert prototypes.

``equinox``
   Candidate for PyTree modules and filtered transforms if it makes model
   objects cleaner without obscuring pure functions.

``diffrax``
   Candidate for FAR3D-like time-evolution and nonlinear transport
   demonstrations. It is not the first choice for eigenvalue solvers.

``jaxopt`` and ``optimistix``
   Candidates for implicit root/least-squares differentiation and optimization
   workflow prototypes.

No ecosystem dependency should become required until it passes a documented
prototype against a native-JAX baseline.

Current demonstrations
----------------------

``examples/research/global_gap_fastion_screen.py`` prints a compact JSON packet
with a constant-profile radial spectrum residual, analytic gap estimates, a
fast-ion resonance peak, and the gradient of a radial-mode avoidance objective.
``tools/generate_validation_figures.py`` turns the same ingredients into the
``global_gap_fastion_dashboard`` documentation figure.

The Boozer global-mode objective test differentiates through local 3-D Boozer
matrices, metric-derived radial stiffness, generalized eigenfrequencies, and a
smooth frequency-window penalty. This is the current differentiable design
screen for avoiding dangerous resonances or moving candidate modes out of
selected frequency windows.

The newest differentiability demonstration is
``examples/research/differentiable_design_study.py``. It uses the same dense
reference path to produce four quantities that are difficult to obtain from
legacy non-differentiable workflows without repeated manual perturbation
studies:

* a full sensitivity matrix ``df_mode / dparameter`` for low-order global
  radial eigenfrequencies,
* linearized uncertainty bands from ``J C J^T`` covariance propagation,
* an inverse-design trajectory from direct gradients of an AE risk objective,
* fast-ion growth screens before and after the differentiable design move.

This remains a reduced radial-model demonstration. The planned publication
path is to preserve the same API while replacing the profile fixture with
validated VMEC/Boozer operators and then cross-code benchmarking the resulting
mode frequencies and structures.

``examples/research/boozmn_differentiable_design_study.py`` now performs that
same calculation on a real ``boozmn`` global radial operator. Its gradients
pass through Boozer surface matrices, metric-derived radial stiffness,
generalized eigenfrequencies, and eigenfunction-weighted fast-ion drive. This
still treats mass density, rotational-transform offset, radial stiffness, and
fast-ion density as reduced design coordinates; the next adapter step is to
map those coordinates to actual ``vmec_jax`` boundary/profile variables and
``booz_xform_jax`` transform outputs.

The live Landreman-Paul ``vmec_jax`` bridge is intentionally a smoke test today:
it verifies object compatibility and finite downstream operators, but it uses a
single VMEC iteration. Differentiable shape-gradient claims require a converged
or otherwise controlled equilibrium solve, a Boozer-transform convergence
study, and finite-difference or adjoint checks against actual VMEC boundary and
profile variables.
