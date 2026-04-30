Validation Traceability
=======================

This page explains how FAX connects equations, implementations, tests,
validation gates, figures, and publication claims. The purpose is to make every
research result reproducible and every numerical shortcut visible.

Traceability Contract
---------------------

Every physics feature should have:

* a written equation in :doc:`equations`,
* an implementation function in :doc:`function_map`,
* a unit or regression test,
* a named validation gate with a tolerance and citation,
* a figure or table when visual inspection is meaningful,
* a statement of what is not yet validated.

If a feature lacks one of these pieces, it should be described as a prototype or
screening model rather than a production result.

Validation Gate Anatomy
-----------------------

Each ``ValidationGate`` contains:

``gate_id``
   Stable identifier used by tests, CLI filters, reports, and figures.

``title``
   Human-readable label used in dashboards.

``layer``
   Physics or numerical category, such as ``operator``, ``gap-physics``,
   ``boozer``, ``global-eigenmode``, ``fast-ion``, or ``cross-code``.

``quantity``
   The metric being measured. Examples are residual norm, relative branch
   error, gradient error, or Hermitian error.

``tolerance``
   The pass/fail threshold. Tolerances should be tight enough to catch
   regressions and loose enough to avoid machine-dependent false failures.

``citation``
   Literature, code artifact, or analytic reference that anchors the gate.

``command``
   The intended local test command.

``tags``
   Filter labels used by ``fax validate --tag`` and by documentation figures.

Analytic Gates
--------------

Analytic gates protect algebra, units, signs, and limiting cases:

* uniform-cylinder continuum branch constancy,
* profile-limit reduction of metric matrices,
* density scaling ``omega proportional rho^(-1/2)``,
* exact scalar and block radial finite-difference spectra,
* TAE/EAE/BAE location and ordering markers,
* exact degenerate cluster splitting and cluster-mean derivatives.

These gates run quickly and should remain in the default local test suite.

Residual and Symmetry Gates
---------------------------

Residual gates check that returned eigenvectors actually solve the discrete
problem. Symmetry gates check Hermiticity and positive mass weights. Together
they prevent common failures:

* wrong harmonic phase convention,
* missing complex conjugation,
* non-Hermitian radial coupling,
* singular or negative mass matrices,
* accidental row/column transposition when reading artifacts.

Convergence Gates
-----------------

Convergence gates compare output under refinement:

* metric angular quadrature,
* radial profile sampling,
* Boozer harmonic truncation,
* local 3-D Boozer operator angular quadrature,
* VMEC iteration ladders propagated through Boozer geometry and FAX continuum
  diagnostics.

The current VMEC/Boozer convergence gates are intentionally smoke-level because
they run locally in development. Publication cases should increase resolution,
record hardware, and archive all input/output artifacts.

Differentiability Gates
-----------------------

Differentiability gates are split by mathematical structure:

Direct autodiff
   Used for small dense reference paths where JAX can differentiate the whole
   calculation.

Isolated adjoint formulas
   Used for one separated generalized eigenvalue. The gate compares custom-JVP,
   explicit adjoint formula, direct autodiff, and finite difference where
   possible.

Cluster/subspace formulas
   Used for repeated or near-crossing eigenvalues. The gate differentiates the
   cluster mean and verifies finite-difference agreement. Splitting directions
   are checked against the projected subspace matrix.

Pipeline gradients
   Used for callable geometry builders that represent future
   ``vmec_jax -> booz_xform_jax -> FAX`` differentiable loops.

Cross-Code Gates
----------------

Cross-code gates currently cover public artifacts:

STELLGAP CTOK
   FAX reads public CTOK Boozer geometry and compares local 3-D continuum
   branches against the STELLGAP artifact. It also checks STELLGAP
   post-processing conventions and largest adjacent-branch local gap centers
   and widths on matched surfaces. A separate executable runner records
   bounded local ``xstgap`` status for runtime packets, but failed, skipped, or
   timed-out runs are not counted as runtime parity.

AE3D
   FAX parses public AE3D matrices, eigenvectors, omega2 tables, and reported
   residuals. It checks matrix symmetry, eigenmode residuals, Rayleigh errors,
   omega2 convention parity, and independent dense-spectrum parity.

Future gates
   Full FAX-vs-AE3D global-mode parity, CAS3D/CAS3D-K global modes and kinetic
   extensions, FAR3D growth rates, and executable runtime comparisons.

Figure Traceability
-------------------

Every generated documentation figure should answer:

* What input case was used?
* Which equation or operator is being shown?
* Which validation gate protects the result?
* What metric and tolerance were used?
* Is the figure a validated result, a smoke test, or a screening example?

``docs/_static/figure_manifest.json`` stores figure metadata. The validation
dashboard visualizes gate metrics normalized by tolerance. The publication
figure suite stores a separate summary JSON for multi-panel examples.

Reviewer Packet
---------------

A reviewer-proof run packet should include:

* TOML config or script path,
* VMEC/Boozer input files or artifact hashes,
* FAX version and commit,
* Python/JAX/scipy/numpy versions,
* hardware and backend,
* validation gate outputs,
* convergence sweep tables,
* figure manifest,
* exact commands used to regenerate results.

The CLI already writes JSON/CSV/Markdown reports for validation, convergence,
benchmark, continuum, and global-eigenmode workflows. Publication examples
should archive these reports alongside plots.

Claim Levels
------------

``local analytic correctness``
   Supported by exact formulas and residual gates.

``operator correctness``
   Supported by Hermiticity, residuals, convergence, and limiting-case gates.

``artifact parity``
   Supported by reproducing published or archived external-code artifacts.

``physics validation``
   Requires matched equilibrium, model assumptions, resolution convergence, and
   cross-code or literature comparison.

``new design result``
   Requires validated forward operators, gradient checks, uncertainty analysis,
   and sensitivity to resolution/model choices.

Current Gaps
------------

The strongest current gaps are:

* full global AE3D/CAS3D/FAR3D parity for the FAX radial block operator,
* calibration and validation of the reduced gyro-Landau closure against
  kinetic energetic-particle growth-rate references,
* production-scale matrix-free eigensolver derivatives,
* converged end-to-end VMEC shape gradients,
* expanded stellarator geometry set beyond Landreman-Paul QA and CTOK.

These are tracked in ``plan.md`` and should remain visible in papers and
release notes until closed.
