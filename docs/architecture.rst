Architecture
============

The planned code is organized around pure, typed, JAX-transformable functions.
The main layers are:

Equilibrium adapters
   Convert ``vmec_jax`` results, VMEC ``wout`` files, or analytic equilibria
   into geometry pytrees with stable shapes.

Coordinate adapters
   Use ``booz_xform_jax`` when a model requires Boozer harmonics, straight
   field-line coordinates, or direct comparison with STELLGAP/AE3D-style
   formulations.

Operators
   Build continuum and global-mode operators from metric coefficients,
   rotational transform, density profiles, and Fourier coupling matrices.
   The current implementation includes both the profile-limit continuum formulas
   and a first angular-grid metric matrix scaffold, so new physics can be
   checked against an analytic reduction before it is treated as stellarator
   validation.

Solvers
   Provide dense reference eigensolvers first, then matrix-free Krylov and
   block-sparse paths with custom differentiation rules where required.

Validation
   Treat reference cases as first-class artifacts: exact analytic cases,
   cross-code comparisons, convergence scans, and experimental benchmarks.
   ``fax.validation`` now provides a registry of named validation gates with
   tolerances, citations, commands, and pass/fail results.
