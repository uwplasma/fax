Numerical Algorithms
====================

FAX keeps dense reference algorithms, sparse scale-up algorithms, and
matrix-free kernels side by side. The dense path is the correctness target; the
sparse and matrix-free paths must match it before any performance claim is made.

Data Flow
---------

The standard forward calculation is:

1. Read or build an equilibrium geometry.
2. Optionally transform it to Boozer coordinates.
3. Select harmonics and radial surfaces.
4. Assemble local surface matrices.
5. Add radial-envelope stiffness when a global mode is requested.
6. Solve ``K x = omega^2 M x``.
7. Compute residuals, mode structures, gap markers, fast-ion overlap, and plots.

The differentiable design calculation wraps the same forward path in a scalar
objective and applies JAX transformations to the selected parameters.

Dense Generalized Eigensolve
----------------------------

``generalized_hermitian_eigensystem`` solves the Hermitian generalized problem
with a Cholesky reduction:

.. math::

   K x = \lambda M x,\qquad M = L L^\dagger,

then diagonalizes

.. math::

   L^{-1} K L^{-\dagger} y = \lambda y,\qquad x=L^{-\dagger}y.

The implementation symmetrizes the standard matrix before calling
``jax.numpy.linalg.eigh``. The returned eigenvectors are not assumed to have a
particular normalization by downstream sensitivity formulas; normalization is
checked explicitly with ``x.H @ M @ x``.

Residual Evaluation
-------------------

All eigenvalue layers use relative residuals of the form

.. math::

   r_j =
   \frac{\|K x_j-\lambda_j M x_j\|}
   {\|K x_j\|+|\lambda_j|\|M x_j\|}.

This avoids accepting an eigenpair only because the absolute scale of the
operator is small. Residual gates are included for local metric matrices,
scalar radial modes, real ``boozmn`` global modes, sparse AE3D artifacts, and
publication figures.

Radial Finite-Difference Energy
-------------------------------

The scalar radial operator uses a centered finite-difference envelope energy.
For constant coefficients and Dirichlet boundaries the exact discrete spectrum
is

.. math::

   \lambda_j =
   \omega_A^2+
   \frac{4 c_r}{\Delta s^2}
   \sin^2\left(\frac{j\pi}{2(N+1)}\right).

The block radial operator uses the same idea with matrices at each face:

.. math::

   E_r = \sum_f (u_{f+1}-u_f)^\dagger C_f (u_{f+1}-u_f)/\Delta s_f^2.

``assemble_radial_block_matrices``, ``assemble_radial_block_sparse_matrices``,
and ``radial_block_stiffness_matvec`` implement the same energy. Tests compare
dense, sparse, and matrix-free paths directly.

Boundary Conditions
-------------------

Dirichlet boundaries
   The first and last radial nodes are fixed to zero and are omitted from the
   active degrees of freedom. This is the default for global-mode examples.

Neumann boundaries
   All radial nodes are active and endpoint face terms omit the missing
   exterior neighbor. The validation registry checks the exact
   constant-coefficient zero-gradient spectrum.

Future variants
   Robin, parity, conducting-wall, and experimentally motivated edge
   conditions should be added as separate boundary policies with analytic
   finite-difference gates before use in publications.

Sparse and Krylov Solves
------------------------

``solve_radial_block_eigenmodes_sparse`` assembles CSR matrices and calls
SciPy's Hermitian generalized sparse eigensolver. It is intentionally not a
JAX-differentiable production path. Its purpose is:

* independent parity with dense assembly,
* benchmark coverage for larger matrix dimensions,
* preparation for cross-code comparisons where matrices are naturally sparse,
* separation of physics error from dense-memory limitations.

The sparse solver reports frequencies, eigenvectors, mode structures,
residuals, and the matrix handles used for verification.

Global Workflow Artifacts
-------------------------

``run_global_eigenmodes`` is the TOML-facing orchestration layer around these
operators. For analytic or VMEC-profile cases it builds
``RadialEigenmodeProblem`` from the selected primary ``(m,n)`` harmonic and
fills boundary amplitudes with the Dirichlet value. For ``boozmn`` cases it
builds ``RadialBlockEigenmodeProblem`` with real Boozer local matrices and the
metric-derived radial stiffness. The workflow writes:

* sorted ``omega^2`` values, frequencies, and residuals,
* ``(mode, surface, harmonic)`` amplitudes with real, imaginary, and magnitude
  columns,
* JSON metrics for residuals, radial-envelope maxima, and harmonic power
  fractions,
* enough geometry and mode-family metadata to reproduce the run.

Matrix-Free Kernels
-------------------

``radial_block_stiffness_matvec`` and ``radial_block_mass_matvec`` apply the
global radial block operator without forming ``K`` or ``M``. The first uses are
Rayleigh quotient and residual checks. The planned production uses are:

* JAX-native Krylov prototypes,
* shift-invert or filtered subspace iteration,
* ``lineax`` or ``jax.lax.custom_linear_solve`` adjoint experiments,
* GPU memory scaling benchmarks.

No matrix-free eigensolver should replace the dense reference until it passes
the same constant-spectrum, residual, and derivative gates.

Custom Derivatives for Isolated Modes
-------------------------------------

For an isolated eigenpair, FAX uses the first-order formula

.. math::

   d\lambda =
   \frac{x^\dagger(dK-\lambda dM)x}{x^\dagger M x}.

``generalized_hermitian_isolated_eigenvalue`` returns the dense eigenvalue but
defines a custom JVP using this expression. This avoids differentiating through
implementation details of the full eigensolver and gives a formula that can be
checked against SciPy, AE3D artifacts, or finite differences.

Rules for use:

* use it only for isolated modes,
* report the mode index and evidence that the mode is separated,
* avoid interpreting single-mode gradients through crossings,
* use cluster objectives when the branch ordering is ambiguous.

Subspace Derivatives for Clusters
---------------------------------

At degeneracy, individual eigenvectors are basis-dependent. FAX therefore
constructs an ``M``-orthonormal cluster basis ``Q`` and projects the tangent
operator:

.. math::

   H_c = Q^\dagger(dK-\lambda_c dM)Q.

The eigenvalues of ``H_c`` are first-order splitting directions. The trace
divided by cluster size is the derivative of the cluster mean. This is the
quantity used by ``generalized_hermitian_cluster_mean_eigenvalue`` and the
radial/Boozer cluster-mean objectives.

Acceptance checks include:

* exact degenerate splitting directions against a hand-computed matrix,
* cluster-mean custom-JVP versus centered finite differences,
* radial-block duplicate-harmonic degeneracy,
* real-Boozer cluster-mean design-gradient finite differences.

Fast-Ion Projection
-------------------

The current fast-ion algorithm evaluates resonance overlap on the radial grid
and weights it by normalized eigenfunction intensity:

.. math::

   w_k(s) = \sum_h |\xi_{kh}(s)|^2.

The radial trapezoid rule is used for profile weighting. This makes drive
screens differentiable with respect to profile parameters, mode frequencies,
and eigenfunction structure. It does not yet solve drift-kinetic equations.

Plot Generation
---------------

Documentation figures are generated from executable scripts and validated by
``tools/qa_figures.py``. The QA script checks that each figure is nonempty and
that PNG/SVG artifacts have plausible rendered content. Figure generation is
part of the local quality gate before pushing.

Performance Measurement
-----------------------

``fax benchmark`` separates cold compile time from warmed runtime where
possible. Benchmark families currently include analytic local branches, global
block dense solves, sparse/Krylov solves, matrix-free application, VMEC/Boozer
adapters, STELLGAP artifacts, and AE3D artifacts. Runtime tables are not
physics claims unless the associated validation metric also passes.
