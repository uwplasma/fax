Differentiability
=================

``fax`` will use several differentiation modes, chosen by problem size and
mathematical structure.

Dense reference path
--------------------

Small continuum and global-mode problems should differentiate directly through
JAX operations. This path is the correctness reference for custom adjoints.

Generalized eigenvalue path
---------------------------

For isolated Hermitian eigenpairs normalized by ``x.H @ M @ x = 1``, the planned
sensitivity is

.. math::

   d\lambda = x^\dagger (dK - \lambda dM) x.

This formula should be implemented as a custom derivative rule and checked
against direct autodiff and finite differences.

Implicit and matrix-free path
-----------------------------

Large problems should avoid differentiating through every Krylov iteration.
Instead, use implicit differentiation or custom VJP/JVP rules that solve the
adjoint linear system directly.

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

