Literature And Codes
====================

The initial survey anchors the code plan in the following families:

* FAR3D: reduced-MHD/gyro-fluid modelling of linear and nonlinear AE/EPM
  stability with energetic-particle moments.
* STELLGAP: shear Alfvén continuum and gap calculations in 3-D equilibria.
* AE3D: shear Alfvén eigenmode and eigenfrequency calculations in 2-D and 3-D
  toroidal configurations.
* CAS3D and CAS3D-K: full ideal-MHD modes and perturbative kinetic growth-rate
  calculations in stellarators.
* EUTERPE/GTC/ORB5/LIGKA-style gyrokinetic approaches: higher-fidelity kinetic
  references for selected cases.
* BOOZ_XFORM and ``booz_xform_jax``: Boozer-coordinate transforms for
  straight-field-line representations.
* JAX differentiable-solver practice: direct autodiff through small dense
  references, custom JVP/VJP rules for eigenvalues and implicit solves, and
  adjoint methods for scalable PDE-constrained optimization.
* JAX ecosystem candidates: ``lineax`` for linear solves, ``equinox`` for
  PyTree model organization, ``diffrax`` for time-dependent reduced models,
  and ``jaxopt``/``optimistix`` for implicit optimization prototypes.

See ``plan.md`` for source URLs, validation targets, and implementation
priorities.

Current cross-code targets
--------------------------

The first external validation targets should be chosen from sources that expose
enough input data, mode labels, and plotted or tabulated frequencies/gaps to
reproduce a quantitative comparison. Current priority targets are:

* Spong, Sanchez & Weller, "Shear Alfvén continua in stellarators",
  Phys. Plasmas 10, 3217 (2003), and the STELLGAP software record:
  https://www.osti.gov/biblio/code-4633
* TJ-II Alfvén-cascade comparisons using STELLGAP and AE3D:
  https://impact.ornl.gov/en/publications/modeling-of-alfven-cascades-in-the-tj-ii-stellarator-with-stellga/
* FAR3D reduced-MHD/energetic-particle studies of CFQS and related
  stellarator cases:
  https://arxiv.org/abs/2104.07987
* Landreman-Paul precise-quasisymmetry and the follow-on quasisymmetric
  continuum analysis used for the first VMEC-backed target:
  https://doi.org/10.1103/PhysRevLett.128.035001 and
  https://doi.org/10.1017/S0022377825100524
* Boozer-coordinate transformation references and implementation conventions:
  https://princetonuniversity.github.io/STELLOPT/BOOZ_XFORM.html and
  https://github.com/hiddenSymmetries/booz_xform. The Boozer-Jacobian bridge
  uses the standard ``sqrt(g)=(G+iota I)/B^2`` identity also documented in the
  GVEC theory notes:
  https://gvec.readthedocs.io/v1.3.x/user/theory.html.
* Basic large-aspect-ratio gap references for the user-facing BAE/TAE/EAE
  example:
  Fu & Cheng, "Theory of a high-n toroidicity-induced shear Alfven eigenmode in
  tokamaks", Physics of Fluids B 2, 985 (1990),
  https://www.osti.gov/biblio/6896532; Betti & Freidberg, "Ellipticity induced
  Alfven eigenmodes", Physics of Fluids B 3, 1865 (1991),
  https://www.osti.gov/biblio/5722321; Turnbull et al., "Global Alfven modes:
  Theory and experiment", Physics of Fluids B 5, 2546 (1993),
  https://www.osti.gov/biblio/6484922.

Differentiability references
----------------------------

The implemented dense JAX path is intentionally small and direct. Larger
operators should follow the JAX custom-derivative guidance and recent
spectral-adjoint literature: validate direct autodiff first, then introduce
custom VJP/JVP or implicit-adjoint rules only when they reproduce the dense
reference and improve memory/runtime. Useful implementation references include
the JAX custom-derivative guide,
https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html,
JAX implicit linear-solve differentiation,
https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html,
the Lineax differentiable linear-solver paper,
https://arxiv.org/abs/2311.17283, and recent differentiable computational
physics examples such as JAX-FEM,
https://arxiv.org/abs/2212.00964, plus spectral-adjoint preprints such as
https://arxiv.org/abs/2506.14792.

The current FAX differentiable-design examples are not just performance
demonstrations: they are intended to create result types that legacy
non-differentiable AE workflows do not provide directly, including full
frequency sensitivity matrices, linearized uncertainty propagation, inverse
design of risk-window placement, and fast-ion growth changes along an
optimization trajectory.

Local External-Code Artifacts
-----------------------------

The following public repositories and answer artifacts have been cloned under
``/Users/rogerio/local`` for reproducible comparison work:

* ``/Users/rogerio/local/Stellgap`` from
  https://github.com/ORNL-Fusion/Stellgap. The repository describes STELLGAP
  as a shear-Alfvén gap/continuum code for 3-D configurations, including
  stellarators, RFPs, and 3-D tokamaks. Its public CTOK test problem now drives
  the FAX ``stellgap_ctok_local_continuum`` validation gate. A local build
  succeeded using Homebrew ``gcc-15`` for preprocessing, ``mpif90``, the local
  STELLOPT ``libstell`` tree, and explicit NetCDF C/Fortran library paths.
  ``xmetric ctok`` runs on the public CTOK input and regenerates
  ``ae_metric.dat``/``tae_data_boozer``. ``xstgap`` still needs build/runtime
  cleanup on this gfortran/OpenMPI stack before it can be an automated
  executable parity gate.
* ``/Users/rogerio/local/AE3D`` from
  https://github.com/ORNL-Fusion/AE3D. The OSTI software record states that
  AE3D solves reduced-MHD shear-Alfvén eigenmodes and eigenfrequencies in 2-D
  and 3-D magnetic confinement devices. The public ``egn_values.dat`` answer
  table, ``omega2.dat`` table, sparse matrix artifacts, and
  ``egn_mode_asci.dat`` eigenvectors are currently used for parser,
  eigenvalue-convention, residual, matrix-symmetry, and matrix/eigenmode
  parity gates. Full FAX-reduced-operator vs AE3D-operator physics parity
  remains open.
* ``/Users/rogerio/local/FAR3d`` from https://github.com/DSfusion/FAR3d. The
  FAR3D review describes a reduced-MHD/gyro-fluid code for AE/EPM stability,
  energetic-particle moment closures, acoustic coupling, and nonlinear studies.
  A local build attempt reached compilation but stopped at the CUDA Fortran
  ``cudafor.mod`` dependency in ``matrix.f90``; a CPU-only build path should be
  resolved before using FAR3D for automated parity gates.
* ``/Users/rogerio/local/shearalfvenwave`` from
  https://github.com/arknyazev/shearalfvenwave. This helper repository
  advertises tools for interacting with STELLGAP, AE3D, and FAR3D and is a
  candidate source of input/output conversion conventions.
* ``/Users/rogerio/local/PyStellgap`` from
  https://github.com/ejpaul/PyStellgap. This is a useful Python-side reference
  for STELLGAP plotting and data handling conventions.

Current quantitative external-code status:

* FAX local 3-D Boozer continuum branches match the public ORNL STELLGAP CTOK
  ``alfven_post`` answer with p95 sorted-spectrum relative error about
  ``2.55e-2`` over 97 matched radial surfaces.
* FAX compares the same CTOK surfaces through largest adjacent-branch local gap
  centers and widths; the current p95 gap metric is below the registered
  ``6.0e-2`` gate.
* FAX reproduces STELLGAP ``alfven_spec`` to ``alfven_post`` post-processing
  with max relative error about ``4.9e-7`` over 3848 public CTOK rows.
* FAX reproduces AE3D ``omega2.dat`` to ``egn_values.dat`` eigenvalue
  conventions with max relative error about ``6.9e-7`` over 1287 modes, checks
  public AE3D sparse matrix symmetry, and verifies the first public AE3D
  eigenvectors against ``A x = lambda B x`` with max residual about
  ``3.3e-6`` over the first 8 modes. This closes AE3D public artifact/matrix
  parity, but not the broader FAX reduced-operator vs AE3D full-operator
  validation lane.
