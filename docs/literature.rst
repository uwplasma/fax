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
and recent spectral-adjoint preprints such as
https://arxiv.org/abs/2506.14792.
