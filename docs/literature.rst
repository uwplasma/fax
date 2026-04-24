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
