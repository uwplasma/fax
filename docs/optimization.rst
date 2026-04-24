Optimization
============

The optimization goal is to make Alfvén-mode risk visible and differentiable
during stellarator design.

Design Variables
----------------

Candidate differentiable variables include VMEC boundary Fourier coefficients,
profile parameters, density and temperature uncertainty parameters, fast-ion
source parameters, and eventually coil-proxy variables.

Objectives
----------

Initial objectives should be smooth functions of the continuum and eigenmode
outputs:

* maximize frequency separation between core fast-ion resonances and weakly
  damped gaps,
* penalize small continuum spacing around expected alpha-particle drive windows,
* reduce perturbative growth-rate estimates,
* maintain validated quasisymmetry, aspect-ratio, volume, beta, and engineering
  constraints supplied by other tools,
* penalize objectives whose uncertainty intervals cross risk thresholds.

Implemented reduced objective
-----------------------------

``fax.optimization.radial_mode_avoidance_objective`` is the first executable
design objective. It maps
``[log_density_scale, iota_shift, log_field_scale]`` to a smooth penalty on
the lowest radial eigenmode frequencies entering a dangerous frequency window.
This is a reduced profile fixture, but it is fully differentiable and gives the
testbed for VMEC/vmec_jax shape and profile variables.

Power-Plant Assessment
----------------------

For reactor design studies, ``fax`` should produce a risk dashboard:

* continuum and gap maps over radius,
* expected alpha and auxiliary-heating resonance overlays,
* mode-family risk ranking,
* sensitivity to density, iota, beta, and boundary-shape parameters,
* uncertainty intervals and worst-case metrics,
* recommended design changes expressed as gradients or shape sensitivities.
