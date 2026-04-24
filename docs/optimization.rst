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

Power-Plant Assessment
----------------------

For reactor design studies, ``fax`` should produce a risk dashboard:

* continuum and gap maps over radius,
* expected alpha and auxiliary-heating resonance overlays,
* mode-family risk ranking,
* sensitivity to density, iota, beta, and boundary-shape parameters,
* uncertainty intervals and worst-case metrics,
* recommended design changes expressed as gradients or shape sensitivities.

