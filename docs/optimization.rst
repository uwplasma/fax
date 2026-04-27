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

New differentiable study tools
------------------------------

The next executable layer is built around publication-style design studies:

``radial_mode_frequency_vector``
   Maps profile parameters to low-order global radial eigenfrequencies.

``radial_mode_frequency_jacobian``
   Returns the autodiff sensitivity matrix for density, rotational transform,
   and magnetic-field variables.

``linearized_radial_mode_uncertainty``
   Propagates a parameter covariance matrix to frequency uncertainty bands via
   ``J C J^T``.

``radial_mode_inverse_design`` and ``gradient_descent_minimize``
   Provide a transparent inverse-design loop for small research examples and
   regression tests.

``radial_fast_ion_growth_screen`` and ``radial_fast_ion_risk_objective``
   Combine frequency-window avoidance with a reduced fast-ion resonance screen
   projected onto radial eigenfunctions.

The example

.. code-block:: bash

   python examples/research/differentiable_design_study.py

generates the current dashboard for sensitivity analysis, uncertainty
quantification, inverse design, and fast-ion risk reduction:

.. figure:: _static/differentiable_design_study.svg
   :alt: Differentiable design study dashboard.

   A differentiable global-mode design study: frequency sensitivities,
   linearized UQ, optimization history, and fast-ion growth before and after
   the design move. This is intended as the first path toward results that
   legacy non-differentiable AE workflows do not provide directly.

Power-Plant Assessment
----------------------

For reactor design studies, ``fax`` should produce a risk dashboard:

* continuum and gap maps over radius,
* expected alpha and auxiliary-heating resonance overlays,
* mode-family risk ranking,
* sensitivity to density, iota, beta, and boundary-shape parameters,
* uncertainty intervals and worst-case metrics,
* recommended design changes expressed as gradients or shape sensitivities.
