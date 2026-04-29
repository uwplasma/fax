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

``boozer_global_mode_frequency_vector`` and related Boozer helpers
   Apply the same differentiable study pattern to real ``boozmn`` global
   radial operators. The design variables currently scale mass density, shift
   rotational transform, and scale radial stiffness; a fourth optional
   parameter scales fast-ion density in the growth-risk objective.

``boozer_global_isolated_mode_frequency_hz`` and ``boozer_global_isolated_mode_avoidance_objective``
   Track a single isolated real-Boozer global mode through the custom-JVP
   generalized-eigenvalue path. This is the preferred objective form when a
   publication study follows one mode away from a resonance window and needs a
   finite-difference reviewer check.

``boozer_global_cluster_mean_frequency_hz`` and ``boozer_global_cluster_mean_avoidance_objective``
   Track a mode family or near crossing through the cluster-mean custom-JVP
   path. This is the preferred objective form when branch labels exchange or
   when an optimizer should move a crossing region rather than one isolated
   eigenvector.

``boozer_pipeline_isolated_mode_avoidance_objective``
   Applies the same isolated-mode objective to a callable Boozer-geometry
   builder, which is the API boundary for differentiable ``vmec_jax`` and
   ``booz_xform_jax`` pipelines.

``boozer_pipeline_cluster_mean_avoidance_objective``
   Applies the crossing-safe cluster objective to the same callable geometry
   builder API, enabling future ``vmec_jax -> booz_xform_jax -> FAX`` design
   studies to optimize mode families without invalid single-mode gradients.

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

The real-Boozer counterpart is:

.. code-block:: bash

   python examples/research/boozmn_differentiable_design_study.py

.. figure:: _static/boozmn_differentiable_design_study.svg
   :alt: Real boozmn differentiable design dashboard.

   Differentiable design on a real ``boozmn`` global 3-D radial operator. This
   is the bridge from validated real-Boozer forward operators to FAX-native
   optimization, UQ, and sensitivity studies on stellarator geometry.

Power-Plant Assessment
----------------------

For reactor design studies, ``fax`` should produce a risk dashboard:

* continuum and gap maps over radius,
* expected alpha and auxiliary-heating resonance overlays,
* mode-family risk ranking,
* sensitivity to density, iota, beta, and boundary-shape parameters,
* uncertainty intervals and worst-case metrics,
* recommended design changes expressed as gradients or shape sensitivities.
