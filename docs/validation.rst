Validation
==========

Validation is a project deliverable, not a final clean-up task. Planned
validation layers include:

* analytic scalings for the shear Alfvén continuum and toroidal gap frequencies,
* axisymmetric tokamak TAE/EAE/NAE benchmarks,
* stellarator continuum comparisons against STELLGAP and recent
  quasisymmetric-continuum results,
* global eigenmode comparisons against AE3D, CAS3D, and EUTERPE cases,
* fast-ion growth-rate comparisons against FAR3D and CAS3D-K limits,
* conservation, symmetry, adjoint, and finite-difference gradient checks,
* regression datasets with tolerances, provenance, and stored plotting scripts.

Each published validation case should include the input files, source citation,
expected metric, tolerance, and the exact command used to regenerate the result.

Current Registered Gates
------------------------

``fax.validation`` now contains a small registry of named validation gates. Each
gate has an identifier, physics layer, measured quantity, tolerance, citation,
tags, and command. The current gates are:

* ``analytic_cylinder_frequency``: verifies the uniform-cylinder continuum
  branch is radially constant.
* ``metric_matches_profile_limit``: checks the angular-grid metric matrix
  reduces to the profile continuum formula in the uniform straight-field-line
  limit.
* ``metric_matrix_hermitian``: checks Hermiticity of the local generalized
  continuum stiffness and mass matrices.
* ``metric_eigensolve_residual``: checks dense generalized Hermitian
  eigenvectors through ``Kx = lambda Mx`` residual norms.
* ``density_scale_gradient``: checks JAX autodiff against the analytic
  ``rho^(-1/2)`` density scaling.
* ``metric_angular_convergence``: checks that the analytic metric fixture is
  insensitive to angular-grid refinement.
* ``metric_radial_convergence``: checks that analytic metric-matrix continuum
  branches converge under radial-grid refinement.
* ``analytic_radial_integral_convergence``: checks radial quadrature convergence
  for a smooth analytic continuum branch.
* ``global_mode_constant_profile``: checks the reduced radial eigenmode
  finite-difference spectrum against the exact constant-profile spectrum and
  eigenpair residuals.
* ``global_mode_density_gradient``: checks JAX autodiff through the radial
  generalized eigenproblem against the analytic ``rho^(-1/2)`` scaling.
* ``analytic_tae_gap_center``: checks the analytic TAE crossing condition
  ``iota=2n/(2m+1)``.
* ``basic_gap_map_ordering``: checks that the fast BAE/TAE/EAE example places
  TAE and EAE at their large-aspect-ratio reference ``q`` surfaces and orders
  the BAE, TAE, and EAE frequencies correctly.
* ``bae_temperature_scaling``: checks that the BAE accumulation marker follows
  the expected ``sqrt(T_i)`` scaling at fixed ``T_e/T_i``.
* ``fast_ion_resonance_peak``: checks that the reduced fast-ion drive peaks at
  the transit/precession resonance.
* ``boozer_harmonic_projection``: checks Boozer ``|B|`` harmonic
  reconstruction/projection and Fourier orthogonality on a synthetic spectrum.
* ``boozer_straight_field_line_phase``: checks that Boozer harmonic phases have
  slope ``m*iota-n`` along a straight field line.
* ``boozer_constant_field_continuum``: checks that the reduced Boozer continuum
  matrix recovers ``omega=v_A |n-m*iota|/R`` for constant ``|B|``.
* ``boozer_sideband_coupling_hermitian``: checks that a known Boozer harmonic
  produces nonzero sideband coupling while keeping the matrix Hermitian.
* ``boozer_jacobian_identity``: checks the Boozer-coordinate identity
  ``J_B |B|^2 = G + iota I`` for a synthetic harmonic spectrum.
* ``boozer_metric_constant_field_continuum``: checks that the
  Boozer-Jacobian weighted matrix recovers the constant-field profile limit.
* ``boozer_metric_mass_coupling``: checks that nonuniform ``J_B`` produces
  mass-matrix coupling while ``J_B |B|^2`` cancels sideband stiffness coupling
  in the reduced bridge.
* ``boozer_3d_constant_field_continuum``: checks that the local 3-D Boozer
  operator recovers ``omega=v_A |n-m*iota|/R`` in the constant-field limit.
* ``boozmn_geometry_jacobian_identity``: reads a real ``boozmn`` fixture and
  checks ``sqrt(g)|B|^2 = G + iota I`` from file harmonics.
* ``boozmn_embedding_jacobian_shape``: compares the physical embedding
  Jacobian shape reconstructed from ``R``, ``Z``, and ``p`` against the
  ``boozmn`` Jacobian harmonics up to one surface-wise radial-coordinate scale.
* ``boozmn_3d_continuum_finite``: checks that a real ``boozmn`` local 3-D
  continuum matrix is Hermitian and produces finite non-negative branches.
* ``landreman_paul_adapter_smoke``: checks the Landreman-Paul QA VMEC ``wout``
  profile and metric adapters when local data are available.
* ``landreman_paul_continuum_finite``: checks that the Landreman-Paul profile
  and metric continuum branches are finite and have small eigensolve residuals.
* ``landreman_paul_radial_sampling_convergence``: checks VMEC-backed radial
  sampling refinement for the current adapter.

Run them from Python:

.. code-block:: python

   from fax.validation import registered_validation_gates, run_validation_suite

   for gate in registered_validation_gates():
       print(gate.gate_id, gate.tolerance, gate.citation)

   results = run_validation_suite()

Run the same gates from the CLI, with rich phase-by-phase output and artifacts:

.. code-block:: bash

   fax validate examples/minimal.toml
   fax validate examples/minimal.toml --tag metric --json

The CLI writes ``validation_summary.json`` and ``validation_report.md`` to
``run.output_dir`` unless ``--no-artifacts`` is used.

Resolution sweeps are available through the convergence workflow:

.. code-block:: bash

   fax convergence examples/landreman_paul_qa_lowres.toml
   fax convergence examples/minimal.toml --radial-points 16 --radial-points 32 --angular-points 8 --angular-points 12 --mode-max 1 --mode-max 2 --json
   fax plot-convergence outputs/convergence --output-dir outputs/figures

This writes a CSV table, JSON summary, and Markdown report comparing each
radial/angular/Fourier truncation against the highest-resolution entry in the
sweep. Branch matrices on coarser radial grids are interpolated onto the
reference radial grid before relative L2 errors are computed.

These are scaffold acceptance gates. Publication claims about 3-D stellarator
continuum gaps still require the planned STELLGAP/AE3D/literature comparison
gates and resolution-convergence reports.

First Local Stellarator Target
------------------------------

The first VMEC-backed stellarator target is
``LandremanPaul2021_QA_lowres``. It is a low-resolution version of the precise
quasi-axisymmetric Landreman-Paul configuration and is already available in the
neighboring ``vmec_jax`` checkout as a VMEC input and ``wout`` file. The bundled
``examples/landreman_paul_qa_lowres.toml`` file points to those local paths.

This case should be used first for adapter smoke tests, continuum-map
development, Boozer-pipeline checks, and early performance measurements. It is
not a validation by itself until the relevant continuum/eigenmode metrics are
compared with analytic limits or literature/cross-code references.

Current Smoke-Test Figures
--------------------------

The current figures are generated by ``tools/generate_validation_figures.py``.
They are smoke tests for the first analytic formulas and local VMEC adapter, not
yet full 3-D Alfvén-continuum validation.

.. figure:: _static/analytic_continuum_scaling.svg
   :alt: Analytic large-aspect-ratio continuum smoke-test branches.

   Analytic large-aspect-ratio continuum branches used to test scaling,
   harmonic labeling, and density/iota differentiability.

.. figure:: _static/landreman_paul_qa_lowres_profiles.svg
   :alt: LandremanPaul2021 QA lowres profiles and local continuum branches.

   First VMEC-backed stellarator reference target. The right panel uses the
   local large-aspect-ratio smoke-test formula, so it is a development check
   rather than a validated stellarator continuum calculation.

.. figure:: _static/gap_sensitivity_dashboard.svg
   :alt: Gap and sensitivity dashboard for the LandremanPaul2021 QA lowres smoke test.

   Prototype dashboard combining a mode-family branch map, nearest-branch
   separation, and autodiff sensitivities. This is a figure-design and
   differentiability smoke test.

.. figure:: _static/global_gap_fastion_dashboard.svg
   :alt: Reduced global-mode, gap, fast-ion, and optimization validation dashboard.

   Reduced global radial spectrum, radial eigenfunctions, analytic gap-center
   estimates, fast-ion resonance screen, and optimization-gradient smoke test.
   This exercises the new end-to-end differentiable reduced path without
   claiming full 3-D stellarator eigenmode validation.

.. figure:: _static/basic_alfven_gap_map.svg
   :alt: Basic BAE, TAE, and EAE Alfvén-continuum gap map.

   Fast user-facing large-aspect-ratio reproduction of standard continuum gaps:
   TAE from toroidal ``m,m+1`` coupling, EAE from elliptic ``m,m+2`` coupling,
   and the low-frequency BAE accumulation marker. This is a pedagogical
   analytic reference, not a 3-D stellarator benchmark.

.. figure:: _static/boozer_harmonic_validation.svg
   :alt: Boozer harmonic reconstruction and phase validation dashboard.

   Boozer harmonic reconstruction/projection, Fourier orthogonality, and
   straight-field-line phase checks. These validate representation conventions
   before Boozer metric operators are used for continuum claims. The plotted
   spectrum is synthetic and is not intended to be quasisymmetric.

.. figure:: _static/boozer_continuum_coupling.svg
   :alt: Reduced Boozer continuum coupling validation dashboard.

   Reduced Boozer continuum matrix checks: constant-field reduction to the
   analytic profile limit, Hermitian sideband coupling from a known Boozer
   harmonic, and gap opening as the harmonic amplitude increases.

.. figure:: _static/boozer_jacobian_identity.svg
   :alt: Boozer-Jacobian identity validation dashboard.

   Boozer-Jacobian metric identity checks: sampled ``|B|``, the corresponding
   ``J_B=(G+iota I)/B^2`` weight, the flux-function product ``J_B |B|^2``,
   and the weighted mass-matrix coupling used by the reduced bridge.

.. figure:: _static/boozmn_3d_geometry_operator.svg
   :alt: Real boozmn geometry, Boozer identity, embedding metric, and local 3-D continuum dashboard.

   Real ``boozmn`` geometry gate using the local ``booz_xform_jax`` fixtures:
   sampled ``|B|``, the file-level Boozer-Jacobian identity residual, the
   embedding-derived ``|\\nabla s|^2`` shape, and the first local 3-D Boozer
   continuum branches. This fixture is used to exercise asymmetric Boozer
   harmonics; it is not a Landreman-Paul QA quasisymmetry claim.

.. figure:: _static/landreman_paul_convergence_dashboard.svg
   :alt: LandremanPaul2021 QA radial, angular, and Fourier convergence dashboard.

   VMEC-backed adapter convergence smoke test generated by ``fax convergence``.
   It checks artifact/report plumbing and current radial/angular/Fourier
   refinement behavior; full stellarator continuum validation still requires
   cross-code or literature comparison.
