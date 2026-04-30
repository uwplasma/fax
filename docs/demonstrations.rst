Demonstrations
==============

Publication-quality examples should be reproducible scripts, not notebooks
alone. Runnable examples are listed in :doc:`research_examples`.

Current demonstrations include:

Differentiability
   Compare JAX gradients, custom adjoints, and finite differences for continuum
   gaps, isolated eigenvalues, and perturbative drive metrics.

Accuracy
   Reproduce analytic cylinder/tokamak limits and selected STELLGAP, AE3D,
   CAS3D, and FAR3D benchmarks with documented tolerances.

Current generated figures
   ``tools/generate_validation_figures.py`` creates the README/docs figures from
   analytic fixtures, LandremanPaul2021 QA lowres adapter data, public
   continuum artifacts, and public convergence artifacts.

Validation packet
   ``examples/research/validation_packet.py`` runs validation, continuum,
   convergence, and plotting workflows and writes a compact local review
   packet under ``outputs/research_validation_packet``.

Density sensitivity
   ``examples/research/density_sensitivity.py`` computes JAX sensitivities of a
   mean branch-frequency diagnostic.

Inverse design
   ``examples/research/inverse_iota_shift.py`` optimizes a bounded scalar
   rotational-transform shift in a transparent analytic fixture.

Basic BAE/TAE/EAE gap map
   ``examples/research/basic_alfven_gap_map.py`` generates a two-panel
   large-aspect-ratio continuum figure showing the TAE, EAE, and BAE reference
   gaps with a compact JSON summary.

Reduced global/gap/fast-ion screen
   ``examples/research/global_gap_fastion_screen.py`` combines the radial
   eigenmode gate, analytic gap estimates, reduced fast-ion resonance screen,
   and radial-mode avoidance gradients.

Boozer harmonic validation
   ``examples/research/boozer_harmonic_validation.py`` checks Boozer scalar
   field reconstruction/projection, harmonic orthogonality, and straight-field
   line phase conventions.

Real Boozer 3-D continuum and global modes
   ``examples/research/boozmn_3d_continuum.py`` and
   ``examples/research/boozmn_global_3d_eigenmodes.py`` apply the real
   ``boozmn`` geometry path to local surface continuum matrices and global
   radial block eigenmodes with plotted mode structures and residuals. The
   same ``BoozerGeometry`` contract can be populated directly from a VMEC
   ``wout`` through ``run_booz_xform_jax_geometry_from_wout`` when
   ``booz_xform_jax`` is available.

Runtime
   ``fax benchmark`` reports cold runtime, warmed runtime, artifact size,
   process memory, JAX backend metadata, and physics-gate margins. The first
   named families cover the analytic continuum limit, dense, matrix-free, and
   sparse-Krylov global radial block operators, public STELLGAP CTOK artifact
   path, bounded STELLGAP executable status capture, public AE3D matrix path,
   and the live ``vmec_jax -> booz_xform_jax -> FAX`` convergence bridge.

Planned demonstrations include:

Sensitivity analysis
   Rank density, iota, beta, and boundary Fourier coefficients by impact on a
   selected gap or growth-rate metric.

Uncertainty quantification
   Propagate profile and equilibrium uncertainty through risk metrics using
   batched JAX evaluations.

Full stellarator inverse design
   Optimize a small set of VMEC boundary/profile parameters to move a dangerous
   mode away from a resonance window while preserving equilibrium constraints.
