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

Planned demonstrations include:

Runtime
   Report compile time, warmed runtime, memory, and scaling with radial and
   Fourier resolution on CPU and GPU.

Sensitivity analysis
   Rank density, iota, beta, and boundary Fourier coefficients by impact on a
   selected gap or growth-rate metric.

Uncertainty quantification
   Propagate profile and equilibrium uncertainty through risk metrics using
   batched JAX evaluations.

Full stellarator inverse design
   Optimize a small set of VMEC boundary/profile parameters to move a dangerous
   mode away from a resonance window while preserving equilibrium constraints.
