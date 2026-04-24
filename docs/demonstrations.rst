Demonstrations
==============

Publication-quality examples should be reproducible scripts, not notebooks
alone. Planned demonstrations include:

Differentiability
   Compare JAX gradients, custom adjoints, and finite differences for continuum
   gaps, isolated eigenvalues, and perturbative drive metrics.

Accuracy
   Reproduce analytic cylinder/tokamak limits and selected STELLGAP, AE3D,
   CAS3D, and FAR3D benchmarks with documented tolerances.

Current generated figures
   ``tools/generate_validation_figures.py`` creates the first analytic
   continuum scaling figure and the LandremanPaul2021 QA lowres smoke-test
   profile figure used in the validation docs. It also creates the first
   README dashboard combining mode-family branches, a gap proxy, and autodiff
   sensitivities.

Runtime
   Report compile time, warmed runtime, memory, and scaling with radial and
   Fourier resolution on CPU and GPU.

Sensitivity analysis
   Rank density, iota, beta, and boundary Fourier coefficients by impact on a
   selected gap or growth-rate metric.

Uncertainty quantification
   Propagate profile and equilibrium uncertainty through risk metrics using
   batched JAX evaluations.

Inverse design
   Optimize a small set of VMEC boundary/profile parameters to move a dangerous
   mode away from a resonance window while preserving equilibrium constraints.
