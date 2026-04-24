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
