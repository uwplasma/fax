Full Research-Grade Roadmap
===========================

FAX is not yet in final no-proxy/full-physics status. The current release is a
validated research foundation with reduced, analytic, local-continuum,
artifact-parity, and adapter/convergence layers clearly labeled. The final
target is stricter: all default publication claims must be protected by native
full-physics operators, multi-geometry cross-code parity, kinetic validation,
and end-to-end design-gradient checks.

The machine-readable source of truth is ``fax.maturity`` and the CLI report:

.. code-block:: bash

   fax maturity
   fax maturity --json

Required Milestones
-------------------

Full native 3-D stellarator ideal-MHD eigenmode parity
   FAX must reproduce matched AE3D, CAS3D/CONTI, and STELLGAP global
   eigenfrequencies, eigenfunctions, residuals, normalizations, harmonic
   conventions, and boundary-condition variants on multiple 3-D equilibria.
   Current status: in progress. STELLGAP CTOK local continuum/gap parity and
   AE3D public matrix/mode-shape artifact parity are implemented, but native
   FAX-vs-code global operator parity remains open.

CAS3D/CAS3D-K/FAR3D kinetic growth-rate validation
   FAX must compare energetic-particle drive/damping and growth-rate trends
   with CAS3D-K and FAR3D on matched distributions, equilibria, and mode
   structures. Current status: planned. Synthetic gyro-Landau closure recovery
   is implemented, but real kinetic growth-rate parity is not.

Fully calibrated fast-ion drive/damping
   The reduced fast-ion closure must be calibrated, cross-validated, and given
   uncertainty/domain-of-validity checks before it can replace kinetic-code
   calls within a declared operating domain. Current status: in progress.

End-to-end differentiable VMEC/Boozer/FAX optimization
   Actual VMEC boundary/profile variables must differentiate through a
   controlled ``vmec_jax`` solve, ``booz_xform_jax`` transform, FAX operator,
   eigenvalue objective, and finite-difference/adjoint verification. Current
   status: in progress through builder APIs and live bridge checks, but not yet
   reviewer-grade shape gradients.

Multi-geometry cross-code benchmark suite
   Reactor-design claims require a curated benchmark matrix covering
   quasi-axisymmetric, quasi-helical, tokamak, heliotron/heliac, and
   reactor-relevant configurations with convergence ladders, runtime metadata,
   external-code parity, and fixture hashes. Current status: in progress.

Final no-proxy/no-reduced-model release gate
   This remains blocked until the previous milestones pass. Reduced models may
   remain as optional screening tools, but no default publication claim should
   depend on an unvalidated proxy.

First Implementation Steps
--------------------------

1. Promote the current CTOK STELLGAP and AE3D public artifacts into the
   :doc:`benchmark_manifest` with hashes, conventions, expected metrics, and
   local paths. This first step is implemented for the current public CTOK and
   AE3D artifact families.
2. Add native global operator parity drivers that compare FAX mode structures
   against external-code eigenfunctions, not just external matrix artifacts.
3. Define the FAR3D/CAS3D-K growth-rate data schema and add the first archived
   calibration/validation case.
4. Choose a small set of VMEC boundary variables for the first end-to-end
   differentiable design check and compare JAX gradients to finite differences.
5. Keep ``fax maturity`` and this page updated whenever a milestone moves from
   planned to in-progress or complete.
