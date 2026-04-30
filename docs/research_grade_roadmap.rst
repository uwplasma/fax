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
   AE3D public matrix/mode-shape plus continuum-placement artifact parity are
   implemented, but native FAX-vs-code global operator parity remains open.

CAS3D/CAS3D-K/FAR3D kinetic growth-rate validation
   FAX must compare energetic-particle drive/damping and growth-rate trends
   with CAS3D-K and FAR3D on matched distributions, equilibria, and mode
   structures. Current status: in progress. Synthetic gyro-Landau closure
   recovery, the held-out ``fax kinetic-parity`` contract, and ``fax
   far3d-ingest`` for FAR3D ``lincheck`` output are implemented. Real FAR3D
   GPU artifacts now cover parser traceability, run-length convergence, and a
   finite-beta scan, but calibrated held-out kinetic growth-rate parity is not
   complete.

Fully calibrated fast-ion drive/damping
   The reduced fast-ion closure must be calibrated, cross-validated, and given
   uncertainty/domain-of-validity checks before it can replace kinetic-code
   calls within a declared operating domain. Current status: in progress.

End-to-end differentiable VMEC/Boozer/FAX optimization
   Actual VMEC boundary/profile variables must differentiate through a
   controlled ``vmec_jax`` solve, ``booz_xform_jax`` transform, FAX operator,
   eigenvalue objective, and finite-difference/adjoint verification. Current
   status: in progress. The first maintained VMEC-boundary gradient gate now
   differentiates Landreman-Paul QA boundary coefficients through the
   differentiable fixed-boundary ``vmec_jax`` path, ``booz_xform_jax`` Boozer
   harmonics, and a FAX global Alfvén risk objective, with centered
   finite-difference parity, a descent check, and a VMEC-iteration refinement
   ladder. A Boozer/angular gradient-refinement gate now also checks drift
   between the current coarse and refined Boozer/operator settings, and a
   nested radial-surface gate checks five-to-seven surface drift. Remaining
   work is to extend this to high-iteration converged VMEC2000 residual
   solves, multi-parameter adjoint/pullback checks, and multi-geometry
   optimization campaigns.

Multi-geometry cross-code benchmark suite
   Reactor-design claims require a curated benchmark matrix covering
   quasi-axisymmetric, quasi-helical, tokamak, heliotron/heliac, and
   reactor-relevant configurations with convergence ladders, runtime metadata,
   external-code parity, and fixture hashes. Current status: in progress.
   A first ``landreman/vmec_equilibria`` VMEC-``wout`` catalog now spans HSX,
   W7-X, NCSX, ARIES-CS, LHD, TJ-II, QHS, and ITER-like references for
   profile-continuum adapter screening; the remaining step is to promote this
   into Boozer/global operator and external-code parity on the same geometries.
   A public ``boozmn`` subset now exercises HSX and Ku-Boozer cases through the
   FAX global radial 3-D Boozer operator with boundary/stiffness variants. The
   remaining acceptance gate is still matched AE3D/CAS3D/STELLGAP physics
   parity on those or equivalent geometries.
   A live ``booz_xform_jax`` generated-Boozer gate now covers public W7-X and
   NCSX VMEC-equilibria ``wout`` files. This should be expanded into a full
   truncation/radial/angular convergence ladder before broad reactor-design
   claims.

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
   calibration/validation case. The schema, ``fax kinetic-parity``, and FAR3D
   ``lincheck`` ingestion are now implemented in :doc:`kinetic_validation`;
   real FAR3D GPU artifacts exist for runtime, convergence, and finite-beta
   scan development. Held-out closure parity against FAR3D/CAS3D-K remains to
   be added.
4. Choose a small set of VMEC boundary variables for the first end-to-end
   differentiable design check and compare JAX gradients to finite differences.
   This is implemented for the Landreman-Paul QA low-resolution input via
   ``fax.vmec_boozer_design`` and
   ``examples/research/vmec_boozer_alfven_design_gradient.py``. The first
   VMEC-iteration gradient ladder is now implemented in
   ``examples/research/vmec_boozer_gradient_convergence.py``. The next step is
   to extend the ladder over additional boundary coefficients, then align the
   path with high-iteration VMEC/discrete-adjoint gradients.
5. Use the VMEC-equilibria public Boozer subset to select the next
   HSX/Ku-Boozer parity cases, then add matching STELLGAP/AE3D/CAS3D/FAR3D
   artifacts where possible. Extend to W7-X/LHD/NCSX once Boozer artifacts are
   generated reproducibly.
6. Keep ``fax maturity`` and this page updated whenever a milestone moves from
   planned to in-progress or complete.
