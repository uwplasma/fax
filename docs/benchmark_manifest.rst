Benchmark Manifest
==================

The benchmark manifest is the first implementation step toward a
multi-geometry cross-code validation suite. It records which external-code
artifacts are used, what claim level each artifact supports, which validation
gates protect it, and which next parity step is required before broader
physics claims are allowed.

Inspect it with:

.. code-block:: bash

   fax benchmark-manifest
   fax benchmark-manifest --json
   fax benchmark-manifest --json --hashes

Current Entries
---------------

``stellgap_ctok_local_continuum_gap``
   STELLGAP CTOK public test problem. This supports local 3-D Boozer continuum
   and largest-local-gap parity, not global eigenmode parity. Required
   artifacts are ``boozmn_ctok.nc`` and ``alfven_post``; ``alfven_spec`` is an
   optional raw-output artifact for post-processing parity.

``ae3d_public_matrix_mode_shape``
   AE3D public generalized matrix problem. This supports matrix, dense-spectrum,
   B-weighted mode-shape artifact parity, and AE3D global-mode peak placement
   against the FAX CTOK local Boozer continuum. It does not yet prove that the
   native FAX reduced/global operator reproduces AE3D's full operator.

``far3d_diiid_gpu_lincheck_artifact``
   FAR3D public DIII-D example built and run on the ``office`` GPU host with
   NVHPC/CUDA Fortran. This supports executable-build, runtime, and
   ``lincheck`` ingestion traceability. It is not a converged FAR3D kinetic
   growth-rate parity result and should not be used as a closure-calibration
   benchmark without a longer documented run.

Why This Matters
----------------

Final reactor-design claims require more than passing tests on one machine.
Each cross-code benchmark needs:

* versioned input/output artifacts,
* file hashes,
* expected metrics and tolerances,
* validation gate IDs,
* claim-level language,
* reproducible commands,
* hardware/runtime metadata when performance is discussed.

The manifest will grow into the multi-geometry benchmark suite required by the
full research-grade roadmap.
