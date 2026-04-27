Research Examples
=================

The examples in ``examples/`` are intended to be runnable from a fresh editable
checkout and to produce files that can be inspected by collaborators and
reviewers. They use the current validated scaffold: analytic continuum limits,
metric-matrix checks, VMEC ``wout`` adapter smoke tests, and differentiability
demonstrations.

Install the development extras first:

.. code-block:: bash

   python -m pip install -e ".[dev]"

Configuration Examples
----------------------

``examples/analytic_circular_tokamak.toml``
   Analytic continuum and metric scaffold case. Use it for first continuum,
   plotting, and local performance checks.

``examples/analytic_convergence.toml``
   Small analytic case sized for radial/angular/Fourier convergence sweeps.

``examples/landreman_paul_qa_lowres.toml``
   VMEC-backed Landreman-Paul QA low-resolution target. This requires the
   neighboring ``vmec_jax`` checkout data path documented in the README.

Validation Packet
-----------------

Run a compact reviewer-facing packet:

.. code-block:: bash

   python examples/research/validation_packet.py

This writes validation, continuum, convergence, and figure artifacts under
``outputs/research_validation_packet``. It exercises the public Python API and
the same artifact plotting path used by the CLI.

The equivalent CLI sequence is:

.. code-block:: bash

   fax validate examples/analytic_circular_tokamak.toml --tag analytic --tag metric
   fax continuum examples/analytic_circular_tokamak.toml
   fax plot-continuum outputs/analytic_circular_tokamak --output-dir outputs/analytic_circular_tokamak/figures
   fax convergence examples/analytic_convergence.toml --radial-points 16 --radial-points 32 --radial-points 48 --angular-points 8 --angular-points 12 --mode-max 1 --mode-max 2
   fax plot-convergence outputs/analytic_convergence/convergence --output-dir outputs/analytic_convergence/figures

Differentiability
-----------------

Run:

.. code-block:: bash

   python examples/research/density_sensitivity.py

This computes JAX gradients of a mean continuum-branch frequency with respect
to density scale, rotational-transform shift, and magnetic-field scale. It is a
small reproducible sensitivity-analysis template.

Inverse Design
--------------

Run:

.. code-block:: bash

   python examples/research/inverse_iota_shift.py

This is a one-parameter inverse-design toy problem. It adjusts a bounded
rotational-transform shift to move a selected analytic continuum diagnostic
toward a target frequency. It is deliberately simple so that users can inspect
the objective, gradients, and iteration history.

Differentiable Design Study
---------------------------

Run:

.. code-block:: bash

   python examples/research/differentiable_design_study.py --output-dir outputs/differentiable_design_study

This is the first integrated design-facing FAX example. It computes autodiff
sensitivities of global radial eigenfrequencies, propagates parameter
covariance into linearized frequency uncertainty bands, runs a small
gradient-based inverse-design loop to move a mode out of a selected risk
window, and projects a reduced fast-ion drive screen onto the eigenfunctions
before and after the design move.

.. figure:: _static/differentiable_design_study.svg
   :alt: Differentiable FAX design study dashboard.

   Differentiable sensitivity, uncertainty quantification, inverse design, and
   fast-ion risk reduction for the current reduced radial model. This is a new
   research-workflow capability, while full 3-D operator parity remains a
   separate validation lane.

Basic BAE/TAE/EAE Gap Map
-------------------------

Run:

.. code-block:: bash

   python examples/research/basic_alfven_gap_map.py

This writes ``outputs/basic_alfven_gap_map/basic_alfven_gap_map.png`` and
``.svg``. It is a fast large-aspect-ratio reproduction of the standard
Alfvén-continuum gap picture: cylindrical branches, toroidicity-induced TAE
coupling, ellipticity-induced EAE coupling, and a finite-beta/acoustic BAE
accumulation-frequency marker. The script also prints the reference ``q``
locations and frequencies as JSON.

Reduced Global/Gaps/Fast Ions
-----------------------------

Run:

.. code-block:: bash

   python examples/research/global_gap_fastion_screen.py

This prints a JSON packet containing a constant-profile radial spectrum gate,
the lowest reduced radial eigenfrequency for an analytic equilibrium, TAE/EAE/
HAE gap estimates, a reduced fast-ion resonance peak, and the gradient of the
radial-mode avoidance objective.

Boozer Harmonic Validation
--------------------------

Run:

.. code-block:: bash

   python examples/research/boozer_harmonic_validation.py

This prints a JSON packet with Boozer ``|B|`` reconstruction/projection error,
harmonic orthogonality error, straight-field-line phase-slope error, and
normalized non-axisymmetric spectral energy for a synthetic spectrum.

Real ``boozmn`` Local 3-D Continuum
-----------------------------------

Run:

.. code-block:: bash

   python examples/research/boozmn_3d_continuum.py

This reads a real ``boozmn`` file from the neighboring ``booz_xform_jax``
checkout by default, checks the file-level Boozer-Jacobian identity, compares
the embedding-derived Jacobian shape, solves the local 3-D Boozer continuum
matrix, writes SVG/PNG figures, and prints a JSON summary. Use
``--boozmn path/to/boozmn.nc`` for a new Boozer geometry.

Real ``boozmn`` Global 3-D Eigenmodes
-------------------------------------

Run:

.. code-block:: bash

   python examples/research/boozmn_global_3d_eigenmodes.py

This reads a real ``boozmn`` file, builds the local 3-D Boozer surface
matrices, assembles the global radial block eigenproblem with Dirichlet radial
boundaries, writes radial-envelope/harmonic-structure figures, and prints
frequencies plus residuals as JSON. Use ``--boozmn path/to/boozmn.nc`` for a
new geometry.

STELLGAP CTOK Cross-Code Comparison
-----------------------------------

Run after cloning/extracting the public ORNL STELLGAP test artifacts under
``/Users/rogerio/local``:

.. code-block:: bash

   python examples/research/stellgap_ctok_comparison.py

This reads the public CTOK ``boozmn_ctok.nc`` and STELLGAP ``alfven_post``
output, recomputes the local 3-D Boozer continuum branches in FAX, writes an
overlay/error figure, and prints JSON metrics. On the current local fixture the
p95 sorted-spectrum error is about 2.55 percent over 97 matched radial
surfaces.

External-Code Artifact Dashboard
--------------------------------

Run after extracting the public ORNL STELLGAP and AE3D answer artifacts under
``/Users/rogerio/local``:

.. code-block:: bash

   python examples/research/external_code_artifact_dashboard.py

This writes a dashboard and JSON metrics for public-code convention checks:
STELLGAP ``alfven_spec`` post-processing to ``alfven_post``, AE3D
``omega2.dat`` to ``egn_values.dat`` eigenvalue conventions, AE3D reported
eigenpair residuals, and sparse matrix symmetry. These checks are intentionally
artifact-level; the next physics step is full global FAX eigenmode parity
against AE3D/CAS3D/FAR3D cases.

Publication Figure Suite
------------------------

Run:

.. code-block:: bash

   python examples/research/publication_figure_suite.py --output-dir outputs/publication_figure_suite

This writes ten publication-style SVG/PNG figures plus a JSON summary. The
cases are intended to cover the figure families normally requested in
Alfvén-eigenmode studies:

* analytic BAE/TAE/EAE gap taxonomy,
* radial/angular/Fourier convergence and eigensolve residuals,
* VMEC ``wout`` profile/metric continuum inspection for the Landreman-Paul QA
  target,
* Boozer harmonic reconstruction and spectral content,
* real ``boozmn`` local 3-D continuum operation,
* global radial eigenfrequencies, eigenfunctions, and residuals,
* fast-ion resonance drive/damping and eigenfunction overlap,
* autodiff sensitivity checks,
* STELLGAP CTOK local-continuum parity with a gap-anchored FAX global-mode
  screen, and
* AE3D public sparse-matrix/eigenmode residual parity.

Generate selected figures with repeated ``--case`` flags, for example:

.. code-block:: bash

   python examples/research/publication_figure_suite.py --case 09 --case 10 --format pdf

The STELLGAP and AE3D cases require the public ORNL artifacts under
``/Users/rogerio/local``. The suite labels those cases as skipped in the JSON
summary if the local artifacts are unavailable.

.. figure:: _static/publication_suite/09_stellgap_ctok_gap_anchored_global_screen.svg
   :alt: STELLGAP CTOK continuum and FAX global-mode screen.

   STELLGAP CTOK local-continuum parity overlaid with the current reduced FAX
   global radial eigenmode screen. The left panel is a parity/inspection
   figure; it should not be described as full AE3D/CAS3D global physics parity.

.. figure:: _static/publication_suite/10_ae3d_matrix_eigenmode_parity.svg
   :alt: AE3D public matrix and eigenmode parity dashboard.

   Public AE3D sparse matrices and ``egn_mode_asci.dat`` eigenvectors checked
   through ``A x = lambda B x`` residuals and Rayleigh-quotient errors. This is
   now a real external matrix/eigenmode parity gate and a prerequisite for
   future FAX-vs-AE3D operator-level validation.

Regenerating Documentation Figures
----------------------------------

The README and documentation figures are generated by:

.. code-block:: bash

   python tools/generate_validation_figures.py
   python tools/qa_figures.py

``tools/qa_figures.py`` parses generated SVG files and checks PNG dimensions and
nonblank pixel coverage. This is the local guard against empty or malformed
review figures.
