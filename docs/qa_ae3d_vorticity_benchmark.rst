Native QA vorticity benchmark
=============================

``examples/research/qa_ae3d_vorticity_benchmark.py`` builds FAX's physical-
potential matrices from the published QA Boozer equilibrium and solves them
with the existing Hermitian eigensolver. It does not import AE3D matrices,
perform particle tracing, download data, or generate plots. It saves each
completed case before advancing and defaults to a 120-second wall cap.

Reference and matching
----------------------

The physical potential reference comes from Knyazev et al. (2026),
`published archive <https://doi.org/10.5281/zenodo.21299908>`_, member
``simulations_fig4_fig5/QA/all_inputs/ae3d_saw_scaled/QA_35harmonics_scale0_001.npy``.
The compact ``examples/research/references/qa_ae3d_potential.npz`` preserves
all 49 radial samples, 35 signed potential harmonics and source hashes.
It has numeric arrays and JSON metadata and loads with ``allow_pickle=False``.
The equilibrium must match its pinned SHA256; a different field is rejected.
The flat QA mass density is 1.88e-6 kg/m^3, from the paper's section 3.

The wave-solver comparison uses 30.308607688 kHz. The
`public AE3D source <https://github.com/ORNL-Fusion/AE3D/blob/master/ae_solve_version31.f>`_
sets ``scale_khz=(1000*2*pi)**2`` in its inertia, so the square root of its
eigenvalue is in kHz. The archived FIRM3D importer instead uses the same
number times 1000 as angular frequency (30308.607688 rad/s). Those conventions
are recorded separately. This benchmark does not modify the imported wave
or any previously recorded FIRM3D/ESSOS loss result.

The `paper appendix A <https://arxiv.org/html/2603.03118v1#A1>`_ uses a cosine
basis. Its QA table 2 contains 158 harmonics, while the transport export
retains 35. ``--harmonics retained`` is a truncation diagnostic;
``--harmonics paper`` selects the complete list given in that table.
The actual archived header, read on 2026-10-21 using a single 64 KiB HTTP
range, contains 162 harmonics. Use ``--harmonics archive`` for new comparisons
to that run. Its n=4 group starts at m=0, and its last group has n=16;
the table instead starts that group at m=4 and labels the last group n=20.
The original table option is retained to reproduce the historical screens.
Both larger-basis options compare only the 35 available reference profiles,
and separately report the native potential norm in the other harmonics.
These comparisons cannot establish full eigenfunction parity without the
remaining AE3D coefficients.

``examples/research/references/qa_ae3d_harmonic_header.json`` records the exact
labels and bounded-header provenance. The archived radial coordinates and
the exported reference coordinates agree exactly, and the header contains
the same eigenvalue as the transport export. No original QA stiffness or
inertia matrices are present in the archive index.

Mode selection maximizes the overlap of the signed physical-potential
profiles, without fitting frequency or individual harmonics. A single global
amplitude/sign (or complex phase) is removed. Comparisons evaluate the actual
piecewise linear finite elements, rather than introducing a cubic spline.
The nearest-frequency mode is also recorded to expose cases where frequency
agreement alone would select the wrong shape.

The preliminary acceptance limits are frequency error <=1%, aligned retained-
profile relative L2 error <=10%, and eigenpair residual <=1e-8. Mesh, harmonic
and boundary convergence are additional requirements for a matched benchmark.

Initial annular CPU screen
--------------------------

The compact results are in
``examples/research/references/qa_vorticity_screen_20261020.json``. These are
measured results, not expected answers used to fit the operator. All cases
use the cosine basis and Dirichlet endpoints at s=0.01 and 0.99.

.. list-table:: Native modes selected by their retained-profile overlap
   :header-rows: 1

   * - Harmonics
     - Radial nodes
     - Angular grid
     - Frequency [kHz]
     - Aligned profile error
   * - 35
     - 8
     - 64 x 64
     - 27.0104
     - 71.2%
   * - 35
     - 12
     - 64 x 64
     - 26.8679
     - 54.1%
   * - 35
     - 16
     - 64 x 64
     - 27.9272
     - 68.5%
   * - 35
     - 24
     - 64 x 64
     - 27.5833
     - 72.8%
   * - 158
     - 6
     - 80 x 80
     - 24.2297
     - 78.5%
   * - 158
     - 8
     - 80 x 80
     - 26.0937
     - 63.7%
   * - 158
     - 12
     - 80 x 80
     - 26.2585
     - 59.7%
   * - 158
     - 16
     - 80 x 80
     - 26.8734
     - 65.8%

**The native QA comparison does not pass.** Eigenpair residuals of the
selected modes are below 3e-13, but frequency and shape agreement fail.
The selected mode can change across refinements; these results do not show
convergence to the reference eigenfunction.

Changing the first 8-surface calculation from complex to cosine harmonics
hardly changes its frequency, so that representation choice does not explain
the dominant discrepancy in the truncated case. An independent physical
field-line-tangent identity checks the Fortran toroidal-offset sign to
2.82e-5 relative RMS, while reversing it gives 0.190 RMS. A sign reversal
is therefore not supported by the input geometry.

Remaining controlled differences include radial discretization and boundaries:
the public STELLGAP metric builder averages half-mesh coefficients onto the
full mesh before producing the AE3D input, and AE3D treats m=0 at the magnetic
axis with a zero-derivative condition. This screen uses linear elements on
the available half-mesh interval with both endpoints fixed to zero. The
public metric builder explicitly identifies s as normalized toroidal flux;
its helper radius sqrt(s) is separate, so no radial-coordinate squaring is
applied here. The full-domain follow-up below tests the axis explanation;
the initial screen does not isolate a missing physical term.

Full-domain axis control
------------------------

The default now uses potential nodes on s=0..1, without coarsening the stored
background geometry. Geometry is reconstructed at interior Gauss points, so
the singular nodal axis metric is never inverted. Interior Fourier coefficients
are interpolated linearly in s, giving the known half-to-full arithmetic
average at full-mesh points. Below the first stored surface, nonzero-m geometry
coefficients continue as r**abs(m). The m=0 axis potential is free, with
dPhi/dr=0 for the linear-in-s basis; all other m vanish there. All harmonics
vanish at the outer conducting shell.

``--radial-domain full-dirichlet`` uses the exact same full-domain geometry
and potential mesh but fixes every axis potential to zero. This isolates
the free m=0 axis condition from the domain and interpolation changes.
``--radial-domain annulus`` reproduces the initial screen above.
These are variational finite elements, not a claim to reproduce the exact
AE3D coefficient-averaging finite-element assembly or its last-interior-point
extrapolation. The source confirms that AE3D also uses linear finite elements;
the earlier description of it as a finite-difference stencil was incorrect.

Completed results and provenance are saved in
``examples/research/references/qa_vorticity_axis_20261020.json``:

.. list-table:: Full 158-harmonic cosine basis, 80 x 80 angular grid
   :header-rows: 1

   * - Axis condition
     - Radial nodes
     - Frequency [kHz]
     - Aligned profile error
   * - Regular m=0
     - 12
     - 26.3768
     - 60.5%
   * - Regular m=0
     - 16
     - 26.8241
     - 65.7%
   * - All values zero
     - 16
     - 26.8290
     - 66.0%

**The m=0 axis condition is not the main cause of this discrepancy.** On the
same 16-node mesh it changes the selected frequency by only 0.0183%, whereas
the regular-axis frequency remains 11.5% below the published 30.3086 kHz.
The large profile error also remains. Extending the domain and keeping the
background resolution fixed has not recovered the AE3D mode. Finer meshes
and agreement of the discrete operator still require verification; no
frequency or density adjustment is made to hide the failed comparison.

An initial paired job hit its 120-second total cap before completing the
fixed-axis control. Batching the Fourier geometry reconstructions then reduced
warm assembly time by 4.4 times, with relative selected-frequency changes
below 5e-15 and aligned-error changes below 1e-13. Both regular-axis cases
and the control completed in 59 seconds under a fresh 120-second total cap.
The compact record preserves both implementations' measured results and hashes.

Independent checks cover cancellation of the circular-axis singularity in
the inertia integral, uniform-wave dispersion, restoration of constrained
degrees of freedom, and compiled reverse derivatives through the full-domain
geometry and an isolated eigenvalue. These validate this boundary treatment;
they do not turn the failed QA comparison into cross-code validation.

The subsequent :doc:`qa_ae3d_operator_audit` isolates source-level mode-index
differences in the magnetic-shear bending terms. It rules out the prefactor
derivative sign as the main explanation and checks FAX independently in
Cartesian coordinates. Agreement with the archived mode therefore must not
be used alone to decide which operator is correct.

Reproduction
------------

Use the QA ``boozmn_betaQA.nc`` from the same archive or the already verified
ESSOS QA packet. No acquisition is performed by this script. Install this
FAX checkout in the chosen environment, then run a bounded CPU screen:

.. code-block:: bash

   python -m pip install -e .
   python -u examples/research/qa_ae3d_vorticity_benchmark.py \
     --boozmn /path/to/boozmn_betaQA.nc --harmonics paper --basis cosine \
     --radial-domain axis --cases 12:80 16:80 --platform cpu --wall-seconds 120 \
     --output outputs/qa_vorticity_axis_cpu

For the control, repeat with ``--radial-domain full-dirichlet --cases 16:80``
and a fresh output directory. For the historical annular screen use
``--radial-domain annulus``.

For a larger, separately authorized GPU refinement on a machine with enough
memory for the dense matrices:

.. code-block:: bash

   python -u examples/research/qa_ae3d_vorticity_benchmark.py \
     --boozmn /path/to/boozmn_betaQA.nc --harmonics paper --basis cosine \
     --radial-domain axis --cases 16:80 24:80 51:80 \
     --platform cuda --wall-seconds 1800 --output outputs/qa_vorticity_axis_gpu

The 51-node case has 7743 active scalar degrees of freedom and uses potential
nodes at the original full-mesh spacing. No GPU run is implied by this
command being documented. Radial refinement alone does not establish
matched AE3D discretization.

Outputs include JSON provenance/timings/gates, a selected mode packet and
spectrum diagnostics. The saved mode is identified as a cosine or complex
spatial basis; it is not automatically a tracing-ready traveling wave.
