Development
===========

Run tests:

.. code-block:: bash

   python -m pytest

Build documentation:

.. code-block:: bash

   sphinx-build -W -b html docs docs/_build/html

Regenerate validation figures:

.. code-block:: bash

   python tools/generate_validation_figures.py
   python tools/qa_figures.py

Inspect registered validation gates:

.. code-block:: python

   from fax.validation import run_validation_suite

   for result in run_validation_suite():
       print(result.gate_id, result.status, result.metric, result.tolerance)

Local quality gate while GitHub Actions is unavailable:

.. code-block:: bash

   python -m ruff check .
   python -m mypy src/fax
   python -m pytest
   sphinx-build -W -b html docs docs/_build/html
   python tools/generate_validation_figures.py
   python tools/qa_figures.py
   python -m build

Local workflow smoke checks:

.. code-block:: bash

   fax convergence examples/minimal.toml --radial-points 16 --radial-points 32 --angular-points 8 --angular-points 12 --mode-max 1 --mode-max 2
   fax benchmark examples/minimal.toml --repeats 1 --warmups 0
   fax benchmark examples/minimal.toml --repeats 1 --warmups 0 --family analytic-local --family global-block --family global-matrix-free --family global-krylov
   fax benchmark examples/minimal.toml --repeats 1 --warmups 0 --family stellgap-ctok --family stellgap-executable --family ae3d-artifact --family ae3d-global-mode

Benchmark packets
-----------------

``fax benchmark`` always includes the public continuum workflow as a baseline.
Named families add timed kernels with reviewer-facing physics gates. This
follows the performance patterns observed in the neighboring ``vmec_jax`` and
``booz_xform_jax`` branches: report cold and warm timings separately, block on
JAX work before stopping timers, avoid speed claims without a passing physics
metric, and keep external-code parity artifacts optional but reproducible.
The ``stellgap-executable`` family is a status-capture benchmark: it attempts a
bounded local ``xstgap 99 2048`` CTOK run when the executable and public inputs
are present. Only ``completed`` runs that produce ``alfven_spec`` should be
used in external-code speed tables; ``failed``, ``timeout``, or ``skipped``
runs remain diagnostic metadata. ``ae3d-global-mode`` is a public-matrix
benchmark that independently solves AE3D's generalized matrix problem and
reports frequency plus ``B``-weighted mode-shape parity.

Progress Output
---------------

Long-running functions should report progress through ``fax.progress`` instead
of ad hoc ``print`` calls. The CLI uses progress output by default and disables
it only for ``--json`` or ``--quiet``.

Future solvers should report:

* configuration and resolved input paths,
* phase starts and completions,
* per-case or per-validation-gate start/done messages,
* elapsed time and estimated remaining time,
* current residuals or convergence metrics,
* current resolution, mode-family, or scan point,
* output paths as artifacts are written.

Before physics kernels are merged, each new operator should include:

* an analytic unit test,
* a shape/JIT test,
* a finite-difference or adjoint-consistency test when differentiable,
* a documented validation path in ``plan.md`` or a dedicated validation note.
