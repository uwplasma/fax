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
   python -m build

Progress Output
---------------

Long-running functions should report progress through ``fax.progress`` instead
of ad hoc ``print`` calls. The CLI uses progress output by default and disables
it only for ``--json`` or ``--quiet``.

Future solvers should report:

* configuration and resolved input paths,
* phase starts and completions,
* elapsed time and estimated remaining time,
* current residuals or convergence metrics,
* current resolution, mode-family, or scan point,
* output paths as artifacts are written.

Before physics kernels are merged, each new operator should include:

* an analytic unit test,
* a shape/JIT test,
* a finite-difference or adjoint-consistency test when differentiable,
* a documented validation path in ``plan.md`` or a dedicated validation note.
