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

Before physics kernels are merged, each new operator should include:

* an analytic unit test,
* a shape/JIT test,
* a finite-difference or adjoint-consistency test when differentiable,
* a documented validation path in ``plan.md`` or a dedicated validation note.
