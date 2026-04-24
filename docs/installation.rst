Installation
============

After the v0.0.1 GitHub release is published to PyPI:

.. code-block:: bash

   python -m pip install uwplasma-fax

The PyPI distribution is ``uwplasma-fax`` because the ``fax`` package name is
already occupied on PyPI. The import name and CLI remain ``fax``.

Install from a local checkout:

.. code-block:: bash

   python -m pip install -e .

For development:

.. code-block:: bash

   python -m pip install -e ".[dev]"

For artifact plotting without the full development toolchain:

.. code-block:: bash

   python -m pip install -e ".[plots]"

Python 3.10 and newer are supported. On Python 3.10, ``fax`` uses ``tomli`` as
the TOML parser fallback. On Python 3.11 and newer, it uses ``tomllib`` from the
standard library.

Local Couplings
---------------

During development, install neighboring projects in editable mode when coupling
tests are needed:

.. code-block:: bash

   python -m pip install -e /Users/rogerio/local/vmec_jax
   python -m pip install -e /Users/rogerio/local/booz_xform_jax
