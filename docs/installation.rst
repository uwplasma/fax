Installation
============

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
