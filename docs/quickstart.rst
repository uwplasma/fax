Quickstart
==========

Print an example configuration:

.. code-block:: bash

   fax --example

Validate and inspect the bundled example:

.. code-block:: bash

   fax examples/minimal.toml --dry-run
   fax examples/minimal.toml --dump-config

The current runner reports a planned run. Physics execution will be added after
the equilibrium adapters, operators, eigensolvers, and validation cases defined
in ``plan.md`` are implemented.

