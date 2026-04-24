Quickstart
==========

Print an example configuration:

.. code-block:: bash

   fax --example

Validate and inspect the bundled example:

.. code-block:: bash

   fax examples/minimal.toml --dry-run

Inspect the first local stellarator reference target:

.. code-block:: bash

   fax examples/landreman_paul_qa_lowres.toml --dry-run

The normal CLI prints rich progress output: solve parameters, phase status,
elapsed time, estimated remaining time, and what the code is doing.

For machine-readable output:

.. code-block:: bash

   fax examples/minimal.toml --dry-run --json
   fax examples/minimal.toml --dump-config
   fax examples/minimal.toml --quiet

Python calls can print the same progress stream:

.. code-block:: python

   from fax import load_config, run

   config = load_config("examples/landreman_paul_qa_lowres.toml")
   summary = run(config, progress=True, dry_run=True)

The current runner reports a planned run. Physics execution will be added after
the equilibrium adapters, operators, eigensolvers, and validation cases defined
in ``plan.md`` are implemented.
