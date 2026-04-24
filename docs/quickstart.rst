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

Run the current validation gates and write a validation report:

.. code-block:: bash

   fax validate examples/minimal.toml
   fax validate examples/minimal.toml --tag metric --json

Compute the current continuum scaffold and write branch/metric artifacts:

.. code-block:: bash

   fax continuum examples/landreman_paul_qa_lowres.toml
   fax plot-continuum outputs/landreman_paul_qa_lowres --output-dir outputs/landreman_paul_qa_lowres/figures

The normal CLI prints rich progress output: solve parameters, phase status,
elapsed time, estimated remaining time, and what the code is doing.

For machine-readable output:

.. code-block:: bash

   fax examples/minimal.toml --dry-run --json
   fax validate examples/minimal.toml --tag analytic --json
   fax continuum examples/minimal.toml --json
   fax plot-continuum outputs --json
   fax examples/minimal.toml --dump-config
   fax examples/minimal.toml --quiet

Python calls can print the same progress stream:

.. code-block:: python

   from fax import load_config, run, run_continuum, run_validation

   config = load_config("examples/landreman_paul_qa_lowres.toml")
   summary = run(config, progress=True, dry_run=True)
   validation = run_validation(config, tags=("metric",), progress=True)
   continuum = run_continuum(config, progress=True)

The original ``fax case.toml --dry-run`` path reports a planned run. The
``validate`` and ``continuum`` subcommands execute the first implemented
validation and continuum workflows, while global eigenmodes and fast-ion drive
remain planned milestones.
