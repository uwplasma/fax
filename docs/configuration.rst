Configuration
=============

``fax`` uses TOML input files.

.. literalinclude:: ../examples/minimal.toml
   :language: toml

Top-level sections:

``[run]``
   Case name, selected physics model, and output directory.

``[equilibrium]``
   Equilibrium source and optional Boozer-coordinate files.

``[grid]``
   Radial and Fourier resolution.

``[physics]``
   Sound-coupling and fast-ion model flags.

``[numerics]``
   Precision, JIT/autodiff flags, and eigensolver choice.

Workflow Artifacts
------------------

The CLI uses ``[run] output_dir`` for durable local artifacts:

``fax validate case.toml``
   Writes ``validation_summary.json`` and ``validation_report.md``.

``fax continuum case.toml``
   Writes ``continuum_summary.json``, ``continuum_metrics.json``, and
   ``continuum_branches.csv``.

``fax plot-continuum artifact_dir``
   Reads the continuum JSON/CSV artifacts and writes publication-draft SVG/PNG
   dashboard figures.

``fax convergence case.toml``
   Writes ``convergence/convergence_summary.json``,
   ``convergence/convergence_metrics.csv``, and
   ``convergence/convergence_report.md`` under ``run.output_dir``. Repeated
   ``--radial-points``, ``--angular-points``, and ``--mode-max`` values define
   the resolution sweep.

``fax benchmark case.toml``
   Writes ``benchmark/benchmark_summary.json`` and
   ``benchmark/benchmark_metrics.csv`` under ``run.output_dir`` while also
   recording the size of the generated continuum artifacts.

The JSON summaries are intended for scripts and regression checks. The
Markdown validation report is intended for review packets and handoff notes.
