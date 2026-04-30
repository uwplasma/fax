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

``fax eigenmodes case.toml``
   Writes ``eigenmodes/global_eigenmode_summary.json``,
   ``eigenmodes/global_eigenmode_metrics.json``,
   ``eigenmodes/global_eigenmodes.csv``, and
   ``eigenmodes/global_mode_structures.csv``. Analytic and VMEC-``wout``
   configurations run the scalar radial envelope operator; configurations with
   ``equilibrium.boozmn`` run the real-Boozer radial block operator.

``fax plot-global artifact_dir``
   Reads ``fax eigenmodes`` artifacts and writes a dashboard with frequencies,
   residuals, radial envelope norms, and first-mode harmonic content.

``fax convergence case.toml``
   Writes ``convergence/convergence_summary.json``,
   ``convergence/convergence_metrics.csv``, and
   ``convergence/convergence_report.md`` under ``run.output_dir``. Repeated
   ``--radial-points``, ``--angular-points``, and ``--mode-max`` values define
   the resolution sweep.

``fax plot-convergence artifact_dir``
   Reads the convergence JSON/CSV artifacts and writes publication-draft
   SVG/PNG dashboard figures.

``fax benchmark case.toml``
   Writes ``benchmark/benchmark_summary.json`` and
   ``benchmark/benchmark_metrics.csv`` under ``run.output_dir`` while also
   recording the size of the generated continuum artifacts. Repeated
   ``--family`` values add named physics benchmark packets. Available families
   currently include ``analytic-local``, ``global-block``,
   ``global-matrix-free``, ``global-krylov``, ``stellgap-ctok``,
   ``ae3d-artifact``, and ``vmec-boozer``; each reports a
   runtime next to the corresponding physics metric and tolerance.

The JSON summaries are intended for scripts and regression checks. The
Markdown validation report is intended for review packets and handoff notes.
