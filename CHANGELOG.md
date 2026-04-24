# Changelog

## v0.0.1 - 2026-04-24

Initial research scaffold release for FAX.

### Added

- MIT-licensed, pip-installable Python package with `fax` CLI.
- TOML configuration loading with Python 3.10 `tomli` fallback and Python 3.11+ `tomllib`.
- Analytic equilibrium fixtures, local shear-Alfven continuum formulas, and metric-matrix continuum scaffold.
- Reduced radial global-eigenmode operator with exact constant-profile spectrum and density-gradient gates.
- Analytic TAE/EAE/HAE gap estimates and reduced fast-ion transit/precession resonance diagnostics.
- Optional Boozer-spectrum reader and local `booz_xform_jax` coupling smoke path.
- Differentiable radial-mode avoidance objective for early design-screen demonstrations.
- VMEC `wout` adapter smoke path for the Landreman-Paul QA low-resolution reference case.
- Validation registry with named gates, tolerances, citations, commands, tags, and JSON/Markdown reports.
- Rich progress output for validation, continuum, convergence, and benchmark workflows.
- Artifact workflows:
  - `fax validate`
  - `fax continuum`
  - `fax plot-continuum`
  - `fax convergence`
  - `fax plot-convergence`
  - `fax benchmark`
- Generated README/docs figures with automated nonblank figure QA.
- Research examples for validation packets, differentiability sensitivity, a small inverse-design toy problem, and a reduced global/gap/fast-ion screening packet.
- ReadTheDocs/Sphinx docs, GitHub Actions definitions for tests/docs/publish, and PyPI trusted-publishing workflow.

The PyPI distribution name is `uwplasma-fax`; the import package and CLI remain
`fax`.

### Validation Status

This release is a validated scaffold, not a final 3-D stellarator eigenmode solver.
Current gates cover analytic continuum limits, metric reduction to the profile
limit, Hermiticity, generalized-eigensolve residuals, JAX density-gradient
consistency, radial/angular convergence, reduced radial eigenmode spectra,
analytic gap centers, reduced fast-ion resonance ordering, and VMEC/Boozer
adapter smoke tests.

### Local Release Gate

The v0.0.1 release candidate was checked locally with:

- `python tools/generate_validation_figures.py`
- `python tools/qa_figures.py`
- `python -m ruff check .`
- `python -m mypy src/fax`
- `python -m pytest -q`
- `sphinx-build -W -b html docs docs/_build/html`
- `python -m build`
- `python -m twine check dist/*`
