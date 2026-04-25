# Changelog

## v0.0.1 - 2026-04-24

Initial research scaffold release for FAX.

### Added

- MIT-licensed, pip-installable Python package with `fax` CLI.
- TOML configuration loading with Python 3.10 `tomli` fallback and Python 3.11+ `tomllib`.
- Analytic equilibrium fixtures, local shear-Alfven continuum formulas, and metric-matrix continuum scaffold.
- Reduced radial global-eigenmode operator with exact constant-profile spectrum and density-gradient gates.
- Analytic TAE/EAE/HAE gap estimates and reduced fast-ion transit/precession resonance diagnostics.
- Fast BAE/TAE/EAE large-aspect-ratio gap-map example with a publication-style figure.
- Optional Boozer-spectrum reader and local `booz_xform_jax` coupling smoke path.
- Boozer harmonic reconstruction/projection, Fourier orthogonality, and straight-field-line phase validation.
- Reduced Boozer continuum matrix with constant-field profile-limit and sideband-coupling gates.
- Boozer-Jacobian identity and Jacobian-weighted reduced continuum gates.
- Real `boozmn` geometry reader for Boozer embedding and Jacobian harmonics, with local 3-D Boozer continuum matrices.
- Real-file Boozer validation gates for `sqrt(g)|B|^2=G+iota I`, embedding-Jacobian shape, and finite Hermitian local 3-D continuum output.
- Global radial 3-D Boozer block eigenmode operator with Dirichlet radial boundaries, radial/harmonic mode structures, exact constant-spectrum gates, density-gradient gates, and real-`boozmn` residual gates.
- Metric-derived Boozer radial stiffness, Neumann radial boundary variant, and eigenfunction-weighted fast-ion drive/damping screen.
- Builder-based Boozer pipeline optimization objective for future differentiable `vmec_jax`/`booz_xform_jax` design workflows.
- Public ORNL STELLGAP CTOK local-continuum comparison gate and generated figure.
- Public ORNL AE3D eigenvalue-answer parser/residual smoke gate.
- README status badges for tests, docs, publish, ReadTheDocs, coverage, PyPI, Python versions, license, Ruff, and JAX.
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
- Research examples for validation packets, differentiability sensitivity, a small inverse-design toy problem, a basic BAE/TAE/EAE gap map, and a reduced global/gap/fast-ion screening packet.
- Research example for Boozer harmonic convention validation.
- Research example for applying the local 3-D Boozer continuum operator to a real `boozmn` file and generating a plot/JSON packet.
- Research example for solving the global radial 3-D Boozer eigenmode problem and plotting radial envelopes/harmonic structure.
- ReadTheDocs/Sphinx docs, GitHub Actions definitions for tests/docs/publish, and PyPI trusted-publishing workflow.

The PyPI distribution name is `uwplasma-fax`; the import package and CLI remain
`fax`.

### Validation Status

This release is a validated scaffold, not a final 3-D stellarator eigenmode solver.
Current gates cover analytic continuum limits, metric reduction to the profile
limit, Hermiticity, generalized-eigensolve residuals, JAX density-gradient
consistency, radial/angular convergence, reduced radial eigenmode spectra,
analytic gap centers, BAE/TAE/EAE reference ordering, Boozer-Jacobian metric
identity checks, real `boozmn` local 3-D Boozer geometry/operator checks,
reduced fast-ion resonance ordering, and VMEC/Boozer adapter smoke tests.

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
