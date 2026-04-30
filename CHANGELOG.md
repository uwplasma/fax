# Changelog

## v0.0.1 - 2026-04-24

Initial research release for FAX.

### Added

- MIT-licensed, pip-installable Python package with `fax` CLI.
- TOML configuration loading with Python 3.10 `tomli` fallback and Python 3.11+ `tomllib`.
- Analytic equilibrium fixtures, local shear-Alfven continuum formulas, and metric-matrix continuum validation gates.
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
- Reduced gyro-Landau fast-ion closure API with resonant, dispersive, diamagnetic, and thermal-Landau damping coefficients plus sign/regression gates.
- Real-`boozmn` global radial convergence gates for Boozer angular quadrature, radial surface coarsening, boundary-condition variants, and radial-stiffness variants.
- Builder-based Boozer pipeline optimization objective for future differentiable `vmec_jax`/`booz_xform_jax` design workflows.
- Public ORNL STELLGAP CTOK local-continuum and largest-local-gap comparison gates and generated figure.
- Public ORNL AE3D eigenvalue-answer parser/residual smoke gate.
- Public ORNL STELLGAP post-processing parity gate and AE3D omega2/matrix artifact gates.
- Public ORNL AE3D global matrix frequency and mode-shape parity gate using the public `A x = lambda B x` artifacts and `B`-weighted eigenvector correlations.
- Public ORNL AE3D global-mode continuum-placement gate against the FAX CTOK local Boozer continuum.
- Bounded STELLGAP CTOK executable runtime/status capture benchmark family.
- FAR3D `lincheck` growth/frequency parser, `fax far3d-ingest` CLI, and conversion into the held-out kinetic growth-rate validation schema with explicit unit normalization.
- Archived FAR3D/NVHPC GPU DIII-D public-example runtime artifact, parser-ready kinetic dataset, benchmark-manifest entry, and generated lincheck dashboard figure.
- Archived FAR3D GPU run-length convergence ladder with 50/100/200/400-step outputs, convergence utility, tests, and generated dashboard figure.
- Archived FAR3D GPU fast-ion beta scan with finite-beta monotonic growth gate, scan utility, tests, benchmark-manifest entry, and generated dashboard figure.
- Differentiable gyro-Landau closure calibration from eigenfunction-weighted growth-rate targets.
- External-code artifact dashboard example and generated README/docs figure.
- README status badges for tests, docs, publish, ReadTheDocs, coverage, PyPI, Python versions, license, Ruff, and JAX.
- Differentiable radial-mode avoidance objective for early design-screen demonstrations.
- VMEC `wout` adapter smoke path for the Landreman-Paul QA low-resolution reference case.
- Validation registry with named gates, tolerances, citations, commands, tags, and JSON/Markdown reports.
- Rich progress output for validation, continuum, convergence, and benchmark workflows.
- Artifact workflows:
  - `fax validate`
  - `fax continuum`
  - `fax plot-continuum`
  - `fax eigenmodes`
  - `fax plot-global`
  - `fax convergence`
  - `fax plot-convergence`
  - `fax benchmark`
- Generated README/docs figures with automated nonblank figure QA.
- Research examples for validation packets, differentiability sensitivity, a small inverse-design toy problem, a basic BAE/TAE/EAE gap map, and a reduced global/gap/fast-ion screening packet.
- Research example for Boozer harmonic convention validation.
- Research example for applying the local 3-D Boozer continuum operator to a real `boozmn` file and generating a plot/JSON packet.
- Research example for solving the global radial 3-D Boozer eigenmode problem and plotting radial envelopes/harmonic structure.
- ReadTheDocs/Sphinx docs, GitHub Actions definitions for tests/docs/publish, and PyPI trusted-publishing workflow.
- Publication-results documentation page tying the ten publication-suite figures to claim levels, metrics, commands, and remaining publication lanes.

The PyPI distribution name is `uwplasma-fax`; the import package and CLI remain
`fax`.

### Validation Status

This release is a validated reduced-model foundation, not a final 3-D stellarator eigenmode solver.
Current gates cover analytic continuum limits, metric reduction to the profile
limit, Hermiticity, generalized-eigensolve residuals, JAX density-gradient
consistency, radial/angular convergence, reduced radial eigenmode spectra,
analytic gap centers, BAE/TAE/EAE reference ordering, Boozer-Jacobian metric
identity checks, real `boozmn` local 3-D Boozer geometry/operator checks,
real-Boozer global radial convergence variants, reduced fast-ion resonance
ordering, reduced gyro-Landau closure signs and coefficient calibration,
STELLGAP CTOK largest local-gap parity, STELLGAP executable status capture, AE3D
public matrix/mode-shape and continuum-placement parity, FAR3D growth-rate
ingestion-schema validation with one real short GPU runtime artifact and a
single-case run-length convergence ladder plus finite-beta parameter scan, and
VMEC/Boozer adapter smoke tests.

### Local Release Gate

The v0.0.1 release candidate was checked locally with:

- `python tools/generate_validation_figures.py`
- `python tools/qa_figures.py`
- `python -m ruff check .`
- `python -m mypy src/fax`
- `python -m coverage run --source=src/fax -m pytest`
- `python -m coverage report --fail-under=95`
- `sphinx-build -W -b html docs docs/_build/html`
- `python -m build`
- `python -m twine check dist/*`
