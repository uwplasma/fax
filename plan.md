# fax Planning And Handoff Log

Repository: <https://github.com/uwplasma/fax>  
Visibility: private  
Local checkout: `/Users/rogerio/local/fax`  
Primary local dependencies: `/Users/rogerio/local/vmec_jax`, `/Users/rogerio/local/booz_xform_jax`  
Initial session date: 2026-04-23 America/Chicago, with online literature survey current to 2026-04-24.

## Mission

Build `fax`: a research-grade, end-to-end differentiable JAX code for Alfvén eigenmodes, fast-ion drive, continuum gaps, and toroidal/helical coupling in stellarators. The code must be fast enough for design iteration, accurate enough for physics publication, and validated strongly enough that a skeptical reviewer can trace every key claim to analytic theory, cross-code comparison, convergence evidence, and experimental or literature benchmarks.

`fax` is not meant to replace full gyrokinetic or hybrid-MHD production codes immediately. Its first role is to be the differentiable, design-facing layer that can:

- map VMEC/`vmec_jax` equilibria to Alfvén-continuum and global-mode metrics,
- expose gradients with respect to equilibrium/design parameters,
- identify gaps, avoided crossings, mode families, and sensitivity to density/iota/geometry,
- estimate fast-ion drive/damping with reduced but controlled closures,
- escalate selected cases to higher-fidelity validation and cross-code comparison.

## Non-Negotiable Design Principles

1. Differentiable by construction: geometry adapters, operators, solvers, diagnostics, and objective functions must be pure JAX-transformable functions unless explicitly documented otherwise.
2. Validation first: no physics model is considered complete until it has analytic tests, convergence tests, and at least one literature or cross-code validation target.
3. Honest capability reporting: CLI and docs must distinguish implemented features from planned features.
4. Static shapes and batchability: expensive paths should support `jax.jit`, `jax.vmap`, and eventually distributed scans without Python loops in hot kernels.
5. Reproducible research artifacts: each validation figure should have an input file, command, tolerance, source citation, and expected output metric.
6. Coupled but modular: `vmec_jax` and `booz_xform_jax` are first-class adapters, but core operators should also support analytic and precomputed reference geometries.

## Initial Literature And Code Survey

### FAR3D

FAR3D is the closest reduced-model precedent for fast, parameter-scannable AE/EPM studies. The 2024 Frontiers review describes FAR3D as a gyro-fluid/reduced-MHD code for linear and nonlinear stability of Alfvén eigenmodes, energetic-particle modes, pressure-gradient-driven modes, and current-driven modes. It has been applied to LHD, JET, DIII-D, EAST, TJ-II, Heliotron J, ITER, CFETR, JT-60SA, and CFQS.

Implication for `fax`: FAR3D should be a validation and model-design anchor for reduced-MHD plus energetic-particle moment closures. `fax` should reproduce selected FAR3D linear limits before attempting nonlinear EP transport claims.

Source: <https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2024.1422411/full>

### STELLGAP

STELLGAP calculates shear Alfvén gap structures for 3-D configurations, including stellarators, RFPs, and 3-D tokamaks, with and without sound-wave coupling. Recent work and APS abstracts continue to use STELLGAP for continuum and gap surveys in quasi-symmetric stellarators.

Implication for `fax`: the first serious milestone should reproduce STELLGAP-style continuum equations and gap locations in Boozer/straight-field-line coordinates, using a dense reference Galerkin path before optimizing.

Sources:

- <https://www.osti.gov/biblio/code-4633>
- <https://meetings-archive.aps.org/dpp/2024/jp12/29/>
- <https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/shear-alfven-continuum-of-quasisymmetric-stellarators/7B50E360D4324D8AF93DA73EB25C35D8>

### AE3D

AE3D solves shear Alfvén eigenmodes and eigenfrequencies in 2-D and 3-D toroidal magnetic confinement devices. The OSTI record states that it uses a reduced-MHD model and does not currently include sound-wave coupling.

Implication for `fax`: AE3D is the natural cross-code comparator for global shear-Alfvén eigenmodes once the continuum operator is working. `fax` should initially match AE3D's reduced model before adding sound and EP physics.

Source: <https://www.osti.gov/biblio/code-4629>

### CAS3D And CAS3D-K

CAS3D is the long-standing full ideal-MHD stability/eigenmode solver for 3-D equilibria. CAS3D-K extends this family with a perturbative kinetic-MHD calculation for growth rates by solving a drift-kinetic response, while neglecting radial orbit drifts away from a flux surface. TJ-II and H-1 literature uses CAS3D/CONTI/CKA/EUTERPE-style comparisons for mode families, sound coupling, and global structures.

Implication for `fax`: CAS3D/CAS3D-K comparisons are necessary for reviewer-proof claims about full-MHD or kinetic growth-rate fidelity. They should be later-stage validations after reduced-MHD benchmarks are stable.

Sources:

- <https://www.researchgate.net/publication/50235854_Alfven_eigenmodes_measured_in_the_TJ-II_stellarator>
- <https://arxiv.org/abs/1202.3204>
- <https://openresearch-repository.anu.edu.au/bitstreams/5ea2947e-b878-4e4b-9f27-692e226f4656/download>

### EUTERPE, GTC, ORB5, LIGKA, HMGC, TRIMEG-GKX

These codes cover higher-fidelity kinetic and gyrokinetic approaches. EUTERPE is a global gyrokinetic code for stellarator geometry; IPP describes it as a nonlinear electromagnetic delta-f PIC code usable in arbitrary magnetic geometry. LIGKA, ORB5, HMGC, TRIMEG-GKX, and GTC are relevant references for damping channels, kinetic thermal effects, EP response, and gyrokinetic validation in selected geometries.

Implication for `fax`: these are not first implementation targets, but they define the escalation ladder. When `fax` reduced models disagree with experiment, the difference should be categorized by comparing to kinetic references, not hidden in tuning parameters.

Sources:

- <https://www.sciencedirect.com/science/article/pii/S0010465523003582>
- <https://www.ipp.mpg.de/1060721/alf>
- <https://www.sciencedirect.com/science/article/abs/pii/S0021999107001660>

### Boozer Coordinates And Booz_xform

Many stellarator AE formulations use Boozer or straight-field-line coordinates because mode coupling and particle invariants are expressed cleanly in those coordinates. STELLOPT documents BOOZ_XFORM as a transformation to Boozer coordinates. Local `booz_xform_jax` is therefore the preferred differentiable coordinate path when needed.

Implication for `fax`: do not make Boozer mandatory for every model. Start with an adapter interface that can accept VMEC real-space geometry, VMEC-derived straight-field-line data, or Boozer harmonics. Require Boozer only for models and validations that need it.

Sources:

- <https://princetonuniversity.github.io/STELLOPT/BOOZ_XFORM.html>
- local checkout: `/Users/rogerio/local/booz_xform_jax`

### JAX Architecture Sources

JAX's key transforms are `jax.jit`, `jax.vmap`, and `jax.grad`. The JAX docs state that `jax.jit` stages a Python function for efficient XLA execution, `jax.vmap` vectorizes functions, and custom JVP/VJP rules support user-defined differentiation for JAX-transformable functions.

Implication for `fax`: implement operators as pure array functions, use static resolution objects, batch parameter scans with `vmap`, and provide custom VJP/JVP rules for generalized eigenvalues, shift-invert solves, and implicit equilibrium/coordinate solves when default autodiff is unstable or mathematically wrong.

Sources:

- <https://docs.jax.dev/en/latest/key-concepts.html>
- <https://docs.jax.dev/en/latest/jit-compilation.html>
- <https://docs.jax.dev/en/latest/automatic-vectorization.html>
- <https://docs.jax.dev/en/latest/jep/2026-custom-derivatives.html>

## Physics Scope

### Phase 1: Shear Alfvén Continuum And Gaps

Goal: compute the local shear Alfvén continuum and identify gaps from poloidal/toroidal/helical coupling in stellarator geometry.

Model ingredients:

- equilibrium flux surfaces, rotational transform, Jacobian, metric coefficients, and density profile,
- Fourier representation over poloidal and toroidal angles,
- family coupling `n = n0 + k * Nfp`,
- optional sound coupling once incompressible benchmarks pass.

Outputs:

- continuum branches versus radius,
- gap centers and widths,
- dominant harmonic labels,
- mode-family coupling matrices,
- gradients of gap metrics with respect to density, iota, and geometry parameters.

Validation targets:

- cylindrical and large-aspect-ratio analytic continuum scaling,
- tokamak TAE gap location and width,
- Spong/STELLGAP stellarator continuum examples,
- recent quasisymmetric stellarator continuum paper figures,
- convergence in radial and Fourier resolution.

### Phase 2: Global Ideal/Reduced-MHD Eigenmodes

Goal: compute radially global eigenmodes in continuum gaps and near extrema.

Model ingredients:

- radial finite elements, B-splines, or high-order finite differences,
- Fourier Galerkin angular coupling,
- generalized Hermitian eigenproblem for reduced-MHD shear-Alfvén modes,
- optional compressibility/sound coupling after the reduced problem is validated.

Outputs:

- eigenfrequencies and eigenfunctions,
- mode localization, harmonic content, and gap association,
- continuum-damping proximity metrics,
- gradients of selected eigenvalues and design objectives.

Validation targets:

- AE3D reduced-MHD examples,
- CAS3D/CONTI/TJ-II and H-1 mode-family comparisons,
- LHD/W7-AS/W7-X cases where public data or literature figures are sufficient,
- convergence and residual-norm thresholds.

### Phase 3: Fast-Ion Drive And Damping

Goal: estimate linear growth/damping from energetic particles and thermal damping in a differentiable reduced model.

Candidate model ladder:

1. FAR3D-style moment/Landau closure for EP density and parallel velocity moments.
2. Perturbative drift-kinetic energy exchange operator for fixed eigenfunctions.
3. Differentiable quadrature over slowing-down, beam-like, Maxwellian, and alpha-like distributions.
4. Optional delta-f marker response only for non-gradient validation paths, unless a differentiable particle representation is justified.

Outputs:

- growth rates, damping rates, threshold metrics,
- resonance maps in energy/pitch/radius,
- sensitivity to EP beta, gradients, birth energy, density, and iota,
- objective functions suitable for reactor-design optimization.

Validation targets:

- FAR3D CFQS, TJ-II, LHD, and published tokamak cases in reduced limits,
- CAS3D-K perturbative growth-rate limits,
- experimental frequency scaling with density and Alfvén speed,
- finite-difference agreement for gradients of smooth drive metrics.

### Phase 4: Nonlinear And Transport Extensions

Goal: only after linear stability is validated, add nonlinear mode amplitude and EP redistribution models.

Candidate scope:

- reduced nonlinear FAR3D-like evolution,
- wave-particle trapping/bump-on-tail reduced models,
- quasilinear diffusion estimates,
- coupling to external orbit/transport tools for validation.

This phase must not block the linear design tool.

## Software Architecture

### Package Layers

`fax.config`
: TOML parsing, typed dataclasses, validation, default handling.

`fax.equilibrium`
: adapters for `vmec_jax`, VMEC `wout`, and analytic equilibria. Must return JAX pytrees with static shapes.

`fax.boozer`
: optional adapter to `booz_xform_jax` and `boozmn` files. Must expose Boozer harmonics and coordinate maps without contaminating non-Boozer workflows.

`fax.geometry`
: metric coefficients, Jacobian, field-line operators, Fourier transforms, radial interpolation.

`fax.continuum`
: continuum matrices, family construction, gap finding, branch tracking.

`fax.eigenmodes`
: global operator assembly, dense reference solver, matrix-free solver path, eigenfunction normalization.

`fax.fast_ions`
: EP distributions, resonance operators, moment closures, perturbative drive/damping.

`fax.objectives`
: differentiable scalar objectives for optimization and design scans.

`fax.validation`
: utilities for convergence scans, reference data, tolerances, and report generation.

`fax.cli`
: TOML-driven command-line entry point.

### Data Model

Use frozen dataclasses for user configuration and lightweight `NamedTuple` or dataclass pytrees for numerical objects. All arrays should be JAX arrays at operator boundaries. Avoid dictionaries inside JIT hot paths unless registered as pytrees with fixed keys.

Suggested core objects:

- `EquilibriumGeometry`: surfaces, iota, pressure, density, metric harmonics, Jacobian, field periods.
- `ModeFamily`: base toroidal mode, coupled harmonics, symmetry flags, truncation metadata.
- `ContinuumOperator`: static metadata plus callable matrix builder.
- `EigenmodeResult`: eigenvalues, eigenvectors, residuals, normalization, harmonic labels.
- `FastIonDistribution`: parameterized distribution and quadrature grid.

### Differentiability Strategy

1. Start with direct autodiff through dense continuum and eigenvalue calculations where eigenvalues are isolated.
2. Add explicit gradient checks against finite differences for every public objective.
3. Use custom VJP/JVP for:
   - generalized Hermitian eigenvalues,
   - eigenvectors with fixed gauge/normalization,
   - shift-invert and Krylov solves,
   - implicit coordinate/equilibrium solves.
4. Document nondifferentiable points:
   - eigenvalue crossings,
   - branch relabeling,
   - gap opening/closing thresholds,
   - hard mode-family truncation changes.
5. Provide smooth alternatives for optimization:
   - soft minimum distance to continuum,
   - smoothed gap objective,
   - spectral-window penalties,
   - mode-family weighted objectives.

### Performance Strategy

- Use JAX x64 by default for validation paths; allow float32 for exploratory scans only after tolerance study.
- Keep all hot kernels shape-static.
- Batch over equilibria, density profiles, mode families, and scan points with `vmap`.
- Use dense eigensolvers as correctness references for small problems.
- Add sparse/block-structured/matrix-free solvers after dense references are validated.
- Cache Fourier-coupling index maps as static metadata.
- Separate compile time from runtime in benchmarks with explicit warmup.
- Track memory as a first-class benchmark metric.
- Maintain CPU and GPU performance dashboards in CI or scheduled workflows once kernels exist.

## Validation Plan

Validation should live in `validation/` once reference data exist. Every case should include:

- `case.toml`
- source citation and DOI/URL
- required local/external data files
- expected metrics with tolerances
- command to regenerate
- plotting script
- short notes on what the case does and does not validate

### Unit Tests

- Fourier indexing and mode-family coupling maps.
- Coordinate transforms and normalization conventions.
- Metric coefficient identities and symmetry/parity checks.
- Density and profile interpolation.
- Operator Hermiticity or documented non-Hermiticity.
- Dimensional units and Alfvén-speed scaling.

### Numerical Accuracy Tests

- residual norm for each eigenpair,
- convergence with radial resolution,
- convergence with Fourier truncation,
- branch tracking robustness,
- gap center/width convergence,
- eigenvalue sensitivity versus finite differences,
- matrix-free solver agreement with dense solver on small cases.

### Physics Validation Tests

1. Uniform cylinder: continuum frequency formula and mode labeling.
2. Circular tokamak: TAE/EAE/NAE gap estimates and mode coupling.
3. STELLGAP stellarator cases: continuum branches, gap locations, harmonic dominance.
4. AE3D cases: reduced global eigenmode frequencies and structures.
5. CAS3D/H-1/TJ-II cases: global mode families and sound-coupling effects.
6. FAR3D reduced cases: growth rates and stability thresholds for EP moment closure.
7. Experimental consistency: frequency scaling with `B / sqrt(n_i)` and mode localization where diagnostics exist.

### Review-Ready Acceptance Criteria

For each major model paper claim:

- provide at least one analytic limit,
- provide at least one cross-code comparison,
- provide at least one grid convergence plot,
- report residuals and conservation errors,
- report sensitivity to uncertain profiles,
- state model limitations explicitly,
- archive exact software commit and input artifacts.

## Milestones

### M0: Repository Scaffold

Status: complete in initial commit.

- Private `uwplasma/fax` repo created.
- Python package skeleton added.
- `fax` CLI added with TOML parsing and Python 3.10 `tomli` fallback.
- ReadTheDocs-ready Sphinx docs added.
- GitHub workflows added for tests, docs, and PyPI publishing.
- Initial tests added.
- This `plan.md` added as the handoff and restart file.

### M1: Geometry And Equilibrium Adapter

- Implement analytic equilibrium fixtures.
- Implement `wout` reader adapter or reuse `vmec_jax` public API if available.
- Implement direct `vmec_jax` adapter for differentiable equilibria.
- Add shape/JIT tests and finite-difference checks.
- Add docs for coordinate conventions.

### M2: Incompressible Continuum

- Implement continuum operator in a small dense reference form.
- Add mode-family construction for stellarator field periods.
- Add branch tracking and gap-finding diagnostics.
- Validate cylinder and circular tokamak limits.
- Start STELLGAP comparison cases.

### M3: Boozer Path And Stellarator Gap Validation

- Couple to `booz_xform_jax` for Boozer harmonics.
- Validate against STELLGAP and recent quasisymmetric-continuum figures.
- Add convergence reports and docs.
- Add differentiable gap objectives.

### M4: Global Eigenmodes

- Implement radial Galerkin/finite-element global solver.
- Validate dense solver against continuum gap predictions and AE3D examples.
- Add custom differentiation for isolated eigenvalues.
- Add mode-structure plotting examples.

### M5: Sound Coupling And Compressibility

- Add sound/Alfvén coupling in the continuum and global operators.
- Validate against STELLGAP with sound coupling, CAS3D/CONTI, and H-1/TJ-II references.
- Document the limits of reduced-MHD versus full-MHD models.

### M6: Fast-Ion Reduced Stability

- Implement FAR3D-style EP moment/Landau closure.
- Add slowing-down and beam distribution parameterizations.
- Validate selected FAR3D cases.
- Add differentiable stability threshold objectives.

### M7: Perturbative Drift-Kinetic Drive

- Implement fixed-mode energy-exchange quadrature.
- Validate CAS3D-K-like limits and simple analytic resonances.
- Add gradient checks for distribution and profile parameters.

### M8: Design Integration

- Expose objectives for optimization over `vmec_jax` equilibrium/design parameters.
- Add batched scans and gradient examples.
- Add reactor-relevant examples for quasi-axisymmetric, quasi-helical, and quasi-isodynamic configurations.

### M9: Publication-Grade Validation Release

- Freeze a set of reference cases.
- Generate validation report figures from CI or reproducible scripts.
- Tag release, publish docs, and optionally publish package to PyPI.

## CI/CD Plan

Current workflows:

- `tests.yml`: Python 3.10 through 3.13 unit tests.
- `docs.yml`: Sphinx build with warnings as errors.
- `publish.yml`: build and publish to PyPI on GitHub release or manual dispatch using trusted publishing.

Future workflows:

- nightly validation cases with cached reference data,
- benchmark workflow on self-hosted GPU/CPU runners,
- scheduled docs link checking,
- optional ReadTheDocs webhook checks,
- artifact upload for validation plots.

## Documentation Plan

Current docs:

- installation,
- quickstart,
- configuration,
- architecture,
- validation,
- literature and code survey,
- development,
- API.

Needed docs:

- coordinate conventions,
- equations and nondimensionalization,
- operator derivations,
- validation case registry,
- differentiability and adjoints,
- performance guide,
- examples gallery,
- coupling guide for `vmec_jax` and `booz_xform_jax`.

## Open Technical Decisions

1. Radial basis: finite differences are fastest to implement, but B-splines or finite elements may match AE3D/STELLGAP literature better.
2. Boozer requirement: decide per model whether Boozer is required or whether VMEC coordinates are sufficient.
3. Eigensolver path: dense reference first; then choose between JAX-native iterative methods, SciPy host callbacks for non-differentiable validation, or custom matrix-free Krylov.
4. Fast-ion model priority: FAR3D-style moment closure is the likely first differentiable reduced model; perturbative drift-kinetic response should follow.
5. Reference data licensing: confirm which code outputs and literature-derived datasets can be committed to a private repo and later redistributed.
6. Package name on PyPI: `fax` may be unavailable or ambiguous; confirm before first public release.

## Handoff Log

### 2026-04-23 America/Chicago

- Confirmed local workspace at `/Users/rogerio/local/tests`.
- Confirmed authenticated GitHub CLI account with `repo` and `workflow` scopes.
- Confirmed local dependency checkouts:
  - `/Users/rogerio/local/vmec_jax`
  - `/Users/rogerio/local/booz_xform_jax`
- Confirmed `uwplasma/fax` did not already exist.
- Created private GitHub repository `uwplasma/fax`.
- Cloned it to `/Users/rogerio/local/fax`.
- Inspected local `vmec_jax` and `booz_xform_jax` packaging, docs, and CLI patterns.
- Performed initial online survey of FAR3D, STELLGAP, AE3D, CAS3D/CAS3D-K, EUTERPE, Boozer-coordinate tooling, and JAX transform documentation.
- Added initial package scaffold, docs scaffold, workflows, examples, tests, and this plan.
- Verified editable install with `python -m pip install -e ".[test,docs]"`.
- Verified CLI with `fax examples/minimal.toml --dry-run`.
- Verified lint with `python -m ruff check .`.
- Verified unit tests with `python -m pytest` on local Python 3.13: 7 passed.
- Verified docs with `sphinx-build -W -b html docs docs/_build/html`.
- Verified local packaging with `python -m build`.
- Pushed initial commit `4facfaf` to `main`.
- GitHub Actions workflows were created and triggered, but GitHub did not start jobs. Check-run annotations report: "The job was not started because recent account payments have failed or your spending limit needs to be increased." This is an account/billing or spending-limit blocker, not a workflow/test failure. Re-run `Tests` and `Docs` after the UWPlasma billing/spending limit is fixed.

### Next Recommended Step

Implement M1 in a narrow branch:

1. Add `fax.equilibrium` with an analytic cylindrical/tokamak fixture.
2. Add a minimal `vmec_jax` adapter that can consume either a local `wout` or a `vmec_jax` Python object.
3. Define the `EquilibriumGeometry` pytree and coordinate/unit conventions.
4. Add unit tests for shape stability, dtype behavior, and Alfvén-speed scaling.
5. Update docs with the coordinate convention before starting the continuum operator.
