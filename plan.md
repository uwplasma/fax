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

## Planning Milestone: Functionality, Equations, Differentiability, And Reactor Design

Date added: 2026-04-24.

This milestone expands the initial scaffold into a concrete product and science plan. It answers what `fax` should compute, what it should read and write, which equations are allowed into the code, how those equations are verified, how differentiability should work, and how the code can become credible for publications and future stellarator power-plant design.

### License

`fax` uses the MIT license. Keep the license permissive unless UWPlasma later decides a different policy is required for non-code reference data. Third-party validation data must retain its own license/provenance metadata and should not be treated as MIT-licensed unless that is explicitly true.

### Planned User-Facing Functionality

The code should eventually support these workflows:

1. `fax continuum case.toml`: compute continuum branches, mode-family coupling, gaps, branch labels, and gap metrics.
2. `fax eigenmodes case.toml`: compute global reduced-MHD or ideal-MHD eigenmodes, eigenfunctions, residuals, harmonic content, and localization metrics.
3. `fax fast-ion case.toml`: estimate perturbative drive/damping or reduced fast-ion-fluid stability metrics for validated mode families.
4. `fax optimize case.toml`: expose differentiable objectives for `vmec_jax`/shape/profile optimization.
5. `fax validate case.toml`: run a registered validation case and compare with analytic, literature, or cross-code references.
6. `fax benchmark case.toml`: separate compile time, warm runtime, memory, and scaling metrics.
7. Python API workflows: call pure functions from optimization loops, notebooks, scripts, or other UWPlasma tools.

The CLI should remain TOML-driven, but every CLI mode should call a documented Python API so research users can script, batch, differentiate, and debug without shell parsing.

### Inputs

Required input classes:

- `run`: case name, model, output directory, random seeds for stochastic/UQ demos, and provenance settings.
- `equilibrium`: one of `vmec_jax` object, VMEC input, VMEC `wout`, analytic equilibrium, or a stored geometry artifact.
- `coordinates`: VMEC coordinates by default; `booz_xform_jax` object or `boozmn` file for Boozer-dependent models.
- `profiles`: mass density, electron/ion temperature, pressure, current/iota controls, impurity/effective mass assumptions, and uncertainty models.
- `mode_family`: base toroidal/poloidal labels, field-period coupling family, parity/symmetry choices, and truncation.
- `fast_ions`: distribution family, density/beta, gradients, birth energy, pitch-angle model, species mass/charge, source localization, and uncertainty.
- `numerics`: radial basis, quadrature, Fourier truncation, precision, eigensolver, differentiation mode, tolerances, and checkpointing.
- `validation`: reference case identifiers, tolerances, source citations, and regeneration commands.
- `optimization`: design variables, bounds, regularization, constraints, objective weights, and optimizer settings.

Input parsing must produce typed dataclasses before numerical work starts. No physics kernel should parse files directly.

### Outputs

Required output classes:

- `summary.json`: version, commit, environment, config hash, run status, timings, and key scalar metrics.
- `results.nc` or `results.zarr`: coordinates, profiles, continuum branches, eigenvalues, eigenvectors, growth rates, mode-family metadata, and derivatives when requested.
- `metrics.csv` or `metrics.parquet`: scalar metrics for scans and optimization loops.
- `validation_report.md`: reference values, errors, tolerances, pass/fail status, and source citations.
- `figures/`: continuum maps, gap maps, mode structures, sensitivity bars, UQ bands, and performance plots.
- Python return objects: JAX pytrees suitable for `jit`, `vmap`, `grad`, and downstream optimization.

All outputs must include enough provenance to reproduce the result: package version, git commit, platform/backend, precision, input hash, source files, and reference data versions.

### Equations And Model Ladder

`fax` should implement equations in a ladder. A higher rung cannot be used for a publication claim until the lower rungs have tests and validation.

#### Local shear-Alfven continuum

Start with the local continuum limit:

```text
omega_A,mn(s) = v_A(s) |k_parallel,mn(s)|,
v_A = B / sqrt(mu0 rho).
```

In a large-aspect-ratio straight-field-line limit, `k_parallel` must reduce to a form proportional to `n - m*iota(s)`. The production implementation must use the full metric/Jacobian/operator representation from VMEC or Boozer coordinates, and the simple formula should be treated as a limiting test, not the whole model.

#### Fourier-coupled continuum operator

Build a Galerkin/Fourier operator for each mode family:

```text
K_A(s; q) a = omega^2 M_A(s; q) a,
```

where `a` is the harmonic amplitude vector and `q` includes geometry, density, iota, and model parameters. For accepted Hermitian reduced-MHD cases, require:

```text
K_A = K_A^H, M_A = M_A^H, M_A > 0.
```

Sound coupling, compressibility, and geodesic acoustic coupling should enter only after the incompressible operator reproduces analytic and STELLGAP-style tests.

#### Radially global eigenmodes

The global reduced-MHD eigenproblem should use a radial Galerkin/FEM/spectral or finite-difference basis:

```text
K(q) x = lambda M(q) x,  lambda = omega^2.
```

Accepted eigenpairs must report residuals:

```text
||K x - lambda M x|| / (||K x|| + |lambda| ||M x||).
```

Eigenfunctions must have a deterministic normalization and phase/gauge convention before derivatives are exposed.

#### Ideal-MHD energy-principle bridge

Before full ideal-MHD claims, document the relation between the reduced operator and the linearized ideal-MHD energy principle. Track which terms are retained or dropped:

- field-line bending,
- pressure-gradient/interchange terms,
- compressibility/sound coupling,
- equilibrium current terms,
- curvature terms,
- boundary conditions.

This bridge is necessary for reviewer-proof comparison with CAS3D/CAS3D-K and full-MHD literature.

#### Fast-ion perturbative drive

The first fast-ion stability path should use a fixed validated eigenfunction:

```text
gamma_f ~= - Im(delta W_f) / (2 omega W_MHD).
```

The drift-kinetic resonance operator must expose the resonant phase-space denominator and sign convention, with tests in transit, bounce, and precession limits:

```text
omega - n omega_phi - m omega_theta - l omega_b - omega_d.
```

FAR3D-style energetic-particle moment/Landau closures should be added as a separate reduced model, not mixed silently with perturbative kinetic estimates.

#### Reduced nonlinear extensions

Nonlinear amplitude, saturation, and transport models should be downstream of linear validation. They should not be used to tune or hide errors in the linear continuum/global-mode/drive pipeline.

### Physics Gates

Each physics capability must pass gates before being presented as mature:

- `G0 dimensions`: units, normalization, species conventions, and coordinate conventions tested.
- `G1 analytic`: cylinder, slab, or large-aspect-ratio tokamak limit reproduced.
- `G2 algebraic`: Hermiticity, positive mass matrix, symmetry/parity, and residual identities tested.
- `G3 convergence`: radial, angular, Fourier, quadrature, and precision convergence demonstrated.
- `G4 cross-code`: matched against STELLGAP, AE3D, CAS3D/CONTI, FAR3D, or another appropriate code.
- `G5 experiment/literature`: selected experimental or literature figure/metric reproduced within stated uncertainty.
- `G6 differentiability`: gradients pass finite-difference, tangent-linear, and adjoint consistency checks.
- `G7 design`: objective is smooth, robust under uncertainty, and meaningful under reactor-relevant constraints.

Gate status should be machine-readable in validation reports.

### Correct Implementation Of Equations

Every equation should be implemented by a documented process:

1. Create an equation note in `docs/equations.rst` or a dedicated derivation page with symbols, assumptions, units, and source references.
2. Create a minimal analytic/reference implementation, even if slow and dense.
3. Add a low-resolution JAX implementation.
4. Check dimensions and limiting behavior.
5. Check algebraic properties: Hermiticity, symmetry, positive definiteness, conservation, and coordinate invariance where applicable.
6. Compare direct matrix application against operator/matrix-free application.
7. Verify residuals for eigenpairs.
8. Verify derivatives against finite differences over a controlled step-size sweep.
9. Run convergence tests in resolution and precision.
10. Freeze a regression artifact only after the reference source and tolerance are recorded.

Docstrings for physics kernels should include:

- equation name or identifier,
- source or derivation page,
- coordinate convention,
- units/normalization,
- input/output shapes,
- differentiability status,
- test names that cover the function.

Comments should explain non-obvious numerical choices, not restate assignments.

### Differentiability Plan

Differentiate at the mathematical level appropriate to each solver:

#### Direct autodiff through dense references

Use JAX direct autodiff for small dense operators and eigensolvers. This provides the baseline for correctness and makes failures in custom adjoints visible.

#### Generalized Hermitian eigenvalue adjoint

For isolated eigenvalues with

```text
K x = lambda M x,   x^H M x = 1,
```

the first-order sensitivity is:

```text
d lambda = x^H (dK - lambda dM) x.
```

Implement this as a custom JVP/VJP after dense direct autodiff tests exist. Treat repeated/near-crossing eigenvalues as a separate subspace problem; do not present single-mode gradients through crossings as physical sensitivities.

#### Non-Hermitian and growth-rate adjoints

For non-Hermitian reduced fast-ion operators, use left/right eigenvectors:

```text
d lambda = y^H (dA) x / (y^H x),
```

with conditioning diagnostics and warnings near defective spectra.

#### Implicit differentiation for solves

Large eigenvalue, shift-invert, equilibrium, Boozer, or nonlinear solves should not rely on backpropagating through arbitrary iteration histories. Use `jax.custom_vjp`, `jax.custom_jvp`, `jax.lax.custom_linear_solve`, JAXopt-style implicit differentiation, or Lineax-backed transposed solves when this is clearer and faster.

#### Discretize-then-optimize vs optimize-then-discretize

Both approaches should be tested:

- direct AD of the discrete algorithm for dense reference problems,
- discrete adjoints/custom VJPs for production solvers,
- continuous/physics adjoint formulas where they are known and reviewable.

Recent spectral-adjoint work in Dedalus shows a useful pattern: build adjoint solvers from symbolic sparse-spectral operator graphs so gradient solves retain sparse/spectral efficiency. `fax` should not copy Dedalus, but it should adopt the lesson: represent operators with enough structure that transposes/adjoints are explicit and testable.

Sources:

- JAX custom derivatives: <https://docs.jax.dev/en/latest/jep/2026-custom-derivatives.html>
- JAX key transforms and tracing: <https://docs.jax.dev/en/latest/key-concepts.html>
- JAX sparse warning/BCOO reference: <https://docs.jax.dev/en/latest/jax.experimental.sparse.html>
- JAX-FEM implicit differentiation pattern: <https://deepmodeling.github.io/jax-fem/learn/compute_gradients/example.html>
- Fast automated adjoints for spectral PDE solvers: <https://arxiv.org/abs/2506.14792>

### JAX Ecosystem Evaluation

Keep base dependencies lean and unpinned. Add optional extras for ecosystem experiments. No external JAX ecosystem library becomes required until it beats or clarifies a native-JAX reference implementation.

#### Lineax

Reason to test: Lineax provides differentiable linear solves and matrix-free/structured linear operators in the JAX+Equinox ecosystem. Its API includes transposed solver state and iterative solvers such as CG, BiCGStab, and GMRES. This is directly relevant to adjoint linear solves and shift-invert prototypes.

Planned trials:

- dense `K - sigma M` solve versus `jax.scipy.linalg`,
- matrix-free operator for radial-Fourier global modes,
- transpose/adjoint solve equality tests,
- gradient correctness and runtime versus `jax.lax.custom_linear_solve`.

Adopt only if it reduces custom code without hiding physics/operator structure.

Source: <https://docs.kidger.site/lineax/api/solvers/>

#### Equinox

Reason to test: Equinox represents models as PyTrees and provides filtered transforms. This may make model/operator objects easier to manage while preserving JAX transformations.

Planned trials:

- define `EquilibriumGeometry`, `ModeFamily`, and `OperatorSpec` as Equinox modules,
- compare compile cache behavior versus dataclass/NamedTuple pytrees,
- verify docstring/type clarity for non-ML scientific users.

Adopt only if it improves ergonomics and does not make simple pure functions harder to inspect.

Source: <https://docs.kidger.site/equinox/all-of-equinox/>

#### Diffrax

Reason to test: Diffrax offers JAX-native ODE/SDE solvers with multiple adjoint modes. It is relevant for FAR3D-like time evolution, nonlinear reduced models, marker/phase-space characteristic tests, and quasilinear transport demonstrations.

Planned trials:

- reduced two-field Alfven wave time integration,
- energetic-particle moment closure time evolution,
- comparison of recursive checkpoint, direct, forward-mode, and backsolve adjoints,
- energy conservation and gradient checks.

Do not use Diffrax for the primary eigenvalue workflow unless a time-domain validation requires it.

Source: <https://docs.kidger.site/diffrax/api/adjoints/>

#### JAXopt, Optimistix, Optax

Reason to test: these may simplify optimization, root-solving, least-squares, and implicit-differentiation workflows.

Planned trials:

- JAXopt implicit root differentiation for branch tracking and nonlinear closures,
- Optimistix least-squares/root solvers for inverse problems,
- Optax for first-order stochastic UQ/inverse-design demonstrations.

Adopt only after examples show clear maintenance value.

#### JAX sparse

JAX BCOO is transform-compatible but the docs describe the sparse module as experimental and not recommended for performance-critical use. `fax` should use dense references first, then matrix-free/block-structured operators. BCOO can be a prototype or storage format but should not be the performance plan by itself.

### Source Code Management Plan

Target package layout:

```text
src/fax/
  config.py
  cli.py
  constants.py
  units.py
  coordinates/
    vmec.py
    boozer.py
    analytic.py
  equilibrium/
    adapters.py
    geometry.py
    profiles.py
  modes/
    families.py
    labels.py
    tracking.py
  operators/
    continuum.py
    global_modes.py
    sound.py
    fast_ion.py
    matrix_free.py
  solvers/
    dense.py
    krylov.py
    eig.py
    adjoints.py
  fast_ions/
    distributions.py
    resonances.py
    moments.py
    drive.py
  objectives/
    gaps.py
    growth.py
    reactor.py
    regularization.py
  validation/
    registry.py
    metrics.py
    reports.py
  demos/
    sensitivity.py
    uncertainty.py
    inverse_design.py
```

Management rules:

- all hot kernels pure and side-effect free,
- no file I/O inside JIT-transformed functions,
- all static choices explicit in config/spec objects,
- public APIs typed and documented,
- tests colocated by behavior, not by implementation accident,
- reference datasets tracked with provenance and checksums,
- examples kept small enough to run in CI unless marked slow,
- larger validation cases separated from unit tests.

### Demonstration Plan

Build demonstrations as scripts with saved outputs and docs pages:

1. `examples/continuum_cylinder`: analytic continuum and derivative with respect to density.
2. `examples/tokamak_tae_gap`: TAE gap scaling and convergence.
3. `examples/stellarator_gap_vmec_jax`: VMEC/`vmec_jax` to continuum map.
4. `examples/booz_xform_pipeline`: `vmec_jax` to `booz_xform_jax` to Boozer-harmonic continuum.
5. `examples/eigenmode_dense_vs_matrix_free`: dense reference versus production solver.
6. `examples/gradient_gap_sensitivity`: JAX gradient versus finite difference over density/iota/shape.
7. `examples/uq_profile_uncertainty`: batched UQ for density and fast-ion profile uncertainty.
8. `examples/inverse_design_gap_placement`: move a gap away from an alpha resonance window.
9. `examples/fast_ion_threshold`: perturbative growth threshold with uncertainty bars.
10. `examples/reactor_assessment_dashboard`: risk ranking for candidate reactor-scale equilibria.

Each demo should state expected runtime and whether it is CI, slow-CI, or manual/GPU.

### Metrics

Numerical metrics:

- eigenpair residual,
- Hermiticity error,
- mass-matrix positive-definiteness margin,
- grid convergence slope,
- Fourier convergence slope,
- branch tracking continuity,
- mode-label purity/mixing,
- gap center/width error,
- growth-rate error,
- gradient finite-difference relative error,
- adjoint consistency inner-product error.

Performance metrics:

- compile time,
- warmed runtime,
- memory peak,
- FLOP proxy or operator applications,
- scaling with radial points and harmonic count,
- CPU/GPU speedup,
- batched scan throughput.

Design metrics:

- distance from continuum resonance windows,
- minimum damping/growth margin under uncertainty,
- alpha-particle resonance overlap score,
- sensitivity to density/iota/shape,
- robustness across operating scenarios,
- compatibility with quasisymmetry, coil, volume, aspect-ratio, beta, and confinement objectives.

### Stellarator Optimization And Power-Plant Assessment

`fax` should become an objective-provider for stellarator design, not an isolated post-processor.

Candidate design variables:

- VMEC boundary Fourier coefficients through `vmec_jax`,
- current/iota and pressure-profile parameters,
- density and temperature profiles,
- fast-ion source/distribution parameters,
- Boozer harmonic targets,
- later, coil-proxy parameters when a differentiable coil/equilibrium bridge is reliable.

Candidate objectives:

- maximize separation between predicted dangerous eigenmodes and alpha-particle resonance windows,
- maximize robust gap/damping margin under profile uncertainty,
- minimize fast-ion growth-rate proxies,
- penalize low-order core continuum resonances,
- penalize sensitivity to uncertain profiles,
- preserve quasisymmetry, coil feasibility, MHD equilibrium quality, aspect ratio, volume, beta, and alpha confinement objectives supplied by other design tools.

Candidate reactor dashboard:

- continuum/gap map by radius and mode family,
- eigenmode risk ranking,
- alpha and auxiliary-heating resonance overlays,
- UQ bands for density/temperature/EP profiles,
- design-gradient arrows or ranked Fourier-shape sensitivities,
- "do not trust" flags when a model gate is not passed.

Optimization demonstrations should start with a small number of shape/profile parameters and explicit constraints. A reactor-scale example should be treated as an assessment workflow until enough validation cases pass.

Sources:

- Stellarator shape adjoints and optimization overview: <https://terpconnect.umd.edu/~mattland/projects/6_optimization/>
- DESC optimization discussion and AD motivation: <https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/desc-stellarator-code-suite-part-3-quasisymmetry-optimization/0B262B685969D2F5697BEE28C8A78DB1>
- SIMSOPT documentation: <https://simsopt.readthedocs.io/stable/>
- STELLOPT documentation: <https://princetonuniversity.github.io/STELLOPT/STELLOPT.html>
- TORAX as a JAX differentiable plasma-simulation precedent: <https://arxiv.org/abs/2406.06718>

### JCP-Grade Publication Plan

A credible JCP or similar methods publication should be scoped around the first validated differentiable continuum/global-mode capability, not the entire future roadmap.

Minimum paper package:

- method derivation and discretization,
- open source code under MIT license,
- reproducible validation inputs,
- analytic continuum and gap tests,
- STELLGAP/AE3D or equivalent cross-code comparisons,
- gradient correctness section with finite-difference and adjoint tests,
- performance/scaling section with CPU/GPU results,
- design demonstration with `vmec_jax` gradients,
- limitations and model validity boundaries,
- archived release tag and documentation snapshot.

Reviewer-proof standard:

- no hidden empirical tuning in validation metrics,
- every sign convention documented,
- every benchmark tolerance justified,
- figures regenerated by scripts,
- failed or out-of-scope cases documented rather than omitted,
- uncertainty in experimental/literature comparisons reported.

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

### M10: Ecosystem And Differentiability Prototype Review

- Prototype Lineax, Equinox, Diffrax, JAXopt, Optimistix, and Optax on small controlled problems.
- Compare against native-JAX references for clarity, runtime, memory, and gradient correctness.
- Decide which libraries remain optional, which become development extras, and which, if any, become required dependencies.
- Document the decision in `docs/differentiability.rst`.

### M11: Demonstration And Reactor Assessment Gallery

- Add the demonstration scripts listed in the 2026-04-24 milestone.
- Add CI-sized examples plus slow/manual validation examples.
- Add a reactor assessment dashboard example with explicit model-gate warnings.
- Add sensitivity, uncertainty quantification, and inverse-design showcases.

### M12: JCP-Ready Methods Release

- Scope a paper around the validated differentiable continuum/global eigenmode capability.
- Freeze validation artifacts, benchmark scripts, and documentation.
- Release an archival version and prepare the manuscript checklist.

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
7. Which optional JAX ecosystem libraries should be adopted after prototypes: Lineax, Equinox, Diffrax, JAXopt, Optimistix, and Optax remain candidates, not required dependencies.
8. Whether production large-scale operators should use custom matrix-free objects, JAX sparse BCOO prototypes, Lineax operators, or a hybrid. Dense references remain mandatory either way.
9. Which first publication target is safest: continuum/gap differentiability, global eigenmodes, or fast-ion reduced stability. Current recommendation is continuum/gap plus differentiable design metrics.

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

### 2026-04-24 America/Chicago

- Added the second planning milestone covering functionality, inputs/outputs, equations, physics gates, equation-correctness workflow, differentiability methods, JAX ecosystem trials, source-code management, demonstrations, optimization, reactor assessment, and JCP-grade publication requirements.
- Added docs pages for equations, differentiability, optimization, and demonstrations.
- Added an optional `ecosystem` dependency extra for future prototypes using Diffrax, Equinox, JAXopt, Lineax, Optimistix, and Optax. These remain optional and unpinned.
- Surveyed additional sources on custom derivatives, implicit differentiation, spectral adjoint solvers, Lineax, Equinox, Diffrax, stellarator adjoints/optimization, SIMSOPT/STELLOPT, DESC, and TORAX.
- Verified after the milestone update with `python -m ruff check .`, `python -m pytest`, `sphinx-build -W -b html docs docs/_build/html`, and `python -m build`.
- Confirmed GitHub Actions will not be available until next month; local checks are the required gate until then.
- Promoted `LandremanPaul2021_QA_lowres` to the first VMEC-backed stellarator reference target.
- Added `src/fax/reference_cases.py` with a local-path discovery helper for `LandremanPaul2021_QA_lowres`.
- Added `examples/landreman_paul_qa_lowres.toml` pointing at the neighboring `vmec_jax` input and `wout`.
- Verified the reference helper resolves the local VMEC input and `wout` on this machine.
- Re-verified after the reference-case update with `python -m ruff check .`, `python -m pytest` (9 passed), `sphinx-build -W -b html docs docs/_build/html`, `fax examples/landreman_paul_qa_lowres.toml --dry-run`, and `python -m build`.
- Added the first executable physics foundation: `EquilibriumGeometry`, analytic cylinder/tokamak fixtures, a minimal VMEC `wout` profile adapter, local large-aspect-ratio Alfvén-continuum formulas, validation metrics, and a lightweight local benchmark helper.
- Added reviewer-style tests for units, JAX pytree/JIT compatibility, density scaling, local continuum formulas, finite-difference versus autodiff gradients, Hermiticity metrics, benchmark timing, and LandremanPaul2021 QA `wout` adapter smoke coverage.
- Added `tools/generate_validation_figures.py` and generated the first docs figures:
  - `docs/_static/analytic_continuum_scaling.svg`
  - `docs/_static/landreman_paul_qa_lowres_profiles.svg`
- These figures are explicitly labeled as smoke-test/development figures, not full 3-D stellarator continuum validation.
- Final local gate for this implementation pass: `python -m ruff check .`, `python -m mypy src/fax`, `python -m pytest` (24 passed), `sphinx-build -W -b html docs docs/_build/html`, `fax examples/landreman_paul_qa_lowres.toml --dry-run`, and `python -m build`.
- Added mode-family helpers for stellarator sideband coupling, gap/separation diagnostics, and tests for both.
- Added `docs/figures.rst`, a figure manifest, and the first README-ready dashboard:
  - `docs/_static/gap_sensitivity_dashboard.svg`
  - `docs/_static/figure_manifest.json`
- Updated the README to show the generated dashboard and list the local quality gate.
- Added rich terminal progress reporting for the CLI and Python `run()` function. Normal runs now print solve parameters, phase starts/completions, elapsed time, estimated remaining time, implementation status, and final execution-phase tables. `--json` and `--quiet` remain machine-readable paths.
- Added the first metric-aware continuum scaffold in `fax.metric`: angular-grid `MetricGeometry`, analytic profile-to-metric lifting, VMEC `wout` Nyquist reconstruction of Jacobian/field/contravariant-field data, Hermitian continuum stiffness/mass matrices, and a dense generalized Hermitian frequency path.
- Promoted the first smoke tests into a validation registry in `fax.validation` with named gates, tolerances, citations, commands, tags, and structured pass/fail/skip results.
- Added tests for metric/profile-limit agreement, Hermiticity, generalized-eigenvalue sanity, density-gradient consistency, the Landreman-Paul metric adapter, and validation-registry metadata/results.
- Added generalized eigensystem residual checks for `Kx = lambda Mx`, an angular-grid convergence gate, and tests for the metric residual path.
- Fixed the VMEC `wout` profile adapter to derive representative radii from Fourier geometry when `Rmajor_p`/`Aminor_p` are zero; this corrected blank Landreman-Paul continuum figures caused by infinite frequencies.
- Expanded generated documentation artifacts to include PNG and SVG versions of workflow, equation, analytic-continuum, VMEC-reference, gap/sensitivity, residual/convergence, and validation-gate figures.
- Added `tools/qa_figures.py` to parse SVGs and inspect rendered PNGs for dimensions and nonblank pixel coverage before README/docs updates.
- Added `src/fax/workflows.py` as the first executable workflow layer above the differentiable kernels.
- Added `fax validate case.toml` to run registered validation gates with rich progress output and write `validation_summary.json` plus `validation_report.md`.
- Added `fax continuum case.toml` to compute profile-limit and metric-matrix continuum branches, mode-family metadata, gap metrics, generalized-eigen residual diagnostics, and artifact files:
  - `continuum_summary.json`
  - `continuum_metrics.json`
  - `continuum_branches.csv`
- Preserved the backward-compatible `fax case.toml --dry-run` CLI path while adding subcommands for real workflow execution.
- Added CLI tests for validation artifacts, continuum artifacts, JSON output, and rich progress output.
- Updated README and ReadTheDocs quickstart/configuration/validation pages with the new workflow commands and artifact contracts.
- Made validation-figure SVG generation deterministic by fixing the Matplotlib SVG hash salt and removing save-time dates from SVG metadata; repeated regeneration now leaves identical file hashes.
- Added `fax.plotting` and `fax plot-continuum artifact_dir` to generate SVG/PNG dashboards directly from `fax continuum` artifacts rather than private plotting-only arrays.
- Added `docs/_static/continuum_artifact_dashboard.*`, generated from a temporary `fax continuum` run and the public artifact plotting path.
- Added `analytic_radial_integral_convergence` and `landreman_paul_continuum_finite` validation gates for radial quadrature convergence and VMEC-backed finite continuum outputs.
- Added `fax convergence case.toml` to sweep angular resolution and Fourier truncation, writing `convergence_summary.json`, `convergence_metrics.csv`, and `convergence_report.md`.
- Added `fax benchmark case.toml` to measure cold continuum workflow time, warm no-artifact repeats, and generated continuum artifact size for local performance tracking.
- Added CLI tests for the convergence and benchmark workflows, using small analytic fixtures so the local gate remains fast.
- Extended `fax convergence` to sweep radial resolution as well as angular resolution and Fourier truncation. Coarse branch matrices are interpolated onto the reference radial grid before relative L2 errors are reported.
- Added `metric_radial_convergence` and `landreman_paul_radial_sampling_convergence` validation gates.
- Added a generated Landreman-Paul QA convergence dashboard for README/docs that visualizes radial refinement, angular refinement, mode truncation, and eigensolve residuals from the VMEC-backed adapter path.
- Added `fax plot-convergence` and `fax.plotting.plot_convergence_artifacts()` so users can generate radial/angular/Fourier convergence dashboards from public convergence artifacts.
- Added research examples:
  - `examples/analytic_circular_tokamak.toml`
  - `examples/analytic_convergence.toml`
  - `examples/research/validation_packet.py`
  - `examples/research/density_sensitivity.py`
  - `examples/research/inverse_iota_shift.py`
- Added `docs/research_examples.rst`, release documentation, `CHANGELOG.md`, `MANIFEST.in`, and GitHub release notes for `v0.0.1`.
- Set package metadata and `fax.__version__` to `0.0.1` for the first release candidate.
- Added the first pass on the remaining open physics lanes:
  - `fax.global_modes`: a one-dimensional radial shear-Alfven generalized eigenproblem with Dirichlet radial boundaries, exact constant-profile finite-difference spectrum, residual diagnostics, JIT tests, and density-gradient autodiff tests.
  - `fax.gap_models`: analytic TAE/EAE/HAE crossing and width estimates based on coupled `m`/`m + Delta m` sidebands.
  - `fax.fast_ions`: reduced energetic-particle transit/precession resonance diagnostics and Lorentzian drive ranking.
  - `fax.optimization`: smooth radial-mode avoidance objective for `[log density scale, iota shift, log field scale]`.
  - `fax.boozer`: optional `boozmn` reader and `booz_xform_jax` runner, plus normalized non-axisymmetric Boozer spectral-energy diagnostics.
- Added reviewer-facing gates for the reduced layers:
  - `global_mode_constant_profile`
  - `global_mode_density_gradient`
  - `analytic_tae_gap_center`
  - `fast_ion_resonance_peak`
- Added tests for the Boozer adapter, reduced global modes, analytic gap models, reduced fast-ion diagnostics, differentiable optimization objective, and the expanded validation registry.
- Added `examples/research/global_gap_fastion_screen.py`, which prints a reproducible JSON packet with radial eigensolve residuals, gap estimates, fast-ion resonance peak, and the radial-mode objective gradient.
- Extended generated documentation figures with `global_gap_fastion_dashboard.*`, covering the exact radial spectrum gate, radial eigenfunctions, analytic gap centers, reduced fast-ion resonance scan, and autodiff objective gradient.
- Updated README and docs to describe the implemented equations and to state explicitly that these are reduced/analytic validation layers, not yet full STELLGAP/AE3D/CAS3D/FAR3D-equivalent 3-D validation.
- Verified this milestone locally with:
  - `python -m ruff check .`
  - `python -m mypy src/fax`
  - `python -m pytest -q` (78 passed)
  - `python tools/generate_validation_figures.py && python tools/qa_figures.py`
  - direct `run_validation_suite()` check with all 15 registered gates passing on this machine
  - `sphinx-build -W -b html docs docs/_build/html`
  - `python -m build`
  - `python -m twine check dist/*`
- Added the next Boozer validation step:
  - `fax.boozer` now reconstructs sampled ``|B|(theta_B,zeta_B)`` from Boozer harmonics, projects sampled fields back to cosine/sine coefficients, checks Fourier-basis orthogonality, and checks the straight-field-line phase slope ``m*iota-n``.
  - Added validation gates `boozer_harmonic_projection` and `boozer_straight_field_line_phase`.
  - Added `examples/research/boozer_harmonic_validation.py` and the generated `boozer_harmonic_validation` README/docs figure.
  - Direct `run_validation_suite()` now has 17 registered gates passing on this machine.
  - Verified after this Boozer step with `python -m ruff check .`, `python -m mypy src/fax`, `python -m pytest -q` (82 passed), `python tools/generate_validation_figures.py && python tools/qa_figures.py`, `sphinx-build -W -b html docs docs/_build/html`, `python -m build`, and `python -m twine check dist/*`.
- Added the first reduced Boozer continuum coupling operator:
  - `boozer_continuum_matrices` assembles a Hermitian reduced continuum matrix from verified Boozer ``|B|`` harmonics, straight-field-line ``k_parallel=(n-m*iota)/R``, and uniform Boozer-angle quadrature.
  - Added `boozer_constant_field_continuum` and `boozer_sideband_coupling_hermitian` validation gates.
  - Added `boozer_continuum_coupling` README/docs figure showing constant-field profile recovery, Hermitian sideband coupling, and reduced gap opening under controlled harmonic amplitude.
  - Direct `run_validation_suite()` now has 19 registered gates passing on this machine.
  - Verified after this Boozer-continuum step with `python -m ruff check .`, `python -m mypy src/fax`, `python -m pytest -q` (85 passed), `python tools/generate_validation_figures.py`, `python tools/qa_figures.py`, `sphinx-build -W -b html docs docs/_build/html`, `python -m build`, and `python -m twine check dist/*`.
- Added a fast basic BAE/TAE/EAE gap-map demonstration:
  - `coupled_gap_branch_map`, `tae_gap_branch_map`, and `eae_gap_branch_map` now solve a minimal Hermitian two-branch avoided crossing for the textbook large-aspect-ratio continuum.
  - `bae_accumulation_frequency_hz` adds the fluid-order finite-beta/acoustic marker ``omega_BAE=sqrt(7/4+Te/Ti) v_ti/(qR)`` for examples and gates.
  - `basic_alfven_gap_map` builds a small analytic tokamak case spanning the ``q=3/2`` TAE surface and ``q=2`` EAE/BAE surface.
  - Added `examples/research/basic_alfven_gap_map.py`, README/docs explanations, and generated `basic_alfven_gap_map` figure showing the TAE, EAE, and BAE gaps clearly.
  - Added validation gates `basic_gap_map_ordering` and `bae_temperature_scaling`; direct `run_validation_suite()` now has 21 registered gates passing on this machine.
  - Verified after this gap-map step with `python -m ruff check .`, `python -m mypy src/fax`, `python -m pytest -q` (89 passed), `python tools/generate_validation_figures.py`, `python tools/qa_figures.py`, `sphinx-build -W -b html docs docs/_build/html`, `python -m build`, and `python -m twine check dist/*`.
- Added the next Boozer metric/Jacobian bridge:
  - `boozer_jacobian_weight` and `boozer_jacobian_identity_error` validate the Boozer-coordinate identity ``J_B=(G+iota I)/B^2`` and ``J_B |B|^2=G+iota I`` on sampled Boozer grids.
  - `boozer_metric_continuum_matrices` and `boozer_metric_continuum_frequencies_hz` use ``J_B`` in reduced matrix weights while keeping the full perpendicular metric/radial operator explicitly out of scope.
  - Added validation gates `boozer_jacobian_identity`, `boozer_metric_constant_field_continuum`, and `boozer_metric_mass_coupling`.
  - Added generated `boozer_jacobian_identity` README/docs figure showing sampled ``|B|``, ``J_B``, the flux-function identity product, and weighted mass coupling.
- Added repository status badge infrastructure:
  - README badges now cover Tests, Docs, Publish, ReadTheDocs, coverage, PyPI version, Python versions, MIT license, Ruff, and JAX.
  - `tools/write_coverage_badge.py` writes `docs/_static/coverage.svg` from `coverage json`.
  - `.github/workflows/tests.yml` now regenerates and commits the coverage badge on `main` pushes when GitHub Actions are available again.
  - Direct `run_validation_suite()` now has 24 registered gates passing on this machine.
  - Verified after this badge/Jacobian step with `python -m coverage run --source=src/fax -m pytest` (94 passed, 90.0% coverage), `python -m ruff check .`, `python -m mypy src/fax`, `python -m pytest -q` (94 passed), `python tools/generate_validation_figures.py`, `python tools/qa_figures.py`, `sphinx-build -W -b html docs docs/_build/html`, `python -m build`, and `python -m twine check dist/*`.
- Added the first real `boozmn` geometry/operator milestone:
  - `BoozerGeometry` reads `|B|`, embedding `R/Z/p`, Boozer-Jacobian harmonics, `iota`, `I`, and `G` from `boozmn` files.
  - `boozer_geometry_grid` reconstructs real Boozer-angle maps and derives the embedding-based normalized-radial metric `|\nabla s|^2`.
  - `boozer_geometry_jacobian_identity_error` validates the file-level identity `sqrt(g)|B|^2=G+iota I`.
  - `boozer_embedding_jacobian_shape_error` compares the embedding Jacobian shape against `boozmn` Jacobian harmonics up to one surface-wise scale/orientation.
  - `boozer_3d_continuum_matrices` and `boozer_3d_continuum_frequencies_hz` assemble a local surface-by-surface 3-D Boozer continuum matrix using `k_parallel=(n-m*iota)|B|/(G+iota I)`.
  - Added validation gates `boozer_3d_constant_field_continuum`, `boozmn_geometry_jacobian_identity`, `boozmn_embedding_jacobian_shape`, and `boozmn_3d_continuum_finite`.
  - Added `examples/research/boozmn_3d_continuum.py` so users can apply the real-geometry local operator to a new `boozmn` file and get a JSON/figure packet.
  - Added generated `boozmn_3d_geometry_operator` README/docs figure showing sampled real `|B|`, the Boozer identity residual, embedding `|\nabla s|^2`, and local 3-D Boozer continuum branches.
  - Important scope note: this is the first true `boozmn` local-operator layer, not yet a global radial eigenmode operator or STELLGAP/AE3D cross-code validation.
  - Direct `run_validation_suite()` now has 28 registered gates passing on this machine.
  - Verified after this real-`boozmn` step with `python -m ruff check .`, `python -m mypy src/fax`, `python -m pytest -q` (100 passed), `python -m coverage run --source=src/fax -m pytest -q` (100 passed, 89.9% coverage badge), `python tools/generate_validation_figures.py`, `python tools/qa_figures.py`, `sphinx-build -W -b html docs docs/_build/html`, `python -m build`, and `python -m twine check dist/*`.
- Added the first global radial 3-D Boozer eigenmode operator:
  - `RadialBlockEigenmodeProblem` carries one coupled harmonic vector per radial surface, local 3-D Boozer stiffness/mass blocks, and Hermitian radial-stiffness blocks.
  - `assemble_radial_block_matrices` builds the Dirichlet radial block matrix `K_global = blockdiag(K_3D) + D_s^H C_s D_s` and `M_global = blockdiag(M_3D)`.
  - `solve_radial_block_eigenmodes` returns global eigenfrequencies, residuals, and full `(mode, surface, harmonic)` mode structures including boundary zeros.
  - `radial_block_problem_from_boozer_geometry` builds the global problem from real `boozmn` local 3-D Boozer matrices.
  - Added exact diagonal constant-coefficient spectrum and density-gradient gates: `global_3d_block_constant_spectrum` and `global_3d_block_density_gradient`.
  - Added real-file gate `boozmn_global_3d_eigenmode_finite`.
  - Added `examples/research/boozmn_global_3d_eigenmodes.py` and generated `boozmn_global_3d_eigenmodes` README/docs figure.
  - Scope note: this closes the global radial operator lane for the current shear-Alfven/Boozer envelope model; STELLGAP/AE3D/CAS3D cross-code benchmarking remains the next validation lane.
- Added the next cross-code/differentiability/fast-ion milestone:
  - Cloned public code references into `/Users/rogerio/local`: ORNL `Stellgap`, ORNL `AE3D`, DSfusion `FAR3d`, `PyStellgap`, and `shearalfvenwave`.
  - Extracted public ORNL STELLGAP and AE3D answer artifacts into `/Users/rogerio/local/Stellgap_test_problem`, `/Users/rogerio/local/Stellgap_test_inputs`, `/Users/rogerio/local/AE3D_test_problem`, and `/Users/rogerio/local/AE3D_test_inputs`.
  - Built STELLGAP locally with:
    - `make PRECOMP=/opt/homebrew/bin/gcc-15 STELLOPT_PATH=/Users/rogerio/local/STELLOPT/ NETCDF_LIB='-L/opt/homebrew/Cellar/netcdf-fortran/4.6.2/lib -lnetcdff -L/opt/homebrew/Cellar/netcdf/4.9.3/lib -lnetcdf'`
    - Produced `/Users/rogerio/local/Stellgap/xmetric`, `xstgap`, `xstgap_snd_ver6`, and `xstgap_snd_ver7`.
    - Ran `/Users/rogerio/local/Stellgap/xmetric ctok` in `/Users/rogerio/local/Stellgap_test_inputs`, which regenerated `ae_metric.dat` and `tae_data_boozer`.
    - `xstgap` starts and prints version/MPI/grid metadata on the CTOK inputs, but exits on this gfortran/OpenMPI stack while opening an invalid `alfven_spec...` path. Do not treat executable STELLGAP parity as closed until this runtime issue is understood.
  - Attempted a local FAR3D build. CMake configured with Homebrew `libomp`, but the Fortran build stopped at the CUDA Fortran `cudafor.mod` dependency in `src/matrix.f90`; a CPU-only build recipe remains open.
  - Confirmed AE3D public build scripts expect Intel `ifort`/`ifx`, which is not available on this machine; the answer artifacts are usable now, while executable parity waits on a portable build path.
  - Added `fax.cross_code` readers for STELLGAP `alfven_post` and AE3D `egn_values.dat`.
  - Added the first quantitative public external-code gate: `stellgap_ctok_local_continuum`. It compares FAX local 3-D Boozer continuum branches with the ORNL STELLGAP CTOK `alfven_post` output using the same `boozmn_ctok.nc` geometry. Current local metric: 97 matched surfaces, mean relative sorted-spectrum error about `1.30e-2`, p95 about `2.55e-2`, max about `3.32e-2`; the gate tolerance is `4.0e-2`.
  - Added `examples/research/stellgap_ctok_comparison.py` and generated `docs/_static/stellgap_ctok_comparison.{svg,png}`.
  - Added the optional AE3D public eigenvalue artifact residual gate. This checks parser/residual handling only; it is not a FAX-vs-AE3D eigenmode parity claim.
  - Added the next artifact-convention gates:
    - `stellgap_public_postprocess_parity`: reproduces STELLGAP `post_process.f` conversion from `alfven_spec` to `alfven_post`; current max relative error is about `4.9e-7` over 3848 rows.
    - `ae3d_public_omega2_eigenvalue_parity`: reproduces AE3D `omega2.dat` to `egn_values.dat`; current max relative error is about `6.9e-7` over 1287 modes.
    - `ae3d_public_matrix_symmetry`: checks public AE3D `a_matrix.dat`/`b_matrix.dat` sparse symmetry; current relative asymmetry is about `1.19e-6` for `A` and `0` for `B`.
  - Added `examples/research/external_code_artifact_dashboard.py` and generated `docs/_static/external_code_artifact_dashboard.{svg,png}`. This is a convention/status figure, not a global-mode physics parity figure.
  - Replaced the old mass-proportional radial coupling default in the real-Boozer global path with a metric-derived radial stiffness matrix `<|J||grad s|^2|B|^2 phi_i^* phi_j/mu0>`, while keeping the previous mass-scaled path available for analytic tests.
  - Added Neumann radial boundary support and an exact constant-coefficient Neumann spectrum gate alongside the existing Dirichlet gate.
  - Added eigenfunction-weighted fast-ion drive/damping diagnostics that project reduced resonance drive onto global `(surface, harmonic)` eigenfunctions.
  - Added `boozer_global_mode_avoidance_objective` and `boozer_pipeline_mode_avoidance_objective` for differentiable Boozer/FAX design screens. The pipeline objective accepts a differentiable Boozer-geometry builder, which is the API boundary for future `vmec_jax`/`booz_xform_jax` end-to-end optimization.
  - Updated README, docs, changelog, release notes, validation registry, and tests to distinguish true STELLGAP local-continuum parity from AE3D artifact smoke tests and still-open full global parity.
  - Verified locally with `python -m ruff check .`, `python -m mypy src/fax`, direct `run_validation_suite(include_skipped=True)` (36 passed, 0 skipped, 0 failed), `python -m pytest -q` (114 passed), `python -m coverage run --source=src/fax -m pytest -q` (114 passed, 90.2% coverage badge), `python tools/generate_validation_figures.py`, `python tools/qa_figures.py`, `sphinx-build -W -b html docs docs/_build/html`, `python -m build`, and `python -m twine check dist/*`.
  - Re-verified after the artifact-convention gates with direct `run_validation_suite(include_skipped=True)` (39 passed, 0 skipped, 0 failed), `python -m pytest -q` (116 passed), coverage run (116 passed, 90.4% coverage badge), figure generation/QA, Sphinx with warnings as errors, build, and twine check.
  - 2026-04-25 coverage-hardening milestone:
    - Added physics/workflow tests rather than line-only scaffolds: analytic BAE/TAE/EAE gap-map figure rendering and ordering, continuum/convergence artifact contract checks, CLI plotting/error paths, validated TOML rejection paths, benchmark synchronization of nested JAX-like outputs, and the `python -m fax` entry point.
    - Added differentiability-oriented PyTree round-trip tests for `MetricGeometry`, `BoozerGeometry`, `CoupledGapEstimate`, and `FastIonPopulation`, each checking that the relevant continuum, gap, Boozer identity, or fast-ion-drive quantity is preserved.
    - Added Boozer/global-mode physics contracts for sine Boozer harmonics, Boozer radial metric profiles/grids, metric-derived and mass-scaled global radial stiffness, explicit radial-stiffness profiles/full blocks, eigenpair residual norms, and invalid operator inputs.
    - Added a local Landreman-Paul QA `wout` CLI continuum test that exercises VMEC `wout` loading, radial resampling, and density scaling when the local fixture is present.
    - Local coverage command now uses `--source=fax` to avoid coverage trying to report generated third-party `netCDF4` internals.
    - Verified locally with `python -m coverage erase && python -m coverage run --source=fax -m pytest && python -m coverage report --skip-empty -m`: 150 passed, 96% FAX source coverage.
    - Also verified `python -m ruff check .` and `python -m mypy src`.
  - 2026-04-25 parity/publication-suite milestone:
    - Added AE3D public matrix/eigenmode parity: FAX now reads `egn_mode_asci.dat`, reconstructs `(mode, surface, harmonic)` eigenvectors, reads public sparse `a_matrix.dat`/`b_matrix.dat`, and checks `A x = lambda B x` residuals plus Rayleigh-quotient/frequency errors. Current local public-artifact metrics over the first 8 modes: max residual about `3.29e-6`, max Rayleigh relative error about `1.63e-6`, and max frequency relative error about `8.17e-7`.
    - Added registered gate `ae3d_public_matrix_eigenmode_parity`. This closes AE3D artifact/matrix/eigenmode parity, not the harder FAX reduced-operator vs AE3D full-operator physics parity.
    - Added `examples/research/publication_figure_suite.py`, a ten-case publication-style figure suite covering: analytic BAE/TAE/EAE gaps, convergence, Landreman-Paul VMEC/wout inspection, Boozer harmonics, real `boozmn` local 3-D continuum, global radial eigenmodes, fast-ion drive/damping, autodiff sensitivities, STELLGAP CTOK gap-anchored global screening, and AE3D matrix/eigenmode parity.
    - Generated and QA-checked `docs/_static/publication_suite/*.svg` and `*.png`; all 10 cases completed locally, with STELLGAP CTOK p95 local-continuum error about `2.55e-2` and global-screen residual about `7.9e-13`.
    - Updated README, research examples, validation docs, and literature notes to distinguish: (1) STELLGAP CTOK local-continuum parity, (2) AE3D public matrix/eigenmode parity, and (3) still-open CAS3D/FAR3D and full AE3D/CAS3D/FAR3D operator-level physics parity.
    - Literature framing checked against AE3D (Spong, D'Azevedo & Todo 2010 / ORNL software record), STELLGAP (Spong, Sanchez & Weller 2003 / ORNL software record), recent quasisymmetric SAW continuum work, and FAR3D energetic-particle stability review literature.
    - Verified locally with `python -m ruff check .`, `python -m mypy src`, `python -m pytest -q` (153 passed), `python -m coverage run --source=fax -m pytest -q` plus coverage report/badge (153 passed, 95.7% exact / 96% rounded), `python tools/qa_figures.py`, `python tools/qa_figures.py --figure-dir docs/_static/publication_suite`, `sphinx-build -W -b html docs docs/_build/html`, `python -m build`, and `python -m twine check dist/*`.
  - 2026-04-27 differentiable-research milestone:
    - Added research-study APIs in `fax.optimization`: radial global-mode frequency maps, autodiff Jacobians, linearized uncertainty propagation, transparent gradient-descent minimization, inverse-design fitting to target frequencies, eigenfunction-projected fast-ion growth screens, and a combined fast-ion/frequency-window risk objective.
    - Added `examples/research/differentiable_design_study.py`, generating a four-panel dashboard for sensitivity analysis, UQ bands, inverse-design objective history, and fast-ion growth before/after the differentiable design move.
    - Generated `docs/_static/differentiable_design_study.{svg,png}` and `docs/_static/differentiable_design_study_summary.json`. Current reduced-model demonstration moves the second radial mode from about 88.4 kHz inside an 80-100 kHz risk window to about 114.8 kHz while reducing the peak reduced fast-ion growth screen from about +1.45 Hz to negative values for all plotted modes.
    - Updated README, research-example docs, optimization docs, differentiability docs, and literature notes to frame these as FAX-native differentiable capabilities: sensitivity matrices, uncertainty quantification, inverse design, and optimization trajectories. This is a new research-workflow lane, not a replacement for the still-open full 3-D AE3D/CAS3D/FAR3D parity lane.
    - Literature framing checked against differentiable scientific-computing practice: JAX custom derivatives/implicit solves, Lineax differentiable linear solvers, JAX-FEM inverse design, and recent spectral-adjoint preprints.
  - 2026-04-28 real-Boozer differentiable design milestone:
    - Extended `fax.optimization` with real-Boozer global-mode frequency maps, Jacobians, linearized uncertainty, inverse design, eigenfunction-projected fast-ion screens, and a combined frequency-window/fast-ion risk objective.
    - Added `examples/research/boozmn_differentiable_design_study.py`, which reads a real `boozmn` file, differentiates through local Boozer matrices, metric-derived radial stiffness, global generalized eigenfrequencies, and fast-ion growth projected on radial/harmonic eigenfunctions.
    - Generated `docs/_static/boozmn_differentiable_design_study.{svg,png}` and summary JSON. On the local Landreman-Sengupta-Plunk `boozmn` fixture, the example moves the first two modes below a 200-300 kHz risk window and reduces the peak reduced fast-ion growth screen from about `+8.87 Hz` to negative.
    - This closes the first real-geometry differentiable-design lane for the current Boozer operator. It is still a reduced-parameter design screen; true shape-gradient publication claims require wiring those parameters to `vmec_jax` boundary/profile variables and validating optimized geometries against external-code/literature targets.
  - 2026-04-28 in-memory VMEC/Boozer bridge milestone:
    - Added FAX `run_booz_xform_jax_from_wout_data` and `run_booz_xform_jax_geometry_from_wout_data` so a live VMEC `wout`-like object can populate the same `BoozerSpectrum`/`BoozerGeometry` contracts as the file-backed adapter.
    - Added a registered physics gate `booz_xform_jax_in_memory_geometry_adapter`, comparing an in-memory asymmetric VMEC object through `booz_xform_jax` against the asymmetric BOOZ_XFORM `boozmn` reference across field, embedding, Boozer-shift, and Jacobian harmonic families.
    - Strengthened neighboring `booz_xform_jax` with `read_wout_data(...).run_jax(...)` parity tests against file-backed `read_wout(...).run_jax(...)` for stellarator-symmetric and asymmetric fixtures, including Jacobian harmonics.
    - Regenerated the validation gate summary figures and manifest so docs/README artifacts include the new in-memory bridge gate.
    - This closes the no-intermediate-file handoff layer needed for `vmec_jax -> booz_xform_jax -> FAX` applications. The next hard step is differentiating through an actual `vmec_jax` solve/transform builder, not only through downstream FAX objectives.
  - 2026-04-28 live Landreman-Paul VMEC-JAX bridge milestone:
    - Added `run_vmec_jax_fixed_boundary_boozer_geometry`, a public integration helper that runs a short `vmec_jax` fixed-boundary solve, converts the live VMEC state to an in-memory `wout`, passes it through `booz_xform_jax`, and returns the normal FAX `BoozerGeometry` contract.
    - Added the registered validation gate `landreman_paul_vmec_jax_boozer_in_memory`, using the local Landreman-Paul QA low-resolution VMEC input with one VMEC iteration as a fast object-handoff smoke test. The gate checks the Boozer Jacobian identity at the percent level and verifies finite non-negative local 3-D Boozer continuum branches.
    - Added a matching physics test in `tests/test_boozer.py` and documented the gate in the README, validation docs, and differentiability notes.
    - Reworked the validation-gate summary figure layout after visual QA showed label overlap with the larger registry; regenerated the figure manifest and coverage badge.
    - Verified locally with the targeted bridge gate (`metric=2.1307e-3 < 1.0e-2`), `python -m pytest -q` (164 passed), coverage (`95.6%` FAX source coverage, fail-under 95 passed), `python -m ruff check .`, `python -m mypy src/fax`, `python tools/generate_validation_figures.py`, `python tools/qa_figures.py`, `python -m sphinx -W --keep-going -b html docs docs/_build/html`, `python -m build`, `python -m twine check dist/*`, and `git diff --check`.
    - Scope note: this proves the end-to-end object bridge is live. It is not yet a converged VMEC/Boozer/continuum publication validation and must be followed by VMEC convergence, Boozer truncation, angular/radial convergence, and cross-code/literature comparisons.
  - 2026-04-29 live VMEC/Boozer convergence milestone:
    - Added `run_vmec_jax_fixed_boundary_wout_data` so FAX can reuse one live `vmec_jax` state across multiple downstream Boozer transforms.
    - Added `run_vmec_jax_boozer_truncation_convergence`, `VmecJaxBoozerConvergenceResult`, and `boozer_geometry_field_relative_error` to check sampled `|B|`, local 3-D continuum frequencies, and Boozer-Jacobian identity under Boozer truncation changes.
    - Added registered gate `landreman_paul_vmec_jax_boozer_truncation_convergence`, comparing `(mboz,nboz)=(4,4)` against `(5,5)` on the same one-iteration Landreman-Paul QA VMEC state. Current local metric: `6.3707e-3 < 1.0e-2`.
    - Added the dedicated `vmec_boozer_bridge_convergence` README/docs figure to make the live bridge convergence evidence visible.
    - Verified locally with `python -m pytest -q` (166 passed), coverage (`95.6%` FAX source coverage, fail-under 95 passed), `python -m ruff check .`, `python -m mypy src/fax`, `python tools/generate_validation_figures.py`, `python tools/qa_figures.py`, `python -m sphinx -W --keep-going -b html docs docs/_build/html`, `python -m build`, `python -m twine check dist/*`, and `git diff --check`.
    - Scope note: this starts the live-bridge convergence ladder. Next required steps are converged VMEC residual/iteration studies, Boozer angular and surface-resolution sweeps, and external-code/literature continuum comparisons on the same geometry family.
  - 2026-04-29 live Boozer operator angular convergence milestone:
    - Added `BoozerOperatorAngularConvergenceResult` and `boozer_3d_operator_angular_convergence` to measure local 3-D Boozer operator angular-grid convergence through frequency errors and Hermitian matrix errors.
    - Added registered gate `landreman_paul_vmec_jax_boozer_operator_angular_convergence`, holding the one-iteration Landreman-Paul live bridge geometry fixed and comparing `12x12`, `16x16`, and `20x20` quadrature against a `32x32` reference grid. Current local metric: `1.7882e-3 < 5.0e-3`.
    - Added `examples/research/vmec_boozer_convergence.py` so users can reproduce the live bridge truncation and operator-angular convergence metrics and generate a compact figure.
    - Added the dedicated `vmec_boozer_operator_angular_convergence` README/docs figure.
    - Verified locally with `python -m pytest -q` (169 passed), coverage (`95.6%` FAX source coverage, fail-under 95 passed), `python -m ruff check .`, `python -m mypy src/fax`, `python tools/generate_validation_figures.py`, `python tools/qa_figures.py`, `python -m sphinx -W --keep-going -b html docs docs/_build/html`, `python -m build`, `python -m twine check dist/*`, and `git diff --check`.
    - Scope note: this strengthens local-operator numerical readiness. It still needs converged VMEC iteration studies, Boozer surface-resolution sweeps, and cross-code/literature comparisons before production physics claims.
  - 2026-04-29 AE3D dense-spectrum parity milestone:
    - Added `Ae3dDenseSpectrumParity` and `ae3d_dense_spectrum_parity`, which form dense copies of the public ORNL AE3D sparse generalized matrices, solve `A x = lambda B x` with SciPy/LAPACK, match the computed real eigenvalues to `egn_mode_asci.dat`, and report eigenvalue/frequency errors.
    - Added registered gate `ae3d_public_dense_spectrum_parity`. Current local metric: max `omega^2` eigenvalue relative error `1.6336e-6 < 2.0e-6` for the first 16 published AE3D modes in the 1287-dimensional public problem.
    - Updated the external-code dashboard, publication-suite AE3D figure, README, validation docs, examples, and tests so AE3D parity now includes matrix symmetry, published-vector residuals/Rayleigh quotients, and an independent dense eigensolve.
    - Verified locally with `python -m pytest -q` (171 passed), coverage (`95.5%` FAX source coverage, fail-under 95 passed), `python -m ruff check .`, `python -m mypy src/fax`, `python tools/generate_validation_figures.py`, `python tools/qa_figures.py`, `python examples/research/publication_figure_suite.py --format svg --format png --output-dir docs/_static/publication_suite`, `python -m sphinx -W --keep-going -b html docs docs/_build/html`, `python -m build`, `python -m twine check dist/*`, and `git diff --check`.
    - Scope note: this closes another public AE3D artifact/parity lane. It is still not a FAX reduced-operator-vs-AE3D physics parity claim; that remains a separate validation target once matching AE3D inputs/operators are fully mapped.
  - 2026-04-29 validation progress-output milestone:
    - Updated `fax validate` so the CLI logs every selected validation gate before and after execution, including gate index, gate id, layer, tolerance, elapsed time, metric, status, and message. This keeps long local validation runs visibly alive while GitHub Actions are unavailable.
    - Updated README/development/validation docs and CLI tests to cover the per-gate progress contract.
    - Verified locally with targeted CLI/progress tests, `python -m coverage erase && python -m coverage run --source=fax -m pytest -q && python -m coverage report --skip-empty -m --fail-under=95` (171 passed, 95.5% FAX source coverage), `python -m ruff check .`, `python -m mypy src/fax`, `python -m sphinx -W --keep-going -b html docs docs/_build/html`, and `git diff --check`.
  - 2026-04-29 VMEC solve-convergence ladder milestone:
    - Added `VmecJaxBoozerSolveConvergenceResult` and `run_vmec_jax_boozer_solve_convergence`, which run a native-grid `vmec_jax` iteration ladder, carry VMEC force residuals into the result object, transform each iterate through `booz_xform_jax`, and compare Boozer `|B|` plus FAX local 3-D continuum frequencies against a converged reference `wout`.
    - Added registered gate `landreman_paul_vmec_jax_solve_convergence`. Current local metric: `2.3407e-2 < 4.0e-2`, with VMEC residual ratio `6.474e-3` and FAX frequency-error ratio `2.365e-1` from requested iterations `(1, 6, 12, 24)`.
    - Added the runtime-comparison plan for matched FAX/STELLGAP/AE3D/CAS3D/FAR3D benchmark families. Speed claims must be coupled to passing physics metrics.
    - Added the `vmec_jax_solve_convergence` README/docs figure to show residual reduction, Boozer/FAX reference errors, and local runtime per ladder point.
    - Verified locally with direct gate execution, `python tools/generate_validation_figures.py`, `python tools/qa_figures.py`, targeted Landreman-Paul registry test, `python -m coverage erase && python -m coverage run --source=fax -m pytest -q && python -m coverage report --skip-empty -m --fail-under=95` (171 passed, 95.5% FAX source coverage), `python -m ruff check .`, `python -m mypy src/fax`, `python -m sphinx -W --keep-going -b html docs docs/_build/html`, `python -m build`, `python -m twine check dist/*`, and `git diff --check`.
  - 2026-04-29 benchmark-family and neighboring-branch numerics milestone:
    - Inspected current local `vmec_jax` `main` and active branches including `origin/gpu_optimization`, `origin/codex/qh-fixed-resolution-jax-main`, and `origin/codex/discrete-adjoint-2506`. Performance/numerics practices to carry into FAX: separate cold and warm runtime, avoid host/device synchronization in quiet scans, keep CPU debug callbacks away from CUDA paths, use small-problem tridiagonal accelerators only behind shape/backend gates, and use chunked Jacobian/adjoint construction before dense autodiff paths become large.
    - Inspected current local `booz_xform_jax` `main` and `origin/fax/jax-geometry-jacobian-outputs`. Useful practices: maintain in-memory VMEC-to-Boozer parity, expose Jacobian/harmonic outputs directly, precompute angular trig tables, and prefer vectorized double-spectral Fourier integrals to repeated Python/dispatch loops.
    - Extended `fax benchmark` with named benchmark families: `analytic-local`, `global-block`, `stellgap-ctok`, `ae3d-artifact`, and `vmec-boozer`. The public continuum workflow remains the baseline, and each added family records cold runtime, warm runtime, process memory, JAX backend metadata, and a physics/parity metric with tolerance.
    - Added the `benchmark_family_packet` README/docs figure and JSON summary so runtime plots are tied to physics gates instead of standalone wall-clock anecdotes.
    - Added tests for dataclass-aware JAX `block_until_ready` synchronization, the analytic named benchmark family, and CLI benchmark-family JSON output.
  - 2026-04-29 sparse/Krylov global radial benchmark milestone:
    - Added sparse CSR assembly for the block global radial operator with the same finite-difference energy as the dense JAX assembly.
    - Added `solve_radial_block_eigenmodes_sparse` as a SciPy sparse/Krylov scale-up path. This is intentionally a non-differentiable benchmark/reference solver; the dense JAX eigensolve remains the differentiable truth path.
    - Added exact-spectrum and dense-assembly parity tests for the sparse matrices and Krylov eigenpairs.
    - Added the `global-krylov` benchmark family, reporting cold/warm runtime, sparse matrix dimension/nonzeros, residuals, and exact-spectrum error.
  - 2026-04-29 matrix-free global radial operator milestone:
    - Added JAX-native matrix-free applications of the block radial stiffness and generalized mass operators. These use the same active degrees of freedom and finite-difference energy as the dense and sparse assemblies, but avoid forming a matrix.
    - Added matrix-free generalized residual and Rayleigh-quotient diagnostics. These are the immediate building blocks for future JAX-native Krylov, shift-invert, Lineax, and custom-adjoint paths.
    - Added Dirichlet and Neumann parity tests against dense assembly, eigenpair residual/Rayleigh tests, and a density-scaling gradient gate through the matrix-free Rayleigh quotient.
    - Added the `global-matrix-free` benchmark family and included it in the generated benchmark packet so runtime claims include dense-matvec and exact-Rayleigh physics gates.
  - 2026-04-29 generalized-eigenvalue sensitivity milestone:
    - Added `generalized_hermitian_eigenvalue_sensitivity`, implementing the isolated Hermitian generalized-eigenvalue formula `x^H(dK - lambda dM)x/(x^H M x)` with an explicit normalization denominator.
    - Added `radial_block_eigenvalue_sensitivity`, which assembles `dK` and `dM` from a radial-block tangent problem so mass, continuum-stiffness, and radial-stiffness tangent directions can be checked without relying on autodiff through the eigensolver.
    - Added tests against direct JAX autodiff and centered finite differences, including normalization-invariance checks and tangent-contract errors.
    - Added registered gate `global_3d_block_eigenvalue_sensitivity` for reviewer-facing adjoint/autodiff agreement on the block radial shear-Alfven eigenproblem.
  - 2026-04-29 custom-JVP isolated eigenvalue milestone:
    - Added `generalized_hermitian_isolated_eigenvalue`, a dense generalized-eigenvalue value path whose JAX JVP uses the explicit isolated-eigenpair adjoint formula instead of differentiating through all eigensolver internals.
    - Added `radial_block_isolated_omega2` and `radial_block_isolated_frequency_hz` so global radial objectives can use the custom-JVP path after matrix assembly.
    - Added tests comparing custom-JVP gradients with explicit adjoint sensitivities and centered finite differences, plus JIT/value parity against the dense solve.
    - Added registered gate `global_3d_block_custom_jvp_eigenvalue`.
  - 2026-04-29 isolated Boozer design-gradient milestone:
    - Added `boozer_global_isolated_mode_omega2`, `boozer_global_isolated_mode_frequency_hz`, and `boozer_global_isolated_mode_avoidance_objective` so real-Boozer global design studies can track one isolated mode through the custom-JVP eigenvalue path.
    - Added `boozer_pipeline_isolated_mode_avoidance_objective` for callable Boozer-geometry builders, preserving the API needed for future differentiable `vmec_jax -> booz_xform_jax -> FAX` shape/profile objectives.
    - Added tests comparing isolated-mode values against the dense frequency vector and gradients against centered finite differences, plus a pipeline-builder differentiation test.
    - Added registered gate `boozer_global_custom_jvp_design_gradient` for reviewer-facing custom-JVP/finite-difference agreement on Boozer design variables.
  - 2026-04-29 degenerate-cluster sensitivity milestone:
    - Added `GeneralizedEigenclusterSensitivity`, `generalized_hermitian_eigencluster_sensitivity`, and `generalized_hermitian_cluster_mean_eigenvalue` for subspace derivatives at exact degeneracy and near crossings.
    - Added `radial_block_cluster_mean_omega2` and `radial_block_cluster_mean_frequency_hz` so global radial block studies can optimize/report a reviewer-safe cluster mean rather than an invalid single-mode gradient through a crossing.
    - Added tests for exact degenerate splitting directions, cluster-mean finite differences, JIT/value parity, and radial-block duplicate-harmonic degeneracy.
    - Added registered gate `global_3d_block_degenerate_cluster_mean`.
  - 2026-04-29 cluster optimization and documentation expansion milestone:
    - Added `boozer_global_cluster_mean_omega2`, `boozer_global_cluster_mean_frequency_hz`, `boozer_global_cluster_mean_avoidance_objective`, and `boozer_pipeline_cluster_mean_avoidance_objective` so real-Boozer and future VMEC/Boozer design studies can optimize crossing-safe mode-family objectives.
    - Added tests for cluster-mean objective gradients against centered finite differences and callable Boozer-builder differentiation.
    - Added registered gate `boozer_global_cluster_mean_design_gradient`.
    - Expanded ReadTheDocs with model inventory, numerical algorithms, validation traceability, user workflows, and function-map pages covering functionality, inputs/outputs, equations, algorithms, tests, plots, and current claim boundaries.
  - 2026-04-29 global eigenmode CLI/artifact milestone:
    - Added `run_global_eigenmodes` as the TOML-facing workflow for scalar profile radial modes and real-`boozmn` Boozer radial block modes.
    - Added `fax eigenmodes` and `fax plot-global` so users can generate global eigenfrequencies, residuals, mode-structure CSV files, JSON metrics, and publication-draft dashboards without writing Python.
    - Added `examples/global_eigenmodes.toml`, public plotting/load helpers, CLI/plot tests, and docs/README coverage for the new workflow.
    - Added `global_eigenmode_artifact_dashboard` to the generated docs figure set and figure manifest.

### Local Testing Policy Until GitHub Actions Resume

GitHub Actions are expected to remain unavailable until the UWPlasma billing/spending-limit issue resets next month. Until then, treat local checks as the required quality gate before every commit:

```bash
python -m ruff check .
python -m pytest
sphinx-build -W -b html docs docs/_build/html
python -m build
```

Do not weaken tests to accommodate the missing CI. Instead, keep local validation commands logged here and use `[skip ci]` only to avoid creating known-failing remote runs while the account-level blocker remains.

### First VMEC-Backed Stellarator Reference Target

Use `LandremanPaul2021_QA_lowres` as the first `vmec_jax`/VMEC-backed stellarator target.

Local paths identified on 2026-04-24:

- VMEC input: `/Users/rogerio/local/vmec_jax/examples/data/input.LandremanPaul2021_QA_lowres`
- VMEC `wout`: `/Users/rogerio/local/vmec_jax/examples/data/wout_LandremanPaul2021_QA_lowres.nc`
- alternate `wout` fixture: `/Users/rogerio/local/simsopt/tests/test_files/wout_LandremanPaul2021_QA_lowres.nc`

Why this case:

- It is a low-resolution version of the Landreman-Paul precise quasi-axisymmetric configuration, so it is small enough for local testing.
- It is literature anchored by Landreman & Paul, "Magnetic fields with precise quasisymmetry for plasma confinement", PRL 128, 035001 (2022), arXiv:2108.03711.
- It is a two-field-period quasi-axisymmetric stellarator, so it is directly relevant to early continuum/gap and toroidal-coupling tests.
- Neighboring `vmec_jax` and `booz_xform_jax` workflows already use the same case, which reduces adapter risk.

Immediate uses:

1. M1 adapter smoke test: read input/wout metadata, extract `nfp`, `ns`, `mpol`, `ntor`, `iota`, `B`, and geometry arrays.
2. M2 continuum prototype: compute local large-aspect-ratio sanity checks before the full 3-D operator.
3. Boozer pipeline check: run/compare `booz_xform_jax` on the same `wout` once Boozer coordinates are needed.
4. Benchmark seed: record local parse/adapter/operator runtime before optimizing.
5. Physics understanding: use the quasisymmetric Alfvén-continuum literature as a nearby reference, while being explicit that this case is not a validation until a concrete metric is matched.

Sources:

- <https://arxiv.org/abs/2108.03711>
- <https://doi.org/10.1103/PhysRevLett.128.035001>
- <https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/shear-alfven-continuum-of-quasisymmetric-stellarators/7B50E360D4324D8AF93DA73EB25C35D8>

### Next Recommended Step

Turn the reduced physics layers into publication-grade stellarator validation
and differentiable design studies that can produce new FAX-native results:

1. Extend the STELLGAP CTOK local-continuum and largest-local-gap gates to additional STELLGAP/literature geometries, then add true AE3D/CAS3D/FAR3D global-mode parity where input/output data and executable builds are available.
2. Extend the new real-`boozmn` angular, radial-surface, boundary-condition, and radial-stiffness convergence gates to larger radial grids, Boozer harmonic truncation sweeps, and multiple geometries.
3. Add additional radial boundary-condition variants beyond Dirichlet/Neumann and validate them on analytic and cross-code targets.
4. Calibrate the reduced gyro-Landau fast-ion drive/damping closure against FAR3D/CAS3D-K comparable outputs, with sign, resonance, growth-rate, and eigenfunction-overlap gates.
5. Wire `boozer_pipeline_mode_avoidance_objective` to a real differentiable `vmec_jax`/`booz_xform_jax` builder and add finite-difference/adjoint consistency plots for VMEC-to-stability gradients.
6. Use the differentiable-design study API on real VMEC/Boozer geometries to produce novel design maps: sensitivity-ranked boundary/profile variables, UQ-aware AE risk bands, inverse-designed resonance avoidance, and alpha/beam resonance robustness scans.
7. Publish the staged `v0.0.1` GitHub release when GitHub Actions are available so the PyPI trusted-publishing workflow can complete.
8. Extend benchmark packets with matrix-free JAX linear-operator families and GPU memory telemetry once those kernels are introduced.

### Runtime Comparison Plan Against Reference Codes

FAX performance claims must be reported on matched physics problems rather
than generic wall-clock anecdotes. Add a runtime table and plot for each
validated comparison family:

1. STELLGAP CTOK local continuum: compare FAX local 3-D Boozer continuum
   runtime against STELLGAP `xstgap` on the same `boozmn_ctok.nc` and matched
   harmonic/surface sets once the local STELLGAP executable issue is resolved.
   Report setup/read time, solve time, post-processing time, total wall time,
   CPU/GPU hardware, thread/MPI counts, and matched p95 continuum error.
2. AE3D public matrix problem: compare FAX/SciPy dense and future FAX sparse or
   matrix-free global solves against AE3D artifact dimensions and, when AE3D
   builds with a portable compiler, against AE3D executable wall time. Report
   matrix assembly, eigensolve, residual check, memory footprint, and
   eigenfrequency parity error.
3. CAS3D/CONTI and CAS3D-K: once runnable inputs or archived outputs are
   available, compare global mode frequency/eigenfunction/growth-rate runtime
   for matched radial/harmonic resolution and physics toggles. Keep these in a
   manual slow benchmark tier until portable execution is proven.
4. FAR3D: after a CPU-only or CUDA Fortran build is available, compare reduced
   fast-ion drive/damping scans on matched equilibria and energetic-particle
   profiles. Report parameter-scan throughput as well as single-case runtime.
5. FAX-native differentiable studies: benchmark gradient, Jacobian, UQ, and
   inverse-design loops against finite-difference baselines and against the
   closest non-differentiable external workflow. Separate compile time from
   warmed runtime and include derivative accuracy/error bars.

Implementation status: `fax benchmark` now supports named benchmark families
(`analytic-local`, `global-block`, `global-matrix-free`, `global-krylov`,
`vmec-boozer`, `ae3d-artifact`, `ae3d-global-mode`, `stellgap-ctok`,
`stellgap-executable`) and writes JSON/CSV artifacts plus the first
`benchmark_family_packet` docs plot. The executable STELLGAP family records
completed/failed/timeout/skipped status; the current local rebuilt CTOK run
completes `xstgap 99 2048`, produces `alfven_spec`, and takes about 47 s on
this Mac. Next, add Markdown reports, JAX-native matrix-free eigensolver
families, executable AE3D timings once portable builds are stable, and GPU
memory telemetry. Do not claim a speedup unless the physics metric also passes
the corresponding validation gate.

### 2026-04-29 Real-Boozer TOML Global Eigenmode Workflow

Added `examples/boozmn_global_eigenmodes.toml` so users can run the
metric-derived real-Boozer radial block operator through the public
`fax eigenmodes` and `fax plot-global` workflow. Added a CLI integration test
that runs this TOML path against the local
`boozmn_LandremanSenguptaPlunk_section5p3.nc` fixture, verifies the operator
is reported as `boozmn_global_3d_radial_block`, checks residuals below the
registered real-geometry gate, and compares the written CSV frequencies and
residuals against a direct call to `radial_block_problem_from_boozer_geometry`
and `solve_radial_block_eigenmodes`.

Docs and README now include the real-Boozer CLI commands and a generated
`boozmn_global_eigenmode_artifact_dashboard` figure. This closes an important
handoff gap: the same validated real-Boozer block operator is now exercised
through the end-user TOML, artifact, and plotting path rather than only through
Python examples and registry gates.

### 2026-04-30 STELLGAP Gap Parity, Boozer Convergence, And Gyro-Landau Closure

Advanced the remaining validation lanes without overstating external-code
parity:

- Added `stellgap_ctok_gap_parity`, which compares FAX and public ORNL
  STELLGAP CTOK largest adjacent-branch local gap centers and widths on
  matched surfaces. This is stronger than artifact parsing and complements the
  existing p95 branch-frequency gate, but it is still a local continuum/gap
  comparison rather than a full global eigenmode comparison.
- Added `run_stellgap_ctok_executable` and the `stellgap-executable` benchmark
  family. The runner copies public CTOK inputs into a temporary directory and
  records return code, output files, stdout/stderr tails, and elapsed time. On
  the current local gfortran/OpenMPI build, `/Users/rogerio/local/Stellgap/xstgap`
  still fails while opening an invalid `alfven_spec...` path, so the benchmark
  records `failed` status and no runtime parity claim is made.
- Added real-`boozmn` global radial block gates for angular quadrature
  convergence, radial surface coarsening, and Dirichlet/Neumann plus
  metric/mass radial-stiffness variants. These keep the global radial 3-D
  operator protected while the external global parity data are being assembled.
- Added `GyroLandauClosure`, `gyro_landau_closure_drive`, and
  `gyro_landau_eigenfunction_drive` as the first reduced FAR3D/CAS3D-K-facing
  fast-ion closure API. The current gate verifies resonance peaking,
  pressure-gradient sign, differentiability, and damping offsets. It is not
  yet calibrated to FAR3D/CAS3D-K growth rates.
- Updated README, ReadTheDocs pages, changelog, release notes, validation
  registry tests, and benchmark tests to make these claim boundaries explicit.

Remaining open lanes after this milestone:

1. Obtain runnable or archived global AE3D/CAS3D/FAR3D/STELLGAP input/output
   pairs for true FAX-vs-code global eigenfrequency and mode-structure parity.
2. Repair or rebuild the local STELLGAP CTOK executable path so `xstgap`
   produces `alfven_spec` reliably; only then add it to runtime comparison
   tables.
3. Calibrate the gyro-Landau closure coefficients against FAR3D/CAS3D-K
   kinetic growth/damping references.
4. Wire a real differentiable VMEC boundary/design-variable builder into the
   existing `boozer_pipeline_*` objectives and add finite-difference checks at
   the VMEC input-variable level.
5. Tag and publish `v0.0.1` through GitHub/PyPI once Actions restart and the
   local release gate is still green.

### 2026-04-30 AE3D Global Mode-Shape Parity And STELLGAP Executable Closure

Closed two more external-code readiness lanes while keeping claim boundaries
explicit:

- Added `Ae3dGlobalModeParity` and `ae3d_global_mode_parity`. FAX now reads
  ORNL AE3D public `a_matrix.dat`, `b_matrix.dat`, and `egn_mode_asci.dat`,
  independently solves the 1287-dimensional generalized eigenproblem with
  SciPy/LAPACK, matches the first 8 published real eigenpairs, and compares
  frequencies plus Euclidean and `B`-weighted mode-shape correlations. Current
  local metrics: max frequency relative error `8.17e-7`, max Euclidean
  mode-shape error `6.58e-8`, and max `B`-weighted mode-shape error
  `5.20e-8`, all below the `2.0e-6` gate.
- Added registered gate `ae3d_public_global_mode_shape_parity` and benchmark
  family `ae3d-global-mode`. This closes public AE3D artifact/matrix/mode-shape
  parity, not native FAX reduced-operator vs AE3D full-operator physics parity.
- Rebuilt the local STELLGAP executable path from the public source so
  `run_stellgap_ctok_executable` can run `xstgap 99 2048`. The FAX benchmark
  now passes when `alfven_spec` is produced; current local runtime is about
  47 s for the public CTOK grid.
- Added `fit_gyro_landau_eigenfunction_closure`, which calibrates reduced
  gyro-Landau resonant, dispersive, and thermal damping coefficients from
  eigenfunction-weighted growth rates through a differentiable least-squares
  solve. The new gate `fast_ion_gyro_landau_calibration` recovers synthetic
  FAR3D/CAS3D-K-facing coefficients to roundoff and protects the sign
  convention before real growth-rate parity data are available.
- Updated README, ReadTheDocs pages, changelog, validation registry tests, API
  exports, benchmark tests, and cross-code/fast-ion tests.

Remaining open lanes after this milestone:

1. True native FAX-vs-AE3D/STELLGAP global operator parity on matched
   equilibrium, radial basis, harmonic truncation, boundary-condition, and
   normalization conventions.
2. CAS3D/CAS3D-K and FAR3D parity once runnable binaries or archived
   input/output pairs are available.
3. Calibration of the reduced gyro-Landau closure against actual FAR3D/CAS3D-K
   energetic-particle growth/damping results.
4. Full end-to-end differentiable VMEC boundary/design-variable objectives
   through vmec_jax, booz_xform_jax, and FAX with finite-difference/adjoint
   consistency checks at the VMEC input-variable level.
5. Final release hygiene: regenerate validation figures, run full local
   release gate, align the `v0.0.1` tag with the final commit, publish the
   GitHub release, and let PyPI trusted publishing run when Actions restart.

### 2026-04-30 v0.0.1 Publication Results Page

Added `docs/publication_results.rst` as the release-facing publication packet.
It collects the ten generated publication figures, claim levels, key metrics,
external-code parity status, exact regeneration commands, local release gate,
and remaining paper-quality lanes. README and the Sphinx index now point to
this page, and the figure docs distinguish the current publication suite from
future publication figures.

### Full Research-Grade Completion Roadmap

This section supersedes the scattered "remaining open lanes" notes as the
single roadmap for reaching the final no-proxy, no-reduced-model, no-scaffold
research-grade FAX target. Until every required milestone below is complete,
FAX must continue to label reduced models, artifact parity, adapter checks,
and screening applications explicitly in README, docs, validation reports, and
publication figures.

The machine-readable tracker is `fax.maturity` and the CLI report:

```bash
fax maturity
fax maturity --json
```

#### RG-1 Full Native 3-D Stellarator Ideal-MHD Eigenmode Parity

Goal: FAX native global 3-D operators reproduce matched AE3D, CAS3D/CONTI,
and STELLGAP global eigenfrequencies, eigenfunctions, residuals,
normalizations, harmonic conventions, mode labels, and boundary-condition
variants on multiple stellarator geometries.

Current status: in progress. Implemented gates include STELLGAP CTOK local
continuum/gap parity and AE3D public matrix/eigenmode/mode-shape artifact
parity. Missing: native FAX-vs-AE3D/STELLGAP/CAS3D global operator parity on
matched operators and geometry.

Acceptance gates:

- analytic limits and residual checks for every retained ideal-MHD term,
- radial, angular, harmonic, and boundary-condition convergence,
- matched-basis AE3D frequency and eigenfunction parity,
- STELLGAP global continuum/eigenmode parity where applicable,
- CAS3D/CONTI ideal-MHD frequency and mode-family parity,
- publication figures showing mode structures, residuals, convergence, and
  code-to-code differences.

Immediate next steps:

1. Promote the current CTOK/STELLGAP and AE3D artifacts into a versioned
   benchmark manifest with file hashes, conventions, expected metrics, and
   local path discovery.
2. Build a native FAX global parity driver that compares FAX modes to external
   eigenfunctions rather than only re-solving external matrices.
3. Add one additional non-CTOK stellarator geometry with literature or
   cross-code global-mode data.

#### RG-2 CAS3D/CAS3D-K/FAR3D Kinetic Growth-Rate Validation

Goal: FAX energetic-particle drive/damping reproduces CAS3D-K and FAR3D
growth rates, damping rates, thresholds, resonance localization, and trend
directions on matched equilibria, distributions, and mode structures.

Current status: planned/in progress. The reduced gyro-Landau closure API,
sign gates, eigenfunction projection, and synthetic coefficient-recovery tests
are implemented. Missing: real FAR3D/CAS3D-K growth-rate parity.

Acceptance gates:

- analytic resonance and sign checks,
- FAR3D linear growth-rate parity on at least one stellarator and one tokamak
  or reactor-relevant benchmark,
- CAS3D-K perturbative growth-rate parity for matched mode structures,
- distribution-function and resonance-width convergence,
- held-out validation cases separate from calibration cases,
- uncertainty estimates for calibrated growth-rate predictions.

Immediate next steps:

1. Define the FAR3D/CAS3D-K benchmark data schema.
2. Create the first archived growth-rate fixture from FAR3D or literature.
3. Add train/test calibration logic so fitted closure coefficients are not
   validated on the same data that set them.

#### RG-3 Fully Calibrated Fast-Ion Drive/Damping Replacement

Goal: The FAX fast-ion model can replace kinetic-code calls within a declared
domain of validity, with calibrated coefficients, cross-validation, uncertainty
bands, and failure-mode diagnostics.

Current status: in progress. Reduced resonance screens and gyro-Landau
coefficient fitting exist, but they are not calibrated against real kinetic
growth/damping references.

Acceptance gates:

- coefficient recovery on synthetic data,
- calibration to real FAR3D/CAS3D-K cases,
- held-out growth-rate prediction error below documented tolerance,
- sign/threshold/localization parity,
- sensitivity and UQ validation for fast-ion inputs,
- docs that state domain of validity and failure modes.

Immediate next steps:

1. Add a `FastIonCalibrationDataset` schema with metadata, distribution
   parameters, mode labels, growth rates, damping rates, and uncertainty.
2. Add a calibration report workflow that writes train/test errors and plots.
3. Add a validation gate that fails if held-out kinetic growth-rate parity is
   missing for a claimed calibrated model.

#### RG-4 End-To-End Differentiable VMEC/Boozer/FAX Optimization

Goal: Actual VMEC boundary/profile variables differentiate through a
controlled `vmec_jax` solve, `booz_xform_jax` transform, FAX operator,
eigenvalue or growth-rate objective, and reviewer-grade finite-difference or
adjoint checks.

Current status: in progress. Builder-based Boozer objectives, live
VMEC/Boozer adapter checks, VMEC solve-convergence ladders, and reduced
parameter design demonstrations exist. Missing: gradients with respect to
actual VMEC boundary/design variables through converged physics.

Acceptance gates:

- converged VMEC residual ladder,
- Boozer transform truncation and quadrature convergence,
- local/global FAX operator convergence,
- isolated-mode and cluster objective gradient checks,
- finite-difference versus JAX gradient agreement at VMEC input-variable level,
- inverse-design example with post-optimization validation and cross-code
  recheck.

Immediate next steps:

1. Select a small, stable VMEC boundary coefficient vector for the first design
   study.
2. Wrap a fixed-shape `vmec_jax -> booz_xform_jax -> FAX` builder that returns
   a differentiable objective and all residual/convergence metadata.
3. Add finite-difference checks for one isolated mode and one cluster-mean
   objective.

#### RG-5 Multi-Geometry Cross-Code Benchmark Suite

Goal: FAX validation covers enough geometry and physics diversity to support
broad stellarator reactor-design claims, not only one CTOK/STELLGAP fixture
and one AE3D public matrix artifact.

Current status: in progress. CTOK/STELLGAP and AE3D public artifacts are
implemented; the benchmark-family CLI exists. Missing: a curated
multi-geometry manifest, additional external-code cases, and runtime tables
for matched physics.

Acceptance gates:

- benchmark manifest with hashes, sources, physics toggles, expected metrics,
  and commands,
- quasi-axisymmetric, quasi-helical, tokamak, heliotron/heliac, and
  reactor-relevant cases,
- convergence and parity figures for each case,
- runtime comparisons with external-code status and hardware metadata,
- no speed claim without a passing physics metric.

Immediate next steps:

1. Add `benchmark_manifest.json` and schema validation.
2. Promote current CTOK and AE3D public artifacts into the manifest.
3. Add at least one QH/heliotron-like geometry once trusted parity data are
   available.

#### RG-6 Final No-Proxy/No-Reduced-Model Release Gate

Goal: Default FAX publication workflows use validated full-physics operators,
and any optional reduced/screening model is explicitly labeled as such. No
claim in README, docs, examples, or figures depends on an unvalidated proxy.

Current status: blocked. This gate cannot pass until RG-1 through RG-5 are
complete.

Acceptance gates:

- `fax maturity --json` reports all required milestones complete,
- every default example is tied to full-physics validation or explicitly
  labeled optional screening,
- figure manifest and publication suite include claim level and validation
  gate IDs for every panel,
- docs contain no unsupported full-physics claims,
- release checklist includes cross-code parity, kinetic validation,
  differentiability validation, benchmark suite, and artifact hashes.

Immediate next steps:

1. Keep `fax maturity` reporting `not_complete` until all full-physics gates
   pass.
2. Add tests that fail if docs claim final no-proxy status prematurely.
3. Move reduced models into explicitly labeled optional workflows once full
   operators are available.

### 2026-04-30 Full-Research-Grade Tracker Milestone

Started implementing the full-research-grade roadmap by adding:

- the machine-readable `fax.maturity` registry,
- the `fax maturity` CLI command,
- `docs/research_grade_roadmap.rst`,
- the machine-readable `fax.benchmark_manifest` registry,
- the `fax benchmark-manifest` CLI command,
- `docs/benchmark_manifest.rst`.

This does not close any physics parity gap by itself; it makes the remaining
gaps explicit, testable, and visible to users and reviewers. The benchmark
manifest now promotes the current CTOK/STELLGAP local-continuum/gap artifacts
and AE3D public matrix/mode-shape artifacts into versioned manifest entries
with claim levels, expected metrics, validation gate IDs, local paths, optional
SHA-256 hashes, and the next parity step. The next coding lane is to use this
manifest to drive native global parity comparisons rather than hard-coded
local path checks.

### 2026-04-30 Kinetic Growth-Rate Validation Contract

Started RG-2/RG-3 implementation by adding a concrete FAR3D/CAS3D-K
growth-rate data contract:

- `fax.kinetic_validation.KineticGrowthRateCase`,
- `fax.kinetic_validation.KineticGrowthRateDataset`,
- uncertainty-weighted differentiable `kinetic_growth_rate_loss`,
- held-out `kinetic_growth_rate_parity`,
- JSON dataset/prediction loaders,
- the `fax kinetic-parity` CLI command,
- `docs/kinetic_validation.rst`.

This is deliberately not marked as completed kinetic validation. It enforces
the structure needed for reviewer-grade kinetic parity: external-code source,
geometry, mode ID, frequency, growth rate, uncertainty, toroidal mode,
distribution parameters, and calibration-vs-validation split. The next step is
to populate this contract with real FAR3D/CAS3D-K or literature growth-rate
artifacts, then fit reduced closure coefficients only on calibration cases and
test them on held-out cases.

### 2026-04-30 AE3D Global-Mode Continuum-Placement Gate

Added a stronger RG-1 intermediate cross-code physics gate:

- `fax.cross_code.ae3d_continuum_placement_parity`,
- `fax.cross_code.ae3d_continuum_placement_from_arrays`,
- validation gate `ae3d_public_continuum_placement_parity`,
- benchmark-manifest expected metric and artifact requirement,
- publication-suite metric update for the AE3D parity figure.

The gate reads public AE3D global eigenmodes, identifies each mode's dominant
harmonic and radial peak surface, evaluates the FAX local CTOK Boozer continuum
for the same harmonics, and checks the distance from the AE3D eigenfrequency to
the closest FAX local continuum branch at that surface. The current acceptance
metric is p95 closest-branch relative error below `3e-2` for the first 128
public AE3D modes. This is a real global-mode placement check and a useful
paper figure metric, but it is still not final native FAX-vs-AE3D global
operator parity. The next RG-1 step remains constructing a matched native FAX
global operator comparison with external eigenfrequencies and eigenfunctions.

### 2026-04-30 FAR3D Lincheck Ingestion

Started turning the RG-2/RG-3 kinetic contract into an external-code ingestion
workflow:

- inspected local FAR3D `src/lincheck.f90` and confirmed the diagnostic rows
  printed as `variable : m=... n=... gam=... om_r=...`,
- added `fax.far3d.Far3dLinearGrowthRow`,
- added `read_far3d_lincheck` for FAR3D run logs/unit-6 output,
- added `far3d_rows_to_kinetic_dataset` and `load_far3d_lincheck_dataset`,
- added the `fax far3d-ingest` CLI command,
- documented the workflow in README and `docs/kinetic_validation.rst`,
- added parser, conversion, public export, and CLI tests,
- reran the local FAR3D CMake build and confirmed it still blocks at
  `src/matrix.f90: use cudafor` because `cudafor.mod` is unavailable on the
  current CPU toolchain; `matrix_cpu.f90` exists upstream, so the next external
  code task is a portable CPU build recipe or archived FAR3D outputs.

Important scope: this is not yet FAR3D kinetic parity. The parser preserves
native FAR3D `gam` and `om_r` code units and requires an explicit
`frequency_scale_hz`/`growth_rate_scale_hz` before creating the FAX
Hz-normalized validation packet. The next step is to obtain or run one portable
FAR3D linear benchmark with documented normalizations, ingest it, and make the
resulting held-out growth-rate gate part of the benchmark manifest.

### 2026-04-30 FAR3D GPU Build/Run Artifact

Closed the first external FAR3D executable-runtime lane on the `office` GPU
host:

- cloned `DSfusion/FAR3d` under `/home/rjorge/local/FAR3d`,
- confirmed the CPU matrix path can build with the existing conda Fortran/MPI
  toolchain when `matrix_cpu.f90` is copied over `src/matrix.f90`,
- installed a user-space minimal NVIDIA HPC SDK 25.9 payload because the full
  installer filled the host filesystem and system compilers did not provide
  CUDA Fortran `cudafor.mod`,
- built upstream FAR3D `src/matrix.f90` with NVHPC/OpenACC/CUDA Fortran using
  the bundled HPC-X MPI runtime,
- resolved the runtime MPI prefix issue by launching with
  `mpirun -np 1 --prefix "$OPAL_PREFIX" ./far3d.x`,
- translated the public DIII-D `Eq_DIIID_RS` example into current `farin`
  namelist form and ran a bounded `maxstp=20` GPU test,
- archived compact run artifacts under
  `external_artifacts/far3d/gpu_diiid_nvhpc`,
- ingested `farprt0000` into
  `external_artifacts/far3d/gpu_diiid_nvhpc/kinetic_dataset.json` using the
  FAR3D-reported conversion factor `599.81341 kHz` per code-frequency unit,
- added a benchmark-manifest entry,
- added parser/ingestion tests for the real artifact,
- added `examples/research/far3d_gpu_lincheck_artifact.py`,
- generated `docs/_static/far3d_gpu_lincheck_artifact.{png,svg}`,
- documented the build/run recipe and scope in `docs/external_codes.rst`,
  `docs/kinetic_validation.rst`, README, release notes, and examples.

Important scope: this is a real external-code executable/build and ingestion
artifact, not converged FAR3D kinetic parity. The run is intentionally short
and exists to prove the CUDA-Fortran build, current-input translation, MPI
launch, parser, and normalization path. The next FAR3D lane is a documented
linear benchmark run long enough for converged growth rates, with complete
input/output hashes, runtime metadata, and calibration-vs-held-out validation
against the FAX fast-ion closure.

### 2026-04-30 FAR3D Run-Length Convergence Ladder

Advanced the FAR3D external-code lane from a single executable trace to a
bounded convergence artifact:

- ran the same public DIII-D `Eq_DIIID_RS` FAR3D GPU case on `office` at
  `maxstp = 50, 100, 200, 400`,
- archived compact `farprt0000`, `farin`, `stdout`, `stderr`, runtime, and
  extracted `lincheck` rows under
  `external_artifacts/far3d/gpu_diiid_nvhpc_convergence`,
- added `Far3dLincheckConvergencePoint` and `far3d_lincheck_convergence`,
- added real-artifact regression tests proving all four run lengths have the
  same 24 common mode labels and that the 200-step output is below `5e-5`
  growth and `2e-5` frequency relative error against the 400-step reference,
- added a second FAR3D benchmark-manifest entry for the convergence ladder,
- added `examples/research/far3d_gpu_convergence_ladder.py`,
- generated `docs/_static/far3d_gpu_convergence_ladder.{png,svg}`,
- updated README, docs, examples, release notes, and the benchmark manifest.

Observed metrics for this DIII-D artifact:

- total ladder runtime on `office`: 15 seconds,
- 200-vs-400 maximum relative growth-rate difference: `2.6e-5`,
- 200-vs-400 maximum relative real-frequency difference: `6.8e-6`,
- 400-step across-field/harmonic spread: zero at printed FAR3D precision.

Important scope: this is still one reduced-MHD/gyrofluid FAR3D input. It is
now stable enough to serve as a parser/convergence fixture, but not enough to
calibrate or validate the FAX fast-ion closure. The next step is a true FAR3D
physics benchmark suite: beta/distribution scans, at least one stellarator
case, documented mode IDs and units, and held-out growth-rate comparisons.

### 2026-04-30 FAR3D Fast-Ion Beta Scan Artifact

Advanced the FAR3D lane from run-length convergence to a first bounded
parameter scan:

- ran the same public DIII-D ``Eq_DIIID_RS`` FAR3D GPU executable/input on
  ``office`` at ``bet0_f = 0.000, 0.010, 0.030, 0.060, 0.090`` with
  ``maxstp = 200``,
- archived compact ``farprt0000``, ``farin``, ``stdout``, ``stderr``,
  runtime, and extracted ``lincheck`` rows under
  ``external_artifacts/far3d/gpu_diiid_nvhpc_beta_scan``,
- added ``Far3dLincheckScanPoint`` and
  ``far3d_lincheck_parameter_scan`` for scalar FAR3D parameter scans,
- added real-artifact regression tests proving all scan points have 24
  ``lincheck`` rows and that the finite-beta branch has monotonically
  increasing mean growth,
- added a benchmark-manifest entry for
  ``far3d_diiid_gpu_fastion_beta_scan``,
- added ``examples/research/far3d_gpu_beta_scan.py``,
- generated ``docs/_static/far3d_gpu_beta_scan.{png,svg}``,
- updated README, docs, examples, release notes, and the benchmark manifest.

Observed metrics for this DIII-D artifact:

- total scan runtime on ``office``: 12 seconds,
- finite-beta mean-growth values: ``0.0147696554``, ``0.0511065958``,
  ``0.1006467917``, ``0.177477`` in FAR3D code units,
- finite-beta linear slope: ``2.0007`` code units per ``bet0_f``,
- beta-zero control growth spread: ``0.8968795`` code units.

Important scope: this is the first FAR3D executable parameter-scan artifact
and is useful for developing the closure-calibration workflow, but it is still
one DIII-D reduced-MHD/gyrofluid case. It is not calibrated FAX/FAR3D kinetic
parity. The next step is to fit FAX reduced gyro-Landau closure coefficients
on a subset of FAR3D beta/distribution points, validate on held-out points,
and repeat the workflow on at least one stellarator geometry.
