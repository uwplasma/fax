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

Implement the first metric-aware stellarator continuum milestone:

1. Add radial resolution-convergence gates for the analytic metric fixture and the Landreman-Paul QA adapter.
2. Add an adapter/figure gate that checks VMEC-derived continuum branches are finite before plotting.
3. Add an optional Boozer path through `booz_xform_jax` for models that need straight-field-line harmonics, while keeping analytic and VMEC-coordinate fixtures runnable without Boozer.
4. Turn the metric matrix path into a CLI-accessible `fax continuum` or `fax validate` workflow with rich progress output, artifact writes, and figure generation.
5. Start the first literature/cross-code comparison gate: STELLGAP-style continuum gaps for a documented quasi-symmetric or axisymmetric reference case.
