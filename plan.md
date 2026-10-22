# FAX 1.0 plan: a lean, differentiable Alfvén-eigenmode code

This plan replaces `plan.md` as the guide for FAX 1.0. Rules that apply to every phase:

- Use as few lines and files as possible.
- Every physics claim needs an analytic or cross-code gate.
- Everything from boundary to growth rate is differentiable in JAX.
- `pip install fax-alfven` (or `uwplasma-fax`) works.
- The same API is available from Python and from a TOML-driven CLI.
- Docs follow the JAX-in-Cell and VMEX templates; the README follows the uwplasma template.

## 0. Where we are (review, 2026-10-21)

**Size.** The package is 17.0k lines in `src`, about 28.6k including tests, tools and examples. It has 28 modules, 29 docs pages, and 15 MB in `docs/_static`. Only about 4k lines are physics:

| Module | Lines |
|---|---|
| boozer | 1.8k |
| global_modes | 0.9k |
| metric | 0.66k |
| fast_ions | 0.45k |
| continuum and gaps | 0.15k |

The rest is scaffolding:

| Module | Lines |
|---|---|
| validation | 3.4k |
| workflows | 1.7k |
| optimization | 1.1k |
| cross_code | 1.0k |
| benchmarks | 0.7k |
| plotting | 0.6k |
| cli | 0.6k |
| far3d, maturity, kinetic_validation, benchmark_manifest, progress, reference_cases | 2.0k combined |

**Physics status.**
1. **Local 3-D Boozer continuum: correct** after PR #4 (STELLGAP weighting). CTOK p95 error is 1.17e-2 and the gap metric 3.17e-2.
2. **Global operator: not ideal MHD.** It solves `K_loc - d_s C d_s = w^2 M_loc`, which has no continuum resonance, so the frequencies depend on a free coefficient (4x spread). This is the main thing to rewrite.
3. **Fast-ion drive: a fitted Lorentzian.** It needs to become a perturbative energy-functional growth rate.
4. **Gradients.** The custom-JVP eigenvalue path works. The end-to-end VMEC gradient fails its radial-refinement gate (10.8% drift). Eigenvalues come from dense `eigh`, so cost grows as N^3.
5. **Performance.** The CTOK continuum takes 0.04 s once compiled (STELLGAP takes 0.8 s). Global block assembly is a Python loop, about 0.4 s. A global eigenvalue gradient with N = 1261 takes 1.9 s.

## 1. Physics model (what 1.0 solves)

Each item is listed with its source in `~/local/papers_fax`.

**(a) Continuum** (Spong, Sanchez and Weller 2003, eq. 5; Paul et al. 2025, eq. 2.3). On each surface solve

`w^2 A x = B x`, with
`A_ij = <mu0 rho |J| g^ss / B^2 e_i e_j>` and
`B_ij = <g^ss / (|J| B^2) (iota d_th + d_ze) e_i (iota d_th + d_ze) e_j>`.

The angle averages are FFT-based quadrature on Boozer grids. Use Paul's normalized form `w_bar = w / w_A`, with `w_A = <B^2> / ((G + iota I) sqrt(mu0 rho))`, for tests. Modes are chosen per family `n0 + k Nfp`, following Paul §5.1.

**(b) Global AEs** (AE3D: Spong, D'Azevedo and Todo 2010, eqs. 3–7). The weak form is

`-w^2 ∫ v_A^-2 grad(phi~) . grad(phi) + ∫ grad(B.grad phi~ / B) . grad(B.grad phi / B) + J_par kink term = 0`.

- Basis: radial linear or cubic-Hermite finite elements times Fourier harmonics, giving `F y = w^2 G y`. Both matrices are symmetric block-tridiagonal with H×H blocks.
- Metric needed: `g^ss`, `g^st`, `g^sz`, `g^tt`, `g^tz`, `g^zz`, B, J, G, I, iota, and their radial derivatives. All come from booz_xform_jax plus the VMEC embedding.
- Boundary conditions: `phi = 0` at the edge; regularity `phi_m ~ s^(m/2)` on the axis.
- The continuum emerges from the singular radial term. TAE, EAE, HAE and MAE gap modes appear as discrete eigenvalues.

**(c) Eigen-solver.** Shift-invert Lanczos or LOBPCG around a target `w^2`, done in JAX:

- A block-tridiagonal LU (Thomas algorithm with H×H blocks, `lax.scan`) costs O(N_s H^3) instead of O((N_s H)^3).
- Gradients use Hellmann–Feynman (`custom_vjp`): `dλ = x^T (dF - λ dG) x / x^T G x`. Nearly degenerate modes use a smooth cluster function (log-sum-exp or trace of the subspace), not a single eigenvalue.
- Modes are tracked between optimizer iterations by eigenvector overlap.

**(d) Fast-ion drive and damping** (Heidbrink 2008; Chen and Zonca 2016, eq. 3.22; Könies et al. 2018).

- Perturbative growth `gamma_h = -Im dW_h / (2 w dK)`.
- Resonances follow `w - (n + k Nfp) w_ze + (m + l) w_th = 0`.
- The distribution is a slowing-down or Maxwellian F(E, P_ze, mu). First model: zero orbit width, passing plus deeply trapped particles. Later: finite orbit width through bounce and transit averages, possibly using ESSOS orbits.
- Damping: continuum damping from the singular radial term, plus ion and electron Landau damping (Chen and Zonca eq. 3.4, FAR3d eq. 46).
- Total growth rate `gamma = gamma_h - gamma_d`.

**(e) Objectives** (Paul 2025, eqs. 7.1–7.4; Gaur 2023).

| Objective | Form |
|---|---|
| Continuum gap shaping | `f_cont` over epsilon harmonics, and gap-edge distance from the fast-ion resonance band |
| Mode avoidance | soft-max over eigenvalues in a frequency window |
| Stability | `softplus(LSE(gamma_k) - gamma_th)` |

Any objective can be combined with QS (vmex) and Γ_c (Nemov 2008; Bader 2019) inside one gradient-based optimizer.

## 2. Target layout (about 2.5k lines of src, 7 files)

```
src/fax/
  geometry.py    # Boozer surfaces (booz_xform_jax in memory or boozmn), metric, analytic cylinder/tokamak/near-axis fixtures
  continuum.py   # local continuum, gap finder, mode families
  eigen.py       # AE3D global operator, block-tridiagonal shift-invert solver, custom_vjp eigen gradients
  drive.py       # perturbative fast-ion drive, damping, resonance maps
  objectives.py  # gap/mode/gamma objectives + vmex boundary pipeline
  io.py          # TOML config, run(), outputs (npz/json), CLI entry
  plot.py        # continuum, mode structure, gap and optimization figures
tests/           # ~10 files, each gate tied to an equation or a published number
examples/        # ~6 TOML cases + 4 scripts (continuum, global AE, Könies TAE, optimization)
docs/            # VMEX/JAX-in-Cell template (Sphinx, MyST, MathJax)
```

**Delete or absorb.** These modules go: validation, benchmarks, benchmark_manifest, maturity, kinetic_validation, cross_code, far3d, progress, reference_cases, workflows, gap_models, runner, modes, metric (profile operators) and equilibrium.

- STELLGAP and AE3D readers become small helpers inside `tests/`.
- FAR3D ingestion is dropped. FAR3D stays an external comparison in `docs`.
- `external_artifacts` shrinks to the minimal test data under `tests/data`, roughly 1 MB or less.
- `docs/_static` keeps only the figures the docs and README use, compressed to 1.5 MB or less.

**API sketch.**
```python
import fax

geo = fax.Geometry.from_vmex(state)  # or .from_boozmn(path), .tokamak(...), .near_axis(...)
cont = fax.continuum(geo, n_family=1, mmax=12)  # frequencies, gaps
modes = fax.eigenmodes(geo, n_family=1, target_khz=80, k=8)
gamma = fax.growth_rate(modes, fax.FastIons(...))
fax.optimize(boundary, objectives=[fax.obj.gap(...), fax.obj.qs(...)])  # thin wrapper; jax.grad works on everything
```
The CLI is `fax run case.toml` plus `fax plot out/`, and nothing else.

## 3. Phases and gates

Each phase is one PR. Each must pass CI on JAX x64, ruff, mypy and docs builds before the next starts.

| Phase | Work | Gate |
|---|---|---|
| P0 | Merge PR #4. Fix the VMEC radial-refinement gradient drift (understand it first: is it Boozer truncation or `ns`?). | CI green |
| P1 Slim | Move to the 7-file layout. Keep only the physics that survives. Remove scaffolding and big artifacts. Rewrite tests as equation and number gates. | ≤ 3k lines of src; CTOK continuum p95 ≤ 1.2e-2; ≥ 90% coverage; full suite under 3 min on CPU |
| P2 Global operator | AE3D weak form, FE radial basis, axis regularity, block-tridiagonal shift-invert. | (i) cylinder: continuum accumulation, with no discrete mode outside the gap; (ii) Könies TAE at n = -6: w ≈ 4.0–4.1e5 rad/s (CAS3D 4.01e5); (iii) AE3D CTOK/LHD/QPS published frequencies (37 and 73.7 kHz LHD TAE, 149 kHz QPS HAE) within 3%; (iv) N_s refinement convergence |
| P3 Gradients | `custom_vjp` eigen gradients through vmex → booz_xform_jax → FAX. Cluster-smooth objectives. | gradients match finite differences to ≤ 1e-4 relative on LP-QA and HSX; one gradient costs ≤ 3 forward solves |
| P4 Drive | Perturbative `gamma_h` (zero orbit width first), continuum damping, Landau damping. | Könies 2018 ZLR band (about 8–10, 25–33 and 38–43 ×10³ s⁻¹ at 100, 200 and 300 keV, within ±20%); dependence on the sign of n and the v_A/3 resonance |
| P5 Optimization | Paul 2025 continuum objective in full 3-D. Gap placement and stability on precise QA/QH and HSX, combined with QS. | reproduce Paul's QH gap trend in full 3-D; show a lower `gamma_max` at fixed QS |
| P6 Performance | `vmap` over n-families and surfaces. GPU check (fp64). Optional sharding. | a CTOK-sized global solve in under 0.2 s compiled; one optimization step under 5 s on CPU at LP-QA resolution |
| P7 Docs, README, release | VMEX/JAX-in-Cell docs with full derivations: continuum → weak form → FE matrices → Hellmann–Feynman → drive. uwplasma README. PyPI trusted publishing. Make the repo public. | docs build with `-W`; README examples run in CI; `pip install` smoke test from a scratch directory |

## 4. Verification ladder

| # | Case | Source | Checks |
|---|---|---|---|
| G1 | Cylinder / uniform field | analytic | `w = k_par v_A`, exact |
| G2 | Paul near-axis gaps | Paul 2025, eqs. 4.5, 4.8 and 4.30; Table 1 | gap width `2 w0 |eps|` |
| G3 | Kolesnichenko Helias | Kolesnichenko 2001, eq. 38 | gap half-width |
| G4 | STELLGAP CTOK | public test problem | local continuum |
| G5 | AE3D | CTOK, LHD #47645, QPS, TJ-II | global frequencies |
| G6 | Könies 2018 ITPA TAE | NF 58 126027 | frequency and growth rate vs T_f |
| G7 | Mishchenko 2011 JET-like | PoP 18 012504 | second growth-rate case |
| G8 | Finite-difference gradient checks | LP-QA, HSX, W7-X | gradient accuracy |

## 5. Missing references to fetch

- Könies and Eremin, Phys. Plasmas 17, 012107 (2010): Alfvén–sound coupling, needed for BAE and slow-sound physics.
- Spong, Nucl. Fusion 53, 053008 (2013): the FAR3d closure coefficients a0 and a1, needed only if we compare against FAR3d.
- Fu and Van Dam (1989), and Betti and Freidberg (1992): analytic TAE drive, used for the P4 sanity check.
- Cheng, Chen and Chance (1985), or Cheng and Chance (1986): TAE gap theory and the tokamak continuum-damping formula.
- Unalmis et al. (2024–25), DESC differentiable Γ_c and bounce integrals: for the combined objective.
- StellDOS slides (Spong, APS 2023): the gap density-of-states objective.

## 6. Logbook

**2026-10-21, P0 (part).** PR #4 makes the local continuum use STELLGAP weighting. CTOK p95 went from 2.55e-2 to 1.17e-2.

**2026-10-21, P1 (slim).** Moved to the 7-file layout: 17.0k → 0.85k lines of src. Tests went from 28 files to 4 files plus a vmex file (15 tests, 60 s, 99% coverage). Docs went from 29 pages to 4. The remaining measures:

- STELLGAP CTOK data (MIT) are vendored in `tests/data` (0.6 MB), so the parity gate runs in CI.
- Dropped: validation/benchmark/maturity registries, FAR3D ingestion, AE3D matrix readers, the profile and metric continuum models, the gyro-Landau calibration fit, the dense radial `radial_coupling_fraction` path, `docs/_static` (15 MB) and `external_artifacts`.
- Python ≥ 3.10 kept (tomli fallback, oldest JAX that pip resolves on 3.10 is tested).

Open items carried forward:

- **P0:** done, see the next entry.
- **P2:**
  - The global solve is dense. The block-tridiagonal shift-invert solver comes with the AE3D operator.

**2026-10-21, P4 (drive).** `drive.py` replaces the Lorentzian screen with a perturbative power-transfer growth rate (zero orbit width, passing particles, large aspect ratio, exact resonance integral in speed at each pitch; Maxwellian and slowing-down; bulk-ion Landau damping by the same integral). Gates:

- Fu–Van Dam flat-Maxwellian scaling `x(1+2x^2+2x^4)e^{-x^2}`: rtol 1e-3.
- Könies 2018 TAE, model Gaussian eigenfunction (m = 10, 11, FWHM 10 cm at q = 1.75), ω = 4.0e5: 4.4, 16.6, 33.1, 65 ×10³ s⁻¹ at 100, 200, 300, 400 keV. 300 keV is within 20% of the ZLR band; 100 and 200 keV are about 40% low. The result changes by under 15% with mode width (3–8 cm) and relative harmonic amplitude. Candidate causes: trapped particles and orbit nonuniformity near the trapped–passing boundary, missing δB_∥, and the model eigenfunction; recheck with the P2 eigenfunction.
- Sign: flat profile gives the same damping for (m, n) and (−m, −n); a peaked profile drives n > 0 and damps n < 0; a hollow profile damps.
- `jax.grad` with respect to fast-ion density scale, T_f and ω matches central differences to 1e-5.

Open items: trapped and precessing particles, finite orbit width (Könies' FOW roll-over above 300 keV is absent, ZOW keeps growing), continuum and electron Landau damping (only a subtraction hook), stellarator helical drift sidebands (only the toroidal ±1 coupling plus orbit harmonics).

**2026-10-21, P0 (diagnosis).** LP-QA low-res, `harmonics [[1,0],[2,1]]`, `rc01`, ntheta = nzeta = 8, FD step 1e-4.

| surfaces | ns | mboz | s used | f (kHz) | df/dp (kHz per unit) | AD vs FD |
|---|---|---|---|---|---|---|
| 5 | 16 | 3 | 0.167 ... 0.833 | 122.2, 138.4, 158.1 | -212.6, -171.4, -130.4 | 5.8e-5 |
| 7 | 16 | 3 | 0.167 ... 0.833 | 122.1, 134.0, 150.2 | -213.8, -183.0, -142.4 | 5.4e-5 |
| 13 | 16 | 3 | two pairs repeat | NaN | NaN | NaN |
| 5 | 16 | 6 | 0.167 ... 0.833 | 122.6, 138.9, 158.7 | -215.8, -174.7, -134.0 | 5.1e-5 |
| 5 | 31 | 3 | 0.183 ... 0.817 | 124.0, 136.9, 154.3 | -208.2, -176.9, -137.0 | 4.4e-5 |
| 7 | 31 | 3 | 0.183 ... 0.817 | 123.5, 141.9, 157.1 | -209.3, -165.5, -133.7 | 4.4e-5 |
| 13 | 31 | 3 | 0.183 ... 0.817 | 123.2, 138.6, 156.6 | -210.9, -172.4, -133.8 | 3.7e-5 |
| 5 | 61 | 3 | 0.208 ... 0.792 | 125.8, 141.6, 151.8 | -204.5, -164.2, -144.0 | 3.3e-5 |
| 7 | 61 | 3 | 0.208 ... 0.792 | 125.3, 145.7, 164.2 | -205.3, -155.2, -122.2 | 3.7e-5 |
| 13 | 61 | 3 | 0.208 ... 0.792 | 125.2, 145.2, 162.9 | -205.9, -156.8, -123.0 | 4.0e-5 |

- The gradient is right: AD matches centered FD to ≤ 6e-5 at every level. The drift is in the objective.
- Main cause (a): the interim operator. Its Dirichlet ends sit on the first and last chosen surfaces, so the radial domain is set by the surface list, and the FD term is first-order in the spacing. Going from 5 to 7 surfaces moves f3 by 8% and its gradient by 15% (ns = 61). From 7 to 13, with the same ends, f moves ≤ 0.8% and df/dp ≤ 1.6%.
- (d) Snapping: `prepare_design` maps each surface to the nearest half-mesh row, so at ns = 16 the ends 0.2 and 0.8 become 0.167 and 0.833. Changing ns moves the domain (f3 at 7 surfaces: 150 → 164 kHz from ns 16 to 61). Snapping two surfaces onto one row gave zero spacing and NaN. It now raises `ValueError`.
- (b) mboz/nboz 3 → 6: ≤ 0.4% in f and 1.5% in the gradient. (c) ns: confounded with snapping. The old gate also ran vmec_jax with `vmec_max_iter=4`, a truncated equilibrium.
- P2 restatement: fix the radial domain (axis to edge, or fixed physical ends), interpolate rather than snap, and refine the FE radial grid at fixed vmex ns and mboz. Gate on the eigenvalue and gradient converging at the expected FE order, with AD-vs-FD ≤ 1e-4 at each level.

LP-QA `NS_ARRAY` ladder: stages 1 and 2 converge (189 and 704 iterations). Stage 3 (ns = 50) converges linearly and reaches ftol 1e-13 at iteration 1115, so 1e-13 is reachable in float64 and NITER = 1000 was just short. VMEC2000 would write the wout anyway. Fix: NITER_ARRAY = 600 1000 2000. Not a vmex bug.

Boozer radial label: booz_xform writes `jlist = compute_surfs + 2` for half-mesh rows (booz_xform_jax `io_utils.py`). Its in-memory `s_in` is `s_half`, so the right label is `s = (j - 1.5)/(ns - 1)`. `read_boozmn` now uses it, which matches `from_wout` and `prepare_design`. The old `(j - 1)/(ns - 1)` label happened to match STELLGAP because STELLGAP is off by one cell too. xmetric averages half rows k and k+1 onto full surface k at `(k - 1)/(ns - 1)`, but stellgap labels it `rho = nsurf/nsurf_max = k/(ns - 1)` (`stellgap_serial_ver5.f`, `rho(ir) = nsurf(ir)/nsurf(irads)`). The parity test now shifts the STELLGAP labels by `-1/(ns - 1)` and interpolates them onto FAX's s. A scan of the label offset has its minimum there. CTOK parity is now mean 6.3e-3 and p95 6.6e-3 (was 4.5e-3 and 1.13e-2), with a 7e-3 gate.
