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

**2026-10-21, P2 (global operator).** The interim envelope (and its `scale`, `boundary` options) is gone. `eigen.py` now assembles the AE3D weak form with linear radial elements, element-averaged coefficient blocks and the full contravariant metric (added to `surface_grid` with `dB` and `sqrt_g`); `F y = w^2 G y` is block tridiagonal. The pressure and J_par (kink) terms are omitted: on CTOK, rerunning AE3D without them moves its global modes by < 0.1%. Solver: block Thomas LU in `lax.scan` plus shift-invert subspace iteration (2k + 8 vectors, 40 iterations) with Rayleigh-Ritz; the dense solver stays for small problems. `targeted_eigenvalues` carries the Hellmann-Feynman JVP with the shift-invert eigenvectors. Src 875 → 1003 lines (eigen 114 → 222, geometry +21).

| Gate | Result |
|---|---|
| (i) uniform field, single harmonic | spectrum inside `[w_A^2(0), w_A^2(1)]`, edges within `0.5/N`; `phi'` participation fixed in cells, so width ∝ 1/N (N = 40, 160) |
| (ii) AE3D CTOK (same boozmn, full mesh + AE3D's edge extrapolation, n = 2, m = 0–12) | matrix leading blocks and the operator continuum within 1%; 6 global modes matched by eigenvector overlap (> 0.9) within 1.4% (343.6, 384.1, 685.6, 703.3, 738.2, 741.2 kHz); AE3D's TAE at 337.9 kHz is not reproduced (FAX 272.8 kHz); real-space quadrature sides with FAX (below) |
| (iii) Könies 2018 ITPA TAE, n = 6, `circular_torus(epsilon_B = a/R0)` | w = 4.0610e5 rad/s (CAS3D 4.01e5 +1.3%, GYGLES ad hoc 4.13e5 −1.7%) |
| (iv) N_s convergence (Könies) | 4.0598 / 4.0607 / 4.0610 / 4.0610 e5 at N = 50 / 100 / 200 / 400 |
| (v) HF gradients vs FD | 1e-5 (dense cluster and shift-invert) |

Timings (CTOK, 99 × 13, one CPU core, compiled): assembly 0.018 s, shift-invert for 6 modes 0.067 s, dense 1.3 s. Full suite 56 s, coverage 99%.

Other changes: `read_boozmn` now labels the half mesh correctly, `s = (j - 1.5)/(ns - 1)`; `Geometry.full_mesh()` averages to the full mesh as STELLGAP/AE3D do. STELLGAP labels half-mesh data with full-mesh `s`, so its parity test shifts by `0.5/(ns - 1)` (unchanged numbers). `circular_torus` now uses the Boozer poloidal angle (`m = 2` terms in R, Z, so `sqrt(g) ~ 1/B^2` to first order) and defaults to interior surfaces.

Open items carried forward:

- **CTOK TAE: FAX is right, AE3D's `iota'` terms are not.** The bending integrand follows from `u = B.grad(phi)/B = i (iota m - n) B/(G + iota I) phi` (Boozer `sqrt(g) = (G + iota I)/B^2`), so `d_s u` carries `i iota' m B/(G + iota I)`; this term is physical. `test_ctok_bending_energy_matches_real_space_quadrature` checks `F` and `G` against finite differences of `phi` on the boozmn embedding, with `B^i` built from `|B|`, `iota` and `g_ij` only: both agree to 0.3%; without the `iota'` term `F` is off by 9–20%. A finer quadrature (1201 × 64 × 16) gives the AE3D TAE eigenvector 329.7 kHz (FAX 328.5–329.3, FAX without `iota'` 342.4, AE3D's own eigenvalue 337.9). AE3D's `a_matrix` matches FAX within 0.3% on same-surface blocks (Frobenius 1.1% overall), but its m = 2/3 couplings between surfaces 45 and 46 (s ≈ 0.46) are 1.071 and 0.820 against FAX's 0.962 and 0.905: the `iota'` part with the wrong sign (FAX with `-iota'` gives 1.066 and 0.806). Neither a global `-iota'` (404.6 kHz) nor swapping the `f' f` blocks (332.1 kHz) reproduces AE3D's eigenvalue, so the AE3D defect is confined to the inter-surface `iota'` cross terms. Continuum: the n = 2 TAE gap at s ≈ 0.45 spans 232–396 kHz (cylinder estimate `v_A/(2 q R)` = 323 kHz with `R = R00`, 334 kHz with `R = G/B`). Both modes peak at s ≈ 0.45 inside it; 272.8 kHz meets the continuum only at s = 0.85–0.94 (1.8% of the FAX mode's norm), 337.9 kHz at s = 0.69–0.82 (14% of AE3D's). Könies is insensitive to `iota'` (4.061e5 vs 4.067e5 rad/s without it), so only CTOK-like shear separates the two forms.
- **vmex gradient gate.** `test_boundary_gradient_matches_finite_differences` is now strict-xfail: FAX's own gradients are exact (≤ 7e-8 vs FD for every geometry leaf), but the vmex/booz_xform `iota` derivative wrt `zs01` is off by 1.1e-4 (FD converged, also at ftol 1e-14), which the resonant AE3D operator amplifies to 2.3e-4. Belongs to P0/P3.
- `circular_torus` has no Shafranov shift; it reproduces the ad hoc equilibrium, not CAS3D's.
- LHD (37, 73.7 kHz) and QPS (149 kHz) published AE3D frequencies need their equilibria.

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

**2026-10-22, P6 (performance).** `examples/benchmark.py [--design]` prints compile+first and steady (mean of 3) times. Changes (numerics unchanged, all tests pass as they were):

- Continuum: `<f exp(i(dm theta - dn zeta))>` is now one `fft2` per surface gathered at `(-dm, dn)` instead of the `(S, H, H, T, Z)` phase tensor. Identical DFT, memory `O(S T Z)`. Steady 0.07 → 0.03 s (CTOK), 0.37 → 0.05 s (N = 200, H = 21), 1.33 → 0.12 s (N = 400, H = 31); compile 1.8 → 0.4 s at N = 400.
- Shift-invert: block Thomas factors (`block_factor`) computed once instead of in each of the 40 iterations; `block_solve(factors, b)`. Gain 5–25% only: per iteration the triangular solves, the `G` matvec and the `(N H, p)` QR cost about the same (10 ms each at N = 400, H = 31); unrolling the scans gave nothing.
- Not done: FFT for the global assembly (its `(S, H, T, Z, 3)` basis is linear in H, the einsum is not the bottleneck); vmap over n-families (one family per solve in all callers).

Laptop CPU (M-series, 10 cores, loaded), seconds, compile+first / steady:

| case | N | H | continuum | assembly | shift-invert (8) | grad frequency_hz | dense |
|---|---|---|---|---|---|---|---|
| CTOK | 98 | 13 | 0.69 / 0.03 | 0.68 / 0.036 | 1.17 / 0.13 | 1.87 / 0.17 | 1.61 / 1.20 |
| torus | 200 | 21 | 0.36 / 0.06 | 0.61 / 0.09 | 1.18 / 0.42 | 2.09 / 0.53 | – |
| torus | 400 | 31 | 0.39 / 0.12 | 0.71 / 0.18 | 2.04 / 1.25 | 3.83 / 1.61 | – |
| LP-QA design `value_and_grad` (ns 16, 4 surfaces, 2 params) | | | | | | 10.96 / 1.46 | |

Office RTX A4000 (fp64, JAX 0.6.2 cuda12) vs the office CPU (36 threads, JAX_PLATFORMS=cpu):

| case | path | GPU | office CPU |
|---|---|---|---|
| CTOK | continuum | 1.04 / 0.010 | 0.81 / 0.14 |
| CTOK | assembly | 1.74 / 0.014 | 0.82 / 0.14 |
| CTOK | shift-invert | 3.09 / 0.29 | 1.44 / 0.33 |
| CTOK | grad | 8.28 / 0.32 | 3.56 / 0.51 |
| CTOK | dense | 1.92 / 0.22 | 8.03 / 5.54 |
| N 200 | shift-invert / grad | 4.64 / 0.86, 11.5 / 0.98 | 2.28 / 1.14, 4.42 / 1.49 |
| N 400 | shift-invert / grad | 5.49 / 1.75, 14.4 / 2.01 | 3.74 / 2.73, 6.04 / 3.43 |

- The GPU wins steady assembly/continuum by 10x and dense by 25x, but shift-invert is a sequential `lax.scan` of small `H x H` blocks and gains only 1.1–1.6x; compile is 2–3x longer on GPU. For one-off solves and optimization loops of < ~50 steps the CPU wins. The design path was not run on office (its Python 3.10 cannot install vmex).

Gates: CTOK-sized global solve (assembly + 8-mode shift-invert) 0.17 s steady on the laptop: **met** (under load; 0.11–0.13 s quiet). One LP-QA optimization step: 1.5 s steady (**met**), but the first step compiles for 11 s (23 s before the shift-invert change cut the traced graph); the 5 s gate holds only after compilation.

**2026-10-22, P3 (gradients).** LP-QA low-res, ns = 16, ftol 1e-12, mboz = nboz = 3.

The 1.1e-4 `iota` gap is not an upstream gradient bug. VMEC's released m = 1 `Z_sin` pair combination is an angle gauge. The adjoint holds it at its converged value; each re-solve leaves it wherever its own path ended, so re-solve FD adds the response to that drift. Evidence (`iota` on half row 4, wrt `zs01` unless noted):

| check | result |
|---|---|
| adjoint vs re-solve FD, h = 1e-3, 1e-4, 1e-5 | -0.1336287 vs -0.1335700 at every h: gap 4.4e-4, reproducible, not FD noise |
| `rc01`, h = 1e-3 vs 1e-4 | gap 1.0e-4 vs 1.5e-4: the gauge drift itself changes with h (1.9e-4 vs 2.5e-4), a solver-path effect |
| constrained-family drift of the re-solves | 2.1e-3 (`zs01`), 1.3e-3 (`rc01`); nothing else frozen moves |
| adjoint + response to that drift (Newton on the frozen residual, as vmex's `test_li383_mean_iota_resolve_fd_gap_is_the_m1_constrained_family`) | closes to 3.3e-9 (`zs01`), 5.8e-9 (`rc01`) |
| global frequency, adjoint vs re-solve FD | 3.0e-5 (`zs01`, `rc01`), 2.6e-5 (`rc10`), 2.7e-3 (`zs11`, an m = 1 boundary mode) |

`frozen_path_directional_fd` is the vmex reference, but it took over 10 min here, so CI compares against re-solve FD on non-m = 1 coefficients with rtol 1e-4 (the gauge term is 3e-5).

**Upstream bug found on the way.** `boozer_input_tables(state, rt, 1)` returns `rmnc`/`zmns` of order 1e13–1e14. The odd-m parity interpolation divides axis round-off by `max(sqrt(0), 1e-30)`. Fix (divide by 1, as booz_xform does) plus regression test: uwplasma/vmex#631. FAX uses half rows j >= 2 until it is released.

**Radial domain.** `prepare_design` no longer snaps surfaces. It transforms every half row j >= 2 and `Geometry.interpolate` maps them, linearly in s and extrapolated to 0 and 1, onto `nodes` uniform FE nodes `k/(nodes + 1)`. So the domain is always [0, 1] and only the FE grid refines. Gate (`test_boundary_gradient_and_radial_refinement`, lowest n = 0 frequency, m = 1–3, 8 × 8 angles; FAX AD vs FD along the re-solved geometry tangent):

| nodes | f (kHz) | df/dzs01 (kHz) | AD vs FD |
|---|---|---|---|
| 16 | 294.648 | -62.528 | 3.4e-8 |
| 32 | 294.219 | -61.230 | 3.5e-8 |
| 64 | 293.989 | -60.583 | 3.6e-8 |
| 128 | 293.873 | -60.261 | 3.6e-8 |

Differences halve, so this is first order, as expected for the continuum minimum. Single harmonics converge the same way, and (2,1)+(3,1) reaches AD vs FD of 1e-10. Before the row fix, any FE grid that sampled row 1 gave AD vs FD of order 1. With the sparse low-lying (2,1) spectrum, eigenvalues inside a dense continuum cross under an h = 1e-4 step, so gates use the lowest eigenvalue or a target away from crossings.

**API for P5.** `design_params` (offsets → vmex params), `state_geometry` (state → half-mesh Boozer geometry; usable as a vmex `metric_fn`), `design_geometry`, and `geometry_frequencies`: FE nodes → dense lowest frequencies, or shift-invert `targeted_eigenvalues` near `target_hz`, both with a Hellmann–Feynman JVP. `design_frequencies` composes them and feeds `window_risk` / `soft_max`. One `jax.grad` is one forward solve plus one adjoint solve. `test_vmex.py` takes about 55 s.

Open: HSX gate; m = 1 boundary coefficients need frozen-path FD (or a gauge-fixed vmex) for a 1e-4 check; drop the j >= 2 workaround after vmex#631 is released.

**2026-10-22, P5 (optimization).** `compression_harmonics` (FFT of `eps = |grad s|^2/<|grad s|^2> - 1` from `surface_grid`) and `continuum_cost` (Paul 2025 eq. 7.4, helicity cut `|iota M - N N_P|`) in `objectives.py`; `design_state` returns the vmex `(state, runtime)` so vmex QS (`QuasisymmetryRatioResidual.total_state`), `implicit.aspect_ratio` and FAX terms share one implicit solve. `examples/optimize_alfven.py`: `scipy.optimize.least_squares`, 4 rows each scaled to ~1 at the start, Jacobian = one forward + one `vjp` per row (independent of the number of coefficients). ns 16, mboz = nboz = 6, office CPU.

| case | coefficients / evaluations / time | Alfvén term | QS total | aspect, mean iota |
|---|---|---|---|---|
| LP-QH, eq. 7.1 (helicity (1, -1)) | 34 / 60 / 31 min | f_cont 7.038 -> 5.226 (-26%) | 1.32e-4 -> 5.72e-4 | 8.000 -> 8.004, -1.2431 -> -1.2393 |
| LP-QA, 3 lowest n = 0 modes out of [293.1, 314.9] kHz | 4 / 30 / 18 min | 294.10-294.92 -> 290.65-291.47 kHz | 1.09e-6 -> 3.96e-6 | 6.000 -> 5.988, 0.4191 -> 0.4159 |

QH gap widths above the cut (Paul eqs. 4.5, 4.16, w/w_A): MAE (0,4) 0.116 -> 0.098, HAE (2,-8) 0.087 -> 0.068, MAE (0,8) 0.069 -> 0.049, but the dominant HAE (3,-8) 0.140 -> 0.139. Traps: the vmex equilibrium is `custom_vjp`, so reverse mode only (jacrev or per-row vjp); a 2% boundary step destroys precise QS (use 0.2% of the minor radius as x_scale); a window centred on the modes has zero gradient (sigmoid plateau), so put the modes near an edge; LP-QH Boozer |B| ~ cos(theta + 4 zeta) (iota < 0), so its helicity is (1, -1). `f_cont` is resolution-converged at 32 x 64 angles (1e-6 vs 48 x 128).

Open: HSX (mpol 18 / ntor 24 decks, not run); lower gamma_max at fixed QS (needs a stellarator drive); resonance-band term; longer/finer QH runs (cost still falling at 60 evaluations).

**2026-10-22, P2b (finite-beta terms).** `global_matrices(..., finite_beta=False)` adds, when on, the Hermitian parts of the ideal-MHD `J_par` and pressure-curvature energies in the AE3D units: `K = int sigma b.(grad phi~* x grad u)`, `P = -2 mu0 int p' (b x grad phi.grad s)(b x grad phi~*.kappa)/B^2`, `sigma = mu0 J.B/B^2`, `kappa = grad_perp(mu0 p + B^2/2)/B^2`. `K` integrated by parts is AE3D eq. 4's third term; the vorticity-equation and `delta W` derivations agree on its sign, and the Newcomb gate fixes it. FAX does not use AE3D's matrix elements. `equilibrium_currents` gets `mu0 p'` from the solvability of the Boozer radial force balance `(iota d_theta + d_zeta) B_s = G' + iota I' + mu0 p' (G + iota I)/B^2` and `B_s` (Pfirsch-Schlueter) by FFT, so `Geometry` is unchanged. On the vmex ITPA equilibrium the inferred `p'` is within 0.4–1.8% of the input. `circular_torus(current=True | pressure=[...])` adds `I = iota s a^2 B0/R0` and the force-balance `G`. `_coefficients` now takes a left and a right vector set and returns Hermitian parts. Src +64 lines (eigen +51, geometry +13).

| Gate | Result |
|---|---|
| vacuum (`I = 0`, `G` const) | matrices bit-identical with and without the flag |
| real-space quadrature, `sigma` from curl of `B_i` on the grid | CTOK: `sigma` to 1e-4 and the kink form to 0.3% (`B_i`, `kappa = (b.grad) b` from the embedding alone); Könies with `p`: 0.16% at N = 128 (0.65% at 64, so O(h^2)) |
| cylinder m = n = 1 rigid shift, q = 0.8 + s | bending + kink = 1.2e-3 of bending (Newcomb `m^2 - 1`); the flipped sign doubles it |
| cylinder Suydam, m/n = 11/6, `p = p0 (1 - s)` | analytic threshold 1.05e4 Pa; 1.0e4 marginal (`1/N^2`), 2.0e4 converged unstable (-7.3e8) |
| HF gradient vs FD with `finite_beta` and the Könies `p` | 1e-7 at step 1e-7. The step must be ≤ 1e-6, since at 1e-4 an eigenvalue crossing gives nonsense. The test step is now 1e-7. |
| Könies ITPA TAE (rad/s), vmex `examples/data/input.ITPA_Konies2018` | `p = 0`: 4.0958 → 4.0978e5 (`J_par`, +0.05%); with `p(s)`: 4.1271 → 3.9482e5 (−4.3%, overlap 0.98, N = 100–400 converged); CAS3D 4.01e5, GYGLES 4.13e5 |
| Könies, `circular_torus` | `J_par` alone +0.07% (4.0610 → 4.0637e5); with `p(s)` Mercier-unstable at q = 11/6 (-3.1e9, converged): no Shafranov shift, so no `(1 - q^2)` stabilization; vmex negatives fall as `1/N^2` |

Findings and open items:

- CAS3D (compressible) lies between FAX's frequencies with and without the finite-beta terms. The 1.5% low bias is plausibly the dropped sound coupling `gamma p |div xi|^2`.
- **CTOK with `finite_beta=True` is not AE3D-like.** `q0 ≈ 0.996`, `sigma ≈ -0.7 /m`. A converged near-axis m = n = 2 instability appears (-2.4e10), and the AE3D-matched modes move 0.1–10% (388 → 428 kHz, overlaps 0.72–0.87). The P2 statement that AE3D's modes move < 0.1% with its terms on does not transfer. Either AE3D's symmetrized `jprl` terms differ from the Freidberg form, or they are as weak as reported. Next: rebuild AE3D (`ae_solve_version31.f`, test-problem `plasma.dat` carries `jprl_coef0..2` and `brho`) with and without `jprl`, and compare the `jprl` blocks with FAX's `K` on CTOK. That was not done here. `test_ae3d_ctok_parity` stays at the default.
- No sound or fast-wave coupling, so no BAE gap.

**2026-10-22, P4b (drive in Boozer geometry).** `drive.py` now evaluates `gamma = -P/2W` on any `fax.Geometry`; the circular-only path and `Plasma` are gone (one API: `growth_rate(omega, phi, geo, harmonics, ions, mass_density, *, sidebands, helical, ...)`). Drift `(B x grad B)^i = eps^ijk B_j d_k B / sqrt(g)` with `B_theta = I`, `B_zeta = G` (radial `I d_zeta B - G d_theta B`, binormal `G d_s B`, `-I d_s B`); passing orbits along `theta - iota zeta = const` with `dt/dzeta = (G + iota I)/(v |xi| B)`; uniform angles from the magnetic differential equation `(iota d_theta + d_zeta) chi = h/<h> - 1` by FFT; resonances `omega = |j iota - k| omega_zeta`, `j = m + l`, `k = n + k' Nfp`; per-harmonic `Q_h F = F_E + n_h F_s/(omega Z e iota psi')` (E_par = 0 turns the radial ExB + v_par dB_r into `-d_zeta phi`); `W` from the eigen inertia `G` (`2 pi^2 y.H G y / mu0`). `resolved=True` gives gamma per (j, k). `objectives.py`: `drive_risk = softplus(soft_max(gamma/scale) - th/scale)`, `geometry_growth_rates` (shift-invert modes near `target_hz`, eigenvectors differentiated through the iteration). Trapped/precessing particles: not done (documented).

| Gate | Result |
|---|---|
| (a) Könies TAE, `circular_torus(epsilon_B = 0.1)`, 100/200/300/400 keV | 4.37 / 16.64 / 33.18 / 65.3 ×10³ s⁻¹ vs the old circular model 4.4 / 16.6 / 33.1 / 65 (≤ 4%); Fu–Van Dam 1e-3; signs unchanged (`n > 0` drives for `iota psi' > 0`) |
| geometry + mode gradients (every leaf, random direction) | 1e-5 vs FD |
| (b) CTOK, FAX TAE 272.8 kHz (n = 2) from shift-invert, D `1e17 (1-s)^2` | gamma/omega = 9.2e-4, 3.6e-3, 1.6e-2 at 25, 50, 100 keV; bulk-ion Landau (1 keV) 2.6 s⁻¹ |
| (c) LP-QA, n = 1 family near 104 kHz, 150 keV D | k ≠ 1 resonances 1e-6–1e-2 of gamma (QA: B nearly axisymmetric); pure sidebands k = -3, 5 nonzero; FAX chain (interp + eigenvectors + drive) AD vs FD 5e-8 (`rc01`); full vmex-adjoint grad vs re-solve FD 1.6e-3 (the P3 m = 1 gauge, amplified; 1e-1 for `rc11`) |
| (c) LP-QH, n = 1 family, slowing-down alphas | 43.9 kHz gap mode: 99% of gamma from k = -7 (absent from the mode); 108.6 kHz: 16% from k = 9 |
| (d) LP-QA `--drive`, 4 coefficients, 16 evals, 13 min laptop | gamma_max 8500 → 8123 s⁻¹ (−4.4%), QS 1.09e-6 → 1.63e-6, aspect 6.000 → 6.005, iota 0.4191 → 0.4204 |

Traps: the grid `max |B|` for the passing boundary has derivative jumps (argmax switches) and made FD noisy at 1e-2; a log-sum-exp bound fixed it. Re-solve FD at h = 1e-4 is off by 4e-4 from curvature (gamma is ~100x more sensitive than f), so gates use h = 1e-5. Without the drive row weighted 10x, least_squares retreats to p = 0: QS has zero slope but huge curvature in `rc11`/`zs11`. `test_vmex.py` drive test takes ~4 min.

Open: trapped/precessing particles (bounce averages), larger gamma reductions (more/other coefficients, n = 0 boundary modes), FOW, `B_s` in the drift.

**2026-10-23, integration of uwplasma/fax#9 (physical-potential vorticity, E. Lascas Neto).** #9 added a second assembly of the same weak form (`vorticity.py`, 305 lines; 1,108 lines with tests, scripts, 0.5 MB of figures and data) on the pre-P1 code.
- Same physics as `eigen.global_matrices`: on CTOK (N = 99, H = 13, 32² angles) every AE3D-matched mode agrees within 5.6e-4 and both give 280 modes in 50-800 kHz. FAX assembly 0.145 s vs 0.352 s steady; dense solve the same.
- Ported: its exact sheared 3-D manufactured solution as one test (FAX nodal error 2.1e-2, 9.2e-3, 3.1e-3, 8.9e-4 at N = 8-64, ratio → 4), and its regular axis extension `(s/s0)^(m/2)` for `m > 0` harmonics in `Geometry.interpolate` (exact for `sqrt(s)` harmonics).
- Recorded, not kept as files: its Knyazev et al. QA comparison (archived AE3D 30.31 kHz; public AE3D formulas 30.30; physical indices 27.57), an independent confirmation of the AE3D shear-index error found on CTOK.
- Not ported: the duplicate operator, its cosine-basis option (no caller; a real basis for stellarator-symmetric cases is a possible later speed-up), Gauss-point geometry sampling (FAX's element averaging already converges at second order), README additions, PNGs (313 kB, 126 kB) and the CSV/NPZ data.
