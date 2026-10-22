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

- **P0:** the radial-refinement drift of the VMEC gradient (old gate, 10.8%) is not yet diagnosed. The LP-QA low-resolution input also does not converge on vmex's own `NS_ARRAY` ladder (ns = 50, ftol 1e-13, 1000 iterations); examples use one stage at ns = 16.
- **P2:**
  - `read_boozmn` labels surfaces `s = (j - 1)/(ns - 1)`; VMEC's half grid is `(j - 1.5)/(ns - 1)`. STELLGAP agreement is better with the current label (p95 1.1e-2 vs 2.4e-2) because STELLGAP interpolates to the full grid. Settle this together with the AE3D operator.
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
