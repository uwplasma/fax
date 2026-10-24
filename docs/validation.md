# Validation

Every physics claim in FAX is a test in `tests/`, tied to an analytic result
or a published number. CI runs them on Python 3.10–3.13 in float64. The numbers
below are the test thresholds and the measured values from the development
logbook (`plan.md`, section 6).

## Summary

| Gate | Test | Result |
|---|---|---|
| Uniform-field continuum, $\omega = \lvert n - m\iota\rvert v_A/R$ | `test_uniform_field_continuum_is_k_parallel_v_alfven` | exact to 1e-9 |
| Angle quadrature converged, $K$, $M$ Hermitian | `test_matrices_are_hermitian_and_resolution_converged` | 32² vs 64² to 1e-9 |
| STELLGAP CTOK continuum | `test_stellgap_ctok_parity` | mean 6.3e-3, p95 6.6e-3 (gate 7e-3) |
| Gap formulas | `test_gaps_and_gap_location` | TAE $\iota = 2n/(2m+1)$, MAE $\iota = (2n - N_{fp})/2m$ |
| Uniform-field global spectrum is a continuum | `test_uniform_field_spectrum_is_a_continuum` | edges within $0.5/N$; mode width ∝ $1/N$ (N = 40, 160) |
| AE3D CTOK global modes | `test_ae3d_ctok_parity` | 6 modes within 1.4%, overlap > 0.9 |
| AE3D CTOK TAE | same | not reproduced; AE3D sign error, see below |
| Independent 3-D spectrum, precise QA of Knyazev et al. 2026 (STELLGAP continuum, AE3D gap band) | `test_stellgap_continuum_parity`, `test_ae3d_gap_band_in_the_sine_basis` | continuum 3.0e-3 mean (gate 4e-3); archived global mode within 2% in energy at 24-64 harmonics, 5.1% at AE3D's 162; no single-mode match (overlap 0.18) |
| Bending and inertia vs real-space quadrature | `test_ctok_bending_energy_matches_real_space_quadrature` | 0.3% (gate 0.5%) |
| Finite-$\beta$ terms vs real-space quadrature ($\sigma$ from curl $\mathbf B$; CTOK and Könies with $p$) | `test_finite_beta_terms_match_real_space_quadrature` | 0.3% CTOK, 0.2% pressure (gate 0.5%) |
| Cylinder: Newcomb $m = 1$ cancellation; Suydam threshold | `test_cylinder_newcomb_m1_and_suydam_threshold` | bending + kink = 1e-3 of bending; stable at 1.0e4 Pa, unstable at 2.0e4 (Suydam 1.05e4) |
| Könies ITPA with $p(s)$, vmex equilibrium | `test_konies_itpa_finite_beta_tae` | 4.127e5 → 3.947e5 rad/s with finite-$\beta$ terms (CAS3D 4.01e5) |
| Könies 2018 ITPA TAE frequency | `test_konies_tae_frequency_and_convergence` | 4.0610e5 rad/s, $N_s$-converged to 1e-4 |
| Shift-invert (SOLVAX) vs dense | `test_shift_invert_matches_dense_and_gradients_match_finite_differences` | 1e-10 |
| Eigenvalue gradients vs finite differences | same | 1e-5 (cluster mean and shift-invert) |
| Fu–Van Dam Maxwellian scaling and closed-form ZOW coefficient (weak toroidicity), flat and with a density gradient (the gradient term carries $j$ = 11, 10) | `test_fu_van_dam_maxwellian_scaling` | 1e-3 |
| Slowing-down $\int F\,d^3v = 1$ with the smoothed edge; duplicated toroidal samples | `test_slowing_down_is_normalized_with_its_smoothed_edge`, `test_duplicated_toroidal_samples_leave_the_drive_unchanged` | 1e-6; 1e-10 |
| Elliptic two-mode gap, full width $2f_0\lvert c\rvert$ | `test_elliptic_two_mode_gap_has_full_width_two_f0_c` | 1e-6 |
| Drive vs Könies 2018 (Boozer form of the circular model) | `test_koenies_2018_tae_growth_rates` | 4.26, 16.27, 32.48, 64.02 ×10³ s⁻¹ (pinned 1e-3); old circular model within 4% |
| 3-D structure: helical relabelling of an axisymmetric torus (with and without $I$), axisymmetric limit of a 3-period description | `test_quasi_helical_relabelling_leaves_the_drive_unchanged`, `test_axisymmetric_limit_ignores_toroidal_sidebands` | 1e-4 (4e-5 measured, FFT wrap); 1e-9; the previous $n_h/\iota$ form is 17–20% off |
| Negative $\iota$ ($\theta$-mirror), $n$-reversal ($\zeta$-mirror, co ↔ counter), amplitude and global phase, density scaling, passivity of a uniform distribution | `test_negative_iota_and_n_reversal_describe_the_same_device`, `test_amplitude_phase_density_and_passivity` | 1e-12; $\gamma < 0$ at every $\omega$ |
| Regularizations: `boundary`, `nodes`, `FastIons.birth_width`, `guard` | `test_passing_boundary_and_node_regularizations_converge`, `test_birth_edge_smoothing_converges_quadratically`, `test_small_divisor_guard_converges_and_is_sensitive_near_rational_surfaces` | see the table below |
| ITPA rung 2: passing-only drive vs the CAS3D-K (ZOW) curve | `test_zow_curve_is_bracketed_by_the_passing_only_drive` | ratio 0.58–0.60, 82–191 keV; unmatched content, not a validation |
| Drive sign rules | `test_drive_sign_and_flat_profile` | flat, peaked, hollow, $\pm n$ |
| Drive gradients vs finite differences | `test_gradients_match_finite_differences`, `test_geometry_and_mode_gradients` | 1e-5 (density, $T_f$, $\omega$, every geometry leaf and the mode) |
| CTOK global TAE driven | `test_ctok_global_tae_drive` | 272.8 kHz mode, $\gamma/\omega$ = 7.0e-4, 3.3e-3 at 25, 50 keV (9.2e-4, 3.6e-3 with the $n/\iota$ form) |
| LP-QA drive: helical sidebands, gradient through vmex geometry | `test_stellarator_drive_gradient_and_sidebands` | $k \ne n$ resonances present; FAX chain AD vs FD ≤ 1e-4 |
| vmex → booz_xform_jax → FAX continuum | `test_vmec_input_to_continuum` | runs, finite |
| Compression harmonics of an elliptic torus, $\epsilon_{\pm2,0} = (\kappa^2-1)/2(\kappa^2+1)$ | `test_continuum_cost_of_an_elliptic_torus` | exact to 1e-12; $f_\mathrm{cont}$ analytic; AD vs FD 1e-7 |
| $f_\mathrm{cont}$ descent step through vmex | `test_continuum_cost_gradient_step_and_vmex_terms` | decreases; vmex aspect on the same state |
| Boundary gradient and FE radial refinement | `test_boundary_gradient_and_radial_refinement` | AD vs FD 1e-4 at 16, 32, 64 nodes; first-order convergence |

## Continuum

**Uniform field.** {func}`fax.circular_torus` with $\epsilon_B = 0$ and
$\iota$ from 0.3 to 0.6, harmonics $n = 1, 2$, $m = 0..4$: every branch equals
$|n - m\iota|\,v_A/R_0$ to 1e-9.

**STELLGAP CTOK.** The public STELLGAP test problem (circular tokamak,
`boozmn_ctok.nc` with $n_s = 101$, $n = 2$, $m = 0..12$, deuterium at
10¹⁹ m⁻³), the STELLGAP output vendored in `tests/data`. Relative error of the
13 branches per surface on 97 surfaces: mean 6.3e-3, 95th percentile 6.6e-3.
STELLGAP labels its full-mesh surface $k$ as $k/(n_s - 1)$ instead of
$(k - 1)/(n_s - 1)$ (`rho = nsurf/nsurf_max` in `stellgap_serial_ver5.f`), so
the test shifts its labels by one cell; a scan of the offset has its minimum
there. The compiled CTOK continuum takes 0.03 s on a laptop CPU (STELLGAP 0.8 s);
`python examples/benchmark.py` prints the timings of every hot path.

(performance)=
### Performance note: SOLVAX eigensolver

FAX's own block Thomas shift-invert (a fixed 40 iterations at `tol = 1e-6`) was replaced by
{func}`solvax.generalized_eigenpairs` (stops at `tol = 1e-10`). Eigenvalues, gradients and the CTOK, Könies and
drive references are unchanged to 1e-9 relative. One CPU core, float64, JAX 0.12; cold is compile plus first call,
warm the median of five (seconds, before / after; forward is `eigenmodes` including assembly, 0.024, 0.039 and 0.079 s,
gradient is `jax.grad` of `frequency_hz`). Iterations: 40 / 40 / 40 before, 11 / 17 / 33 after.

| case | $N$, $H$ | forward cold | forward warm | gradient cold | gradient warm |
|---|---|---|---|---|---|
| CTOK, 6 modes, 384 kHz | 98, 13 | 0.70 / 0.85 | 0.076 / 0.047 | 1.76 / 1.93 | 0.115 / 0.074 |
| Könies, 3 modes, 64.6 kHz | 200, 12 | 0.89 / 1.04 | 0.110 / 0.088 | 1.78 / 2.07 | 0.171 / 0.134 |
| Könies, 3 modes, 64.6 kHz | 400, 12 | 1.11 / 1.25 | 0.220 / 0.236 | 2.10 / 2.20 | 0.324 / 0.324 |

Warm times are equal or better, cold compile about 0.15 s longer, and the iteration count now adapts to the accuracy
rather than being fixed. Timings on a shared laptop vary by about 10%.

## Global operator

**Continuum accumulation.** For a uniform field with sheared $\iota$ and a
single harmonic, the global spectrum fills
$[\omega_A^2(0), \omega_A^2(1)]$ with edges within $0.5/N$, and each mode's
$\phi'$ jump is localized to a fixed number of cells, so its width shrinks as
$1/N$ ($N = 40$ and $160$). No discrete mode appears outside the continuum.

**AE3D CTOK.** The same `boozmn` file averaged to AE3D's 99-point full mesh
with AE3D's edge extrapolation, $n = 2$, $m = 0..12$. The matrix leading
blocks and the operator continuum agree within 1%. Six global modes, matched
to AE3D's eigenvectors by overlap > 0.9, agree within 1.4%:

AE3D's 343.6, 384.1, 685.6, 703.3, 738.2 and 741.2 kHz modes are all
reproduced within 1.4% (gate 1.5%).

**The CTOK TAE: an AE3D sign error.** AE3D's TAE at 337.9 kHz is not
reproduced; FAX's best-overlap mode is at 272.8 kHz. The evidence that FAX is
right:

- The shear term $i\iota' m_jB/(G + \iota I)$ in $\partial_sc_j$
  ({eq}`shear-term`) follows from $u = \mathbf B\cdot\nabla\phi/B$ and is
  physical.
- `test_ctok_bending_energy_matches_real_space_quadrature` builds $u$ from
  finite differences of a test $\phi$ on the boozmn embedding, with $B^i$
  constructed from $|B|$, $\iota$ and $g_{ij}$ only, and integrates
  $\int|\nabla u|^2dV$ and $\int\mu_0\rho|\nabla\phi|^2/B^2dV$ in real space.
  $F$ and $G$ agree to 0.3%; without the $\iota'$ term $F$ is off by 9–20%.
- With a finer quadrature (1201 × 64 × 16) the Rayleigh quotient of AE3D's own
  TAE eigenvector is 329.7 kHz (FAX's operator: 328.5–329.3 kHz; FAX without
  $\iota'$: 342.4 kHz), not AE3D's eigenvalue 337.9 kHz. The test checks the
  quotient within 3%.
- AE3D's matrix matches FAX within 0.3% on same-surface blocks (1.1% Frobenius
  overall), but its $m = 2/3$ couplings between surfaces 45 and 46
  ($s \approx 0.46$) are 1.071 and 0.820 against FAX's 0.962 and 0.905. FAX
  with $-\iota'$ gives 1.066 and 0.806: the $\iota'$ cross term with the wrong
  sign. Neither a global $-\iota'$ (404.6 kHz) nor swapped $f'f$ blocks
  (332.1 kHz) reproduce AE3D's eigenvalue, so the defect is confined to its
  inter-surface $\iota'$ cross terms.
- The $n = 2$ TAE gap at $s \approx 0.45$ spans 232–396 kHz. Both modes peak
  there; the 272.8 kHz mode meets the continuum only at $s = 0.85$–$0.94$
  (1.8% of its norm), AE3D's 337.9 kHz mode at $s = 0.69$–$0.82$ (14%).

The Könies case is insensitive to $\iota'$ (4.061e5 vs 4.067e5 rad/s without
it), so only strongly sheared cases such as CTOK separate the two forms.

**Könies 2018 ITPA TAE.** {func}`fax.circular_torus` with $R_0 = 10$ m,
$a = 1$ m, $B_0 = 3$ T, $q = 1.71 + 0.16s$, $\epsilon_B = a/R_0$, hydrogen at
2·10¹⁹ m⁻³, $n = 6$, $m = 5..16$:

| $N_s$ | 50 | 100 | 200 | 400 |
|---|---|---|---|---|
| $\omega$ (10⁵ rad/s) | 4.0598 | 4.0607 | 4.0610 | 4.0610 |

against CAS3D 4.01e5 (+1.3%, shifted equilibrium) and GYGLES 4.13e5 (−1.7%,
ad hoc equilibrium as here). The test requires $4.01 < \omega/10^5 < 4.13$.

**Finite-$\beta$ terms.** The $J_\parallel$ term is tested in a cylinder
($\epsilon_B = 0$, $q = 0.8 + s$): for an $m = n = 1$ rigid shift inside
$q = 1$ it cancels the bending energy to 1e-3, Newcomb's $(m^2 - 1)$
marginality. This also fixes its sign; the opposite sign doubles the energy.
The pressure term reproduces the Suydam threshold for $m/n = 11/6$ with
$p = p_0(1 - s)$: $p_0 = 1.05\cdot10^4$ Pa from
$rB^2(q'/q)^2/8\mu_0 + p' = 0$. At $1.0\cdot10^4$ Pa the lowest
$\omega^2$ falls as $1/N^2$ (a marginal continuum), and at $2\cdot10^4$ Pa it is
a converged instability. Suydam growth is exponentially small near
threshold, so the test brackets it rather than locating it.

Könies ITPA, $p(s) = 7.17\cdot10^3 - 6.811\cdot10^3 s - 358.5 s^2$ Pa
(Könies et al. 2018), TAE in rad/s:

| equilibrium | no finite-$\beta$ terms | $J_\parallel$ and pressure |
|---|---|---|
| `circular_torus`, `current=True`, $N_s = 400$ | 4.0610e5 | 4.0637e5 ($J_\parallel$ only, +0.07%) |
| `circular_torus` with $p(s)$ | 4.0601e5 | Mercier-unstable (see below) |
| vmex, $p = 0$ (`PRES_SCALE = 0`), $N_s = 400$ | 4.0958e5 | 4.0978e5 (+0.05%) |
| vmex with $p(s)$, $N_s = 100$–400 | 4.1271e5 | 3.9482e5 (−4.3%, overlap 0.98) |

CAS3D, which is compressible and uses the shifted finite-$\beta$
equilibrium, gives 4.01e5. FAX with both terms is 1.5% below it, and 2.9% above
it without them. The remaining gap is plausibly the sound coupling
($\gamma p|\nabla\cdot\boldsymbol\xi|^2$), which the reduced model drops
and which raises the frequency. In the ad hoc equilibrium, which has no
Shafranov shift, the pressure term gives a converged localized instability
($\omega^2 = -3.1\cdot10^9$ s⁻²) at $q = 11/6$, below the cylindrical
Suydam threshold. That equilibrium lacks the $O(\epsilon^2)$ terms behind
the $(1 - q^2)$ Mercier stabilization. On the vmex equilibrium the
negative eigenvalues fall as $1/N^2$. Use a real equilibrium with
`finite_beta=True`.

On CTOK ($q_0 \approx 0.996$, $\sigma \approx -0.7$ m⁻¹), the $J_\parallel$ term
is large. It gives a converged near-axis $m = n = 2$ instability
($\omega^2 \approx -2.4\cdot10^{10}$ s⁻²), and it moves the AE3D-matched
modes by 0.1–10% (388 → 428 kHz; overlaps with AE3D fall to 0.72–0.87). The earlier logbook
statement that AE3D moves its modes by less than 0.1% with these terms
therefore does not carry over to this form. AE3D has not been rerun with
its $J_\parallel$ term on to compare; `test_ae3d_ctok_parity` keeps the default
`finite_beta=False`.

## Independent 3-D reference: precise QA of Knyazev et al.

The archive of Knyazev et al. (2026; Zenodo DOI 10.5281/zenodo.21299907, CC-BY-4.0) holds a precise-QA equilibrium
(`nfp = 2`, aspect 6, finite-$\beta$ VMEC, `boozmn_betaQA.nc`), its STELLGAP continuum and its AE3D spectrum with the kink
term removed. `tests/data/knyazev_qa.npz` (0.67 MB) reduces it; `tests/data/PROVENANCE.md` has hashes and switches, and
`examples/knyazev_qa.py` rebuilds the fixture and reproduces every number below (15 min, 8 GB). FAX runs with
`finite_beta=False`, a flat density of 1.88e-6 kg/m$^3$ (the decks are not in the archive; the value reproduces the
continuum), AE3D's 49 nodes $s = 0.02..0.98$ and its 162 harmonics. AE3D's sine basis is FAX's operator on
$(m, n), (-m, -n)$ restricted to the odd subspace.

```{image} _static/knyazev_qa.png
:alt: Precise-QA continuum and gap band, the archived global mode in FAX's eigenbasis, convergence with the harmonics
:width: 100%
```

**Continuum.** STELLGAP labels surface $k$ one cell (0.02) outward, as for CTOK. On its 98 stored surfaces (162 sorted
branches each) the relative L2 error is mean 3.0e-3, p95 3.4e-3, max 4.3e-3 in the complex basis of the harmonic list
(3.5e-3 mean, 4.4e-3 max in the odd sector), unchanged between grids of 48 x 32 and 112 x 72 where unaliased. The branches
that bracket the archived global mode's gap at $s > 0.25$ agree to 1.0% (the lower, median 0.4%) and 3.0% (the upper, near 35 kHz, median 1%). The
spectrum of AE3D is dense: only 3 of its 7938 eigenvalues above 5 kHz are 1% from both neighbours, so **no isolated
frequency exists** in this case, and 56 eigenvalues are negative (AE3D's own, down to $-84$ kHz$^2$); FAX's operator is
positive by construction.

**The gap band.** The paper follows one radially global mode at 30.309 kHz. It sits in a continuum gap (26-35 kHz at
mid-radius) that holds 82 AE3D eigenvalues, of which 20 are radially extended (inverse participation ratio of at least 0.3
of the nodes) and mid-gap (mean position at least 0.25 of the way between the STELLGAP branches that bracket them, radius-weighted).
FAX is judged on the same rule. $\hat f$ is the Rayleigh quotient of the archived eigenvector in FAX's $F$, $G$;
"best" is the FAX mode with the largest $G$-overlap with the archived global mode; "contained" is the mean fraction of each
archived band mode inside FAX's band subspace.

| nodes | harmonics | grid | modes in gap (AE3D 82) | band modes (AE3D 20) | global $\hat f$ shift | best FAX mode kHz (overlap) | band mean $\hat f$ shift | contained |
|---|---|---|---|---|---|---|---|---|
| 49 | 24 | 48 x 32 | 73 | 14 | +0.9% | 29.58 (0.14) | +8.6% | 0.22 |
| 49 | 40 | 48 x 32 | 74 | 22 | -0.0% | 29.79 (0.19) | +3.5% | 0.49 |
| 49 | 64 | 48 x 32, 64 x 48 | 91 | 21 | -1.1% | 28.58 (0.14) | +1.3% | 0.39 |
| 49 | 96 | 48 x 32 | 73 | 20 | -4.2% | 27.77 (0.19) | -0.9% | 0.40 |
| 49 | 162 (AE3D's) | 48 x 32 | 78 | 18 | -5.1% | 27.61 (0.18) | n/a | 0.17 (global mode) |
| 24 | 40 | 48 x 32 | 39 | 18 | -0.2% | 30.11 (0.33) | +5.1% | 0.44 |
| 99 | 40 | 48 x 32 | 139 | 29 | -0.5% | 30.35 (0.14) | +2.9% | 0.40 |

Harmonics are the largest by the energy of the archived band. The last 162-harmonic, 49-node row is AE3D's own setting;
its global-mode quotient is 28.76 kHz against the archived 30.31. The localization of the best FAX mode matches the archived
one (IPR 32 against 35 nodes, centroid $s$ 0.41 against 0.49).

**Convergence.** The angular grid is converged: 48 x 32 and 64 x 48 give identical numbers to the digits printed, and the band
subspaces coincide (cosines 1.00). Harmonics and nodes are not: FAX's own band subspace is 0.55 inside itself between 40 and
64 harmonics, 0.55 between 64 and 96, 0.64 between 24 and 49 nodes and 0.58 between 49 and 99. The number of modes in the gap
scales with the nodes (39, 74, 139 at 24, 49, 99), so counts compare only at AE3D's 49 nodes; the quotient of the archived
mode converges only when all of its harmonics are kept (+0.9% to -5.1%).

**Verdict.** What agrees: the continuum (0.3%, gap edges 1-3%), the count (18-22 against 20), frequency range
(28.4-34.9 against 28.6-33.2 kHz) and localization of the gap band at AE3D's resolution. What does not: the archived global
mode is not a FAX eigenmode. At matched settings its energy in FAX's operator is 5.1% below its AE3D eigenvalue, and it is
spread over many FAX modes (27.6-30.7 kHz, largest overlap 0.18 at 27.61 kHz). PR #9 recorded the same from a separate
assembly: FAX with physical shear indices 27.57 kHz, and a reconstruction of AE3D's public formulas 30.30 kHz (diagnostic
numbers, not rerun here; its operator is not in this repository). The 27.6 kHz mode is therefore reproducible, but it is
not unique: in this near-degenerate band the mode-by-mode eigenvectors are ill-conditioned, FAX's own band subspace moves
by 36-45% under refinement, and the overlap with AE3D (0.4) is of that size. This is consistent with, but does not
establish, a difference through AE3D's shear cross terms (the CTOK discussion above): that needs a pinned AE3D build and its
blocks (plan, B). `finite_beta` terms are off, as in the archive.

## Gradients

- **Eigenvalues.** For the Könies operator at $N = 40$, the Hellmann–Feynman
  derivative of a dense cluster mean and of a (SOLVAX) shift-invert eigenvalue with
  respect to density and an $\iota$ offset matches central differences to
  1e-5, and $\partial f/\partial\ln\rho = -f/2$ to 1e-10. The same holds with
  `finite_beta=True` and the Könies pressure (central step 1e-7).
- **Drive.** `jax.grad` of $\gamma_h$ with respect to the fast-ion density
  scale, $T_f$ and $\omega$ matches central differences to 1e-5; the
  slowing-down gradient is finite through the smoothed birth edge.
- **Boundary.** On low-resolution precise QA ($n_s = 16$, ftol 1e-12,
  `mboz = 3`), with the lowest $n = 0$ global frequency ($m = 1$–3), only the
  FE radial grid is refined:

  | nodes | f (kHz) | df/d`zs01` (kHz) | FAX AD vs FD |
  |---|---|---|---|
  | 16 | 294.648 | -62.528 | 3.4e-8 |
  | 32 | 294.219 | -61.230 | 3.5e-8 |
  | 64 | 293.989 | -60.583 | 3.6e-8 |
  | 128 | 293.873 | -60.261 | 3.6e-8 |

  Both converge at first order (differences halve), as expected for the
  continuum minimum. The full adjoint gradient matches re-solve finite
  differences to 3e-5 for `zs01`, `rc01` and `rc10`; the remainder is the
  $m = 1$ gauge drift of the re-solves ({ref}`vmex-gradients`), which is
  2.7e-3 for `zs11`.

## Optimization

`examples/optimize_alfven.py` ({doc}`tutorials`, section 6), $n_s = 16$,
`mboz = nboz = 6`, office CPU (`JAX_PLATFORMS=cpu`).

**QH: Paul et al. (2025) eq. 7.1 on LP-QH** (reactor scale, $N_P = 4$, helicity
$(1, -1)$, cut $|\iota + 4|$), 34 coefficients, 60 evaluations, 31 min:

| | initial | optimized |
|---|---|---|
| $f_\mathrm{cont}$ | 7.038 | 5.226 (−26%) |
| QS total ($f_{QS}$) | 1.32e-4 | 5.72e-4 |
| aspect ratio | 8.0000 | 8.0037 |
| mean $\iota$ | −1.2431 | −1.2393 |

Largest gaps above the cut, centre $\omega^{(0)}/\omega_A$ and width
$2\omega^{(0)}|\epsilon|/\omega_A$ (full width, surface-averaged):

| $(\delta m, n)$ | centre | width before | width after |
|---|---|---|---|
| (3, −8) HAE | 2.14 | 0.280 | 0.278 |
| (0, 4) MAE | 2.00 | 0.232 | 0.196 |
| (2, −8) HAE | 2.76 | 0.174 | 0.136 |
| (0, 8) MAE | 4.00 | 0.138 | 0.098 |
| (3, −12) HAE | 4.14 | 0.132 | 0.116 |

The trend of Paul's figures 8–9 is reproduced: the high-frequency gaps
narrow (up to 29%) while QS degrades moderately (×4.3, still 5.7e-4) and
aspect ratio and $\iota$ hold. The reduction is smaller than Paul's
(Wistell-A start, many more evaluations): the cost was still falling at
60 evaluations, and the dominant (3, −8) gap barely moved, since closing it
costs QS on this precise-QS start (QS row weight 0.1).

**QA: global modes out of a window.** The three lowest $n = 0$ global modes
($m = 1$–3, 24 nodes) start at 294.10–294.92 kHz inside
[293.10, 314.92] kHz; 4 coefficients (`rc11, zs11, rc12, zs12`),
30 evaluations, 18 min:

| | initial | optimized |
|---|---|---|
| modes (kHz) | 294.10, 294.44, 294.92 | 290.65, 290.99, 291.47 |
| window risk row (cost) | 143.2 | 5.17 |
| QS total | 1.09e-6 | 3.96e-6 |
| aspect ratio | 6.0000 | 5.9881 |
| mean $\iota$ | 0.4191 | 0.4159 |

All three modes leave the window (2.4 kHz below its edge, 2.4 softness
widths) with QS still at the 4e-6 level; the shift comes mainly from
$\iota$ (−0.8%), as $f \propto |\iota m - n|$ near the continuum minimum.

## Fast-ion drive

The drive is evaluated in Boozer geometry ({doc}`theory`, section 8); the
tokamak gates use {func}`fax.circular_torus`, axisymmetric (`helical = 0`,
`nzeta = 1`).

- **Fu–Van Dam.** Weak toroidicity ($\varepsilon_B = 10^{-4}$, $R_0 = 10^4a$,
  nonzero drift coupling) and a flat Maxwellian with the two
  $v_\parallel = v_A$ resonances: $\gamma$ at fixed $\beta_h$ scales as
  $x(1 + 2x^2 + 2x^4)e^{-x^2}$, $x = v_A/v_{th}$, to 1e-3, and its absolute
  value equals the closed-form ZOW integral (NumPy, derived in the test
  docstring: orbit-averaged kernel, $\int 2\xi\,d\xi\,\xi^{-8}(1 + \xi^2)^2e^{-x^2/\xi^2}
  = 2(1 + 2x^2 + 2x^4)e^{-x^2}/x^6$, independent $W$) to 1e-3. The prefactor
  $\pi^2\mu_0m_f^2/2W$ is that of {eq}`gamma-zow`, pinned by the Könies gate,
  not derived there. The earlier test used $\varepsilon_B = 0$, where $\gamma = 0$
  exactly (no $|B|$ variation, no drift coupling), so it checked nothing.
- **Könies et al. (2018).** Model TAE ($m = 10, 11$, $n = 6$, Gaussian
  envelope of 10 cm FWHM at $q = 1.75$), $\omega = 4.0$e5 rad/s, the paper's
  fast-ion density profile (its eq. 1), `circular_torus(epsilon_B = a/R0)`:

  | $T_f$ (keV) | 100 | 200 | 300 | 400 |
  |---|---|---|---|---|
  | FAX, Boozer geometry, $j$-form (10³ s⁻¹) | 4.26 | 16.27 | 32.48 | 64.02 |
  | FAX, Boozer geometry, previous $n_h/\iota$ form | 4.33 | 16.52 | 32.94 | 64.85 |
  | FAX, previous circular large-aspect-ratio model | 4.4 | 16.6 | 33.1 | 65 |
  | Könies ZLR band (10³ s⁻¹) | 8–10 | 25–33 | 38–43 | |

  The geometric path and the circular model differ in $O(\varepsilon)$ terms
  ($dl/d\zeta = G/B$ instead of $R_0$, the Boozer angle, the exact $\nabla B$) and in the
  label of the gradient term ({doc}`theory`, section 8: $-1.3$ to $-1.7\%$); they agree within 4%. The pooled ZLR band mixes codes with
  trapped particles, FOW-free but FLR-free kinetics and different damping, and is **not** the comparison used
  here (rule 2: no fitting to the pooled band).
- **ITPA ladder, rung 2 (fixed-mode ZOW).** Könies et al. (2018) fig. 1a has one zero-orbit-width curve, CAS3D-K (ZOW), valid at small
  $T_f$ (table 1: no FOW, no FLR). The paper does not say whether it contains trapped particles, and FAX is passing-only,
  so the content is **not matched**. The dotted curve read off the vector figure (the points between 100 and 135 keV are hidden
  behind other markers) and FAX (same mode, $\omega = 4.0\cdot10^5$ rad/s, eq. 1 density, Maxwellian):

  | $T_f$ (keV) | 82 | 139 | 163 | 191 |
  |---|---|---|---|---|
  | CAS3D-K ZOW (10³ s⁻¹) | 4.26 | 14.60 | 19.98 | 25.17 |
  | FAX, passing only (10³ s⁻¹) | 2.47 | 8.72 | 11.63 | 15.12 |
  | ratio | 0.58 | 0.60 | 0.58 | 0.60 |
  | $\gamma/\omega$ | 0.006 | 0.022 | 0.029 | 0.038 |

  FAX is a constant 0.58–0.60 of the ZOW curve over a factor 2.3 in $T_f$, which is what a missing trapped contribution of
  about 40% would look like; that attribution is not established and needs either a passing-only CAS3D-K ZOW run (Könies and
  Kleiber 2012, not in the reference folder) or trapped particles in FAX. Missing damping is unknown, not zero: the paper
  subtracts each code's damping so that $\gamma(T_f = 0) = 0$, FAX has no continuum, electron or ion Landau damping of the
  mode ($\gamma_d$ would be subtracted). Frozen-mode validity: $\gamma/\omega$ stays below 5% up to 200 keV and reaches 8% and 16%
  at 300 and 400 keV, where the perturbative picture also loses the frequency shift of fig. 2 and finite orbit width. Rungs 3–5
  (FOW/ZLR, FLR, damping) are not started.
- **Signs.** A flat profile gives equal damping for $(m, n)$ and $(-m, -n)$;
  a peaked profile drives $n > 0$ and damps $n < 0$ ($\iota\psi' > 0$); a
  hollow profile damps; bulk-ion Landau damping is positive.
- **Gradients.** `jax.grad` with respect to every {class}`fax.Geometry` leaf and
  the mode, along a random direction, matches central differences to 1e-5.
- **3-D structure** ({doc}`theory`, section 8). The gradient term is $j\,\partial_sF/(\omega Ze\psi')$ with the orbit's poloidal harmonic $j$,
  not the mode's $n_h/\iota$. (i) A tokamak (with and without $I$) and its helical relabelling $\theta' = \theta - N\zeta$
  ($N$ = 2, 4; $\iota \to \iota + N$, $n \to n + mN$) have the same $\gamma$ to 4e-5 (the old form: 17–20%).
  (ii) Weak-toroidicity closed form with a density tilt: the gradient term, 27% of the profile factor at $x = 1$, is
  reproduced with $j = 11, 10$ to 1e-3 (the $n/\iota$ form is 1–2% off). (iii) $\theta$- and $\zeta$-mirrors
  ($\iota \to -\iota$ with $m \to -m$, or $n \to -n$, $I \to -I$) agree to 1e-12 and the second maps each $(j, k)$ to $(j, -k)$ with
  the same $\gamma_{jk}$, so co- and counter-passing resonances both carry drive. (iv) Amplitude and global phase of $\phi$,
  and density scaling, exact to 1e-12; a uniform Maxwellian or slowing-down distribution has $\gamma < 0$ at every $\omega$.
  The model uncertainty of the label ($F$ of the mean flux versus of the $P_\zeta$ surface) is $O(\omega/n\Omega_\varphi)$: 1.5% for $n = 6$, 23% and 7% at 25 and 50 keV for the CTOK
  $n = 2$ mode.
- **Regularizations** (explicit arguments, defaults in brackets; relative error against the converged value):

  | Parameter | 0.1 | 0.01 | 0.001 | 1e-4 | note |
  |---|---|---|---|---|---|
  | `boundary` [0] | 1.5e-1 | 1.8e-2 | 1.0e-3 | 2e-6 | first order up to a log; Könies-like torus |
  | `guard` [0.01], away from rationals | 6e-3 | 8e-5 | 1e-7 | 1e-8 | helical test torus, $\iota \in [0.43, 0.46]$ |
  | `guard`, $\iota = 1/2$ crossed (shear 0.3) | 5e-3 | 2e-4 | 1e-5 | 1e-7 | |
  | `guard`, $\iota \approx 1/2$ on the whole mode (shear 0.02) | 1.7e-2 | 1.1e-2 | 3.1e-2 | 2e-3 | not converged above 1e-4 |

  `FastIons.birth_width` [0.02] converges as $O(w^2)$ (1865.6, 1848.6, 1845.0 s⁻¹ at 0.04, 0.02, 0.01;
  difference ratios 5.0 and 4.7 per halving) provided the pitch nodes resolve the edge ($\gtrsim 2/w$); at 32 nodes and $w \le 10^{-3}$ the
  edge falls between nodes and its share of $\gamma$ (about 9% in that case) is lost without any warning. `nodes` [32]: 3.5e-3, 9e-4, 3.5e-4, 2e-5 at 8, 16, 32, 64.
  Near a rational surface of a 3-D equilibrium $\gamma$ depends on `guard` at the percent level (a true singularity, not
  noise): refine it there.
- **CTOK.** The FAX TAE of the CTOK `boozmn` (272.8 kHz, $n = 2$, $m = 0..12$,
  D at 10¹⁹ m⁻³) from the shift-invert solve, driven by deuterons with
  $n_f = 10^{17}(1 - s)^2$ m⁻³: $\gamma/\omega$ = 7.0e-4 (25 keV), 3.3e-3
  (50 keV), 1.57e-2 (100 keV); 9.2e-4, 3.6e-3, 1.6e-2 with the previous $n_h/\iota$ form; bulk-ion Landau damping at 1 keV is 2.6 s⁻¹.
- **Stellarators.** LP-QA (ns 16, 16 FE nodes, $n = 1$ family $m = 0..5$,
  $n \in \{-1, 1, 3\}$, the four modes nearest 104 kHz, 150 keV Maxwellian
  deuterons, $n_f = 10^{15}(1 - s)^2$): resonances with $k \ne 1$ carry a
  fraction $10^{-6}$–$10^{-2}$ of $\gamma$, as expected for quasi-axisymmetry,
  and the pure helical sidebands $k = -3, 5$ (outside the mode's own $n$) are
  nonzero. `jax.jvp` of {func}`fax.drive_risk` of {func}`fax.geometry_growth_rates`
  (FE interpolation, shift-invert eigenvectors differentiated implicitly,
  drive) along the re-solved geometry tangent of `rc01` matches FD
  to ≤ 1e-4 (5e-8 measured at 24 nodes). The full `jax.grad` through the vmex
  adjoint differs from re-solve FD by 1.6e-3: vmex's released $m = 1$ gauge
  ({ref}`vmex-gradients`), amplified by the drive's sensitivity. In LP-QH
  (helical $B \sim \cos(\theta + 4\zeta)$, $N_{fp} = 4$) the helical sidebands
  dominate some modes: for the 43.9 kHz gap mode of the $n = 1$ family
  driven by slowing-down alphas, 99% of $\gamma$ comes from $k = -7$, a
  toroidal number absent from the mode.

## Not done

- **Sound and fast-wave coupling** (no BAE gap; $\gamma p|\nabla\cdot\boldsymbol\xi|^2$
  dropped), and an AE3D rerun with its $J_\parallel$ term for CTOK.
- **Trapped and precessing fast ions** in the drive (bounce-averaged
  precession resonances), and the covariant $B_s$ and $\delta B_\parallel$ in the drift.
- **Finite orbit width** and finite Larmor radius in the drive.
- **Continuum and electron Landau damping** are not computed (subtract an
  external $\gamma_d$).
- **AE3D LHD (37, 73.7 kHz) and QPS (149 kHz)** published frequencies need
  their equilibria; Mishchenko et al. (2011) is not yet a gate.
- **P5 open items**: HSX optimization (the HSX fixed-boundary decks are
  mpol 18, ntor 24, and were not reduced or run); a larger $\gamma_\mathrm{max}$ reduction (the 4-coefficient LP-QA
  `--drive` run lowers it by only 4.4%, 8500 → 8123 s⁻¹, at QS 1.1e-6 → 1.6e-6); a
  fast-ion resonance-band term; higher resolution ($n_s \ge 25$) and longer
  QH runs.
- {func}`fax.circular_torus` has no Shafranov shift: it reproduces Könies'
  ad hoc equilibrium, not the CAS3D one.
