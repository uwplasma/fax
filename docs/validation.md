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
| Bending and inertia vs real-space quadrature | `test_ctok_bending_energy_matches_real_space_quadrature` | 0.3% (gate 0.5%) |
| Finite-$\beta$ terms vs real-space quadrature ($\sigma$ from curl $\mathbf B$; CTOK and Könies with $p$) | `test_finite_beta_terms_match_real_space_quadrature` | 0.3% CTOK, 0.2% pressure (gate 0.5%) |
| Cylinder: Newcomb $m = 1$ cancellation; Suydam threshold | `test_cylinder_newcomb_m1_and_suydam_threshold` | bending + kink = 1e-3 of bending; stable at 1.0e4 Pa, unstable at 2.0e4 (Suydam 1.05e4) |
| Könies ITPA with $p(s)$, vmex equilibrium | `test_konies_itpa_finite_beta_tae` | 4.127e5 → 3.947e5 rad/s with finite-$\beta$ terms (CAS3D 4.01e5) |
| Könies 2018 ITPA TAE frequency | `test_konies_tae_frequency_and_convergence` | 4.0610e5 rad/s, $N_s$-converged to 1e-4 |
| Shift-invert vs dense | `test_shift_invert_matches_dense_and_gradients_match_finite_differences` | 1e-10 |
| Eigenvalue gradients vs finite differences | same | 1e-5 (cluster mean and shift-invert) |
| Fu–Van Dam Maxwellian scaling | `test_fu_van_dam_maxwellian_scaling` | 1e-3 |
| Drive vs Könies 2018 | `test_koenies_2018_tae_growth_rates` | 300 keV within 20%; 100, 200 keV about 40% low |
| Drive sign rules | `test_drive_sign_and_flat_profile` | flat, peaked, hollow, $\pm n$ |
| Drive gradients vs finite differences | `test_gradients_match_finite_differences` | 1e-5 |
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

## Gradients

- **Eigenvalues.** For the Könies operator at $N = 40$, the Hellmann–Feynman
  derivative of a dense cluster mean and of a shift-invert eigenvalue with
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
$\omega^{(0)}|\epsilon|/\omega_A$ (surface-averaged):

| $(\delta m, n)$ | centre | width before | width after |
|---|---|---|---|
| (3, −8) HAE | 2.14 | 0.140 | 0.139 |
| (0, 4) MAE | 2.00 | 0.116 | 0.098 |
| (2, −8) HAE | 2.76 | 0.087 | 0.068 |
| (0, 8) MAE | 4.00 | 0.069 | 0.049 |
| (3, −12) HAE | 4.14 | 0.066 | 0.058 |

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

- **Fu–Van Dam.** A flat Maxwellian with one $v_\parallel = v_A$ resonance
  scales as $x(1 + 2x^2 + 2x^4)e^{-x^2}$, $x = v_A/v_{th}$, to 1e-3.
- **Könies et al. (2018).** Model TAE ($m = 10, 11$, $n = 6$, Gaussian
  envelope of 10 cm FWHM at $q = 1.75$), $\omega = 4.0$e5 rad/s, the paper's
  fast-ion density profile (its eq. 1):

  | $T_f$ (keV) | 100 | 200 | 300 | 400 |
  |---|---|---|---|---|
  | FAX (10³ s⁻¹) | 4.4 | 16.6 | 33.1 | 65 |
  | Könies ZLR band (10³ s⁻¹) | 8–10 | 25–33 | 38–43 | |

  300 keV is within 20%; 100 and 200 keV are about 40% low. The result changes
  by less than 15% with mode width (3–8 cm) and relative harmonic amplitude.
  Candidate causes are trapped particles and orbit non-uniformity near the
  trapped–passing boundary, the missing $\delta B_\parallel$, and the model
  eigenfunction. Above 300 keV the published finite-orbit-width results roll
  over and FAX (ZOW) keeps growing.
- **Signs.** A flat profile gives equal damping for $(m, n)$ and $(-m, -n)$;
  a peaked profile drives $n > 0$ and damps $n < 0$; a hollow profile damps;
  bulk-ion Landau damping is positive.

## Not done

- **Sound and fast-wave coupling** (no BAE gap; $\gamma p|\nabla\cdot\boldsymbol\xi|^2$
  dropped), and an AE3D rerun with its $J_\parallel$ term for CTOK.
- **Trapped and precessing fast ions** in the drive.
- **Finite orbit width** and finite Larmor radius in the drive.
- **Stellarator drift sidebands**: the drive uses a circular large-aspect-ratio
  model with only the toroidal $\pm1$ coupling plus orbit harmonics; helical
  $B$ harmonics do not enter the resonance or the drift.
- **Continuum and electron Landau damping** are not computed (subtract an
  external $\gamma_d$).
- **AE3D LHD (37, 73.7 kHz) and QPS (149 kHz)** published frequencies need
  their equilibria; Mishchenko et al. (2011) is not yet a gate.
- **P5 open items**: HSX optimization (the HSX fixed-boundary decks are
  mpol 18, ntor 24, and were not reduced or run); a lower $\gamma_\mathrm{max}$ at fixed QS through
  {func}`fax.growth_rate` (the drive is circular large-aspect-ratio); a
  fast-ion resonance-band term; higher resolution ($n_s \ge 25$) and longer
  QH runs.
- {func}`fax.circular_torus` has no Shafranov shift: it reproduces Könies'
  ad hoc equilibrium, not the CAS3D one.
