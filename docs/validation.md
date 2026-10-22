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
| Könies 2018 ITPA TAE frequency | `test_konies_tae_frequency_and_convergence` | 4.0610e5 rad/s, $N_s$-converged to 1e-4 |
| Shift-invert vs dense | `test_shift_invert_matches_dense_and_gradients_match_finite_differences` | 1e-10 |
| Eigenvalue gradients vs finite differences | same | 1e-5 (cluster mean and shift-invert) |
| Fu–Van Dam Maxwellian scaling | `test_fu_van_dam_maxwellian_scaling` | 1e-3 |
| Drive vs Könies 2018 | `test_koenies_2018_tae_growth_rates` | 300 keV within 20%; 100, 200 keV about 40% low |
| Drive sign rules | `test_drive_sign_and_flat_profile` | flat, peaked, hollow, $\pm n$ |
| Drive gradients vs finite differences | `test_gradients_match_finite_differences` | 1e-5 |
| vmex → booz_xform_jax → FAX continuum | `test_vmec_input_to_continuum` | runs, finite |
| Boundary gradient vs finite differences | `test_boundary_gradient_matches_finite_differences` | strict xfail at 1e-4 (upstream, below) |

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

## Gradients

- **Eigenvalues.** For the Könies operator at $N = 40$, the Hellmann–Feynman
  derivative of a dense cluster mean and of a shift-invert eigenvalue with
  respect to density and an $\iota$ offset matches central differences to
  1e-5, and $\partial f/\partial\ln\rho = -f/2$ to 1e-10.
- **Drive.** `jax.grad` of $\gamma_h$ with respect to the fast-ion density
  scale, $T_f$ and $\omega$ matches central differences to 1e-5; the
  slowing-down gradient is finite through the smoothed birth edge.
- **Boundary.** On low-resolution precise QA, FAX's own derivatives are exact
  (at most 7e-8 against finite differences for every geometry leaf). The
  end-to-end check against finite differences in `rc01`, `zs01` is a strict
  xfail at 1e-4: the vmex/booz_xform_jax derivative of $\iota$ with respect to
  `zs01` is off by 1.1e-4 (finite differences converged, also at ftol 1e-14),
  and the resonant operator amplifies that to 2.3e-4.

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

- **Pressure-gradient and $J_\parallel$ (kink) terms** of the AE3D weak form
  (below 0.1% on CTOK, needed at finite $\beta$; no BAE or sound coupling).
- **Trapped and precessing fast ions** in the drive.
- **Finite orbit width** and finite Larmor radius in the drive.
- **Stellarator drift sidebands**: the drive uses a circular large-aspect-ratio
  model with only the toroidal $\pm1$ coupling plus orbit harmonics; helical
  $B$ harmonics do not enter the resonance or the drift.
- **Continuum and electron Landau damping** are not computed (subtract an
  external $\gamma_d$).
- **AE3D LHD (37, 73.7 kHz) and QPS (149 kHz)** published frequencies need
  their equilibria; Mishchenko et al. (2011) is not yet a gate.
- {func}`fax.circular_torus` has no Shafranov shift: it reproduces Könies'
  ad hoc equilibrium, not the CAS3D one.
