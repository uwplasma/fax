# FAX Research Examples

These scripts are small, reproducible entry points for users who want to learn
the current FAX workflows and generate validation artifacts themselves.

Run them from the repository root after installing the development extras:

```bash
python -m pip install -e ".[dev]"
```

Start with these three commands:

```bash
python examples/research/basic_alfven_gap_map.py
python examples/research/global_gap_fastion_screen.py
python examples/research/publication_figure_suite.py --case 01 --case 02
```

They require no external geometry artifacts and produce figures plus JSON
summaries under `outputs/`.

The full local example set is:

```bash
python examples/research/validation_packet.py
python examples/research/density_sensitivity.py
python examples/research/inverse_iota_shift.py
python examples/research/basic_alfven_gap_map.py
python examples/research/global_gap_fastion_screen.py
python examples/research/differentiable_design_study.py
python examples/research/boozer_harmonic_validation.py
python examples/research/boozmn_3d_continuum.py
python examples/research/boozmn_global_3d_eigenmodes.py
python examples/research/boozmn_differentiable_design_study.py
python examples/research/vmec_boozer_alfven_design_gradient.py
python examples/research/vmec_boozer_gradient_convergence.py
python examples/research/vmec_boozer_convergence.py
python examples/research/vmec_equilibria_catalog.py
python examples/research/vmec_equilibria_boozer_global.py
python examples/research/vmec_equilibria_live_boozer.py
python examples/research/stellgap_ctok_comparison.py
python examples/research/external_code_artifact_dashboard.py
python examples/research/far3d_gpu_lincheck_artifact.py
python examples/research/far3d_gpu_convergence_ladder.py
python examples/research/far3d_gpu_beta_scan.py
python examples/research/publication_figure_suite.py
```

The examples intentionally use the analytic and Landreman-Paul low-resolution
paths that are already covered by local tests and validation gates. They are
research workflow templates, not final 3-D stellarator eigenmode validations.

Scripts that take `--boozmn` or read `~/local` use optional local
STELLGAP/AE3D/BOOZ_XFORM artifacts. When those files are absent, use the
analytic examples first and treat the external-code examples as parity checks
to run after the artifacts are installed.

## Differentiable Design Study

```bash
python examples/research/differentiable_design_study.py --output-dir outputs/differentiable_design_study
```

This writes a four-panel publication-style dashboard and JSON summary:
autodiff sensitivity matrix, linearized frequency uncertainty bands,
gradient-based inverse-design history, and reduced fast-ion growth before/after
optimization. This is the first FAX example built around capabilities that are
hard to reproduce in legacy non-differentiable AE workflows.

## Real `boozmn` Differentiable Design Study

```bash
python examples/research/boozmn_differentiable_design_study.py --boozmn /path/to/boozmn.nc
```

This applies the same differentiable workflow to a real Boozer global-mode
operator: frequency sensitivities, linearized UQ bands, a gradient-based design
move, and fast-ion growth projected onto the optimized global eigenfunctions.
It is the current bridge from validated real-Boozer operators toward
publication-grade differentiable stellarator design studies.

## VMEC Boundary Alfvén Design Gradient

```bash
python examples/research/vmec_boozer_alfven_design_gradient.py --output-dir outputs/vmec_boozer_alfven_design_gradient
```

This is the first full in-memory differentiability showcase from actual VMEC
boundary Fourier coefficients through `vmex`, `booz_xform_jax`, and a FAX
global Alfvén risk objective. It compares the JAX gradient with centered finite
differences, then takes one line-searched boundary-shape step that lowers the
smooth mode-in-window risk. Increase `--vmec-max-iter`, Boozer truncation, and
surface count for production studies after the gradient gate passes.

## VMEC Boundary Gradient Convergence

```bash
python examples/research/vmec_boozer_gradient_convergence.py --output-dir outputs/vmec_boozer_gradient_convergence
```

This repeats the VMEC-boundary Alfvén gradient check across VMEC-iteration and
Boozer/angular refinement levels. It writes a JSON packet plus a figure showing
mode-frequency drift, AD/finite-difference gradient error, and relative drift
to the reference level.

## Basic Alfvén Gap Map

```bash
python examples/research/basic_alfven_gap_map.py
```

This writes `outputs/basic_alfven_gap_map/basic_alfven_gap_map.png` and `.svg`.
It is the shortest visual introduction to the current gap model: cylindrical
continuum branches, a TAE avoided crossing, an EAE avoided crossing, and a
low-frequency BAE accumulation marker. The JSON output reports the reference
`q`, radial surface, center frequency, and width/accumulation frequency.

## Real `boozmn` Local 3-D Continuum

```bash
python examples/research/boozmn_3d_continuum.py --boozmn /path/to/boozmn.nc
```

This reads Boozer harmonics, checks `sqrt(g)|B|^2=G+iota I`, compares the
embedding-derived Jacobian shape, solves the local 3-D Boozer continuum matrix,
and writes a compact figure. It is a local surface-by-surface geometry/operator
example, not a global radial eigenmode or STELLGAP cross-code benchmark.

## Real `boozmn` Global 3-D Eigenmodes

```bash
python examples/research/boozmn_global_3d_eigenmodes.py --boozmn /path/to/boozmn.nc
```

This builds the radial block eigenproblem from real local 3-D Boozer matrices,
solves for global frequencies and radial/harmonic mode structures, writes a
figure, and prints residuals. It is the current global radial shear-Alfven
operator path, still awaiting STELLGAP/AE3D cross-code validation.

## Live VMEC/Boozer Convergence

```bash
python examples/research/vmec_boozer_convergence.py --output-dir outputs/vmec_boozer_convergence
```

This runs the local Landreman-Paul QA ``vmex`` input through the in-memory
``booz_xform_jax`` bridge, then checks two convergence layers: Boozer harmonic
truncation and angular quadrature for the local 3-D Boozer operator. It is a
production-readiness smoke check for the live bridge, not a converged VMEC
physics benchmark.

## VMEC-Equilibria Multi-Geometry Catalog

```bash
python examples/research/vmec_equilibria_catalog.py --root ~/local/vmec_equilibria
```

This reads selected `wout*.nc` files from the external
`landreman/vmec_equilibria` clone and writes a profile-continuum catalog over
HSX, W7-X, NCSX, ARIES-CS, LHD, TJ-II, QHS, and ITER-like references. It is the
first broad geometry-matrix screen for FAX examples. It is still a
profile-level VMEC-`wout` catalog, not full Boozer/global cross-code parity.

## VMEC-Equilibria Boozer Global Operator

```bash
python examples/research/vmec_equilibria_boozer_global.py --root ~/local/vmec_equilibria
```

This reads public `boozmn` artifacts from the external VMEC-equilibria clone
and runs the FAX global radial 3-D Boozer operator on HSX and Ku-Boozer QHS
cases. The figure compares lowest global frequencies, radial envelopes,
Boozer-Jacobian identity errors, and boundary/stiffness variants. This is a
public-artifact global-operator gate, not yet AE3D/CAS3D/STELLGAP parity.

## VMEC-Equilibria Live Boozer Generation

```bash
python examples/research/vmec_equilibria_live_boozer.py --root ~/local/vmec_equilibria --jit
```

This runs `booz_xform_jax` directly on public W7-X and NCSX VMEC `wout` files,
then sends the generated Boozer geometry into FAX's global radial 3-D Boozer
operator. It compares coarse/reference Boozer truncations, runtime,
Boozer-Jacobian identity errors, and lowest global-mode convergence. This is a
generated-Boozer convergence gate, not external-code parity.

## STELLGAP CTOK Cross-Code Comparison

```bash
python examples/research/stellgap_ctok_comparison.py
```

This reads the public ORNL STELLGAP CTOK `boozmn_ctok.nc` and `alfven_post`
artifacts when they are available under `~/local`, computes FAX
local 3-D Boozer continuum branches for the same harmonic family, and writes a
comparison figure plus JSON metrics. This is the first quantitative external
code parity gate in FAX; it is local-continuum parity, not a global AE3D/CAS3D
eigenmode benchmark.

## External-Code Artifact Dashboard

```bash
python examples/research/external_code_artifact_dashboard.py
```

This reads the public ORNL STELLGAP and AE3D answer artifacts under
`~/local`, reproduces STELLGAP and AE3D post-processing/eigenvalue
conventions, checks AE3D sparse matrix symmetry, independently re-solves the
public AE3D generalized matrix spectrum and global mode shapes, and writes a
figure plus JSON metrics.
These are prerequisite convention gates before full physics parity against
AE3D/CAS3D/FAR3D.

## FAR3D GPU Lincheck Artifact

```bash
python examples/research/far3d_gpu_lincheck_artifact.py
```

This reads the archived FAR3D DIII-D GPU run under
`external_artifacts/far3d/gpu_diiid_nvhpc`, parses the `lincheck` rows, applies
the run-reported code-to-kHz conversion, and writes a compact dashboard. It is a
build/run/parser traceability example, not a converged kinetic parity result.

## FAR3D GPU Convergence Ladder

```bash
python examples/research/far3d_gpu_convergence_ladder.py
```

This reads the archived FAR3D DIII-D `maxstp = 50, 100, 200, 400` ladder and
plots convergence of the parsed `lincheck` rows against the 400-step reference.
It shows that the 200-step artifact is stable to better than `3e-5` in maximum
relative growth-rate difference for common mode labels. This is still a
single-case convergence screen, not FAR3D/CAS3D-K kinetic parity.

## FAR3D GPU Fast-Ion Beta Scan

```bash
python examples/research/far3d_gpu_beta_scan.py
```

This reads the archived FAR3D DIII-D `bet0_f = 0.000, 0.010, 0.030, 0.060,
0.090` scan and plots the finite-beta growth trend, real-frequency trend,
growth spread, and highest-beta mode cloud. The beta-zero case is shown as a
control rather than a calibrated branch point. This is the first executable
FAR3D parameter-scan artifact for closure-calibration development, not a
held-out FAX/FAR3D kinetic parity result.

## Publication Figure Suite

```bash
python examples/research/publication_figure_suite.py --output-dir outputs/publication_figure_suite
```

This writes ten SVG/PNG figures plus a JSON summary covering the common
publication lanes for FAX studies: gap taxonomy, convergence, VMEC/wout
inspection, Boozer harmonics, local 3-D Boozer continua, global radial
eigenmodes, fast-ion drive/damping, differentiable sensitivities, STELLGAP
CTOK gap-anchored global screening, and AE3D public matrix/eigenmode/spectrum
plus mode-shape parity.
Run selected cases with repeated `--case` flags, for example `--case 09 --case
10`.
