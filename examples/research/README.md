# FAX Research Examples

These scripts are small, reproducible entry points for users who want to learn
the current FAX scaffold and generate validation artifacts themselves.

Run them from the repository root after installing the development extras:

```bash
python -m pip install -e ".[dev]"
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
python examples/research/stellgap_ctok_comparison.py
python examples/research/external_code_artifact_dashboard.py
python examples/research/publication_figure_suite.py
```

The examples intentionally use the analytic and Landreman-Paul low-resolution
paths that are already covered by local tests and validation gates. They are
research workflow templates, not final 3-D stellarator eigenmode validations.

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

## STELLGAP CTOK Cross-Code Comparison

```bash
python examples/research/stellgap_ctok_comparison.py
```

This reads the public ORNL STELLGAP CTOK `boozmn_ctok.nc` and `alfven_post`
artifacts when they are available under `/Users/rogerio/local`, computes FAX
local 3-D Boozer continuum branches for the same harmonic family, and writes a
comparison figure plus JSON metrics. This is the first quantitative external
code parity gate in FAX; it is local-continuum parity, not a global AE3D/CAS3D
eigenmode benchmark.

## External-Code Artifact Dashboard

```bash
python examples/research/external_code_artifact_dashboard.py
```

This reads the public ORNL STELLGAP and AE3D answer artifacts under
`/Users/rogerio/local`, reproduces STELLGAP and AE3D post-processing/eigenvalue
conventions, checks AE3D sparse matrix symmetry, and writes a figure plus JSON
metrics. These are prerequisite convention gates before full physics parity
against AE3D/CAS3D/FAR3D.

## Publication Figure Suite

```bash
python examples/research/publication_figure_suite.py --output-dir outputs/publication_figure_suite
```

This writes ten SVG/PNG figures plus a JSON summary covering the common
publication lanes for FAX studies: gap taxonomy, convergence, VMEC/wout
inspection, Boozer harmonics, local 3-D Boozer continua, global radial
eigenmodes, fast-ion drive/damping, differentiable sensitivities, STELLGAP
CTOK gap-anchored global screening, and AE3D public matrix/eigenmode parity.
Run selected cases with repeated `--case` flags, for example `--case 09 --case
10`.
