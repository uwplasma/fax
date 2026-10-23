# Tutorials

The five examples in `examples/` cover the whole code. Run them from the
repository root; the figures below were made by these exact commands
(one CPU core, times include JAX compilation).

## 1. A continuum from a `boozmn` file

`examples/ctok_continuum.toml` is the STELLGAP CTOK test problem: a circular
tokamak, `n = 2`, `m = 0..12`, deuterium at 10¹⁹ m⁻³.

```console
fax run examples/ctok_continuum.toml --plot      # about 20 s, mostly compilation
```

```{image} _static/ctok_continuum.png
:alt: CTOK n = 2 shear-Alfvén continuum
:width: 60%
```

Each dot is one eigenvalue of the surface problem {eq}`continuum-galerkin` on
one of the 99 half-mesh surfaces. Branches with neighbouring `m` repel where
they would cross, which opens the TAE gaps (`m, m + 1`) and, higher up, the
EAE gaps. The same continuum in Python:

```python
import fax

geo = fax.read_boozmn("tests/data/boozmn_ctok.nc")
f_hz = fax.continuum(geo, fax.mode_family(2, 12), fax.mass_density(1e19, 2.0))
center, width = fax.gaps(f_hz)  # widest gap on each surface
iota, w_hat = fax.gap_location(2, 3, 1)  # cylindrical (3,2)-(4,2) TAE crossing: iota, omega R / v_A
```

## 2. Global eigenmodes near a target frequency

`examples/ctok_eigenmodes.toml` sets `task = "eigenmodes"`,
`target_khz = 384` and `count = 6`, so the block-tridiagonal shift-invert
solver returns the six eigenpairs of `F y = ω² G y` nearest 384 kHz.

```console
fax run examples/ctok_eigenmodes.toml --plot     # about 8 s
```

```{image} _static/ctok_modes.png
:alt: radial structure of the four lowest CTOK modes near 384 kHz
```

The modes found here (377.6, 378.6, 386.7, 388.1, 389.8, 390.9 kHz) sit
inside the continuum, so most of them are continuum-like: their harmonics peak
where `ω` crosses a continuum branch (red lines in the figure on the
{doc}`index`). The second one has the broad `m = 2` component of AE3D's
384.1 kHz global mode. This case runs on the 99 half-mesh surfaces of the file;
the AE3D comparison in {doc}`validation` uses AE3D's own full mesh.

```python
modes = fax.eigenmodes(geo, fax.mode_family(2, 12), fax.mass_density(1e19, 2.0), target_hz=384e3, count=6, ntheta=32, nzeta=4)
modes.frequencies_hz, modes.structures.shape  # (6,), (6, 99, 13)
```

Omit `target_hz` (or set `target_khz = 0`) for the dense solve of every
mode; it is `O((N H)^3)` and only sensible for small problems.

## 3. A stellarator from a VMEC input

`examples/qa_continuum.toml` solves the low-resolution precise-QA input
(Landreman & Paul 2022) with vmex at `ns = 16`, transforms nine surfaces to
Boozer coordinates with booz_xform_jax (`mboz = nboz = 8`) and computes the
continuum of the `n = 0, ±2` family (`Nfp = 2`, so these couple).

```console
pip install "uwplasma-fax[vmex]"
fax run examples/qa_continuum.toml --plot        # about 40 s
```

```{image} _static/qa_continuum.png
:alt: precise-QA continuum for the n = 0, 2, -2 family
:width: 60%
```

`iota` falls from 0.422 to 0.417 across these surfaces. In Python the geometry
step is `fax.from_vmec_input(path, ns=16, mboz=8, nboz=8, surfaces=...)`;
`fax.from_wout` does the same from a `wout` file or in-memory wout.

## 4. A gradient with respect to the plasma boundary

`examples/boundary_gradient.py` differentiates a smooth frequency-window
objective, {func}`fax.window_risk`, of the three lowest global frequencies of
precise QA with respect to the boundary coefficients `rc01` (`RBC(n=1, m=0)`)
and `zs01` (`ZBS(n=1, m=0)`).

```console
python examples/boundary_gradient.py             # about 40 s
```

```text
frequencies (kHz): [294.36787292 294.71040894 295.20747717]
risk = 7.1802e-02, d risk / d ('rc01', 'zs01') = [-0.6231568  0.5461438]
risk after a 1e-3 step along -grad: 7.0970e-02
```

{func}`fax.prepare_design` builds the static part (vmex configuration, Boozer
tables, FE nodes) once; {func}`fax.design_frequencies` is then a traceable
function of the boundary offsets, differentiated through the vmex implicit
equilibrium ({ref}`vmex-gradients`). This example uses a
tiny model (three harmonics, 8 × 8 angles, 24 radial nodes) to stay fast;
{doc}`validation` lists what its gradient is checked against.

## 5. Fast-ion drive of the Könies ITPA TAE

`examples/koenies_tae_drive.py` evaluates the perturbative growth rate
{func}`fax.growth_rate` of a model TAE (harmonics `m = 10, 11`, `n = 6`,
Gaussian radial envelope centred on `q = 1.75`, `ω = 4.0e5` rad/s) driven by
the Maxwellian fast-ion profile of Könies et al. (2018), for fast-ion
temperatures from 50 to 500 keV.

```console
cd examples && python koenies_tae_drive.py       # about 8 s, writes koenies_tae_drive.png
```

```{image} _static/koenies_tae_drive.png
:alt: FAX growth rate vs fast-ion temperature compared with Könies 2018
:width: 60%
```

The dots mark the zero-Larmor-radius band of Könies et al. (2018, fig. 1a).
FAX is a zero-orbit-width, passing-only model, so it keeps growing above
300 keV where the finite-orbit-width codes roll over; see {doc}`validation`.
Every argument of `growth_rate` (density profile, temperature, `ω`, the mode)
is differentiable.

## 6. Optimizing the Alfvén spectrum with quasisymmetry

`examples/optimize_alfven.py` runs `scipy.optimize.least_squares` on VMEC
boundary coefficients with residual rows (aspect ratio, $\sqrt{f_{QS}}$,
$\iota$ floor, Alfvén term), each scaled to about one at the start, and an exact
Jacobian: one vmex + FAX forward pass and one adjoint per row
({ref}`continuum-objective`).

```console
python examples/optimize_alfven.py --case qh    # Paul's f_cont on LP-QH, about 3 min
python examples/optimize_alfven.py --case qa    # LP-QA global modes out of a window, about 3 min
python examples/optimize_alfven.py --nfev 60 --ns 25    # longer, finer
```

- **QH** (Paul et al. 2025, eq. 7.1): 34 coefficients with $m \le 3$, $|n| \le 2$;
  rows $10(A/A_0 - 1)$, $0.1\sqrt{f_{QS}/f_{QS,0}}$, $10\sqrt{f_\iota}$ with
  $|\iota| \ge 1.03$, and $\sqrt{f_\mathrm{cont}/f_{\mathrm{cont},0}}$.
- **QA**: the three lowest $n = 0$ global modes ($m = 1$–3, 24 FE nodes) sit
  just inside the lower edge of a window $[f_\mathrm{min} - 1, f_\mathrm{max} + 20]$ kHz;
  the Alfvén row is $30\sqrt{\text{window\_risk}}$ (1 kHz softness) at fixed
  aspect ratio and QS.

The script prints the objective history, aspect, $\iota$, QS, $f_\mathrm{cont}$,
the largest gaps above the cut (centre and width in $\omega_A$ units) and, for
QA, the mode frequencies, and writes the figures below; {doc}`validation`
lists the numbers.

```{image} _static/optimize_alfven_qh.png
:alt: LP-QH continuum before and after continuum optimization, and the objective history
```

```{image} _static/optimize_alfven_qa.png
:alt: LP-QA continuum and global modes before and after moving them out of a window
```
