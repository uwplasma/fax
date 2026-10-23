# FAX

[![Tests](https://github.com/uwplasma/fax/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/uwplasma/fax/actions/workflows/tests.yml)
[![Docs](https://readthedocs.org/projects/fax/badge/?version=latest)](https://fax.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

FAX is a lean, differentiable JAX code for shear-Alfvén physics in
stellarators. It computes shear-Alfvén continua, AE3D-form global Alfvén
eigenmodes and a perturbative fast-ion drive, and differentiates all of them
with respect to profiles and, through [vmex](https://github.com/uwplasma/vmex)
and [booz_xform_jax](https://github.com/uwplasma/booz_xform_jax), the VMEC
plasma boundary.

<p align="center">
  <img src="docs/_static/ctok_eigenmodes_continuum.png" width="44%" alt="CTOK continuum and global modes">
  <img src="docs/_static/koenies_tae_drive.png" width="48%" alt="Könies TAE drive">
</p>

*Left: the n = 2 continuum of the CTOK tokamak with the six global modes
nearest 384 kHz (red). Right: fast-ion growth rate of the Könies et al. (2018)
ITPA TAE against the published zero-Larmor-radius band.*

## What it does

- **Continuum.** On each Boozer surface, the Galerkin form of the low-β
  continuum equation (Spong, Sanchez & Weller 2003), $K x = \omega^2 M x$, with
  FFT-exact angle averages and STELLGAP weighting. Gaps give the TAE, EAE, MAE
  and HAE windows.
- **Global modes.** The AE3D weak form (Spong, D'Azevedo & Todo 2010) with the
  full metric and the magnetic-shear term, linear radial finite elements and
  axis regularity, giving a block-tridiagonal $F y = \omega^2 G y$ solved near
  a target by shift-invert with a block Thomas factorization ($O(N H^3)$).
- **Gradients.** Hellmann–Feynman eigenvalue derivatives
  $\delta\lambda = y^\dagger(\delta F - \lambda\,\delta G)y$, cluster means
  that stay smooth through crossings, and vmex's implicit-function derivative
  of the converged equilibrium.
- **Drive.** The perturbative growth rate $\gamma_h = -P/2W$ for zero-orbit-width
  passing fast ions (Maxwellian or slowing-down), with an exact resonance
  integral, plus bulk-ion Landau damping.

## Install

Python 3.10 or newer.

```bash
pip install uwplasma-fax              # core
pip install "uwplasma-fax[vmex]"      # VMEC solves and boundary gradients (Python >= 3.11)
```

## Run

From the command line, with a TOML case:

```bash
fax run examples/ctok_continuum.toml --plot     # results.npz, case.json, continuum.png
fax run examples/ctok_eigenmodes.toml --plot    # six global modes nearest 384 kHz
```

From Python:

```python
import jax, fax

geo = fax.read_boozmn("tests/data/boozmn_ctok.nc")
harmonics = fax.mode_family(2, 12)  # (m, n) for m = 0..12, n = 2
rho = fax.mass_density(1e19, ion_mass_amu=2.0)
f_hz = fax.continuum(geo, harmonics, rho)  # (surfaces, harmonics)
center_hz, width_hz = fax.gaps(f_hz)
modes = fax.eigenmodes(geo, harmonics, rho, target_hz=384e3, count=6)
dfdrho = jax.grad(lambda r: fax.frequency_hz(geo, harmonics, r, 2, 3, target_hz=384e3, count=6))(rho)
```

## Examples

| Example | What it shows |
|---|---|
| `examples/ctok_continuum.toml` | STELLGAP CTOK continuum from a `boozmn` file |
| `examples/ctok_eigenmodes.toml` | AE3D-form global modes by shift-invert |
| `examples/qa_continuum.toml` | precise QA: vmex solve → booz_xform_jax → continuum |
| `examples/boundary_gradient.py` | gradient of a frequency objective with respect to VMEC boundary coefficients |
| `examples/optimize_alfven.py` | QS + aspect + Paul's continuum-gap objective (QH) or a global-mode window (QA), `least_squares` with exact Jacobians |
| `examples/koenies_tae_drive.py` | fast-ion growth rate of the Könies ITPA TAE vs temperature |

## Validation

| Gate | Result |
|---|---|
| Uniform field, $\omega = \lvert n-m\iota\rvert v_A/R$ | exact to 1e-9 |
| STELLGAP CTOK continuum (97 surfaces, n = 2, m = 0–12) | mean 6.3e-3, p95 6.6e-3 relative |
| Uniform-field global spectrum fills $[\omega_A(0), \omega_A(1)]$ | edges within $0.5/N$, singular modes of width $\propto 1/N$ |
| AE3D CTOK global modes (n = 2, m = 0–12) | 6 modes within 1.4%; AE3D's TAE is off by its $\iota'$ sign error (checked by real-space quadrature to 0.3%) |
| Könies et al. 2018 ITPA TAE (n = 6) | $\omega$ = 4.061e5 rad/s (CAS3D 4.01e5, GYGLES 4.13e5), converged to 1e-4 |
| Könies ITPA with $p(s)$, vmex equilibrium, `finite_beta=True` | 3.947e5 rad/s (4.127e5 without the $J_\parallel$ and pressure terms) |
| Könies et al. 2018 drive, 100/200/300 keV | 4.4/16.6/33.1 vs 8–10/25–33/38–43 ×10³ s⁻¹ (ZOW, passing only) |
| Eigenvalue and drive gradients vs finite differences | 1e-5 |

The compiled CTOK continuum takes 0.04 s on one CPU core (STELLGAP 0.8 s); the
six-mode CTOK shift-invert takes 0.07 s. Not yet included: sound and
fast-wave coupling, trapped particles, finite orbit width and stellarator
drift sidebands in the drive. Details are on the
[validation page](https://fax.readthedocs.io/en/latest/validation.html).

## Documentation

[fax.readthedocs.io](https://fax.readthedocs.io): getting started and the TOML
reference, tutorials, full derivations, validation and the API.

## Testing

```bash
pip install -e ".[dev,vmex]"
python -m pytest -q
ruff check . && ruff format --check .
sphinx-build -W -b html docs docs/_build/html
```

## Contributing and citing

Issues and pull requests are welcome; every physics change needs a test against
an analytic result or a published number. If you use FAX, please cite this
repository and Spong, Sanchez & Weller (2003), Spong, D'Azevedo & Todo (2010)
and Könies et al. (2018); see the
[references](https://fax.readthedocs.io/en/latest/references.html).

## License

MIT.
