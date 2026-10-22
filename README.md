# FAX

[![Tests](https://github.com/uwplasma/fax/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/uwplasma/fax/actions/workflows/tests.yml)
[![Docs](https://readthedocs.org/projects/fax/badge/?version=latest)](https://fax.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

FAX computes shear-Alfvén continua and Alfvén eigenmodes of stellarators in
JAX, so that frequencies, gaps and fast-ion resonance screens can be
differentiated with respect to the plasma boundary and profiles. It sits
downstream of [vmex](https://github.com/uwplasma/vmex) and
[booz_xform_jax](https://github.com/uwplasma/booz_xform_jax).

## What it does

On each flux surface FAX solves the low-beta continuum equation in Boozer
coordinates (Spong, Sanchez & Weller 2003) as the Hermitian problem
$K x = \omega^2 M x$ with

$$
K_{ij}=\Big\langle \tfrac{|\sqrt g|\,|\nabla s|^2}{\mu_0}\,k_{\parallel i}k_{\parallel j}\,e_i^*e_j\Big\rangle,\qquad
M_{ij}=\Big\langle \tfrac{\rho\,|\sqrt g|\,|\nabla s|^2}{B^2}\,e_i^*e_j\Big\rangle,\qquad
k_{\parallel j}=\frac{(n_j-m_j\iota)\,|B|}{G+\iota I}.
$$

Gaps between branches give the TAE, EAE, MAE and HAE frequency windows.
Eigenvalue derivatives use the Hellmann–Feynman identity
$\delta\lambda = x^\dagger(\delta K-\lambda\,\delta M)x$, and the VMEC boundary
enters through vmex's implicit-function derivative of the converged equilibrium.

| Module | Contents |
|---|---|
| `geometry` | Boozer geometry from `boozmn`, `wout`, a vmex solve, or an analytic torus |
| `continuum` | continuum matrices and frequencies, gaps, gap locations |
| `eigen` | global radial operator, differentiable eigenvalues |
| `drive` | fast-ion resonance screen |
| `objectives` | smooth objectives and the VMEC-boundary design pipeline |
| `io`, `plot` | TOML cases, `fax` command, figures |

## Install

```bash
pip install uwplasma-fax            # core
pip install "uwplasma-fax[vmex]"    # VMEC solves and boundary gradients
```

## Run

```bash
fax run examples/ctok_continuum.toml --plot
```

```python
import jax, fax

geo = fax.read_boozmn("tests/data/boozmn_ctok.nc")
harmonics = fax.mode_family(2, 12)  # (m, n) for m = 0..12, n = 2
rho = fax.mass_density(1e19, ion_mass_amu=2.0)
f_hz = fax.continuum(geo, harmonics, rho)  # (surfaces, harmonics)
center_hz, width_hz = fax.gaps(f_hz)
dfdrho = jax.grad(lambda r: fax.frequency_hz(geo, harmonics, r))(rho)
```

`examples/boundary_gradient.py` differentiates global frequencies with respect
to VMEC boundary coefficients of the precise-QA configuration.

## Validation

| Gate | Result |
|---|---|
| Uniform field, $\omega = \lvert n-m\iota\rvert v_A/R$ | exact to 1e-9 |
| STELLGAP CTOK continuum (97 surfaces, n = 2, m = 0–12) | mean 4.6e-3, p95 1.2e-2 relative |
| Constant-coefficient global spectrum | exact to 1e-12 |
| Eigenvalue and boundary gradients vs. finite differences | 1e-5 (eigen), 1e-4 (vmex) |

The CTOK continuum takes 0.04 s per call after compilation on one CPU core
(STELLGAP: 0.8 s).

## Status

The local continuum is complete. The global operator is an interim radial
envelope without continuum resonance, and the drive is a ranking screen, not a
calibrated growth rate. Both are being replaced by the AE3D weak form and a
perturbative energetic-particle growth rate; see [plan.md](plan.md).

## Documentation

[fax.readthedocs.io](https://fax.readthedocs.io) has the derivations and API.

## License

MIT.
