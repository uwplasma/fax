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
Global eigenmodes solve the AE3D reduced ideal-MHD weak form (Spong, D'Azevedo
& Todo 2010) with radial finite elements, giving a Hermitian block-tridiagonal
problem $F y = \omega^2 G y$ that is solved near a target frequency by
shift-invert with a block Thomas factorization.
Eigenvalue derivatives use the Hellmann–Feynman identity
$\delta\lambda = x^\dagger(\delta K-\lambda\,\delta M)x$, and the VMEC boundary
enters through vmex's implicit-function derivative of the converged equilibrium.

| Module | Contents |
|---|---|
| `geometry` | Boozer geometry from `boozmn`, `wout`, a vmex solve, or an analytic torus |
| `continuum` | continuum matrices and frequencies, gaps, gap locations |
| `eigen` | AE3D global operator, block-tridiagonal shift-invert, differentiable eigenvalues |
| `drive` | perturbative fast-ion growth rate and ion Landau damping |
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
modes = fax.eigenmodes(geo, harmonics, rho, target_hz=384e3, count=6)
dfdrho = jax.grad(lambda r: fax.frequency_hz(geo, harmonics, r, 2, 3, target_hz=384e3, count=6))(rho)
```

`examples/boundary_gradient.py` differentiates global frequencies with respect
to VMEC boundary coefficients of the precise-QA configuration.

## Validation

| Gate | Result |
|---|---|
| Uniform field, $\omega = \lvert n-m\iota\rvert v_A/R$ | exact to 1e-9 |
| STELLGAP CTOK continuum (97 surfaces, n = 2, m = 0–12) | mean 4.6e-3, p95 1.2e-2 relative |
| Uniform field, global spectrum fills $[\omega_A^2(0), \omega_A^2(1)]$, singular modes | edges within $0.5/N$, width $\propto 1/N$ |
| AE3D CTOK global modes (same boozmn, 99 surfaces, n = 2, m = 0–12) | 6 modes within 1.4%; TAE at 338 kHz unresolved (see plan) |
| Könies et al. 2018 ITPA TAE (n = 6, m = 10, 11) | $\omega$ = 4.061e5 rad/s (CAS3D 4.01e5, GYGLES 4.13e5), converged to 1e-4 |
| Eigenvalue gradients vs. finite differences | 1e-5 (dense and shift-invert) |

The CTOK continuum takes 0.04 s per call after compilation on one CPU core
(STELLGAP: 0.8 s).

## Status

The local continuum and the AE3D global operator (without the pressure and
parallel-current terms) are in place. The drive is a perturbative
zero-orbit-width growth rate for passing fast ions (Könies 2018 TAE: 33×10³ s⁻¹
at 300 keV vs 38–43×10³ published). See [plan.md](plan.md).

## Documentation

[fax.readthedocs.io](https://fax.readthedocs.io) has the derivations and API.

## License

MIT.
