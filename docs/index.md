# FAX

FAX is a lean, differentiable JAX code for shear-Alfvén physics in
stellarators. From a Boozer-coordinate equilibrium it computes

- the **shear-Alfvén continuum** on each flux surface and its TAE, EAE, MAE
  and HAE gaps;
- **global Alfvén eigenmodes** from the reduced ideal-MHD weak form of AE3D,
  with radial finite elements and a block-tridiagonal shift-invert solver;
- a **perturbative fast-ion drive** (power transfer over mode energy) and
  bulk-ion Landau damping;
- **derivatives** of all of these with respect to profiles and, through
  [vmex](https://github.com/uwplasma/vmex) and
  [booz_xform_jax](https://github.com/uwplasma/booz_xform_jax), with respect to
  the VMEC plasma boundary.

```console
pip install uwplasma-fax
fax run examples/ctok_continuum.toml --plot
```

```python
import jax, fax

geo = fax.read_boozmn("tests/data/boozmn_ctok.nc")
harmonics = fax.mode_family(2, 12)  # (m, n), m = 0..12, n = 2
rho = fax.mass_density(1e19, ion_mass_amu=2.0)
f_hz = fax.continuum(geo, harmonics, rho)  # (surfaces, harmonics)
modes = fax.eigenmodes(geo, harmonics, rho, target_hz=384e3, count=6)
dfdrho = jax.grad(lambda r: fax.frequency_hz(geo, harmonics, r, target_hz=384e3, count=6))(rho)
```

```{image} _static/ctok_eigenmodes_continuum.png
:alt: CTOK n = 2 continuum with the six global modes nearest 384 kHz
:width: 60%
```

| Page | Contents |
|---|---|
| {doc}`getting-started` | install, command line, Python, every TOML key |
| {doc}`tutorials` | the five examples, step by step, with their figures |
| {doc}`theory` | derivations: Boozer metric, continuum, gaps, AE3D weak form, solver, gradients, drive, objectives |
| {doc}`validation` | every gate with its numbers, and what is not done |
| {doc}`api` | API reference |
| {doc}`development` | tests, contributing, citing |
| {doc}`references` | papers FAX implements or compares with |

```{toctree}
:hidden:
:maxdepth: 2

getting-started
tutorials
theory
validation
api
development
references
```
