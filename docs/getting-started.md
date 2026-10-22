# Getting started

## Install

FAX needs Python 3.10 or newer.

```console
pip install uwplasma-fax              # core: JAX, NumPy, netCDF4
pip install "uwplasma-fax[vmex]"      # VMEC solves and boundary gradients (Python >= 3.11)
pip install "uwplasma-fax[plot]"      # matplotlib for figures
```

From a clone, `pip install -e ".[dev,vmex]"` installs everything used by the
tests and the docs. FAX switches JAX to float64 on import: Alfvén operators are
stiff, and their eigenvalues are not reliable in float32.

## Command line

```console
fax run case.toml            # writes <run.output>/results.npz and case.json
fax run case.toml --plot     # also continuum.png (and modes.png for eigenmodes)
fax plot outputs/case        # figures for an existing results folder
```

`results.npz` always holds `s`, `iota` and `harmonics`. A `continuum` run adds
`frequencies_hz` (surfaces × harmonics), `gap_center_hz` and `gap_width_hz`
(the widest gap on each surface). An `eigenmodes` run adds `frequencies_hz`,
`structures` (modes × surfaces × harmonics), `mode_s` and `continuum_hz`.

## Python

The same steps from Python ({func}`fax.run` takes a TOML path or a dict):

```python
import fax

out = fax.run("examples/ctok_continuum.toml", save=False)  # dict of NumPy arrays
config = fax.load({"modes": {"n": [2], "m_max": 12}})  # defaults merged, unknown keys rejected
```

Every function below `run` is a pure JAX function, so `jax.grad`, `jax.jit`
and `jax.vmap` apply; see {doc}`tutorials`.

## TOML reference

A case has four tables. Every key is optional and takes the default below
(`fax.io.DEFAULTS`); unknown tables or keys raise `ValueError`. Relative
`geometry.path` and `run.output` are resolved against the case file.

### `[geometry]`

| Key | Default | Meaning |
|---|---|---|
| `source` | `"circular"` | `"boozmn"` (BOOZ_XFORM file), `"wout"` (VMEC file through booz_xform_jax), `"vmec"` (VMEC input solved by vmex, then booz_xform_jax), `"circular"` ({func}`fax.circular_torus` with its defaults) |
| `path` | none | file for `boozmn`, `wout` and `vmec` |
| `mboz`, `nboz` | `8`, `8` | Boozer resolution for `wout` and `vmec` |
| `surfaces` | `[0.1, ..., 0.9]` | surfaces `s` for `wout` and `vmec` |
| `ns` | none | `vmec` only: one stage at `ns` surfaces; none runs the input's `NS_ARRAY` ladder |
| `ftol` | `1e-12` | `vmec` only: force tolerance of that stage |
| `max_iter` | `5000` | `vmec` only: iteration limit of that stage |

### `[modes]`

| Key | Default | Meaning |
|---|---|---|
| `n` | `[1]` | toroidal mode numbers (a Boozer family `n0 + k Nfp` is a list) |
| `m_max` | `8` | largest poloidal mode number |
| `m_min` | `0` | smallest poloidal mode number |

The harmonics are every `(m, n)` with `m_min <= m <= m_max` and `n` in the list
({func}`fax.mode_family`).

### `[plasma]`

| Key | Default | Meaning |
|---|---|---|
| `density` | `1e19` | ion density in m⁻³ (uniform) |
| `ion_mass` | `1.0` | ion mass in proton masses |

### `[run]`

| Key | Default | Meaning |
|---|---|---|
| `task` | `"continuum"` | `"continuum"` or `"eigenmodes"` |
| `output` | `"outputs"` | results folder |
| `ntheta`, `nzeta` | `32`, `32` | angle grid for the surface averages |
| `target_khz` | `0.0` | `eigenmodes`: shift-invert target; `0` solves densely for all modes |
| `count` | `8` | `eigenmodes`: number of modes kept |
