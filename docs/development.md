# Development

## Layout

| File | Contents |
|---|---|
| `src/fax/geometry.py` | Boozer geometry, readers, the vmex/booz_xform_jax path, metric grids |
| `src/fax/continuum.py` | continuum matrices and frequencies, gaps |
| `src/fax/eigen.py` | AE3D operator, block solver, Hellmann–Feynman derivatives |
| `src/fax/drive.py` | perturbative fast-ion drive, Landau damping |
| `src/fax/objectives.py` | smooth objectives, VMEC-boundary design |
| `src/fax/io.py`, `plot.py` | TOML cases, `fax` command, figures |
| `tests/` | one file per module; each test is an equation or a published number |
| `examples/` | the cases of {doc}`tutorials` |

## Tests and checks

```console
pip install -e ".[dev,vmex]"
python -m pytest -q                                  # about 1 min on CPU
ruff check . && ruff format --check .
python -m coverage run --source=src/fax -m pytest -q && python -m coverage report --fail-under=90
sphinx-build -W -b html docs docs/_build/html
```

CI runs the suite on Python 3.10–3.13, on the oldest supported JAX (0.4.30 on
Python 3.10), and the vmex path, lint, coverage and docs on Python 3.12. Keep
Python 3.10 working (the `tomli` fallback, no newer-JAX-only functions).

## Contributing

- Keep the code small: add physics to the existing modules rather than new
  layers.
- Every new physics claim needs a test against an analytic result or a
  published number, and an entry in {doc}`validation`.
- Every equation in {doc}`theory` must match the code; update both together.
- Figures in `docs/_static` stay compressed (under 200 kB each).

## Citing

If you use FAX, please cite the repository
(<https://github.com/uwplasma/fax>) and the papers whose models it implements:
Spong, Sanchez & Weller (2003) for the continuum, Spong, D'Azevedo & Todo
(2010) for the global operator, and Könies et al. (2018) for the TAE
benchmark ({doc}`references`).

## License

MIT.
