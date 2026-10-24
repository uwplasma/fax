# Development

## Layout

| File | Contents |
|---|---|
| `src/fax/geometry.py` | Boozer geometry, readers, the vmex/booz_xform_jax path, metric grids |
| `src/fax/continuum.py` | continuum matrices and frequencies, gaps |
| `src/fax/eigen.py` | AE3D operator, a thin wrapper of `solvax.generalized_eigenpairs`, `Status`, derivative rejection |
| `src/fax/drive.py` | perturbative fast-ion drive, Landau damping |
| `src/fax/objectives.py` | smooth objectives, VMEC-boundary design |
| `src/fax/io.py`, `plot.py` | TOML cases, `fax` command, figures |
| `tests/` | one file per module; each test is an equation or a published number |
| `examples/` | the cases of {doc}`tutorials` |

## Tests and checks

```console
pip install -e ".[dev,vmex]"
python -m pytest -q                                  # fast suite, about 1 min on CPU
python -m pytest -q -m slow                          # vmex end-to-end tests, several minutes
python -m pytest -q -m scheduled -s                  # resolution budget and gauge floors (about 30 min); weekly workflow
ruff check . && ruff format --check .
python -m coverage run --branch --source=src/fax -m pytest -q -m '' && python -m coverage report -m
sphinx-build -W -b html docs docs/_build/html
```

CI runs the suite on Python 3.10–3.13, on the oldest supported JAX (0.4.38 on
Python 3.10, the SOLVAX floor), and the vmex path, lint, coverage and docs on Python 3.12. SOLVAX comes
from PyPI (0.32.1 or newer). Keep
Python 3.10 working (the `tomli` fallback, no newer-JAX-only functions).

Coverage is measured with branches on the full suite (fast and slow tests,
`src/fax` only, no excluded lines or modules): statements 100% (705/705) and
branches 100% (72/72). CI fails if either falls below 95%, checked separately.
The fast suite alone reaches 94% of statements, because the vmex paths
(`from_vmec_input`, `prepare_design`, `design_geometry`) run only in the slow
tests.

## Reference fixtures

`tests/data/PROVENANCE.md` lists every external reference (identifier, URL, SHA-256, model switches, units, license,
generation command, tolerance). The Knyazev et al. (2026) precise-QA fixture (0.67 MB) is rebuilt from the Zenodo archive
(DOI 10.5281/zenodo.21299907, 1.1 GB, CC-BY-4.0) by `examples/knyazev_qa.py`: download `Knyazev_JPP2026.zip` from
zenodo.org, set `ARCHIVE` and `REBUILD = True` at the top of the script and run it. The archive is data, never code: the
script reads members with `zipfile` and parses them as text. The same script holds the comparison helpers the test uses
and, with `STUDY = True`, regenerates the tables and the figure of {doc}`validation`.

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
