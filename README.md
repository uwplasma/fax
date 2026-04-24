# fax

`fax` is a planned end-to-end differentiable JAX code for Alfvén eigenmodes,
fast-ion drive, continuum gaps, and toroidal/helical coupling in stellarators.

The project is intentionally coupled to the UWPlasma JAX ecosystem:

- `vmec_jax` supplies differentiable MHD equilibria and VMEC-compatible output.
- `booz_xform_jax` supplies Boozer-coordinate transforms when the selected
  model needs straight-field-line Boozer geometry.
- `fax` supplies Alfvén-continuum, global-eigenmode, and fast-ion stability
  models with validation artifacts suitable for design studies.

This first repository state is a research-grade scaffold: packaging, CLI,
docs, CI/CD, examples, tests, and the detailed development plan are in place.
Physics kernels will be added behind the interfaces described in `plan.md`.

## Install

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev]"
```

The package supports Python 3.10 and newer. On Python 3.10 it uses `tomli` as
the TOML parser fallback; on Python 3.11+ it uses the standard-library
`tomllib`.

## CLI

```bash
fax examples/minimal.toml --dry-run
fax examples/minimal.toml --dump-config
```

`fax` reads TOML configuration files. The current CLI validates and reports the
planned run configuration; solver implementations will be added incrementally.

## Repository Layout

- `plan.md`: literature survey, physics goals, architecture, validation plan,
  milestones, and running handoff log.
- `src/fax`: importable Python package and CLI.
- `docs`: Sphinx/ReadTheDocs source.
- `examples`: runnable TOML examples.
- `tests`: unit tests for the scaffold.
- `.github/workflows`: tests, docs, and PyPI publishing workflows.

## Current Scope

The initial implementation deliberately avoids pretending the full physics model
exists. It establishes:

- installable package metadata with unpinned library dependencies,
- a TOML-driven `fax` command,
- ReadTheDocs-ready documentation,
- GitHub Actions for tests, docs, and PyPI publishing,
- unit tests for configuration loading and the CLI,
- a handoff plan grounded in the existing literature and comparison codes.

