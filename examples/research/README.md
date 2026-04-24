# FAX Research Examples

These scripts are small, reproducible entry points for users who want to learn
the current FAX scaffold and generate validation artifacts themselves.

Run them from the repository root after installing the development extras:

```bash
python -m pip install -e ".[dev]"
python examples/research/validation_packet.py
python examples/research/density_sensitivity.py
python examples/research/inverse_iota_shift.py
```

The examples intentionally use the analytic and Landreman-Paul low-resolution
paths that are already covered by local tests and validation gates. They are
research workflow templates, not final 3-D stellarator eigenmode validations.
