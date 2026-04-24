# FAX Research Examples

These scripts are small, reproducible entry points for users who want to learn
the current FAX scaffold and generate validation artifacts themselves.

Run them from the repository root after installing the development extras:

```bash
python -m pip install -e ".[dev]"
python examples/research/validation_packet.py
python examples/research/density_sensitivity.py
python examples/research/inverse_iota_shift.py
python examples/research/basic_alfven_gap_map.py
python examples/research/global_gap_fastion_screen.py
python examples/research/boozer_harmonic_validation.py
```

The examples intentionally use the analytic and Landreman-Paul low-resolution
paths that are already covered by local tests and validation gates. They are
research workflow templates, not final 3-D stellarator eigenmode validations.

## Basic Alfvén Gap Map

```bash
python examples/research/basic_alfven_gap_map.py
```

This writes `outputs/basic_alfven_gap_map/basic_alfven_gap_map.png` and `.svg`.
It is the shortest visual introduction to the current gap model: cylindrical
continuum branches, a TAE avoided crossing, an EAE avoided crossing, and a
low-frequency BAE accumulation marker. The JSON output reports the reference
`q`, radial surface, center frequency, and width/accumulation frequency.
