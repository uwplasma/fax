from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from fax.config import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_example_toml_files_load():
    for path in [
        REPO_ROOT / "examples" / "minimal.toml",
        REPO_ROOT / "examples" / "analytic_circular_tokamak.toml",
        REPO_ROOT / "examples" / "analytic_convergence.toml",
        REPO_ROOT / "examples" / "landreman_paul_qa_lowres.toml",
    ]:
        config = load_config(path)
        assert config.run.case


def test_density_sensitivity_example_runs():
    result = subprocess.run(
        [sys.executable, "examples/research/density_sensitivity.py"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    assert payload["mean_frequency_hz"] > 0.0
    assert payload["gradient"]["density_scale"] < 0.0
    assert payload["gradient"]["field_scale"] > 0.0


def test_inverse_iota_shift_example_runs():
    result = subprocess.run(
        [sys.executable, "examples/research/inverse_iota_shift.py"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    assert payload["final_iota_shift"] < 0.0
    assert payload["final_mean_frequency_hz"] > payload["initial_mean_frequency_hz"]
    assert len(payload["history"]) == 40


def test_global_gap_fastion_screen_example_runs():
    result = subprocess.run(
        [sys.executable, "examples/research/global_gap_fastion_screen.py"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    assert payload["constant_fixture_relative_omega2_error"] < 3.0e-6
    assert payload["constant_fixture_max_residual"] < 1.0e-5
    assert payload["lowest_radial_mode_frequency_hz"] > 0.0
    assert len(payload["gap_estimates"]) == 3
    assert payload["fast_ion_peak"]["drive"] > 0.0


def test_boozer_harmonic_validation_example_runs():
    result = subprocess.run(
        [sys.executable, "examples/research/boozer_harmonic_validation.py"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    assert payload["projection_error_surface_0"] < 2.0e-6
    assert payload["projection_error_surface_1"] < 2.0e-6
    assert payload["orthogonality_error"] < 2.0e-6
    assert payload["straight_field_line_phase_error"] < 1.0e-4
    assert 0.0 < payload["normalized_nonaxisymmetric_energy"] < 1.0
