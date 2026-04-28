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


def test_differentiable_design_study_example_runs(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "examples/research/differentiable_design_study.py",
            "--output-dir",
            str(tmp_path),
            "--format",
            "png",
            "--steps",
            "8",
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["final_objective"] < payload["initial_objective"]
    assert len(payload["frequency_jacobian_hz_per_parameter"]) == 4
    assert max(payload["initial_net_growth_hz"]) > max(payload["optimized_net_growth_hz"])
    assert all(Path(path).exists() for path in payload["figure_files"])


def test_basic_alfven_gap_map_example_runs(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "examples/research/basic_alfven_gap_map.py",
            "--output-dir",
            str(tmp_path),
            "--format",
            "png",
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    assert payload["tae_gap"]["center_frequency_hz"] > payload["bae_gap"]["accumulation_frequency_hz"]
    assert payload["eae_gap"]["center_frequency_hz"] > payload["tae_gap"]["center_frequency_hz"]
    assert Path(payload["figure_files"][0]).exists()


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


def test_boozmn_3d_continuum_example_runs(tmp_path):
    fixture = Path("/Users/rogerio/local/booz_xform_jax/tests/test_files/boozmn_LandremanSenguptaPlunk_section5p3.nc")
    if not fixture.exists():
        return

    result = subprocess.run(
        [
            sys.executable,
            "examples/research/boozmn_3d_continuum.py",
            "--boozmn",
            str(fixture),
            "--output-dir",
            str(tmp_path),
            "--theta-points",
            "24",
            "--zeta-points",
            "24",
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    assert payload["jacobian_identity_error"] < 2.0e-5
    assert payload["embedding_jacobian_shape_error"] < 5.0e-2
    assert payload["min_frequency_hz"] >= 0.0
    assert payload["max_frequency_hz"] > payload["min_frequency_hz"]
    assert all(Path(path).exists() for path in payload["figure_files"])


def test_boozmn_global_3d_eigenmodes_example_runs(tmp_path):
    fixture = Path("/Users/rogerio/local/booz_xform_jax/tests/test_files/boozmn_LandremanSenguptaPlunk_section5p3.nc")
    if not fixture.exists():
        return

    result = subprocess.run(
        [
            sys.executable,
            "examples/research/boozmn_global_3d_eigenmodes.py",
            "--boozmn",
            str(fixture),
            "--output-dir",
            str(tmp_path),
            "--theta-points",
            "12",
            "--zeta-points",
            "12",
            "--modes",
            "3",
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    assert payload["max_residual"] < 2.0e-5
    assert len(payload["frequencies_hz"]) == 3
    assert payload["mode_structure_shape"][1] == payload["surface_count"]
    assert all(Path(path).exists() for path in payload["figure_files"])


def test_boozmn_differentiable_design_study_example_runs(tmp_path):
    fixture = Path("/Users/rogerio/local/booz_xform_jax/tests/test_files/boozmn_LandremanSenguptaPlunk_section5p3.nc")
    if not fixture.exists():
        return

    result = subprocess.run(
        [
            sys.executable,
            "examples/research/boozmn_differentiable_design_study.py",
            "--boozmn",
            str(fixture),
            "--output-dir",
            str(tmp_path),
            "--format",
            "png",
            "--steps",
            "5",
            "--theta-points",
            "8",
            "--zeta-points",
            "8",
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["final_objective"] < payload["initial_objective"]
    assert max(payload["initial_net_growth_hz"]) > max(payload["optimized_net_growth_hz"])
    assert len(payload["frequency_jacobian_hz_per_parameter"]) == 4
    assert all(Path(path).exists() for path in payload["figure_files"])


def test_stellgap_ctok_comparison_example_runs(tmp_path):
    boozmn = Path("/Users/rogerio/local/Stellgap_test_inputs/boozmn_ctok.nc")
    alfven_post = Path("/Users/rogerio/local/Stellgap_test_problem/alfven_post")
    if not boozmn.exists() or not alfven_post.exists():
        return

    result = subprocess.run(
        [
            sys.executable,
            "examples/research/stellgap_ctok_comparison.py",
            "--boozmn",
            str(boozmn),
            "--alfven-post",
            str(alfven_post),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    assert payload["matched_surfaces"] >= 90
    assert payload["p95_relative_error"] < 4.0e-2
    assert all(Path(path).exists() for path in payload["figure_files"])


def test_external_code_artifact_dashboard_example_runs(tmp_path):
    stellgap_dir = Path("/Users/rogerio/local/Stellgap_test_problem")
    ae3d_dir = Path("/Users/rogerio/local/AE3D_test_problem")
    required = [
        stellgap_dir / "alfven_spec",
        stellgap_dir / "alfven_post",
        ae3d_dir / "omega2.dat",
        ae3d_dir / "egn_values.dat",
        ae3d_dir / "a_matrix.dat",
        ae3d_dir / "b_matrix.dat",
    ]
    if not all(path.exists() for path in required):
        return

    result = subprocess.run(
        [
            sys.executable,
            "examples/research/external_code_artifact_dashboard.py",
            "--stellgap-dir",
            str(stellgap_dir),
            "--ae3d-dir",
            str(ae3d_dir),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    assert payload["stellgap_postprocess"]["max_relative_error"] < 1.0e-5
    assert payload["ae3d_omega2_eigenvalue"]["max_relative_error"] < 1.0e-5
    assert payload["ae3d_a_matrix"]["relative_asymmetry"] < 1.0e-5
    assert payload["ae3d_b_matrix"]["relative_asymmetry"] < 1.0e-12
    assert all(Path(path).exists() for path in payload["figures"])


def test_publication_figure_suite_selected_cases_run(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "examples/research/publication_figure_suite.py",
            "--case",
            "01",
            "--case",
            "06",
            "--case",
            "08",
            "--format",
            "png",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["case_count"] == 3
    assert payload["completed_count"] == 3
    assert payload["skipped_count"] == 0
    assert all(Path(path).exists() for result in payload["results"] for path in result["figures"])


def test_publication_suite_external_parity_cases_run_if_artifacts_exist(tmp_path):
    stellgap_required = [
        Path("/Users/rogerio/local/Stellgap_test_inputs/boozmn_ctok.nc"),
        Path("/Users/rogerio/local/Stellgap_test_problem/alfven_post"),
    ]
    ae3d_required = [
        Path("/Users/rogerio/local/AE3D_test_problem/a_matrix.dat"),
        Path("/Users/rogerio/local/AE3D_test_problem/b_matrix.dat"),
        Path("/Users/rogerio/local/AE3D_test_problem/egn_mode_asci.dat"),
    ]
    if not all(path.exists() for path in (*stellgap_required, *ae3d_required)):
        return

    result = subprocess.run(
        [
            sys.executable,
            "examples/research/publication_figure_suite.py",
            "--case",
            "09",
            "--case",
            "10",
            "--format",
            "png",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    by_case = {result["case"]: result for result in payload["results"]}

    assert payload["completed_count"] == 2
    assert by_case["09_stellgap_ctok_gap_anchored_global_screen"]["stellgap_p95_relative_error"] < 4.0e-2
    assert by_case["09_stellgap_ctok_gap_anchored_global_screen"]["max_residual"] < 1.0e-10
    assert by_case["10_ae3d_matrix_eigenmode_parity"]["max_residual"] < 5.0e-6
    assert all(Path(path).exists() for result in payload["results"] for path in result["figures"])
