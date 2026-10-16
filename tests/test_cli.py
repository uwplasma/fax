from __future__ import annotations

import json
import runpy
import sys

import numpy as np
import pytest

from fax.boozer import read_boozmn_geometry
from fax.cli import main
from fax.global_modes import radial_block_problem_from_boozer_geometry, solve_radial_block_eigenmodes
from fax.modes import stellarator_mode_family
from fax.reference_cases import EXTERNAL_ROOT, landreman_paul_qa_lowres


BOOZ_XFORM_TEST_FILES = EXTERNAL_ROOT / "booz_xform_jax/tests/test_files"


def test_cli_example(capsys):
    assert main(["--example"]) == 0
    captured = capsys.readouterr()
    assert "[run]" in captured.out
    assert "model = \"continuum\"" in captured.out


def test_cli_dry_run(tmp_path, capsys):
    config_path = tmp_path / "case.toml"
    config_path.write_text(
        """
[run]
case = "cli"
model = "continuum"
""",
        encoding="utf-8",
    )

    assert main([str(config_path), "--dry-run", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["case"] == "cli"
    assert payload["status"] == "planned"
    assert payload["phases"]


def test_cli_dry_run_default_prints_progress(tmp_path, capsys):
    config_path = tmp_path / "case.toml"
    config_path.write_text(
        """
[run]
case = "progress"
model = "continuum"
""",
        encoding="utf-8",
    )

    assert main([str(config_path), "--dry-run"]) == 0
    captured = capsys.readouterr()
    assert "FAX dry run" in captured.out
    assert "Solve parameters" in captured.out
    assert "estimated remaining" in captured.out
    assert "Execution phases" in captured.out


def test_cli_quiet_prints_json_only(tmp_path, capsys):
    config_path = tmp_path / "case.toml"
    config_path.write_text("[run]\ncase = \"quiet\"\n", encoding="utf-8")

    assert main([str(config_path), "--quiet"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["case"] == "quiet"


def test_cli_dump_config(tmp_path, capsys):
    config_path = tmp_path / "case.toml"
    config_path.write_text("[run]\ncase = \"dump\"\n", encoding="utf-8")

    assert main([str(config_path), "--dump-config"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run"]["case"] == "dump"


def test_cli_validate_json_writes_artifacts(tmp_path, capsys):
    output_dir = tmp_path / "validation_outputs"
    config_path = tmp_path / "case.toml"
    config_path.write_text(
        f"""
[run]
case = "validate_cli"
model = "continuum"
output_dir = "{output_dir}"
""",
        encoding="utf-8",
    )

    assert main(["validate", str(config_path), "--tag", "analytic", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["case"] == "validate_cli"
    assert payload["status"] == "passed"
    assert payload["gate_count"] >= 1
    assert payload["failed_count"] == 0
    assert (output_dir / "validation_summary.json").exists()
    assert (output_dir / "validation_report.md").exists()


def test_cli_maturity_json_reports_not_complete(capsys):
    assert main(["maturity", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "not_complete"
    assert payload["ready"] is False
    ids = {milestone["milestone_id"] for milestone in payload["milestones"]}
    assert "native_3d_ideal_mhd_parity" in ids


def test_cli_benchmark_manifest_json_reports_current_cases(capsys):
    assert main(["benchmark-manifest", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    ids = {entry["benchmark_id"] for entry in payload["entries"]}
    assert "stellgap_ctok_local_continuum_gap" in ids
    assert "ae3d_public_matrix_mode_shape" in ids


def test_cli_kinetic_parity_reports_held_out_growth_rate_gate(tmp_path, capsys):
    dataset_path = tmp_path / "kinetic_dataset.json"
    predictions_path = tmp_path / "predictions.json"
    dataset_path.write_text(
        json.dumps(
            {
                "dataset_id": "cli_kinetic_fixture",
                "cases": [
                    {
                        "case_id": "far3d_calibration",
                        "source_code": "FAR3D",
                        "geometry_id": "qa",
                        "mode_id": "n1",
                        "frequency_hz": 80_000.0,
                        "growth_rate_hz": 100.0,
                        "growth_rate_uncertainty_hz": 10.0,
                        "toroidal_mode": 1,
                        "split": "calibration",
                    },
                    {
                        "case_id": "cas3dk_validation",
                        "source_code": "CAS3D-K",
                        "geometry_id": "qa",
                        "mode_id": "n2",
                        "frequency_hz": 120_000.0,
                        "growth_rate_hz": -20.0,
                        "growth_rate_uncertainty_hz": 5.0,
                        "toroidal_mode": 2,
                        "split": "validation",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    predictions_path.write_text(json.dumps({"far3d_calibration": 104.0, "cas3dk_validation": -21.0}), encoding="utf-8")

    assert main(["kinetic-parity", str(dataset_path), str(predictions_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "passed"
    assert payload["validation_count"] == 1
    assert payload["validation_sign_accuracy"] == 1.0


def test_cli_far3d_ingest_writes_kinetic_dataset(tmp_path, capsys):
    far3d_output = tmp_path / "far3d.out"
    dataset_path = tmp_path / "far3d_dataset.json"
    far3d_output.write_text(
        """
 psi     : m=   4 n=   1 gam=  1.00000E-03 om_r=  4.00000E-02
 nfast   : m=   5 n=   1 gam= -2.00000D-03 om_r=  5.00000D-02
""",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "far3d-ingest",
                str(far3d_output),
                "--dataset-id",
                "cli_far3d",
                "--geometry-id",
                "ctok",
                "--frequency-scale-hz",
                "2000000",
                "--growth-rate-scale-hz",
                "1000000",
                "--variable",
                "psi",
                "--output",
                str(dataset_path),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    written = json.loads(dataset_path.read_text(encoding="utf-8"))
    assert payload == written
    assert payload["dataset_id"] == "cli_far3d"
    assert len(payload["cases"]) == 1
    assert payload["cases"][0]["frequency_hz"] == 80_000.0
    assert payload["cases"][0]["growth_rate_hz"] == 1_000.0


def test_cli_validate_default_prints_progress(tmp_path, capsys):
    config_path = tmp_path / "case.toml"
    config_path.write_text("[run]\ncase = \"validate_progress\"\n", encoding="utf-8")

    assert main(["validate", str(config_path), "--tag", "analytic", "--no-artifacts"]) == 0
    captured = capsys.readouterr()
    assert "FAX run" in captured.out
    assert "Run validation gates" in captured.out
    assert "gate 1/" in captured.out
    assert "START analytic_cylinder_frequency" in captured.out
    assert "DONE  analytic_cylinder_frequency" in captured.out
    assert "analytic_cylinder_frequency" in captured.out
    assert "Execution phases" in captured.out


def test_cli_continuum_json_writes_artifacts(tmp_path, capsys):
    output_dir = tmp_path / "continuum_outputs"
    config_path = tmp_path / "case.toml"
    config_path.write_text(
        f"""
[run]
case = "continuum_cli"
model = "continuum"
output_dir = "{output_dir}"

[equilibrium]
source = "analytic"

[grid]
radial_points = 12
m_max = 2
n_max = 2
field_periods = 1
""",
        encoding="utf-8",
    )

    assert main(["continuum", str(config_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["case"] == "continuum_cli"
    assert payload["status"] == "completed"
    assert payload["radial_points"] == 12
    assert payload["harmonic_count"] >= 1
    assert payload["metric_residual_p95"] < 1.0e-5
    assert (output_dir / "continuum_summary.json").exists()
    assert (output_dir / "continuum_metrics.json").exists()
    branches = output_dir / "continuum_branches.csv"
    assert branches.exists()
    assert len(branches.read_text(encoding="utf-8").splitlines()) == 13


def test_cli_eigenmodes_json_writes_global_artifacts(tmp_path, capsys):
    output_dir = tmp_path / "global_outputs"
    config_path = tmp_path / "case.toml"
    config_path.write_text(
        f"""
[run]
case = "global_cli"
model = "global_eigenmode"
output_dir = "{output_dir}"

[equilibrium]
source = "analytic"

[grid]
radial_points = 12
m_max = 2
n_max = 1
field_periods = 1
""",
        encoding="utf-8",
    )

    assert main(["eigenmodes", str(config_path), "--max-modes", "5", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    artifact_dir = output_dir / "eigenmodes"
    assert payload["case"] == "global_cli"
    assert payload["status"] == "completed"
    assert payload["operator"] == "profile_radial_envelope"
    assert payload["reported_mode_count"] == 5
    assert payload["residual_p95"] < 1.0e-5
    assert (artifact_dir / "global_eigenmode_summary.json").exists()
    assert (artifact_dir / "global_eigenmodes.csv").exists()
    assert (artifact_dir / "global_mode_structures.csv").exists()
    assert len((artifact_dir / "global_eigenmodes.csv").read_text(encoding="utf-8").splitlines()) == 6


@pytest.mark.requires_booz_xform_jax
def test_cli_eigenmodes_real_boozmn_artifacts_match_direct_block_solve(tmp_path, capsys):
    boozmn = BOOZ_XFORM_TEST_FILES / "boozmn_LandremanSenguptaPlunk_section5p3.nc"
    if not boozmn.exists():
        pytest.skip(f"missing local booz_xform_jax fixture: {boozmn}")

    output_dir = tmp_path / "boozmn_global_outputs"
    config_path = tmp_path / "boozmn_global.toml"
    config_path.write_text(
        f"""
[run]
case = "boozmn_global_cli"
model = "global_eigenmode"
output_dir = "{output_dir}"

[equilibrium]
source = "wout"
use_boozer = true
boozmn = "{boozmn}"

[grid]
radial_points = 5
m_max = 2
n_max = 1
field_periods = 3

[physics]
density_scale = 1.0
""",
        encoding="utf-8",
    )

    assert main(["eigenmodes", str(config_path), "--max-modes", "4", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    artifact_dir = output_dir / "eigenmodes"

    assert payload["case"] == "boozmn_global_cli"
    assert payload["operator"] == "boozmn_global_3d_radial_block"
    assert payload["geometry"]["source"] == "boozmn"
    assert payload["reported_mode_count"] == 4
    assert payload["harmonic_count"] == 2
    assert payload["residual_p95"] < 2.0e-5
    assert (artifact_dir / "global_eigenmode_summary.json").exists()
    assert (artifact_dir / "global_eigenmode_metrics.json").exists()
    assert (artifact_dir / "global_mode_structures.csv").exists()

    geometry = read_boozmn_geometry(boozmn)
    harmonics = stellarator_mode_family(
        poloidal_modes=range(1, 3),
        base_n=1,
        field_periods=geometry.field_periods,
        sidebands=(0,),
    ).as_array()
    problem = radial_block_problem_from_boozer_geometry(
        geometry,
        harmonics,
        mass_density=3.4e-8,
        radial_stiffness_method="metric",
        theta_points=16,
        zeta_points=16,
        boundary_condition="dirichlet",
    )
    direct = solve_radial_block_eigenmodes(problem)
    modes_csv = np.loadtxt(artifact_dir / "global_eigenmodes.csv", delimiter=",", skiprows=1)
    np.testing.assert_allclose(modes_csv[:4, 2], np.asarray(direct.frequencies_hz[:4]), rtol=1.0e-10, atol=1.0e-6)
    np.testing.assert_allclose(modes_csv[:4, 3], np.asarray(direct.residuals[:4]), rtol=1.0e-10, atol=1.0e-12)


def test_cli_convergence_json_writes_artifacts(tmp_path, capsys):
    output_dir = tmp_path / "convergence_outputs"
    config_path = tmp_path / "case.toml"
    config_path.write_text(
        f"""
[run]
case = "convergence_cli"
model = "continuum"
output_dir = "{output_dir}"

[equilibrium]
source = "analytic"

[grid]
radial_points = 10
m_max = 2
n_max = 2
field_periods = 1
""",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "convergence",
                str(config_path),
                "--angular-points",
                "8",
                "--angular-points",
                "12",
                "--mode-max",
                "1",
                "--mode-max",
                "2",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["case"] == "convergence_cli"
    assert payload["status"] == "completed"
    assert payload["radial_points"] == [10]
    assert len(payload["records"]) == 4
    assert payload["reference"]["theta_points"] == 12
    assert payload["reference"]["radial_points"] == 10
    assert (output_dir / "convergence" / "convergence_summary.json").exists()
    assert (output_dir / "convergence" / "convergence_metrics.csv").exists()
    assert (output_dir / "convergence" / "convergence_report.md").exists()


def test_cli_convergence_radial_sweep_json(tmp_path, capsys):
    output_dir = tmp_path / "convergence_radial_outputs"
    config_path = tmp_path / "case.toml"
    config_path.write_text(
        f"""
[run]
case = "convergence_radial_cli"
model = "continuum"
output_dir = "{output_dir}"

[equilibrium]
source = "analytic"

[grid]
radial_points = 10
m_max = 1
n_max = 1
field_periods = 1
""",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "convergence",
                str(config_path),
                "--radial-points",
                "8",
                "--radial-points",
                "12",
                "--angular-points",
                "8",
                "--mode-max",
                "1",
                "--json",
                "--no-artifacts",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["case"] == "convergence_radial_cli"
    assert payload["status"] == "completed"
    assert payload["radial_points"] == [8, 12]
    assert payload["reference"]["radial_points"] == 12
    assert len(payload["records"]) == 2
    assert {record["radial_points"] for record in payload["records"]} == {8, 12}
    assert all(record["reference_radial_points_used"] == 12 for record in payload["records"])


def test_cli_benchmark_json_writes_artifacts(tmp_path, capsys):
    output_dir = tmp_path / "benchmark_outputs"
    config_path = tmp_path / "case.toml"
    config_path.write_text(
        f"""
[run]
case = "benchmark_cli"
model = "continuum"
output_dir = "{output_dir}"

[equilibrium]
source = "analytic"

[grid]
radial_points = 8
m_max = 1
n_max = 1
field_periods = 1
""",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "benchmark",
                str(config_path),
                "--repeats",
                "1",
                "--warmups",
                "0",
                "--family",
                "global-block",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["case"] == "benchmark_cli"
    assert payload["status"] == "completed"
    assert payload["warm_result"]["repeats"] == 1
    families = {result["family"]: result for result in payload["family_results"]}
    assert families["continuum"]["gate_passed"]
    assert families["global-block"]["gate_passed"]
    assert payload["artifact_size_bytes"] > 0
    assert (output_dir / "benchmark" / "benchmark_summary.json").exists()
    assert (output_dir / "benchmark" / "benchmark_metrics.csv").exists()


def test_cli_rejects_missing_top_level_config(capsys):
    assert main([str("__missing_fax_case__.toml"), "--json"]) == 2

    captured = capsys.readouterr()
    assert "configuration file not found" in captured.err


def test_cli_rejects_missing_subcommand_config(capsys):
    assert main(["validate", str("__missing_fax_case__.toml"), "--json"]) == 2

    captured = capsys.readouterr()
    assert "configuration file not found" in captured.err


def test_cli_requires_config_when_no_example_is_requested():
    with pytest.raises(SystemExit) as excinfo:
        main([])

    assert excinfo.value.code == 2


def test_module_entrypoint_prints_example(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["python -m fax", "--example"])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("fax.__main__", run_name="__main__")

    assert excinfo.value.code == 0
    assert "[run]" in capsys.readouterr().out


def test_cli_plot_commands_have_plain_output_and_error_contracts(tmp_path, capsys):
    artifact_dir = tmp_path / "artifacts"
    figure_dir = tmp_path / "figures"
    run_config = tmp_path / "case.toml"
    run_config.write_text(
        f"""
[run]
case = "plot_plain_cli"
model = "continuum"
output_dir = "{artifact_dir}"

[equilibrium]
source = "analytic"

[grid]
radial_points = 8
m_max = 1
n_max = 1
field_periods = 1
""",
        encoding="utf-8",
    )
    assert main(["continuum", str(run_config), "--json"]) == 0
    capsys.readouterr()

    assert (
        main(
            [
                "plot-continuum",
                str(artifact_dir),
                "--output-dir",
                str(figure_dir),
                "--stem",
                "plain_continuum",
                "--format",
                "svg",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "FAX continuum artifact figures" in captured.out
    assert (figure_dir / "plain_continuum.svg").exists()

    assert main(["plot-continuum", str(tmp_path / "does_not_exist")]) == 2
    assert "missing continuum artifact" in capsys.readouterr().err

    assert (
        main(
            [
                "convergence",
                str(run_config),
                "--angular-points",
                "8",
                "--mode-max",
                "1",
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    convergence_dir = artifact_dir / "convergence"
    assert (
        main(
            [
                "plot-convergence",
                str(convergence_dir),
                "--output-dir",
                str(figure_dir),
                "--stem",
                "plain_convergence",
                "--format",
                "svg",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "FAX convergence artifact figures" in captured.out
    assert (figure_dir / "plain_convergence.svg").exists()

    assert main(["plot-convergence", str(tmp_path / "does_not_exist")]) == 2
    assert "missing convergence artifact" in capsys.readouterr().err


def test_cli_benchmark_rejects_empty_repeat_sweep(tmp_path, capsys):
    config_path = tmp_path / "case.toml"
    config_path.write_text(
        """
[run]
case = "benchmark_invalid"
model = "continuum"

[equilibrium]
source = "analytic"

[grid]
radial_points = 8
m_max = 1
n_max = 1
field_periods = 1
""",
        encoding="utf-8",
    )

    assert main(["benchmark", str(config_path), "--repeats", "0", "--json"]) == 2
    assert "repeats must be at least 1" in capsys.readouterr().err


@pytest.mark.requires_vmex
def test_cli_continuum_wout_landreman_qa_resamples_geometry(tmp_path, capsys):
    case = landreman_paul_qa_lowres()
    if case.wout is None or not case.wout.exists():
        pytest.skip(f"missing local Landreman-Paul QA wout: {case.wout}")

    output_dir = tmp_path / "lpqa_wout_outputs"
    config_path = tmp_path / "lpqa.toml"
    config_path.write_text(
        f"""
[run]
case = "lpqa_wout_coverage"
model = "continuum"
output_dir = "{output_dir}"

[equilibrium]
source = "wout"
wout = "{case.wout}"

[grid]
radial_points = 8
m_max = 1
n_max = 1
field_periods = 2

[physics]
density_scale = 1.1
""",
        encoding="utf-8",
    )

    assert main(["continuum", str(config_path), "--json", "--no-artifacts"]) == 0
    payload = json.loads(capsys.readouterr().out)
    profile_metadata = dict(payload["geometry"]["profile_metadata"])
    metric_metadata = dict(payload["geometry"]["metric_metadata"])

    assert payload["geometry"]["source"] == "vmec_wout"
    assert payload["radial_points"] == 8
    assert payload["metric_residual_p95"] < 1.0e-4
    assert profile_metadata["radial_adapter"] == "linear_s_interpolation"
    assert metric_metadata["radial_adapter"] == "linear_s_interpolation"
    assert payload["profile_max_frequency_hz"] > payload["profile_min_frequency_hz"]
