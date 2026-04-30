from __future__ import annotations

import json
import runpy
import sys

import pytest

from fax.cli import main
from fax.reference_cases import landreman_paul_qa_lowres


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


@pytest.mark.requires_vmec_jax
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
