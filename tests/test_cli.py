from __future__ import annotations

import json

from fax.cli import main


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

    assert main(["benchmark", str(config_path), "--repeats", "1", "--warmups", "0", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["case"] == "benchmark_cli"
    assert payload["status"] == "completed"
    assert payload["warm_result"]["repeats"] == 1
    assert payload["artifact_size_bytes"] > 0
    assert (output_dir / "benchmark" / "benchmark_summary.json").exists()
    assert (output_dir / "benchmark" / "benchmark_metrics.csv").exists()
