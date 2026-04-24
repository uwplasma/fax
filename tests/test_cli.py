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
