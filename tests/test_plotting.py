from __future__ import annotations

import json
import xml.etree.ElementTree as ET

from fax.cli import main
from fax.config import EquilibriumConfig, FaxConfig, GridConfig, RunConfig
from fax.plotting import load_continuum_artifacts, plot_continuum_artifacts
from fax.workflows import run_continuum


def _artifact_config(output_dir) -> FaxConfig:
    return FaxConfig(
        run=RunConfig(case="plotting", model="continuum", output_dir=str(output_dir)),
        equilibrium=EquilibriumConfig(source="analytic"),
        grid=GridConfig(radial_points=10, m_max=2, n_max=2, field_periods=1),
    )


def test_plot_continuum_artifacts_from_workflow_outputs(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    figure_dir = tmp_path / "figures"
    run_continuum(_artifact_config(artifact_dir), progress=False)

    artifacts = load_continuum_artifacts(artifact_dir)
    assert artifacts.summary["case"] == "plotting"
    assert "s" in artifacts.branches

    paths = plot_continuum_artifacts(artifact_dir, output_dir=figure_dir, stem="plotting_check")
    assert {path.suffix for path in paths} == {".svg", ".png"}
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 1000
    ET.parse(figure_dir / "plotting_check.svg")


def test_cli_plot_continuum_json(tmp_path, capsys):
    artifact_dir = tmp_path / "artifacts"
    figure_dir = tmp_path / "figures"
    run_continuum(_artifact_config(artifact_dir), progress=False)

    assert (
        main(
            [
                "plot-continuum",
                str(artifact_dir),
                "--output-dir",
                str(figure_dir),
                "--stem",
                "cli_plot",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    assert len(payload["figures"]) == 2
    assert (figure_dir / "cli_plot.svg").exists()
    assert (figure_dir / "cli_plot.png").exists()
