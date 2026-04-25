from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import pytest

from fax.cli import main
from fax.config import EquilibriumConfig, FaxConfig, GridConfig, RunConfig
from fax.gap_models import basic_alfven_gap_map
from fax.plotting import (
    load_continuum_artifacts,
    load_convergence_artifacts,
    plot_basic_alfven_gap_map,
    plot_continuum_artifacts,
    plot_convergence_artifacts,
)
from fax.workflows import run_continuum, run_convergence


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


def test_plot_convergence_artifacts_from_workflow_outputs(tmp_path):
    artifact_root = tmp_path / "artifacts"
    figure_dir = tmp_path / "figures"
    run_convergence(
        _artifact_config(artifact_root),
        progress=False,
        radial_points=(8, 10),
        angular_points=(8, 12),
        mode_max_values=(1, 2),
    )
    convergence_dir = artifact_root / "convergence"

    artifacts = load_convergence_artifacts(convergence_dir)
    assert artifacts.summary["case"] == "plotting"
    assert "frequency_l2_error_vs_reference" in artifacts.metrics

    paths = plot_convergence_artifacts(convergence_dir, output_dir=figure_dir, stem="convergence_check")
    assert {path.suffix for path in paths} == {".svg", ".png"}
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 1000
    ET.parse(figure_dir / "convergence_check.svg")


def test_cli_plot_convergence_json(tmp_path, capsys):
    artifact_root = tmp_path / "artifacts"
    figure_dir = tmp_path / "figures"
    run_convergence(
        _artifact_config(artifact_root),
        progress=False,
        radial_points=(8, 10),
        angular_points=(8, 12),
        mode_max_values=(1, 2),
    )

    assert (
        main(
            [
                "plot-convergence",
                str(artifact_root / "convergence"),
                "--output-dir",
                str(figure_dir),
                "--stem",
                "cli_convergence_plot",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    assert len(payload["figures"]) == 2
    assert (figure_dir / "cli_convergence_plot.svg").exists()
    assert (figure_dir / "cli_convergence_plot.png").exists()


def test_basic_gap_map_plot_is_literature_ordered_and_renderable(tmp_path):
    demo = basic_alfven_gap_map()

    paths = plot_basic_alfven_gap_map(tmp_path, formats=("svg", "png", "pdf"))

    assert {path.suffix for path in paths} == {".svg", ".png", ".pdf"}
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 1000
    ET.parse(tmp_path / "basic_alfven_gap_map.svg")
    assert demo.bae_frequency_at_surface_hz < demo.tae.estimate.center_frequency_hz
    assert demo.tae.estimate.center_frequency_hz < demo.eae.estimate.center_frequency_hz
    assert demo.tae.estimate.width_frequency_hz > 0.0
    assert demo.eae.estimate.width_frequency_hz > 0.0


def test_plotting_rejects_missing_and_malformed_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing continuum artifact"):
        load_continuum_artifacts(tmp_path / "missing_continuum")
    with pytest.raises(FileNotFoundError, match="missing convergence artifact"):
        load_convergence_artifacts(tmp_path / "missing_convergence")

    continuum_dir = tmp_path / "bad_continuum"
    continuum_dir.mkdir()
    (continuum_dir / "continuum_summary.json").write_text('{"case": "bad"}', encoding="utf-8")
    (continuum_dir / "continuum_metrics.json").write_text("{}", encoding="utf-8")
    (continuum_dir / "continuum_branches.csv").write_text("s,branch_hz\n0.0,1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="profile_"):
        plot_continuum_artifacts(continuum_dir, output_dir=tmp_path / "bad_figures")

    convergence_dir = tmp_path / "bad_convergence"
    convergence_dir.mkdir()
    (convergence_dir / "convergence_summary.json").write_text('{"records": []}', encoding="utf-8")
    (convergence_dir / "convergence_metrics.csv").write_text("metric,value\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not contain any records"):
        plot_convergence_artifacts(convergence_dir, output_dir=tmp_path / "bad_convergence_figures")


def test_plotting_artifact_contract_handles_sparse_records_and_bad_formats(tmp_path):
    convergence_dir = tmp_path / "sparse_convergence"
    convergence_dir.mkdir()
    (convergence_dir / "convergence_summary.json").write_text(
        json.dumps(
            {
                "case": "sparse",
                "records": [
                    "ignored record from a corrupt writer",
                    {
                        "radial_points": 8,
                        "theta_points": 8,
                        "m_max": 1,
                        "branch_count": 2,
                        "frequency_l2_error_vs_reference": 1.0e-2,
                        "mode_truncation_l2_error_vs_global_reference": 4.0e-2,
                        "residual_p95": 0.0,
                        "accepted": True,
                    },
                    {
                        "radial_points": 10,
                        "theta_points": 12,
                        "m_max": 2,
                        "branch_count": 4,
                        "frequency_l2_error_vs_reference": 1.0e-3,
                        "mode_truncation_l2_error_vs_global_reference": 2.0e-3,
                        "residual_p95": 3.0e-10,
                        "accepted": True,
                    },
                ],
                "reference": {"radial_points": 10, "theta_points": 12, "m_max": 2},
            }
        ),
        encoding="utf-8",
    )
    (convergence_dir / "convergence_metrics.csv").write_text(
        "radial_points,theta_points,m_max,frequency_l2_error_vs_reference\n8,8,1,0.01\n",
        encoding="utf-8",
    )

    paths = plot_convergence_artifacts(convergence_dir, output_dir=tmp_path / "sparse_figures", formats=("svg",))
    assert paths == (tmp_path / "sparse_figures" / "convergence_artifact_dashboard.svg",)
    ET.parse(paths[0])

    with pytest.raises(ValueError, match="unsupported figure format"):
        plot_basic_alfven_gap_map(tmp_path, formats=("eps",))

    convergence_dir_no_header = tmp_path / "no_header"
    convergence_dir_no_header.mkdir()
    (convergence_dir_no_header / "convergence_summary.json").write_text('{"records": []}', encoding="utf-8")
    (convergence_dir_no_header / "convergence_metrics.csv").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="has no header"):
        load_convergence_artifacts(convergence_dir_no_header)
