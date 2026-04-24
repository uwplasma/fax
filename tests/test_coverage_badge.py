from __future__ import annotations

import json
from pathlib import Path

from tools.write_coverage_badge import coverage_percent, main, render_badge


def test_coverage_badge_writer(tmp_path: Path):
    coverage_json = tmp_path / "coverage.json"
    output_svg = tmp_path / "coverage.svg"
    coverage_json.write_text(json.dumps({"totals": {"percent_covered": 87.654}}), encoding="utf-8")

    assert coverage_percent(coverage_json) == 87.654
    assert main([str(coverage_json), str(output_svg)]) == 0

    content = output_svg.read_text(encoding="utf-8")
    assert "coverage" in content
    assert "87.7%" in content


def test_coverage_badge_color_thresholds():
    assert "#4c1" in render_badge(95.0)
    assert "#e05d44" in render_badge(10.0)
