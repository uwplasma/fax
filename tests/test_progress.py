from __future__ import annotations

from io import StringIO

from fax.config import FaxConfig
from fax.progress import format_seconds
from fax.runner import run


def test_python_run_progress_output_contains_phase_information():
    stream = StringIO()
    summary = run(FaxConfig.from_mapping({"run": {"case": "python_progress"}}), stream=stream, dry_run=True)
    output = stream.getvalue()

    assert summary.case == "python_progress"
    assert summary.phases
    assert "FAX dry run" in output
    assert "estimated remaining" in output
    assert "Execution phases" in output


def test_python_run_can_disable_progress_output():
    stream = StringIO()
    summary = run(FaxConfig.from_mapping({"run": {"case": "silent"}}), stream=stream, progress=False)

    assert summary.case == "silent"
    assert summary.phases
    assert stream.getvalue() == ""


def test_format_seconds():
    assert format_seconds(0.2) == "<1s"
    assert format_seconds(2.25) == "2.2s"
    assert format_seconds(125.0) == "2m 5s"

