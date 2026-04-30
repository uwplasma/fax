from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

from fax.config import FaxConfig
from fax.progress import ProgressReporter, format_seconds
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
    assert format_seconds(3700.0) == "1h 1m"


def test_progress_reporter_plain_text_fallback_reports_status_and_summary():
    stream = StringIO()
    config = FaxConfig.from_mapping({"run": {"case": "fallback_progress"}})
    reporter = ProgressReporter(enabled=True, stream=stream, total_estimate_seconds=2.0)
    reporter._rich_available = False

    reporter.banner(config)
    reporter.phase_start("assemble operator", detail="radial_points=8", estimate_seconds=1.0)
    reporter.log("residual=1e-8")
    record = reporter.phase_done(status="skipped")
    reporter.summary(SimpleNamespace(status="completed", message="fallback checked"))

    output = stream.getvalue()
    assert record.status == "skipped"
    assert "FAX run" in output
    assert "Solve parameters" in output
    assert "START assemble operator" in output
    assert "residual=1e-8" in output
    assert "DONE  assemble operator" in output
    assert "Execution phases" in output
    assert "fallback checked" in output
