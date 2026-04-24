from __future__ import annotations

from fax.benchmarks import time_callable


def test_time_callable_reports_positive_timings():
    result = time_callable("unit", lambda: 1 + 1, warmups=0, repeats=2)

    assert result.name == "unit"
    assert result.repeats == 2
    assert result.warmups == 0
    assert result.min_seconds >= 0.0
    assert result.max_seconds >= result.min_seconds

