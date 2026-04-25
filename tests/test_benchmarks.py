from __future__ import annotations

from fax.benchmarks import time_callable


def test_time_callable_reports_positive_timings():
    result = time_callable("unit", lambda: 1 + 1, warmups=0, repeats=2)

    assert result.name == "unit"
    assert result.repeats == 2
    assert result.warmups == 0
    assert result.min_seconds >= 0.0
    assert result.max_seconds >= result.min_seconds


def test_time_callable_blocks_nested_jax_like_results():
    class ReadyValue:
        def __init__(self) -> None:
            self.calls = 0

        def block_until_ready(self):
            self.calls += 1
            return self

    ready_values = [ReadyValue(), ReadyValue(), ReadyValue()]

    def physics_kernel():
        return {"continuum": [ready_values[0], ready_values[1]], "global_mode": ready_values[2]}

    result = time_callable("nested", physics_kernel, warmups=1, repeats=2)

    assert result.name == "nested"
    assert result.repeats == 2
    assert result.warmups == 1
    assert [value.calls for value in ready_values] == [3, 3, 3]
