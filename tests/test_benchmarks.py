from __future__ import annotations

from dataclasses import dataclass

import pytest

from fax.benchmarks import known_benchmark_families, run_named_benchmark_family, time_callable


def test_time_callable_reports_positive_timings():
    result = time_callable("unit", lambda: 1 + 1, warmups=0, repeats=2)

    assert result.name == "unit"
    assert result.repeats == 2
    assert result.warmups == 0
    assert result.min_seconds >= 0.0
    assert result.max_seconds >= result.min_seconds


def test_time_callable_blocks_nested_jax_like_results():
    @dataclass(frozen=True)
    class KernelPayload:
        value: object

    class ReadyValue:
        def __init__(self) -> None:
            self.calls = 0

        def block_until_ready(self):
            self.calls += 1
            return self

    ready_values = [ReadyValue(), ReadyValue(), ReadyValue()]

    def physics_kernel():
        return {"continuum": [ready_values[0], ready_values[1]], "global_mode": KernelPayload(ready_values[2])}

    result = time_callable("nested", physics_kernel, warmups=1, repeats=2)

    assert result.name == "nested"
    assert result.repeats == 2
    assert result.warmups == 1
    assert [value.calls for value in ready_values] == [3, 3, 3]


def test_named_benchmark_family_runs_physics_gate():
    assert "analytic-local" in known_benchmark_families()

    result = run_named_benchmark_family("analytic-local", warmups=0, repeats=1)

    assert result.family == "analytic-local"
    assert result.status == "completed"
    assert result.gate_passed
    assert result.metric_value is not None
    assert result.metric_value < result.tolerance
    assert result.cold_seconds is not None
    assert result.warm_result is not None
    assert result.warm_result.repeats == 1


def test_named_benchmark_family_rejects_invalid_controls():
    with pytest.raises(ValueError, match="repeats"):
        run_named_benchmark_family("analytic-local", warmups=0, repeats=0)
    with pytest.raises(ValueError, match="warmups"):
        run_named_benchmark_family("analytic-local", warmups=-1, repeats=1)
    with pytest.raises(ValueError, match="unknown benchmark family"):
        run_named_benchmark_family("not-a-family", warmups=0, repeats=1)


@pytest.mark.parametrize("family", ["global-block", "ae3d-artifact", "stellgap-ctok"])
def test_named_benchmark_family_physics_parity_or_clean_skip(family: str):
    result = run_named_benchmark_family(family, warmups=0, repeats=1)

    assert result.family == family
    assert result.status in {"completed", "skipped"}
    assert result.physics_metric
    assert result.process_max_rss_bytes is not None
    assert result.environment["jax_backend"]
    if result.status == "completed":
        assert result.gate_passed
        assert result.metric_value is not None
        assert result.tolerance is not None
        assert result.metric_value < result.tolerance
        assert result.cold_seconds is not None
        assert result.warm_result is not None
    else:
        assert not result.gate_passed
        assert result.metric_value is None
        assert "reason" in result.details
