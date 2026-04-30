from __future__ import annotations

from fax.benchmark_manifest import (
    benchmark_manifest_markdown,
    benchmark_manifest_summary,
    default_benchmark_manifest,
)


def test_default_benchmark_manifest_has_current_external_cases():
    entries = default_benchmark_manifest()
    ids = {entry.benchmark_id for entry in entries}

    assert "stellgap_ctok_local_continuum_gap" in ids
    assert "ae3d_public_matrix_mode_shape" in ids
    assert all(entry.validation_gate_ids for entry in entries)
    assert all(entry.expected_metrics for entry in entries)


def test_benchmark_manifest_summary_is_json_serializable():
    summary = benchmark_manifest_summary(include_hashes=False)

    assert summary["entry_count"] >= 2
    assert isinstance(summary["all_required_available"], bool)
    for entry in summary["entries"]:
        assert "available" in entry
        assert entry["claim_level"]
        assert all("exists" in artifact for artifact in entry["artifacts"])


def test_benchmark_manifest_markdown_mentions_claim_boundaries():
    report = benchmark_manifest_markdown()

    assert "Benchmark Manifest" in report
    assert "STELLGAP" in report
    assert "AE3D" in report
    assert "not native FAX operator parity" in report
