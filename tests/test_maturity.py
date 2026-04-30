from __future__ import annotations

from fax.maturity import (
    registered_research_grade_milestones,
    research_grade_ready,
    research_grade_report_markdown,
    research_grade_summary,
)


def test_research_grade_milestones_cover_required_open_lanes():
    milestone_ids = {milestone.milestone_id for milestone in registered_research_grade_milestones()}

    assert {
        "native_3d_ideal_mhd_parity",
        "kinetic_growth_rate_parity",
        "fully_calibrated_fast_ion_model",
        "end_to_end_vmec_boozer_optimization",
        "multi_geometry_benchmark_suite",
        "no_proxy_full_physics_gate",
    }.issubset(milestone_ids)


def test_research_grade_summary_is_honest_before_full_physics_completion():
    summary = research_grade_summary()

    assert summary["status"] == "not_complete"
    assert summary["ready"] is False
    assert research_grade_ready() is False
    assert summary["complete_count"] < summary["required_count"]
    assert all(milestone["blocking_items"] for milestone in summary["milestones"])


def test_research_grade_report_mentions_cross_code_and_design_blockers():
    report = research_grade_report_markdown()

    assert "Full Research-Grade Roadmap" in report
    assert "AE3D" in report
    assert "CAS3D" in report
    assert "FAR3D" in report
    assert "VMEC" in report
