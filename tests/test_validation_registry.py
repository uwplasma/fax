from __future__ import annotations

import math

import pytest

from fax.validation import registered_validation_gates, run_validation_gate, run_validation_suite


def test_validation_registry_has_unique_gate_ids_and_metadata():
    gates = registered_validation_gates()
    gate_ids = [gate.gate_id for gate in gates]

    assert len(gates) >= 9
    assert len(gate_ids) == len(set(gate_ids))
    for gate in gates:
        assert gate.title
        assert gate.layer
        assert gate.quantity
        assert gate.command.startswith("python")
        assert gate.tags


def test_core_validation_gates_pass():
    gate_ids = [
        "analytic_cylinder_frequency",
        "metric_matches_profile_limit",
        "metric_matrix_hermitian",
        "metric_eigensolve_residual",
        "density_scale_gradient",
        "metric_angular_convergence",
        "metric_radial_convergence",
        "analytic_radial_integral_convergence",
        "global_mode_constant_profile",
        "global_mode_density_gradient",
        "analytic_tae_gap_center",
        "basic_gap_map_ordering",
        "bae_temperature_scaling",
        "fast_ion_resonance_peak",
        "boozer_harmonic_projection",
        "boozer_straight_field_line_phase",
        "boozer_constant_field_continuum",
        "boozer_sideband_coupling_hermitian",
    ]

    for gate_id in gate_ids:
        result = run_validation_gate(gate_id)
        assert result.status == "passed"
        assert math.isfinite(result.metric)
        assert result.metric <= result.tolerance


@pytest.mark.requires_vmec_jax
def test_landreman_paul_validation_gate_passes_or_skips_cleanly():
    for gate_id in (
        "landreman_paul_adapter_smoke",
        "landreman_paul_continuum_finite",
        "landreman_paul_radial_sampling_convergence",
    ):
        result = run_validation_gate(gate_id)

        assert result.status in {"passed", "skipped"}
        assert result.message


def test_validation_suite_tag_filter():
    results = run_validation_suite(tags=("metric",), include_skipped=False)

    assert results
    assert {result.status for result in results} <= {"passed", "failed"}
    assert any(result.gate_id == "metric_matches_profile_limit" for result in results)
