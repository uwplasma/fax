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
        "global_3d_block_constant_spectrum",
        "global_3d_block_neumann_spectrum",
        "global_3d_block_density_gradient",
        "global_3d_block_eigenvalue_sensitivity",
        "global_3d_block_custom_jvp_eigenvalue",
        "boozer_global_custom_jvp_design_gradient",
        "global_3d_block_degenerate_cluster_mean",
        "boozer_global_cluster_mean_design_gradient",
        "analytic_tae_gap_center",
        "basic_gap_map_ordering",
        "bae_temperature_scaling",
        "fast_ion_resonance_peak",
        "fast_ion_eigenfunction_drive_localization",
        "fast_ion_gyro_landau_closure_signs",
        "fast_ion_gyro_landau_calibration",
        "boozer_harmonic_projection",
        "boozer_straight_field_line_phase",
        "boozer_constant_field_continuum",
        "boozer_sideband_coupling_hermitian",
        "boozer_jacobian_identity",
        "boozer_metric_constant_field_continuum",
        "boozer_metric_mass_coupling",
        "boozer_3d_constant_field_continuum",
    ]

    for gate_id in gate_ids:
        result = run_validation_gate(gate_id)
        assert result.status == "passed"
        assert math.isfinite(result.metric)
        assert result.metric <= result.tolerance


@pytest.mark.requires_booz_xform_jax
def test_boozmn_validation_gates_pass_or_skip_cleanly():
    for gate_id in (
        "boozmn_geometry_jacobian_identity",
        "booz_xform_jax_live_geometry_adapter",
        "booz_xform_jax_in_memory_geometry_adapter",
        "boozmn_embedding_jacobian_shape",
        "boozmn_3d_continuum_finite",
        "boozmn_global_3d_eigenmode_finite",
        "boozmn_metric_radial_stiffness_hermitian",
        "boozmn_global_angular_convergence",
        "boozmn_global_radial_surface_coarsening",
        "boozmn_global_boundary_and_stiffness_variants",
    ):
        result = run_validation_gate(gate_id)

        assert result.status in {"passed", "skipped"}
        assert result.message


def test_cross_code_validation_gates_pass_or_skip_cleanly():
    for gate_id in (
        "stellgap_ctok_local_continuum",
        "stellgap_ctok_largest_gap_parity",
        "stellgap_public_postprocess_parity",
        "ae3d_public_eigenvalue_artifact",
        "ae3d_public_omega2_eigenvalue_parity",
        "ae3d_public_matrix_symmetry",
        "ae3d_public_matrix_eigenmode_parity",
        "ae3d_public_dense_spectrum_parity",
        "ae3d_public_global_mode_shape_parity",
    ):
        result = run_validation_gate(gate_id)

        assert result.status in {"passed", "skipped"}
        assert result.message


@pytest.mark.requires_vmec_jax
def test_landreman_paul_validation_gate_passes_or_skips_cleanly():
    for gate_id in (
        "landreman_paul_adapter_smoke",
        "landreman_paul_vmec_jax_boozer_in_memory",
        "landreman_paul_vmec_jax_boozer_truncation_convergence",
        "landreman_paul_vmec_jax_boozer_operator_angular_convergence",
        "landreman_paul_vmec_jax_solve_convergence",
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
