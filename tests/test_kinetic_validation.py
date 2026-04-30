from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fax.far3d import (
    far3d_lincheck_convergence,
    far3d_lincheck_parameter_scan,
    far3d_rows_to_kinetic_dataset,
    load_far3d_lincheck_dataset,
    read_far3d_lincheck,
)
from fax.kinetic_validation import (
    KineticGrowthRateCase,
    KineticGrowthRateDataset,
    kinetic_growth_rate_loss,
    kinetic_growth_rate_parity,
    kinetic_growth_rate_parity_markdown,
    load_kinetic_growth_rate_dataset,
    load_kinetic_growth_rate_predictions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FAR3D_GPU_ARTIFACT = REPO_ROOT / "external_artifacts/far3d/gpu_diiid_nvhpc/farprt0000"
FAR3D_GPU_DATASET = REPO_ROOT / "external_artifacts/far3d/gpu_diiid_nvhpc/kinetic_dataset.json"
FAR3D_GPU_CONVERGENCE_DIR = REPO_ROOT / "external_artifacts/far3d/gpu_diiid_nvhpc_convergence"
FAR3D_GPU_BETA_SCAN_DIR = REPO_ROOT / "external_artifacts/far3d/gpu_diiid_nvhpc_beta_scan"


def _dataset() -> KineticGrowthRateDataset:
    return KineticGrowthRateDataset(
        dataset_id="far3d_cas3dk_contract_fixture",
        citation="synthetic contract fixture for FAR3D/CAS3D-K data ingestion",
        cases=(
            KineticGrowthRateCase(
                case_id="far3d_train_core",
                source_code="FAR3D",
                geometry_id="qa_lowres",
                mode_id="n1_core_tae",
                frequency_hz=72_000.0,
                growth_rate_hz=120.0,
                growth_rate_uncertainty_hz=12.0,
                toroidal_mode=1,
                split="calibration",
                resonance_radius=0.42,
                distribution_parameters={"beta_fast": 0.01, "v_alpha_over_va": 1.1},
            ),
            KineticGrowthRateCase(
                case_id="cas3dk_train_edge",
                source_code="CAS3D-K",
                geometry_id="qa_lowres",
                mode_id="n1_edge_eae",
                frequency_hz=118_000.0,
                growth_rate_hz=-35.0,
                growth_rate_uncertainty_hz=8.0,
                toroidal_mode=1,
                split="train",
                resonance_radius=0.78,
            ),
            KineticGrowthRateCase(
                case_id="far3d_validation_core",
                source_code="FAR3D",
                geometry_id="heldout_qh",
                mode_id="n2_core_tae",
                frequency_hz=91_000.0,
                growth_rate_hz=80.0,
                growth_rate_uncertainty_hz=10.0,
                toroidal_mode=2,
                split="validation",
                resonance_radius=0.50,
            ),
            KineticGrowthRateCase(
                case_id="cas3dk_validation_stable",
                source_code="CAS3D_K",
                geometry_id="heldout_qh",
                mode_id="n2_stable_gap",
                frequency_hz=146_000.0,
                growth_rate_hz=-25.0,
                growth_rate_uncertainty_hz=5.0,
                toroidal_mode=2,
                split="holdout",
                resonance_radius=0.63,
            ),
        ),
    )


def test_kinetic_growth_rate_dataset_enforces_reviewer_contracts():
    dataset = _dataset()

    assert dataset.case_ids == (
        "far3d_train_core",
        "cas3dk_train_edge",
        "far3d_validation_core",
        "cas3dk_validation_stable",
    )
    assert int(jnp.sum(dataset.calibration_mask)) == 2
    assert int(jnp.sum(dataset.validation_mask)) == 2
    assert dataset.cases[1].split == "calibration"
    assert dataset.cases[3].source_code == "CAS3D-K"
    assert dict(dataset.cases[0].distribution_parameters)["beta_fast"] == 0.01

    with pytest.raises(ValueError, match="uncertainty"):
        KineticGrowthRateCase(
            case_id="bad",
            source_code="FAR3D",
            geometry_id="geom",
            mode_id="mode",
            frequency_hz=10.0,
            growth_rate_hz=1.0,
            growth_rate_uncertainty_hz=0.0,
            toroidal_mode=1,
        )
    with pytest.raises(ValueError, match="case_id"):
        KineticGrowthRateDataset(cases=(dataset.cases[0], dataset.cases[0]))


def test_kinetic_growth_rate_case_rejects_nonphysical_reference_metadata():
    base = {
        "case_id": "case",
        "source_code": "FAR3D",
        "geometry_id": "geom",
        "mode_id": "mode",
        "frequency_hz": 10.0,
        "growth_rate_hz": 1.0,
        "growth_rate_uncertainty_hz": 0.5,
        "toroidal_mode": 1,
    }

    invalid_edits = [
        ({"case_id": ""}, "case_id"),
        ({"geometry_id": ""}, "geometry_id"),
        ({"mode_id": ""}, "mode_id"),
        ({"frequency_hz": 0.0}, "frequency_hz"),
        ({"growth_rate_hz": float("nan")}, "growth_rate_hz"),
        ({"resonance_radius": 1.2}, "resonance_radius"),
        ({"source_code": "UNKNOWN"}, "source_code"),
        ({"split": "bad-split"}, "split"),
    ]
    for edit, message in invalid_edits:
        with pytest.raises(ValueError, match=message):
            KineticGrowthRateCase(**(base | edit))


def test_kinetic_growth_rate_parity_uses_held_out_cases_and_uncertainty_weights():
    dataset = _dataset()
    predicted = jnp.asarray([132.0, -31.0, 86.0, -28.0])

    parity = kinetic_growth_rate_parity(
        dataset,
        predicted,
        max_validation_normalized_rms_error=1.0,
        min_validation_count=2,
        min_validation_sign_accuracy=1.0,
    )

    assert parity.passed
    assert parity.status == "passed"
    assert parity.calibration_count == 2
    assert parity.validation_count == 2
    np.testing.assert_allclose(parity.validation_normalized_rms_error, np.sqrt((0.6**2 + 0.6**2) / 2.0))
    assert parity.validation_sign_accuracy == 1.0

    failed = kinetic_growth_rate_parity(dataset, jnp.asarray([132.0, -31.0, -86.0, 28.0]), min_validation_count=2)
    assert not failed.passed
    assert failed.status == "failed"
    assert failed.validation_sign_accuracy == 0.0


def test_kinetic_growth_rate_parity_markdown_reports_reviewer_metrics():
    dataset = _dataset()
    parity = kinetic_growth_rate_parity(dataset, jnp.asarray([132.0, -31.0, 86.0, -28.0]), min_validation_count=2)

    report = kinetic_growth_rate_parity_markdown(parity)

    assert "# Kinetic Growth-Rate Parity" in report
    assert "Validation normalized RMS error" in report
    assert "| `far3d_validation_core` |" in report
    assert "86" in report


def test_kinetic_growth_rate_loss_is_differentiable():
    observed = jnp.asarray([10.0, -4.0, 2.0])
    uncertainty = jnp.asarray([2.0, 1.0, 0.5])

    def loss(scale):
        return kinetic_growth_rate_loss(scale * observed, observed, uncertainty)

    gradient = jax.grad(loss)(1.1)
    expected = 2.0 * 0.1 * float(jnp.mean((observed / uncertainty) ** 2))

    np.testing.assert_allclose(float(gradient), expected, rtol=1.0e-12)


def test_kinetic_growth_rate_dataset_and_predictions_load_from_json(tmp_path: Path):
    dataset = _dataset()
    dataset_path = tmp_path / "kinetic_dataset.json"
    predictions_path = tmp_path / "predictions.json"
    dataset_path.write_text(json.dumps(dataset.to_dict()), encoding="utf-8")
    predictions_path.write_text(
        json.dumps({"predictions": {case.case_id: case.growth_rate_hz for case in dataset.cases}}),
        encoding="utf-8",
    )

    loaded = load_kinetic_growth_rate_dataset(dataset_path)
    predictions = load_kinetic_growth_rate_predictions(predictions_path, dataset=loaded)
    parity = kinetic_growth_rate_parity(loaded, predictions, min_validation_count=2)

    assert loaded.dataset_id == dataset.dataset_id
    assert parity.passed
    np.testing.assert_allclose(np.asarray(predictions), np.asarray(dataset.growth_rate_hz), rtol=1.0e-12)

    missing_path = tmp_path / "missing_predictions.json"
    missing_path.write_text(json.dumps({"predictions": {"far3d_train_core": 1.0}}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing kinetic predictions"):
        load_kinetic_growth_rate_predictions(missing_path, dataset=loaded)

    list_dataset_path = tmp_path / "list_dataset.json"
    list_dataset_path.write_text(json.dumps([dataset.cases[0].to_dict()]), encoding="utf-8")
    assert load_kinetic_growth_rate_dataset(list_dataset_path).case_ids == ("far3d_train_core",)

    invalid_dataset_path = tmp_path / "invalid_dataset.json"
    invalid_dataset_path.write_text(json.dumps(3.14), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object or list"):
        load_kinetic_growth_rate_dataset(invalid_dataset_path)

    short_predictions_path = tmp_path / "short_predictions.json"
    short_predictions_path.write_text(json.dumps([1.0]), encoding="utf-8")
    with pytest.raises(ValueError, match="prediction list length"):
        load_kinetic_growth_rate_predictions(short_predictions_path, dataset=loaded)

    invalid_predictions_path = tmp_path / "invalid_predictions.json"
    invalid_predictions_path.write_text(json.dumps("bad"), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON mapping or list"):
        load_kinetic_growth_rate_predictions(invalid_predictions_path, dataset=loaded)

    with pytest.raises(ValueError, match="matching shapes"):
        kinetic_growth_rate_loss(jnp.asarray([1.0]), jnp.asarray([1.0, 2.0]), jnp.asarray([1.0, 1.0]))
    with pytest.raises(ValueError, match="one value per dataset case"):
        kinetic_growth_rate_parity(loaded, jnp.asarray([1.0]))


def test_far3d_lincheck_parser_reads_native_growth_and_frequency_rows(tmp_path: Path):
    output = tmp_path / "far3d.out"
    output.write_text(
        """
 copy of in:
 psi     : m=   4 n=   1 gam=  1.25000E-03 om_r=  5.00000E-02
 phi     : m=   5 n=   1 gam= -2.50000D-04 om_r= -6.00000D-02
 uzt     : this is not a lincheck row
 nfast   : m=   6 n=   2 gam=  0.00000E+00 om_r=  7.00000E-02
""",
        encoding="utf-8",
    )

    rows = read_far3d_lincheck(output)
    assert [row.variable for row in rows] == ["psi", "phi", "nfast"]
    assert rows[0].mode_label == "psi_m4_n1"
    assert rows[0].to_dict()["variable"] == "psi"
    np.testing.assert_allclose(rows[0].growth_rate_code_units, 1.25e-3)
    np.testing.assert_allclose(rows[1].omega_code_units, -6.0e-2)
    assert rows[2].line_number == 6

    filtered = read_far3d_lincheck(output, variables=("PHI",))
    assert len(filtered) == 1
    assert filtered[0].variable == "phi"


def test_far3d_lincheck_rows_convert_to_kinetic_growth_rate_dataset(tmp_path: Path):
    output = tmp_path / "far3d.out"
    output.write_text(
        """
 psi     : m=   4 n=   1 gam=  1.00000E-03 om_r=  4.00000E-02
 phi     : m=   5 n=   2 gam= -2.00000E-03 om_r= -8.00000E-02
""",
        encoding="utf-8",
    )
    rows = read_far3d_lincheck(output)

    dataset = far3d_rows_to_kinetic_dataset(
        rows,
        dataset_id="far3d_fixture",
        geometry_id="ctok",
        frequency_scale_hz=2.0e6,
        growth_rate_scale_hz=1.0e6,
        relative_growth_rate_uncertainty=0.2,
        split="calibration",
        citation="FAR3D lincheck fixture",
    )

    assert dataset.dataset_id == "far3d_fixture"
    assert dataset.cases[0].case_id == "far3d_fixture_psi_m4_n1"
    assert dataset.cases[0].frequency_hz == 80_000.0
    assert dataset.cases[0].growth_rate_hz == 1_000.0
    assert dataset.cases[0].growth_rate_uncertainty_hz == 200.0
    assert dataset.cases[0].split == "calibration"
    assert dataset.cases[1].toroidal_mode == 2
    assert dataset.cases[1].frequency_hz == 160_000.0
    assert "omega_code_units=-0.08" in dataset.cases[1].notes

    loaded = load_far3d_lincheck_dataset(
        output,
        dataset_id="far3d_loaded",
        geometry_id="ctok",
        frequency_scale_hz=2.0e6,
        growth_rate_uncertainty_hz=50.0,
        variables=("psi",),
    )
    assert len(loaded.cases) == 1
    assert loaded.cases[0].growth_rate_uncertainty_hz == 50.0

    with pytest.raises(ValueError, match="at least one FAR3D"):
        far3d_rows_to_kinetic_dataset((), dataset_id="empty", geometry_id="ctok", frequency_scale_hz=1.0)
    with pytest.raises(ValueError, match="scales must be positive"):
        far3d_rows_to_kinetic_dataset(rows, dataset_id="bad", geometry_id="ctok", frequency_scale_hz=0.0)
    with pytest.raises(ValueError, match="relative_growth_rate_uncertainty"):
        far3d_rows_to_kinetic_dataset(
            rows,
            dataset_id="bad",
            geometry_id="ctok",
            frequency_scale_hz=1.0,
            relative_growth_rate_uncertainty=0.0,
        )
    with pytest.raises(ValueError, match="growth_rate_uncertainty_hz"):
        far3d_rows_to_kinetic_dataset(
            rows,
            dataset_id="bad",
            geometry_id="ctok",
            frequency_scale_hz=1.0,
            growth_rate_uncertainty_hz=0.0,
        )


def test_archived_far3d_gpu_lincheck_artifact_is_parser_ready():
    rows = read_far3d_lincheck(FAR3D_GPU_ARTIFACT)

    assert len(rows) == 24
    assert {row.variable for row in rows} == {"psi", "phi", "pr", "nfast", "vfast", "vthfast"}
    assert {row.n for row in rows} == {1}
    assert min(row.m for row in rows) == 1
    assert max(row.m for row in rows) == 4
    assert max(row.growth_rate_code_units for row in rows) == pytest.approx(0.258901)
    assert max(abs(row.omega_code_units) for row in rows) == pytest.approx(0.474362)

    psi_rows = read_far3d_lincheck(FAR3D_GPU_ARTIFACT, variables=("psi",))
    assert [row.m for row in psi_rows] == [1, 2, 3, 4]


def test_archived_far3d_gpu_ingestion_packet_preserves_run_normalization():
    dataset = load_kinetic_growth_rate_dataset(FAR3D_GPU_DATASET)

    assert dataset.dataset_id == "far3d_diiid_gpu_nvhpc_20step"
    assert len(dataset.cases) == 24
    assert all(case.source_code == "FAR3D" for case in dataset.cases)
    assert all(case.split == "validation" for case in dataset.cases)
    assert dataset.cases[0].case_id == "far3d_diiid_gpu_nvhpc_20step_psi_m1_n1"
    assert dataset.cases[0].frequency_hz == pytest.approx(0.365588 * 599_813.41)
    assert dataset.cases[0].growth_rate_hz == pytest.approx(-0.417541 * 599_813.41)
    assert "Short 20-step executable/build artifact" in dataset.cases[0].notes


def test_archived_far3d_gpu_convergence_ladder_reaches_stable_lincheck_values():
    paths = {
        50: FAR3D_GPU_CONVERGENCE_DIR / "step_50/farprt0000",
        100: FAR3D_GPU_CONVERGENCE_DIR / "step_100/farprt0000",
        200: FAR3D_GPU_CONVERGENCE_DIR / "step_200/farprt0000",
        400: FAR3D_GPU_CONVERGENCE_DIR / "step_400/farprt0000",
    }

    points = far3d_lincheck_convergence(paths)

    assert [point.step_count for point in points] == [50, 100, 200, 400]
    assert all(point.row_count == 24 for point in points)
    assert all(point.common_mode_count == 24 for point in points)
    assert points[-1].spread_growth_code_units == pytest.approx(0.0)
    assert points[-1].spread_omega_code_units == pytest.approx(0.0)
    assert points[-2].max_relative_growth_delta_to_reference < 5.0e-5
    assert points[-2].max_relative_omega_delta_to_reference < 2.0e-5
    assert points[0].max_relative_growth_delta_to_reference > 0.4
    assert points[0].to_dict()["step_count"] == 50


def test_far3d_lincheck_convergence_rejects_missing_or_inconsistent_inputs(tmp_path: Path):
    with pytest.raises(ValueError, match="at least one"):
        far3d_lincheck_convergence({})
    with pytest.raises(ValueError, match="reference_step"):
        far3d_lincheck_convergence({1: FAR3D_GPU_ARTIFACT}, reference_step=2)

    empty = tmp_path / "empty.out"
    empty.write_text("no lincheck rows here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no FAR3D lincheck rows"):
        far3d_lincheck_convergence({1: empty})


def test_archived_far3d_gpu_beta_scan_has_monotonic_finite_beta_growth_branch():
    paths = {
        0.00: FAR3D_GPU_BETA_SCAN_DIR / "beta_0p000/farprt0000",
        0.01: FAR3D_GPU_BETA_SCAN_DIR / "beta_0p010/farprt0000",
        0.03: FAR3D_GPU_BETA_SCAN_DIR / "beta_0p030/farprt0000",
        0.06: FAR3D_GPU_BETA_SCAN_DIR / "beta_0p060/farprt0000",
        0.09: FAR3D_GPU_BETA_SCAN_DIR / "beta_0p090/farprt0000",
    }

    points = far3d_lincheck_parameter_scan(paths)

    assert [point.parameter_value for point in points] == [0.0, 0.01, 0.03, 0.06, 0.09]
    assert all(point.row_count == 24 for point in points)
    finite_beta_growth = [point.mean_growth_code_units for point in points[1:]]
    assert finite_beta_growth == sorted(finite_beta_growth)
    assert points[1].mean_growth_code_units == pytest.approx(0.014769655416666666)
    assert points[2].mean_growth_code_units == pytest.approx(0.05110659583333333)
    assert points[-1].mean_growth_code_units == pytest.approx(0.17747700000000002)
    assert points[-1].spread_growth_code_units == pytest.approx(0.0)
    assert points[0].spread_growth_code_units > 0.8
    assert points[-1].to_dict()["dominant_mode_label"]


def test_far3d_lincheck_parameter_scan_rejects_empty_inputs(tmp_path: Path):
    with pytest.raises(ValueError, match="at least one"):
        far3d_lincheck_parameter_scan({})

    empty = tmp_path / "empty.out"
    empty.write_text("no lincheck rows here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no FAR3D lincheck rows"):
        far3d_lincheck_parameter_scan({0.1: empty})
