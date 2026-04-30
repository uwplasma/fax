from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fax.kinetic_validation import (
    KineticGrowthRateCase,
    KineticGrowthRateDataset,
    kinetic_growth_rate_loss,
    kinetic_growth_rate_parity,
    load_kinetic_growth_rate_dataset,
    load_kinetic_growth_rate_predictions,
)


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
