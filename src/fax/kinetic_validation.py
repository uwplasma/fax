"""Kinetic growth-rate benchmark contracts for fast-ion validation.

The objects here are intentionally validation infrastructure, not a replacement
for CAS3D-K or FAR3D. They define the data contract needed to calibrate and
hold out energetic-particle growth-rate references before any reduced FAX
closure can be advertised as calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np


VALID_KINETIC_SOURCE_CODES = ("CAS3D-K", "FAR3D", "CAS3D", "LITERATURE")
VALID_KINETIC_SPLITS = ("calibration", "validation")


@dataclass(frozen=True)
class KineticGrowthRateCase:
    """One external kinetic-code or literature growth-rate reference."""

    case_id: str
    source_code: str
    geometry_id: str
    mode_id: str
    frequency_hz: float
    growth_rate_hz: float
    growth_rate_uncertainty_hz: float
    toroidal_mode: int
    split: str = "validation"
    damping_rate_hz: float | None = None
    resonance_radius: float | None = None
    distribution_parameters: Mapping[str, float] | tuple[tuple[str, float], ...] = ()
    citation: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", str(self.case_id))
        object.__setattr__(self, "source_code", _canonical_source_code(self.source_code))
        object.__setattr__(self, "geometry_id", str(self.geometry_id))
        object.__setattr__(self, "mode_id", str(self.mode_id))
        object.__setattr__(self, "frequency_hz", float(self.frequency_hz))
        object.__setattr__(self, "growth_rate_hz", float(self.growth_rate_hz))
        object.__setattr__(self, "growth_rate_uncertainty_hz", float(self.growth_rate_uncertainty_hz))
        object.__setattr__(self, "toroidal_mode", int(self.toroidal_mode))
        object.__setattr__(self, "split", _canonical_split(self.split))
        object.__setattr__(self, "distribution_parameters", _parameter_tuple(self.distribution_parameters))
        if self.damping_rate_hz is not None:
            object.__setattr__(self, "damping_rate_hz", float(self.damping_rate_hz))
        if self.resonance_radius is not None:
            object.__setattr__(self, "resonance_radius", float(self.resonance_radius))
        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        if not self.geometry_id:
            raise ValueError("geometry_id must be non-empty")
        if not self.mode_id:
            raise ValueError("mode_id must be non-empty")
        if not np.isfinite(self.frequency_hz) or self.frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be positive and finite")
        if not np.isfinite(self.growth_rate_hz):
            raise ValueError("growth_rate_hz must be finite")
        if not np.isfinite(self.growth_rate_uncertainty_hz) or self.growth_rate_uncertainty_hz <= 0.0:
            raise ValueError("growth_rate_uncertainty_hz must be positive and finite")
        if self.resonance_radius is not None and not 0.0 <= self.resonance_radius <= 1.0:
            raise ValueError("resonance_radius must be between 0 and 1 when provided")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "KineticGrowthRateCase":
        """Build a case from a JSON-style mapping."""
        return cls(**dict(payload))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable case record."""
        return {
            "case_id": self.case_id,
            "source_code": self.source_code,
            "geometry_id": self.geometry_id,
            "mode_id": self.mode_id,
            "frequency_hz": self.frequency_hz,
            "growth_rate_hz": self.growth_rate_hz,
            "growth_rate_uncertainty_hz": self.growth_rate_uncertainty_hz,
            "toroidal_mode": self.toroidal_mode,
            "split": self.split,
            "damping_rate_hz": self.damping_rate_hz,
            "resonance_radius": self.resonance_radius,
            "distribution_parameters": dict(self.distribution_parameters),
            "citation": self.citation,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class KineticGrowthRateDataset:
    """A calibration/validation packet for kinetic growth-rate parity."""

    cases: tuple[KineticGrowthRateCase, ...]
    dataset_id: str = "unnamed_kinetic_growth_rate_dataset"
    citation: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))
        object.__setattr__(self, "dataset_id", str(self.dataset_id))
        case_ids = [case.case_id for case in self.cases]
        if not self.cases:
            raise ValueError("at least one kinetic growth-rate case is required")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("case_id values must be unique")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "KineticGrowthRateDataset":
        """Build a dataset from a JSON-style mapping."""
        cases = tuple(KineticGrowthRateCase.from_dict(case) for case in payload["cases"])
        return cls(cases=cases, dataset_id=str(payload.get("dataset_id", "unnamed_kinetic_growth_rate_dataset")), citation=str(payload.get("citation", "")))

    @property
    def case_ids(self) -> tuple[str, ...]:
        """Return case IDs in dataset order."""
        return tuple(case.case_id for case in self.cases)

    @property
    def growth_rate_hz(self) -> jax.Array:
        """Return observed growth rates in dataset order."""
        return jnp.asarray([case.growth_rate_hz for case in self.cases])

    @property
    def growth_rate_uncertainty_hz(self) -> jax.Array:
        """Return one-sigma growth-rate uncertainties in dataset order."""
        return jnp.asarray([case.growth_rate_uncertainty_hz for case in self.cases])

    @property
    def validation_mask(self) -> jax.Array:
        """Return a boolean mask for held-out validation cases."""
        return jnp.asarray([case.split == "validation" for case in self.cases], dtype=bool)

    @property
    def calibration_mask(self) -> jax.Array:
        """Return a boolean mask for calibration/training cases."""
        return jnp.asarray([case.split == "calibration" for case in self.cases], dtype=bool)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dataset record."""
        return {
            "dataset_id": self.dataset_id,
            "citation": self.citation,
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(frozen=True)
class KineticGrowthRateParity:
    """Parity metrics between FAX predictions and kinetic-code references."""

    dataset_id: str
    status: str
    passed: bool
    count: int
    calibration_count: int
    validation_count: int
    rms_error_hz: float
    normalized_rms_error: float
    validation_rms_error_hz: float
    validation_normalized_rms_error: float
    sign_accuracy: float
    validation_sign_accuracy: float
    max_abs_error_hz: float
    max_normalized_error: float
    case_ids: tuple[str, ...]
    predicted_growth_rate_hz: jax.Array
    observed_growth_rate_hz: jax.Array
    error_hz: jax.Array
    normalized_error: jax.Array

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable parity report."""
        return {
            "dataset_id": self.dataset_id,
            "status": self.status,
            "passed": self.passed,
            "count": self.count,
            "calibration_count": self.calibration_count,
            "validation_count": self.validation_count,
            "rms_error_hz": self.rms_error_hz,
            "normalized_rms_error": self.normalized_rms_error,
            "validation_rms_error_hz": self.validation_rms_error_hz,
            "validation_normalized_rms_error": self.validation_normalized_rms_error,
            "sign_accuracy": self.sign_accuracy,
            "validation_sign_accuracy": self.validation_sign_accuracy,
            "max_abs_error_hz": self.max_abs_error_hz,
            "max_normalized_error": self.max_normalized_error,
            "case_ids": list(self.case_ids),
            "predicted_growth_rate_hz": np.asarray(self.predicted_growth_rate_hz).tolist(),
            "observed_growth_rate_hz": np.asarray(self.observed_growth_rate_hz).tolist(),
            "error_hz": np.asarray(self.error_hz).tolist(),
            "normalized_error": np.asarray(self.normalized_error).tolist(),
        }


def load_kinetic_growth_rate_dataset(path: str | Path) -> KineticGrowthRateDataset:
    """Load a kinetic growth-rate dataset from JSON."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        payload = {"cases": payload}
    if not isinstance(payload, Mapping):
        raise ValueError("kinetic growth-rate dataset must be a JSON object or list of cases")
    return KineticGrowthRateDataset.from_dict(payload)


def load_kinetic_growth_rate_predictions(
    path: str | Path,
    *,
    dataset: KineticGrowthRateDataset,
) -> jax.Array:
    """Load predicted growth rates from JSON in dataset order.

    The prediction file may be either a mapping from ``case_id`` to predicted
    growth rate, ``{"predictions": {"case_id": value}}``, or a list ordered
    like the dataset.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and "predictions" in payload:
        payload = payload["predictions"]
    if isinstance(payload, Mapping):
        missing = [case_id for case_id in dataset.case_ids if case_id not in payload]
        if missing:
            raise ValueError("missing kinetic predictions for case IDs: " + ", ".join(missing))
        return jnp.asarray([float(payload[case_id]) for case_id in dataset.case_ids])
    if isinstance(payload, list):
        if len(payload) != len(dataset.cases):
            raise ValueError("prediction list length must match the dataset case count")
        return jnp.asarray([float(value) for value in payload])
    raise ValueError("kinetic predictions must be a JSON mapping or list")


def kinetic_growth_rate_loss(
    predicted_growth_rate_hz: Any,
    observed_growth_rate_hz: Any,
    uncertainty_hz: Any,
) -> jax.Array:
    """Return differentiable mean squared normalized growth-rate error."""
    predicted = jnp.asarray(predicted_growth_rate_hz)
    observed = jnp.asarray(observed_growth_rate_hz)
    uncertainty = jnp.maximum(jnp.asarray(uncertainty_hz), 1.0e-30)
    if predicted.shape != observed.shape or predicted.shape != uncertainty.shape:
        raise ValueError("predicted, observed, and uncertainty arrays must have matching shapes")
    return jnp.mean(((predicted - observed) / uncertainty) ** 2)


def kinetic_growth_rate_parity(
    dataset: KineticGrowthRateDataset,
    predicted_growth_rate_hz: Any,
    *,
    max_validation_normalized_rms_error: float = 2.0,
    min_validation_count: int = 1,
    min_validation_sign_accuracy: float = 1.0,
) -> KineticGrowthRateParity:
    """Compare FAX-predicted growth rates with held-out kinetic references."""
    predicted = jnp.asarray(predicted_growth_rate_hz)
    observed = dataset.growth_rate_hz
    uncertainty = dataset.growth_rate_uncertainty_hz
    if predicted.shape != observed.shape:
        raise ValueError("predicted_growth_rate_hz must have one value per dataset case")
    error = predicted - observed
    normalized_error = error / jnp.maximum(uncertainty, 1.0e-30)
    validation_mask = np.asarray(dataset.validation_mask, dtype=bool)
    calibration_mask = np.asarray(dataset.calibration_mask, dtype=bool)
    error_np = np.asarray(error, dtype=float)
    normalized_np = np.asarray(normalized_error, dtype=float)
    predicted_np = np.asarray(predicted, dtype=float)
    observed_np = np.asarray(observed, dtype=float)

    validation_error = error_np[validation_mask]
    validation_normalized = normalized_np[validation_mask]
    validation_predicted = predicted_np[validation_mask]
    validation_observed = observed_np[validation_mask]
    validation_count = int(validation_mask.sum())
    sign_accuracy = _sign_accuracy(predicted_np, observed_np)
    validation_sign_accuracy = _sign_accuracy(validation_predicted, validation_observed) if validation_count else float("nan")
    validation_normalized_rms = _rms(validation_normalized) if validation_count else float("nan")
    validation_rms = _rms(validation_error) if validation_count else float("nan")
    has_required_validation = validation_count >= int(min_validation_count)
    passed = (
        has_required_validation
        and validation_normalized_rms <= float(max_validation_normalized_rms_error)
        and validation_sign_accuracy >= float(min_validation_sign_accuracy)
    )
    status = "passed" if passed else ("failed" if validation_count else "incomplete")
    return KineticGrowthRateParity(
        dataset_id=dataset.dataset_id,
        status=status,
        passed=bool(passed),
        count=len(dataset.cases),
        calibration_count=int(calibration_mask.sum()),
        validation_count=validation_count,
        rms_error_hz=_rms(error_np),
        normalized_rms_error=_rms(normalized_np),
        validation_rms_error_hz=validation_rms,
        validation_normalized_rms_error=validation_normalized_rms,
        sign_accuracy=sign_accuracy,
        validation_sign_accuracy=validation_sign_accuracy,
        max_abs_error_hz=float(np.max(np.abs(error_np))),
        max_normalized_error=float(np.max(np.abs(normalized_np))),
        case_ids=dataset.case_ids,
        predicted_growth_rate_hz=predicted,
        observed_growth_rate_hz=observed,
        error_hz=error,
        normalized_error=normalized_error,
    )


def kinetic_growth_rate_parity_markdown(parity: KineticGrowthRateParity) -> str:
    """Return a Markdown report for a kinetic growth-rate parity check."""
    lines = [
        f"# Kinetic Growth-Rate Parity: {parity.dataset_id}",
        "",
        f"Status: **{parity.status}**.",
        f"Validation cases: **{parity.validation_count}**.",
        f"Validation normalized RMS error: **{parity.validation_normalized_rms_error:.6g}**.",
        f"Validation sign accuracy: **{parity.validation_sign_accuracy:.6g}**.",
        "",
        "| Case | Observed [Hz] | Predicted [Hz] | Error [Hz] | Normalized Error |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for case_id, observed, predicted, error, normalized in zip(
        parity.case_ids,
        np.asarray(parity.observed_growth_rate_hz),
        np.asarray(parity.predicted_growth_rate_hz),
        np.asarray(parity.error_hz),
        np.asarray(parity.normalized_error),
        strict=True,
    ):
        lines.append(f"| `{case_id}` | {observed:.6g} | {predicted:.6g} | {error:.6g} | {normalized:.6g} |")
    return "\n".join(lines)


def _canonical_source_code(source_code: str) -> str:
    source = str(source_code).upper()
    if source == "CAS3D_K":
        source = "CAS3D-K"
    if source not in VALID_KINETIC_SOURCE_CODES:
        raise ValueError(f"source_code must be one of {VALID_KINETIC_SOURCE_CODES}")
    return source


def _canonical_split(split: str) -> str:
    value = str(split).lower()
    if value in {"train", "training"}:
        value = "calibration"
    if value in {"test", "holdout", "held-out"}:
        value = "validation"
    if value not in VALID_KINETIC_SPLITS:
        raise ValueError(f"split must be one of {VALID_KINETIC_SPLITS}")
    return value


def _parameter_tuple(values: Mapping[str, float] | tuple[tuple[str, float], ...]) -> tuple[tuple[str, float], ...]:
    items = values.items() if isinstance(values, Mapping) else values
    return tuple(sorted((str(key), float(value)) for key, value in items))


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def _sign_accuracy(predicted: np.ndarray, observed: np.ndarray) -> float:
    if predicted.size == 0:
        return float("nan")
    return float(np.mean(np.sign(predicted) == np.sign(observed)))
