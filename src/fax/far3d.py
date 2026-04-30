"""FAR3D benchmark-output ingestion helpers.

FAR3D's ``lincheck`` diagnostic writes one line per retained field harmonic,
with the format defined in ``src/lincheck.f90``:

``psi     : m=   4 n=   1 gam=  1.23456E-03 om_r=  5.67890E-02``

The parser in this module deliberately keeps the native FAR3D code-unit growth
rate and real frequency separate from the conversion into FAX's Hz-based
kinetic validation schema. A caller must provide the normalization scale before
rows become publication-facing growth-rate cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping

from .kinetic_validation import KineticGrowthRateCase, KineticGrowthRateDataset


_FORTRAN_FLOAT = r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[EeDd][+-]?\d+)?"
_LINCHECK_RE = re.compile(
    rf"^\s*(?P<variable>[A-Za-z0-9_ -]+?)\s*:\s*"
    rf"m\s*=\s*(?P<m>[+-]?\d+)\s+"
    rf"n\s*=\s*(?P<n>[+-]?\d+)\s+"
    rf"gam\s*=\s*(?P<growth>{_FORTRAN_FLOAT})\s+"
    rf"om_r\s*=\s*(?P<omega>{_FORTRAN_FLOAT})\s*$"
)


@dataclass(frozen=True)
class Far3dLinearGrowthRow:
    """One FAR3D ``lincheck`` growth/frequency row in native code units."""

    variable: str
    m: int
    n: int
    growth_rate_code_units: float
    omega_code_units: float
    line_number: int
    raw_line: str = ""

    @property
    def mode_label(self) -> str:
        """Return a stable mode label suitable for case IDs."""
        return f"{self.variable}_m{self.m}_n{self.n}"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable row record."""
        return {
            "variable": self.variable,
            "m": self.m,
            "n": self.n,
            "growth_rate_code_units": self.growth_rate_code_units,
            "omega_code_units": self.omega_code_units,
            "line_number": self.line_number,
            "raw_line": self.raw_line,
        }


@dataclass(frozen=True)
class Far3dLincheckConvergencePoint:
    """Convergence metrics for one FAR3D ``lincheck`` run length."""

    step_count: int
    row_count: int
    common_mode_count: int
    mean_growth_code_units: float
    mean_omega_code_units: float
    spread_growth_code_units: float
    spread_omega_code_units: float
    max_abs_growth_delta_to_reference: float
    max_abs_omega_delta_to_reference: float
    max_relative_growth_delta_to_reference: float
    max_relative_omega_delta_to_reference: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serializable convergence record."""
        return {
            "step_count": self.step_count,
            "row_count": self.row_count,
            "common_mode_count": self.common_mode_count,
            "mean_growth_code_units": self.mean_growth_code_units,
            "mean_omega_code_units": self.mean_omega_code_units,
            "spread_growth_code_units": self.spread_growth_code_units,
            "spread_omega_code_units": self.spread_omega_code_units,
            "max_abs_growth_delta_to_reference": self.max_abs_growth_delta_to_reference,
            "max_abs_omega_delta_to_reference": self.max_abs_omega_delta_to_reference,
            "max_relative_growth_delta_to_reference": self.max_relative_growth_delta_to_reference,
            "max_relative_omega_delta_to_reference": self.max_relative_omega_delta_to_reference,
        }


def read_far3d_lincheck(path: str | Path, *, variables: tuple[str, ...] | None = None) -> tuple[Far3dLinearGrowthRow, ...]:
    """Read FAR3D ``lincheck`` rows from a run log or ``confil`` output.

    Parameters
    ----------
    path:
        Text file containing FAR3D stdout or the configured unit-6 output file.
    variables:
        Optional whitelist of FAR3D field names, such as ``("psi", "phi")``.

    Returns
    -------
    tuple[Far3dLinearGrowthRow, ...]
        Parsed rows in file order. Lines that are not FAR3D ``lincheck`` rows
        are ignored.
    """
    source = Path(path)
    requested = {variable.strip().lower() for variable in variables} if variables else None
    rows: list[Far3dLinearGrowthRow] = []
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        match = _LINCHECK_RE.match(raw_line)
        if match is None:
            continue
        variable = " ".join(match.group("variable").strip().split()).lower()
        if requested is not None and variable not in requested:
            continue
        rows.append(
            Far3dLinearGrowthRow(
                variable=variable,
                m=int(match.group("m")),
                n=int(match.group("n")),
                growth_rate_code_units=_parse_fortran_float(match.group("growth")),
                omega_code_units=_parse_fortran_float(match.group("omega")),
                line_number=line_number,
                raw_line=raw_line.rstrip(),
            )
        )
    return tuple(rows)


def far3d_lincheck_convergence(
    paths_by_step: Mapping[int, str | Path],
    *,
    reference_step: int | None = None,
) -> tuple[Far3dLincheckConvergencePoint, ...]:
    """Compare FAR3D ``lincheck`` rows against a reference run length.

    The convergence metric is intentionally mode-by-mode. Only labels present
    in every run are compared, so a growing harmonic set cannot hide a
    non-converged common mode.
    """
    if not paths_by_step:
        raise ValueError("at least one FAR3D lincheck output is required")
    rows_by_step = {int(step): read_far3d_lincheck(path) for step, path in paths_by_step.items()}
    empty_steps = [step for step, rows in rows_by_step.items() if not rows]
    if empty_steps:
        raise ValueError(f"no FAR3D lincheck rows found for steps: {empty_steps}")
    selected_reference_step = max(rows_by_step) if reference_step is None else int(reference_step)
    if selected_reference_step not in rows_by_step:
        raise ValueError(f"reference_step {selected_reference_step} is not present in paths_by_step")

    labels = set(_rows_by_mode_label(rows_by_step[selected_reference_step]))
    for rows in rows_by_step.values():
        labels &= set(_rows_by_mode_label(rows))
    if not labels:
        raise ValueError("FAR3D lincheck outputs have no common mode labels")
    ordered_labels = tuple(sorted(labels))
    reference_rows = _rows_by_mode_label(rows_by_step[selected_reference_step])
    tiny = 1.0e-300

    points: list[Far3dLincheckConvergencePoint] = []
    for step in sorted(rows_by_step):
        rows_by_label = _rows_by_mode_label(rows_by_step[step])
        growth = tuple(rows_by_label[label].growth_rate_code_units for label in ordered_labels)
        omega = tuple(rows_by_label[label].omega_code_units for label in ordered_labels)
        reference_growth = tuple(reference_rows[label].growth_rate_code_units for label in ordered_labels)
        reference_omega = tuple(reference_rows[label].omega_code_units for label in ordered_labels)
        growth_delta = tuple(abs(value - reference) for value, reference in zip(growth, reference_growth, strict=True))
        omega_delta = tuple(abs(value - reference) for value, reference in zip(omega, reference_omega, strict=True))
        growth_relative_delta = tuple(
            delta / max(abs(reference), tiny) for delta, reference in zip(growth_delta, reference_growth, strict=True)
        )
        omega_relative_delta = tuple(
            delta / max(abs(reference), tiny) for delta, reference in zip(omega_delta, reference_omega, strict=True)
        )
        points.append(
            Far3dLincheckConvergencePoint(
                step_count=step,
                row_count=len(rows_by_step[step]),
                common_mode_count=len(ordered_labels),
                mean_growth_code_units=_mean(growth),
                mean_omega_code_units=_mean(omega),
                spread_growth_code_units=max(growth) - min(growth),
                spread_omega_code_units=max(omega) - min(omega),
                max_abs_growth_delta_to_reference=max(growth_delta),
                max_abs_omega_delta_to_reference=max(omega_delta),
                max_relative_growth_delta_to_reference=max(growth_relative_delta),
                max_relative_omega_delta_to_reference=max(omega_relative_delta),
            )
        )
    return tuple(points)


def far3d_rows_to_kinetic_dataset(
    rows: tuple[Far3dLinearGrowthRow, ...],
    *,
    dataset_id: str,
    geometry_id: str,
    frequency_scale_hz: float,
    growth_rate_scale_hz: float | None = None,
    growth_rate_uncertainty_hz: float | None = None,
    relative_growth_rate_uncertainty: float = 0.1,
    split: str = "validation",
    citation: str = "",
    notes: str = "",
) -> KineticGrowthRateDataset:
    """Convert FAR3D ``lincheck`` rows into the FAX kinetic validation schema.

    FAR3D's output is normalized. FAX only creates Hz-based validation cases
    after the caller supplies the conversion scale. If the growth-rate scale is
    omitted, the frequency scale is used for both ``gam`` and ``om_r``.
    """
    if not rows:
        raise ValueError("at least one FAR3D lincheck row is required")
    frequency_scale = float(frequency_scale_hz)
    growth_scale = frequency_scale if growth_rate_scale_hz is None else float(growth_rate_scale_hz)
    if frequency_scale <= 0.0 or growth_scale <= 0.0:
        raise ValueError("frequency and growth-rate scales must be positive")
    relative_uncertainty = float(relative_growth_rate_uncertainty)
    if growth_rate_uncertainty_hz is None and relative_uncertainty <= 0.0:
        raise ValueError("relative_growth_rate_uncertainty must be positive")
    if growth_rate_uncertainty_hz is not None and float(growth_rate_uncertainty_hz) <= 0.0:
        raise ValueError("growth_rate_uncertainty_hz must be positive")

    cases: list[KineticGrowthRateCase] = []
    for row in rows:
        frequency_hz = abs(row.omega_code_units) * frequency_scale
        growth_rate_hz = row.growth_rate_code_units * growth_scale
        uncertainty_hz = (
            float(growth_rate_uncertainty_hz)
            if growth_rate_uncertainty_hz is not None
            else max(abs(growth_rate_hz) * relative_uncertainty, growth_scale * 1.0e-12)
        )
        cases.append(
            KineticGrowthRateCase(
                case_id=f"{dataset_id}_{row.mode_label}",
                source_code="FAR3D",
                geometry_id=geometry_id,
                mode_id=row.mode_label,
                frequency_hz=frequency_hz,
                growth_rate_hz=growth_rate_hz,
                growth_rate_uncertainty_hz=uncertainty_hz,
                toroidal_mode=row.n,
                split=split,
                citation=citation,
                notes=_join_notes(
                    notes,
                    (
                        "Imported from FAR3D lincheck output; "
                        f"variable={row.variable}, m={row.m}, n={row.n}, "
                        f"gam_code_units={row.growth_rate_code_units:.16g}, "
                        f"omega_code_units={row.omega_code_units:.16g}, "
                        f"source_line={row.line_number}."
                    ),
                ),
            )
        )
    return KineticGrowthRateDataset(cases=tuple(cases), dataset_id=dataset_id, citation=citation)


def load_far3d_lincheck_dataset(
    path: str | Path,
    *,
    dataset_id: str,
    geometry_id: str,
    frequency_scale_hz: float,
    growth_rate_scale_hz: float | None = None,
    growth_rate_uncertainty_hz: float | None = None,
    relative_growth_rate_uncertainty: float = 0.1,
    split: str = "validation",
    citation: str = "",
    notes: str = "",
    variables: tuple[str, ...] | None = None,
) -> KineticGrowthRateDataset:
    """Read a FAR3D log and return a Hz-normalized kinetic dataset."""
    rows = read_far3d_lincheck(path, variables=variables)
    return far3d_rows_to_kinetic_dataset(
        rows,
        dataset_id=dataset_id,
        geometry_id=geometry_id,
        frequency_scale_hz=frequency_scale_hz,
        growth_rate_scale_hz=growth_rate_scale_hz,
        growth_rate_uncertainty_hz=growth_rate_uncertainty_hz,
        relative_growth_rate_uncertainty=relative_growth_rate_uncertainty,
        split=split,
        citation=citation,
        notes=notes,
    )


def _parse_fortran_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def _rows_by_mode_label(rows: tuple[Far3dLinearGrowthRow, ...]) -> dict[str, Far3dLinearGrowthRow]:
    rows_by_label: dict[str, Far3dLinearGrowthRow] = {}
    for row in rows:
        if row.mode_label in rows_by_label:
            raise ValueError(f"duplicate FAR3D lincheck mode label: {row.mode_label}")
        rows_by_label[row.mode_label] = row
    return rows_by_label


def _mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values)


def _join_notes(prefix: str, detail: str) -> str:
    prefix = prefix.strip()
    return f"{prefix} {detail}" if prefix else detail
