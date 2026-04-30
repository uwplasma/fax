"""Summarize public ``landreman/vmec_equilibria`` geometries with FAX."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fax.continuum import continuum_branch_matrix
from fax.equilibrium import from_vmec_wout
from fax.modes import stellarator_mode_family
from fax.reference_cases import VmecEquilibriaCase, vmec_equilibria_catalog, vmec_equilibria_root


def _select_cases(case_ids: tuple[str, ...], root: Path) -> tuple[VmecEquilibriaCase, ...]:
    catalog = {case.case_id: case for case in vmec_equilibria_catalog()}
    selected = tuple(catalog[case_id] for case_id in case_ids) if case_ids else tuple(catalog.values())
    missing_ids = [case_id for case_id in case_ids if case_id not in catalog]
    if missing_ids:
        raise ValueError(f"unknown VMEC-equilibria case IDs: {missing_ids}")
    available = tuple(case for case in selected if case.is_available(root=root))
    if not available:
        raise FileNotFoundError(f"no selected VMEC-equilibria cases are available under {root}")
    return available


def _profile_record(case: VmecEquilibriaCase) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "family": case.family,
        "description": case.description,
        "tags": list(case.tags),
    }


def _short_label(case_id: str) -> str:
    return (
        case_id.replace("nuhrenberg_zille_qhs_ns51", "NZ-QHS")
        .replace("lhd_090493t2330", "LHD")
        .replace("tjii_100_44_64", "TJ-II")
        .replace("_", " ")
    )


def build_catalog_figure(root: Path, output_dir: Path, *, case_ids: tuple[str, ...] = ()) -> dict[str, object]:
    """Build a multi-geometry VMEC-equilibria continuum catalog figure."""
    cases = _select_cases(case_ids, root)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    profiles: list[tuple[VmecEquilibriaCase, np.ndarray, np.ndarray]] = []
    for case in cases:
        geometry = from_vmec_wout(case.wout_path(root=root), name=case.case_id)
        iota = np.asarray(geometry.iota, dtype=float)
        sign = -1 if float(np.nanmedian(iota)) < 0.0 else 1
        family = stellarator_mode_family(
            poloidal_modes=(1, 2, 3, 4),
            base_n=sign,
            field_periods=geometry.field_periods,
            sidebands=(-1, 0, 1),
        )
        branches = np.asarray(continuum_branch_matrix(geometry, family.as_array()), dtype=float)
        record = {
            **_profile_record(case),
            "wout": str(case.wout_path(root=root)),
            "vmec_input": str(case.vmec_input_path(root=root)) if case.vmec_input_path(root=root) else None,
            "field_periods": int(geometry.field_periods),
            "radial_points": int(geometry.s.shape[0]),
            "major_radius_m": float(geometry.major_radius),
            "minor_radius_m": float(geometry.minor_radius),
            "aspect_ratio": float(geometry.major_radius / geometry.minor_radius),
            "iota_min": float(np.nanmin(iota)),
            "iota_max": float(np.nanmax(iota)),
            "iota_edge": float(iota[-1]),
            "median_b_tesla": float(np.nanmedian(np.asarray(geometry.magnetic_field, dtype=float))),
            "continuum_p05_khz": float(np.nanpercentile(branches, 5.0) / 1.0e3),
            "continuum_median_khz": float(np.nanmedian(branches) / 1.0e3),
            "mode_family_base_n": sign,
            "mode_family_harmonics": [[int(m), int(n)] for m, n in family.harmonics],
        }
        records.append(record)
        profiles.append((case, np.asarray(geometry.s, dtype=float), iota))

    csv_path = output_dir / "vmec_equilibria_catalog.csv"
    fieldnames = [
        "case_id",
        "family",
        "field_periods",
        "radial_points",
        "major_radius_m",
        "minor_radius_m",
        "aspect_ratio",
        "iota_min",
        "iota_max",
        "iota_edge",
        "median_b_tesla",
        "continuum_p05_khz",
        "continuum_median_khz",
        "wout",
        "vmec_input",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record[name] for name in fieldnames})

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.2), constrained_layout=True)
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, len(records)))
    for color, (case, s_grid, iota) in zip(colors, profiles, strict=True):
        axes[0, 0].plot(s_grid, iota, lw=1.6, color=color, label=_short_label(case.case_id))
    axes[0, 0].axhline(0.0, color="0.25", lw=0.8)
    axes[0, 0].set_title("Rotational-transform profiles")
    axes[0, 0].set_xlabel("s")
    axes[0, 0].set_ylabel("iota")
    axes[0, 0].legend(fontsize=7, ncol=2)

    nfp = np.asarray([record["field_periods"] for record in records], dtype=float)
    aspect = np.asarray([record["aspect_ratio"] for record in records], dtype=float)
    axes[0, 1].scatter(nfp, aspect, c=colors, s=70, edgecolor="0.2")
    axes[0, 1].set_title("Device geometry spread")
    axes[0, 1].set_xlabel("field periods")
    axes[0, 1].set_ylabel("aspect ratio R/a")
    axes[0, 1].grid(alpha=0.25)

    order = np.argsort([float(record["continuum_p05_khz"]) for record in records])
    labels = [_short_label(str(records[index]["case_id"])) for index in order]
    p05 = np.asarray([records[index]["continuum_p05_khz"] for index in order], dtype=float)
    median = np.asarray([records[index]["continuum_median_khz"] for index in order], dtype=float)
    x = np.arange(len(order))
    axes[1, 0].bar(x - 0.18, p05, width=0.36, label="p05")
    axes[1, 0].bar(x + 0.18, median, width=0.36, label="median")
    axes[1, 0].set_title("Profile-continuum frequency scale")
    axes[1, 0].set_ylabel("frequency [kHz]")
    axes[1, 0].set_xticks(x, labels, rotation=35, ha="right")
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.25)

    b_median = np.asarray([record["median_b_tesla"] for record in records], dtype=float)
    continuum = np.asarray([record["continuum_median_khz"] for record in records], dtype=float)
    size = 25.0 + 3.0 * aspect
    axes[1, 1].scatter(b_median, continuum, c=colors, s=size, edgecolor="0.2")
    axes[1, 1].set_title("Field strength versus continuum scale")
    axes[1, 1].set_xlabel("median |B| [T]")
    axes[1, 1].set_ylabel("median profile-continuum [kHz]")
    axes[1, 1].grid(alpha=0.25)

    fig.suptitle("FAX multi-geometry VMEC-equilibria profile-continuum catalog", fontsize=14)
    figure_paths = []
    for suffix in ("png", "svg"):
        path = output_dir / f"vmec_equilibria_catalog.{suffix}"
        fig.savefig(path, dpi=180)
        figure_paths.append(str(path))
    plt.close(fig)

    summary = {
        "root": str(root),
        "case_count": len(records),
        "cases": records,
        "csv": str(csv_path),
        "figures": figure_paths,
        "scope": (
            "Multi-geometry VMEC wout profile-continuum catalog from landreman/vmec_equilibria; "
            "not a full Boozer/global cross-code parity claim."
        ),
    }
    (output_dir / "vmec_equilibria_catalog_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(vmec_equilibria_root()), help="Path to a landreman/vmec_equilibria clone.")
    parser.add_argument("--output-dir", default="outputs/vmec_equilibria_catalog", help="Output directory.")
    parser.add_argument("--case", action="append", default=[], help="Case ID to include; may be repeated.")
    args = parser.parse_args()
    summary = build_catalog_figure(Path(args.root), Path(args.output_dir), case_ids=tuple(args.case))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
