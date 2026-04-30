"""Run FAX global 3-D Boozer operators on public VMEC-equilibria artifacts.

The script reads cataloged ``boozmn`` files from a local
``landreman/vmec_equilibria`` checkout, assembles FAX's radial block
shear-Alfven operator, and compares boundary-condition and radial-stiffness
variants. It is a reproducible public-artifact gate toward multi-geometry
Boozer/global parity. It is not, by itself, an AE3D/CAS3D/STELLGAP parity
claim.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from fax.boozer import (
    boozer_geometry_jacobian_identity_error,
    dominant_boozer_modes,
    normalized_nonaxisymmetric_boozer_energy,
    read_boozmn_geometry,
    slice_boozer_geometry_surfaces,
)
from fax.global_modes import radial_block_problem_from_boozer_geometry, solve_radial_block_eigenmodes
from fax.reference_cases import available_vmec_equilibria_boozmn_cases, vmec_equilibria_root

jax.config.update("jax_enable_x64", True)


DEFAULT_HARMONICS = ((1.0, 0.0), (2.0, 1.0), (3.0, 2.0))


def _surface_indices(total: int, requested: int) -> np.ndarray:
    count = max(4, min(int(total), int(requested)))
    return np.unique(np.linspace(0, int(total) - 1, count, dtype=int))


def _solve_variant(
    geometry: Any,
    harmonics: jax.Array,
    *,
    mass_density: float,
    theta_points: int,
    zeta_points: int,
    boundary_condition: str,
    radial_stiffness_method: str,
    radial_stiffness_scale: float,
    radial_coupling_fraction: float,
) -> tuple[Any, Any]:
    problem = radial_block_problem_from_boozer_geometry(
        geometry,
        harmonics,
        mass_density=mass_density,
        radial_stiffness_method=radial_stiffness_method,
        radial_stiffness_scale=radial_stiffness_scale,
        radial_coupling_fraction=radial_coupling_fraction,
        boundary_condition=boundary_condition,
        theta_points=theta_points,
        zeta_points=zeta_points,
    )
    return problem, solve_radial_block_eigenmodes(problem)


def _case_packet(case: Any, args: argparse.Namespace) -> dict[str, Any]:
    path = case.boozmn_path(root=args.root)
    if path is None:
        raise ValueError(f"case {case.case_id} has no cataloged boozmn path")
    geometry = read_boozmn_geometry(path)
    indices = _surface_indices(int(geometry.s.shape[0]), int(args.surface_count))
    sampled = slice_boozer_geometry_surfaces(geometry, indices)
    harmonics = jnp.asarray(DEFAULT_HARMONICS, dtype=float)

    variants: dict[str, tuple[Any, Any]] = {}
    for name, boundary, method in (
        ("metric_dirichlet", "dirichlet", "metric"),
        ("metric_neumann", "neumann", "metric"),
        ("mass_dirichlet", "dirichlet", "mass"),
    ):
        variants[name] = _solve_variant(
            sampled,
            harmonics,
            mass_density=float(args.mass_density),
            theta_points=int(args.theta_points),
            zeta_points=int(args.zeta_points),
            boundary_condition=boundary,
            radial_stiffness_method=method,
            radial_stiffness_scale=float(args.radial_stiffness_scale),
            radial_coupling_fraction=float(args.radial_coupling_fraction),
        )

    baseline_problem, baseline = variants["metric_dirichlet"]
    frequency_hz = {name: float(result.frequencies_hz[0]) for name, (_, result) in variants.items()}
    residuals = {name: float(np.max(np.asarray(result.residuals[: min(4, result.residuals.shape[0])]))) for name, (_, result) in variants.items()}
    envelope = np.linalg.norm(np.asarray(baseline.mode_structures[0]), axis=1)
    envelope = envelope / max(float(np.max(envelope)), 1.0e-30)
    nonaxisymmetric_energy = float(normalized_nonaxisymmetric_boozer_energy(sampled.as_spectrum()))
    jacobian_identity_error = float(
        boozer_geometry_jacobian_identity_error(
            sampled,
            theta_points=int(args.theta_points),
            zeta_points=int(args.zeta_points),
        )
    )
    dominant = dominant_boozer_modes(sampled.as_spectrum(), count=4)
    return {
        "case_id": case.case_id,
        "family": case.family,
        "description": case.description,
        "boozmn": str(path),
        "surface_count_available": int(geometry.s.shape[0]),
        "surface_count_used": int(sampled.s.shape[0]),
        "mode_count": int(geometry.xm.shape[0]),
        "field_periods": int(sampled.field_periods),
        "s": [float(value) for value in np.asarray(sampled.s)],
        "iota_axis": float(sampled.iota[0]),
        "iota_edge": float(sampled.iota[-1]),
        "jacobian_identity_error": jacobian_identity_error,
        "normalized_nonaxisymmetric_b_energy": nonaxisymmetric_energy,
        "lowest_frequency_hz": frequency_hz,
        "residuals": residuals,
        "envelope": [float(value) for value in envelope],
        "harmonics": [[int(m), int(n)] for m, n in np.asarray(baseline_problem.harmonics, dtype=int)],
        "dominant_boozer_modes": [{"m": m, "n": n, "amplitude_t": amp} for m, n, amp in dominant],
    }


def _write_outputs(output_dir: Path, packets: list[dict[str, Any]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [packet["case_id"].replace("hsx_", "HSX\n").replace("ku_boozer_", "Ku-Boozer\n").replace("_", "\n") for packet in packets]
    x = np.arange(len(packets))
    baseline = np.asarray([packet["lowest_frequency_hz"]["metric_dirichlet"] / 1.0e3 for packet in packets])
    neumann = np.asarray([packet["lowest_frequency_hz"]["metric_neumann"] / 1.0e3 for packet in packets])
    mass = np.asarray([packet["lowest_frequency_hz"]["mass_dirichlet"] / 1.0e3 for packet in packets])
    jacobian = np.asarray([packet["jacobian_identity_error"] for packet in packets])
    nonaxis = np.asarray([packet["normalized_nonaxisymmetric_b_energy"] for packet in packets])

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), constrained_layout=True)

    ax = axes[0, 0]
    width = 0.26
    ax.bar(x - width, baseline, width, label="metric, Dirichlet")
    ax.bar(x, neumann, width, label="metric, Neumann")
    ax.bar(x + width, mass, width, label="mass scaled, Dirichlet")
    ax.set_title("lowest global radial frequency")
    ax.set_ylabel("frequency [kHz]")
    ax.set_xticks(x, labels, rotation=0)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for packet in packets:
        ax.plot(
            packet["s"],
            packet["envelope"],
            marker="o",
            markersize=3,
            linewidth=1.4,
            label=packet["case_id"].replace("_", " "),
        )
    ax.set_title("mode-1 radial envelopes")
    ax.set_xlabel("normalized toroidal flux s")
    ax.set_ylabel("normalized envelope")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.semilogy(x, jacobian, marker="o", linewidth=1.5, label=r"$|\sqrt{g}B^2-(G+\iota I)|$")
    ax2 = ax.twinx()
    ax2.plot(x, nonaxis, marker="s", color="tab:orange", linewidth=1.5, label="non-axisymmetric |B| energy")
    ax.set_title("Boozer representation checks")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Jacobian identity error")
    ax2.set_ylabel("normalized spectral energy")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    ax = axes[1, 1]
    neumann_delta = np.abs(neumann - baseline) / np.maximum(np.abs(baseline), 1.0e-30)
    mass_delta = np.abs(mass - baseline) / np.maximum(np.abs(baseline), 1.0e-30)
    ax.bar(x - 0.18, neumann_delta, 0.36, label="Neumann vs Dirichlet")
    ax.bar(x + 0.18, mass_delta, 0.36, label="mass vs metric stiffness")
    ax.set_title("operator-variant spread")
    ax.set_ylabel("relative lowest-mode shift")
    ax.set_xticks(x, labels)
    ax.legend(fontsize=8)

    figure_paths = []
    for suffix in ("svg", "png"):
        path = output_dir / f"vmec_equilibria_boozer_global.{suffix}"
        fig.savefig(path, facecolor="white")
        figure_paths.append(str(path))
    plt.close(fig)

    json_path = output_dir / "vmec_equilibria_boozer_global_summary.json"
    json_path.write_text(json.dumps({"cases": packets}, indent=2, sort_keys=True) + "\n")

    csv_path = output_dir / "vmec_equilibria_boozer_global.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "case_id",
                "family",
                "surface_count_used",
                "mode_count",
                "jacobian_identity_error",
                "normalized_nonaxisymmetric_b_energy",
                "metric_dirichlet_hz",
                "metric_neumann_hz",
                "mass_dirichlet_hz",
                "metric_dirichlet_residual",
            ),
        )
        writer.writeheader()
        for packet in packets:
            writer.writerow(
                {
                    "case_id": packet["case_id"],
                    "family": packet["family"],
                    "surface_count_used": packet["surface_count_used"],
                    "mode_count": packet["mode_count"],
                    "jacobian_identity_error": packet["jacobian_identity_error"],
                    "normalized_nonaxisymmetric_b_energy": packet["normalized_nonaxisymmetric_b_energy"],
                    "metric_dirichlet_hz": packet["lowest_frequency_hz"]["metric_dirichlet"],
                    "metric_neumann_hz": packet["lowest_frequency_hz"]["metric_neumann"],
                    "mass_dirichlet_hz": packet["lowest_frequency_hz"]["mass_dirichlet"],
                    "metric_dirichlet_residual": packet["residuals"]["metric_dirichlet"],
                }
            )
    return [*figure_paths, str(json_path), str(csv_path)]


def main(argv: list[str] | None = None) -> int:
    """Run the public VMEC-equilibria Boozer/global artifact comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(vmec_equilibria_root()), help="Local landreman/vmec_equilibria checkout.")
    parser.add_argument("--output-dir", default="outputs/vmec_equilibria_boozer_global", help="Directory for artifacts.")
    parser.add_argument("--case", action="append", default=None, help="Optional case_id filter; may be repeated.")
    parser.add_argument("--surface-count", type=int, default=24, help="Maximum radial surfaces used per case.")
    parser.add_argument("--theta-points", type=int, default=8, help="Boozer theta quadrature points.")
    parser.add_argument("--zeta-points", type=int, default=8, help="Boozer zeta quadrature points.")
    parser.add_argument("--mass-density", type=float, default=3.4e-8, help="Mass density in kg/m^3.")
    parser.add_argument("--radial-stiffness-scale", type=float, default=1.0e-8, help="Scale applied to metric radial stiffness.")
    parser.add_argument("--radial-coupling-fraction", type=float, default=0.004, help="Mass-scaled radial stiffness fraction.")
    args = parser.parse_args(argv)

    requested = set(args.case or [])
    cases = available_vmec_equilibria_boozmn_cases(root=args.root)
    if requested:
        cases = tuple(case for case in cases if case.case_id in requested)
    if not cases:
        raise FileNotFoundError(f"no cataloged boozmn cases are available under {args.root}")

    packets = [_case_packet(case, args) for case in cases]
    artifact_files = _write_outputs(Path(args.output_dir), packets)
    payload = {
        "diagnostic": "VMEC-equilibria Boozer/global radial-operator artifact comparison",
        "root": str(Path(args.root).expanduser()),
        "case_count": len(packets),
        "cases": [packet["case_id"] for packet in packets],
        "artifact_files": artifact_files,
        "scope": "public boozmn artifact gate for FAX global radial 3-D operator; not an external-code ideal-MHD parity claim",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
