"""Generate live Boozer geometry from VMEC-equilibria ``wout`` files.

This example runs ``booz_xform_jax`` directly on public
``landreman/vmec_equilibria`` VMEC outputs, then passes the generated
``BoozerGeometry`` into FAX's global radial 3-D operator. It is the
reproducible generated-Boozer companion to the public ``boozmn`` artifact
example.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from fax.boozer import (
    boozer_geometry_field_relative_error,
    boozer_geometry_jacobian_identity_error,
    run_booz_xform_jax_geometry_from_wout,
)
from fax.global_modes import radial_block_problem_from_boozer_geometry, solve_radial_block_eigenmodes
from fax.reference_cases import vmec_equilibria_catalog, vmec_equilibria_root

jax.config.update("jax_enable_x64", True)


DEFAULT_CASES = ("w7x_standard", "ncsx_li383")
DEFAULT_SURFACES = (0.2, 0.35, 0.5, 0.65, 0.8)
DEFAULT_HARMONICS = ((1.0, 0.0), (2.0, 1.0))


def _catalog_by_id() -> dict[str, Any]:
    return {case.case_id: case for case in vmec_equilibria_catalog()}


def _global_packet(geometry: Any, args: argparse.Namespace) -> tuple[Any, Any]:
    problem = radial_block_problem_from_boozer_geometry(
        geometry,
        jnp.asarray(DEFAULT_HARMONICS, dtype=float),
        mass_density=float(args.mass_density),
        theta_points=int(args.theta_points),
        zeta_points=int(args.zeta_points),
        radial_stiffness_scale=float(args.radial_stiffness_scale),
    )
    return problem, solve_radial_block_eigenmodes(problem)


def _run_case(case: Any, args: argparse.Namespace) -> dict[str, Any]:
    wout = case.wout_path(root=args.root)
    if not wout.exists():
        raise FileNotFoundError(f"missing VMEC-equilibria wout for {case.case_id}: {wout}")

    start = time.perf_counter()
    coarse = run_booz_xform_jax_geometry_from_wout(
        wout,
        mboz=int(args.coarse_mboz),
        nboz=int(args.coarse_nboz),
        surfaces=tuple(float(value) for value in args.surface),
        jit=bool(args.jit),
    )
    coarse_seconds = time.perf_counter() - start

    start = time.perf_counter()
    reference = run_booz_xform_jax_geometry_from_wout(
        wout,
        mboz=int(args.reference_mboz),
        nboz=int(args.reference_nboz),
        surfaces=tuple(float(value) for value in args.surface),
        jit=bool(args.jit),
    )
    reference_seconds = time.perf_counter() - start

    _, coarse_result = _global_packet(coarse, args)
    _, reference_result = _global_packet(reference, args)
    frequency_relative_error = float(
        abs(coarse_result.frequencies_hz[0] - reference_result.frequencies_hz[0])
        / max(abs(float(reference_result.frequencies_hz[0])), 1.0e-30)
    )
    envelope = np.linalg.norm(np.asarray(reference_result.mode_structures[0]), axis=1)
    envelope = envelope / max(float(np.max(envelope)), 1.0e-30)

    return {
        "case_id": case.case_id,
        "family": case.family,
        "wout": str(wout),
        "surfaces": [float(value) for value in np.asarray(reference.s)],
        "field_periods": int(reference.field_periods),
        "coarse_modes": int(coarse.xm.shape[0]),
        "reference_modes": int(reference.xm.shape[0]),
        "coarse_runtime_seconds": float(coarse_seconds),
        "reference_runtime_seconds": float(reference_seconds),
        "coarse_jacobian_identity_error": float(
            boozer_geometry_jacobian_identity_error(coarse, theta_points=int(args.theta_points), zeta_points=int(args.zeta_points))
        ),
        "reference_jacobian_identity_error": float(
            boozer_geometry_jacobian_identity_error(reference, theta_points=int(args.theta_points), zeta_points=int(args.zeta_points))
        ),
        "field_relative_error": float(
            jnp.max(
                boozer_geometry_field_relative_error(
                    coarse,
                    reference,
                    theta_points=int(args.theta_points),
                    zeta_points=int(args.zeta_points),
                )
            )
        ),
        "lowest_frequency_hz_coarse": float(coarse_result.frequencies_hz[0]),
        "lowest_frequency_hz_reference": float(reference_result.frequencies_hz[0]),
        "lowest_frequency_relative_error": frequency_relative_error,
        "reference_residual": float(np.max(np.asarray(reference_result.residuals[:4]))),
        "reference_envelope": [float(value) for value in envelope],
    }


def _write_outputs(output_dir: Path, packets: list[dict[str, Any]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [packet["case_id"].replace("_", "\n") for packet in packets]
    x = np.arange(len(packets))

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), constrained_layout=True)

    ax = axes[0, 0]
    ax.bar(x - 0.18, [packet["coarse_runtime_seconds"] for packet in packets], 0.36, label="coarse")
    ax.bar(x + 0.18, [packet["reference_runtime_seconds"] for packet in packets], 0.36, label="reference")
    ax.set_title("live booz_xform_jax runtime")
    ax.set_ylabel("seconds")
    ax.set_xticks(x, labels)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.semilogy(x, [packet["coarse_jacobian_identity_error"] for packet in packets], marker="o", label="coarse")
    ax.semilogy(x, [packet["reference_jacobian_identity_error"] for packet in packets], marker="s", label="reference")
    ax.set_title("Boozer-Jacobian identity")
    ax.set_ylabel("relative error")
    ax.set_xticks(x, labels)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.bar(x - 0.18, [packet["field_relative_error"] for packet in packets], 0.36, label="|B| field")
    ax.bar(x + 0.18, [packet["lowest_frequency_relative_error"] for packet in packets], 0.36, label="lowest global mode")
    ax.set_title("coarse-to-reference convergence")
    ax.set_ylabel("relative difference")
    ax.set_xticks(x, labels)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for packet in packets:
        ax.plot(packet["surfaces"], packet["reference_envelope"], marker="o", linewidth=1.5, label=packet["case_id"].replace("_", " "))
    ax.set_title("reference mode-1 envelopes")
    ax.set_xlabel("normalized toroidal flux s")
    ax.set_ylabel("normalized envelope")
    ax.legend(fontsize=8)

    paths = []
    for suffix in ("svg", "png"):
        path = output_dir / f"vmec_equilibria_live_boozer.{suffix}"
        fig.savefig(path, facecolor="white")
        paths.append(str(path))
    plt.close(fig)

    json_path = output_dir / "vmec_equilibria_live_boozer_summary.json"
    json_path.write_text(json.dumps({"cases": packets}, indent=2, sort_keys=True) + "\n")
    csv_path = output_dir / "vmec_equilibria_live_boozer.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "case_id",
                "coarse_runtime_seconds",
                "reference_runtime_seconds",
                "coarse_jacobian_identity_error",
                "reference_jacobian_identity_error",
                "field_relative_error",
                "lowest_frequency_relative_error",
                "lowest_frequency_hz_reference",
                "reference_residual",
            ),
        )
        writer.writeheader()
        for packet in packets:
            writer.writerow({key: packet[key] for key in writer.fieldnames or ()})
    return [*paths, str(json_path), str(csv_path)]


def main(argv: list[str] | None = None) -> int:
    """Run live VMEC-equilibria Boozer generation and FAX global-mode checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(vmec_equilibria_root()), help="Local landreman/vmec_equilibria checkout.")
    parser.add_argument("--output-dir", default="outputs/vmec_equilibria_live_boozer", help="Directory for artifacts.")
    parser.add_argument("--case", action="append", default=None, help="Case id; defaults to W7-X standard and NCSX li383.")
    parser.add_argument("--surface", action="append", type=float, default=None, help="Boozer surface in normalized toroidal flux.")
    parser.add_argument("--coarse-mboz", type=int, default=4)
    parser.add_argument("--coarse-nboz", type=int, default=4)
    parser.add_argument("--reference-mboz", type=int, default=5)
    parser.add_argument("--reference-nboz", type=int, default=5)
    parser.add_argument("--theta-points", type=int, default=12)
    parser.add_argument("--zeta-points", type=int, default=12)
    parser.add_argument("--mass-density", type=float, default=3.4e-8)
    parser.add_argument("--radial-stiffness-scale", type=float, default=1.0e-8)
    parser.add_argument("--jit", action="store_true", help="JIT the booz_xform_jax transform.")
    args = parser.parse_args(argv)
    args.surface = tuple(args.surface or DEFAULT_SURFACES)

    catalog = _catalog_by_id()
    case_ids = tuple(args.case or DEFAULT_CASES)
    packets = [_run_case(catalog[case_id], args) for case_id in case_ids]
    artifact_files = _write_outputs(Path(args.output_dir), packets)
    payload = {
        "diagnostic": "live VMEC-equilibria booz_xform_jax to FAX global operator",
        "root": str(Path(args.root).expanduser()),
        "cases": case_ids,
        "artifact_files": artifact_files,
        "scope": "generated-Boozer convergence gate for public VMEC wout files; not external-code parity",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
