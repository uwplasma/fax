"""Plot the archived FAR3D GPU fast-ion beta scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fax.far3d import Far3dLincheckScanPoint, far3d_lincheck_parameter_scan, read_far3d_lincheck

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "external_artifacts/far3d/gpu_diiid_nvhpc_beta_scan"


def _discover_outputs(input_dir: Path) -> dict[float, Path]:
    outputs: dict[float, Path] = {}
    for path in sorted(input_dir.glob("beta_*/farprt0000")):
        beta = float(path.parent.name.split("_", maxsplit=1)[1].replace("p", "."))
        outputs[beta] = path
    if not outputs:
        raise FileNotFoundError(f"no beta_*/farprt0000 files found under {input_dir}")
    return outputs


def _runtime_seconds(step_dir: Path) -> float:
    fields = (step_dir / "runtime.txt").read_text(encoding="utf-8").split()
    return float(fields[1])


def _finite_beta(points: tuple[Far3dLincheckScanPoint, ...]) -> tuple[Far3dLincheckScanPoint, ...]:
    return tuple(point for point in points if point.parameter_value > 0.0)


def build_figure(input_dir: Path, output_dir: Path) -> dict[str, object]:
    """Create a FAR3D fast-ion beta scan dashboard."""
    outputs = _discover_outputs(input_dir)
    points = far3d_lincheck_parameter_scan(outputs)
    finite_beta = _finite_beta(points)
    output_dir.mkdir(parents=True, exist_ok=True)

    beta = np.asarray([point.parameter_value for point in points], dtype=float)
    mean_growth = np.asarray([point.mean_growth_code_units for point in points], dtype=float)
    max_growth = np.asarray([point.max_growth_code_units for point in points], dtype=float)
    mean_omega = np.asarray([point.mean_omega_code_units for point in points], dtype=float)
    spread_growth = np.asarray([point.spread_growth_code_units for point in points], dtype=float)
    runtime = np.asarray([_runtime_seconds(outputs[point.parameter_value].parent) for point in points], dtype=float)

    finite_beta_values = np.asarray([point.parameter_value for point in finite_beta], dtype=float)
    finite_growth = np.asarray([point.mean_growth_code_units for point in finite_beta], dtype=float)
    slope, intercept = np.polyfit(finite_beta_values, finite_growth, deg=1)
    fitted_growth = slope * finite_beta_values + intercept
    finite_mask = beta > 0.0
    control_mask = ~finite_mask

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0), constrained_layout=True)
    axes[0, 0].plot(finite_beta_values, finite_growth, marker="o", label="finite-beta mean gamma")
    axes[0, 0].plot(beta[finite_mask], max_growth[finite_mask], marker="s", label="finite-beta max gamma")
    axes[0, 0].plot(finite_beta_values, fitted_growth, color="0.25", ls="--", label="finite-beta linear fit")
    if np.any(control_mask):
        axes[0, 0].scatter(
            beta[control_mask],
            mean_growth[control_mask],
            marker="D",
            facecolors="none",
            edgecolors="#4c78a8",
            s=70,
            label="beta=0 control",
            zorder=3,
        )
    axes[0, 0].set_title("FAR3D finite-beta growth trend")
    axes[0, 0].set_xlabel("bet0_f")
    axes[0, 0].set_ylabel("gamma [code units]")
    axes[0, 0].legend()

    axes[0, 1].plot(beta[finite_mask], mean_omega[finite_mask], marker="o", color="#f58518", label="finite beta")
    if np.any(control_mask):
        axes[0, 1].scatter(
            beta[control_mask],
            mean_omega[control_mask],
            marker="D",
            facecolors="none",
            edgecolors="#f58518",
            s=70,
            label="beta=0 control",
            zorder=3,
        )
    axes[0, 1].axhline(0.0, color="0.25", lw=1.0)
    axes[0, 1].set_title("Mean real frequency")
    axes[0, 1].set_xlabel("bet0_f")
    axes[0, 1].set_ylabel("omega_r [code units]")
    axes[0, 1].legend()

    axes[1, 0].semilogy(beta, np.maximum(spread_growth, 1.0e-16), marker="o", color="#54a24b")
    axes[1, 0].set_title("Across-field/harmonic spread")
    axes[1, 0].set_xlabel("bet0_f")
    axes[1, 0].set_ylabel("gamma spread [code units]")

    high_beta_rows = read_far3d_lincheck(outputs[max(outputs)])
    high_beta_growth = np.asarray([row.growth_rate_code_units for row in high_beta_rows], dtype=float)
    high_beta_omega = np.asarray([row.omega_code_units for row in high_beta_rows], dtype=float)
    axes[1, 1].scatter(high_beta_growth, high_beta_omega, s=30, alpha=0.75)
    axes[1, 1].set_title("Highest-beta lincheck mode cloud")
    axes[1, 1].set_xlabel("gamma [code units]")
    axes[1, 1].set_ylabel("omega_r [code units]")
    axes[1, 1].annotate(f"scan runtime: {runtime.sum():.0f} s", xy=(0.04, 0.90), xycoords="axes fraction")

    fig.suptitle("FAR3D GPU DIII-D fast-ion beta scan", fontsize=13)
    paths = []
    for suffix in ("png", "svg"):
        path = output_dir / f"far3d_gpu_beta_scan.{suffix}"
        fig.savefig(path, dpi=180)
        paths.append(str(path))
    plt.close(fig)

    metrics = {
        "input_dir": str(input_dir),
        "beta_values": [point.parameter_value for point in points],
        "total_runtime_seconds": float(runtime.sum()),
        "finite_beta_linear_slope": float(slope),
        "finite_beta_linear_intercept": float(intercept),
        "beta_zero_control_growth_spread": float(spread_growth[control_mask][0]) if np.any(control_mask) else None,
        "scan": [point.to_dict() for point in points],
        "figures": paths,
        "scope": (
            "FAR3D GPU single-geometry finite-beta scan with beta=0 flagged as a control; "
            "not calibrated FAX/FAR3D kinetic parity"
        ),
    }
    (output_dir / "far3d_gpu_beta_scan_summary.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Directory containing beta_*/farprt0000.")
    parser.add_argument("--output-dir", default="outputs/far3d_gpu_beta_scan", help="Output directory.")
    args = parser.parse_args()
    metrics = build_figure(Path(args.input_dir), Path(args.output_dir))
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
