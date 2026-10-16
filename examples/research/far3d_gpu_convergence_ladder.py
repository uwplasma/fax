"""Plot the archived FAR3D GPU run-length convergence ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fax.reference_cases import portable_paths
from fax.far3d import far3d_lincheck_convergence, read_far3d_lincheck

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "external_artifacts/far3d/gpu_diiid_nvhpc_convergence"


def _discover_outputs(input_dir: Path) -> dict[int, Path]:
    outputs: dict[int, Path] = {}
    for path in sorted(input_dir.glob("step_*/farprt0000")):
        step = int(path.parent.name.split("_", maxsplit=1)[1])
        outputs[step] = path
    if not outputs:
        raise FileNotFoundError(f"no step_*/farprt0000 files found under {input_dir}")
    return outputs


def _runtime_seconds(step_dir: Path) -> float:
    fields = (step_dir / "runtime.txt").read_text(encoding="utf-8").split()
    return float(fields[1])


def build_figure(input_dir: Path, output_dir: Path) -> dict[str, object]:
    """Create a FAR3D run-length convergence dashboard."""
    outputs = _discover_outputs(input_dir)
    convergence = far3d_lincheck_convergence(outputs)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = np.asarray([point.step_count for point in convergence], dtype=float)
    runtimes = np.asarray([_runtime_seconds(outputs[point.step_count].parent) for point in convergence], dtype=float)
    growth_error = np.asarray([point.max_relative_growth_delta_to_reference for point in convergence], dtype=float)
    omega_error = np.asarray([point.max_relative_omega_delta_to_reference for point in convergence], dtype=float)
    growth_spread = np.asarray([point.spread_growth_code_units for point in convergence], dtype=float)
    omega_spread = np.asarray([point.spread_omega_code_units for point in convergence], dtype=float)
    mean_growth = np.asarray([point.mean_growth_code_units for point in convergence], dtype=float)
    mean_omega = np.asarray([point.mean_omega_code_units for point in convergence], dtype=float)

    final_rows = read_far3d_lincheck(outputs[int(steps[-1])])
    final_growth = np.asarray([row.growth_rate_code_units for row in final_rows], dtype=float)
    final_omega = np.asarray([row.omega_code_units for row in final_rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0), constrained_layout=True)
    axes[0, 0].plot(steps, mean_growth, marker="o", label="mean gamma")
    axes[0, 0].plot(steps, mean_omega, marker="s", label="mean omega_r")
    axes[0, 0].set_title("Run-length convergence of lincheck means")
    axes[0, 0].set_xlabel("FAR3D maxstp")
    axes[0, 0].set_ylabel("code units")
    axes[0, 0].legend()

    axes[0, 1].semilogy(steps, np.maximum(growth_error, 1.0e-16), marker="o", label="gamma")
    axes[0, 1].semilogy(steps, np.maximum(omega_error, 1.0e-16), marker="s", label="omega_r")
    axes[0, 1].axhline(1.0e-3, color="0.2", ls="--", lw=1.0, label="screening gate")
    axes[0, 1].set_title("Max mode-by-mode error vs 400-step reference")
    axes[0, 1].set_xlabel("FAR3D maxstp")
    axes[0, 1].set_ylabel("relative error")
    axes[0, 1].legend()

    axes[1, 0].semilogy(steps, np.maximum(growth_spread, 1.0e-16), marker="o", label="gamma spread")
    axes[1, 0].semilogy(steps, np.maximum(omega_spread, 1.0e-16), marker="s", label="omega_r spread")
    axes[1, 0].set_title("Across-field/harmonic collapse")
    axes[1, 0].set_xlabel("FAR3D maxstp")
    axes[1, 0].set_ylabel("max-min spread [code units]")
    axes[1, 0].legend()

    axes[1, 1].scatter(final_growth, final_omega, s=30, alpha=0.75)
    axes[1, 1].set_title("400-step lincheck mode cloud")
    axes[1, 1].set_xlabel("gamma [code units]")
    axes[1, 1].set_ylabel("omega_r [code units]")
    axes[1, 1].annotate(f"runtime ladder: {runtimes.sum():.0f} s", xy=(0.03, 0.92), xycoords="axes fraction")

    fig.suptitle("FAR3D GPU DIII-D run-length convergence ladder", fontsize=13)
    paths = []
    for suffix in ("png", "svg"):
        path = output_dir / f"far3d_gpu_convergence_ladder.{suffix}"
        fig.savefig(path, dpi=180)
        paths.append(str(path))
    plt.close(fig)

    metrics = {
        "input_dir": str(input_dir),
        "steps": [int(step) for step in steps],
        "total_runtime_seconds": float(runtimes.sum()),
        "convergence": [point.to_dict() for point in convergence],
        "figures": paths,
        "scope": "FAR3D GPU run-length convergence artifact; still not multi-geometry kinetic parity",
    }
    (output_dir / "far3d_gpu_convergence_ladder_summary.json").write_text(
        json.dumps(portable_paths(metrics), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Directory containing step_*/farprt0000.")
    parser.add_argument("--output-dir", default="outputs/far3d_gpu_convergence_ladder", help="Output directory.")
    args = parser.parse_args()
    metrics = build_figure(Path(args.input_dir), Path(args.output_dir))
    print(json.dumps(portable_paths(metrics), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
