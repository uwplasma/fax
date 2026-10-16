"""Plot the archived FAR3D GPU ``lincheck`` artifact.

This example uses the short DIII-D FAR3D run archived under
``external_artifacts/far3d/gpu_diiid_nvhpc``.  It is intentionally a build and
ingestion traceability figure, not a completed kinetic parity claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

from fax.reference_cases import portable_paths
from fax.far3d import read_far3d_lincheck

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "external_artifacts/far3d/gpu_diiid_nvhpc/farprt0000"


def _frequency_scale_khz(path: Path) -> float | None:
    pattern = re.compile(r"Conversion factor from code frequency to kHz:\s*([0-9.EeDd+-]+)")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match is not None:
            return float(match.group(1).replace("D", "E").replace("d", "e"))
    return None


def build_figure(input_path: Path, output_dir: Path) -> dict[str, object]:
    """Create the FAR3D ``lincheck`` dashboard and return plotted metrics."""
    rows = read_far3d_lincheck(input_path)
    if not rows:
        raise ValueError(f"no FAR3D lincheck rows found in {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    scale_khz = _frequency_scale_khz(input_path)
    variables = tuple(dict.fromkeys(row.variable for row in rows))
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, len(variables)))

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.2), constrained_layout=True)
    for variable, color in zip(variables, colors, strict=True):
        selected = [row for row in rows if row.variable == variable]
        m = np.asarray([row.m for row in selected], dtype=float)
        growth = np.asarray([row.growth_rate_code_units for row in selected], dtype=float)
        omega = np.asarray([abs(row.omega_code_units) for row in selected], dtype=float)
        axes[0, 0].plot(m, growth, marker="o", lw=1.2, ms=4, label=variable, color=color)
        if scale_khz is not None:
            axes[0, 1].plot(m, omega * scale_khz, marker="o", lw=1.2, ms=4, label=variable, color=color)
        else:
            axes[0, 1].plot(m, omega, marker="o", lw=1.2, ms=4, label=variable, color=color)

    max_growth = [max(row.growth_rate_code_units for row in rows if row.variable == variable) for variable in variables]
    max_frequency = [max(abs(row.omega_code_units) for row in rows if row.variable == variable) for variable in variables]
    if scale_khz is not None:
        max_frequency = [value * scale_khz for value in max_frequency]

    axes[0, 0].axhline(0.0, color="0.2", lw=1.0)
    axes[0, 0].set_title("FAR3D lincheck growth estimates")
    axes[0, 0].set_xlabel("poloidal harmonic m")
    axes[0, 0].set_ylabel("growth rate gamma [code units]")
    axes[0, 0].legend(ncol=2, fontsize=8)

    axes[0, 1].set_title("FAR3D real frequency estimates")
    axes[0, 1].set_xlabel("poloidal harmonic m")
    axes[0, 1].set_ylabel("frequency [kHz]" if scale_khz is not None else "omega_r [code units]")

    axes[1, 0].bar(variables, max_growth, color=colors)
    axes[1, 0].axhline(0.0, color="0.2", lw=1.0)
    axes[1, 0].set_title("Largest growth by evolved field")
    axes[1, 0].set_ylabel("max gamma [code units]")
    axes[1, 0].tick_params(axis="x", rotation=25)

    axes[1, 1].bar(variables, max_frequency, color=colors)
    axes[1, 1].set_title("Largest absolute frequency by evolved field")
    axes[1, 1].set_ylabel("max |f| [kHz]" if scale_khz is not None else "max |omega_r| [code units]")
    axes[1, 1].tick_params(axis="x", rotation=25)

    title = "FAR3D GPU DIII-D public-example lincheck artifact"
    subtitle = "RTX A4000/NVHPC build-run trace; not a calibrated kinetic parity gate"
    fig.suptitle(f"{title}\n{subtitle}", fontsize=13)

    paths = []
    for suffix in ("png", "svg"):
        path = output_dir / f"far3d_gpu_lincheck_artifact.{suffix}"
        fig.savefig(path, dpi=180)
        paths.append(str(path))
    plt.close(fig)

    metrics = {
        "input_path": str(input_path),
        "row_count": len(rows),
        "variables": list(variables),
        "frequency_scale_khz_per_code_unit": scale_khz,
        "max_growth_code_units": float(max(row.growth_rate_code_units for row in rows)),
        "min_growth_code_units": float(min(row.growth_rate_code_units for row in rows)),
        "max_abs_omega_code_units": float(max(abs(row.omega_code_units) for row in rows)),
        "figures": paths,
        "scope": "FAR3D executable/build and ingestion artifact; not calibrated kinetic parity",
    }
    (output_dir / "far3d_gpu_lincheck_artifact_summary.json").write_text(
        json.dumps(portable_paths(metrics), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="FAR3D farprt/lincheck output.")
    parser.add_argument("--output-dir", default="outputs/far3d_gpu_lincheck_artifact", help="Output directory.")
    args = parser.parse_args()
    metrics = build_figure(Path(args.input), Path(args.output_dir))
    print(json.dumps(portable_paths(metrics), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
