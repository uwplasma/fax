"""Generate a fast BAE/TAE/EAE Alfvén-continuum gap map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fax.gap_models import basic_alfven_gap_map
from fax.plotting import plot_basic_alfven_gap_map


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "basic_alfven_gap_map",
        help="Directory for the generated PNG/SVG figure.",
    )
    parser.add_argument(
        "--format",
        dest="formats",
        action="append",
        choices=("png", "svg", "pdf"),
        help="Figure format to write. Repeat to write multiple formats. Defaults to png and svg.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Write the figure and print a compact JSON summary."""
    args = _parser().parse_args(argv)
    formats = tuple(args.formats) if args.formats else ("png", "svg")
    demo = basic_alfven_gap_map()
    paths = plot_basic_alfven_gap_map(args.output_dir, demo=demo, formats=formats)

    payload = {
        "diagnostic": "basic large-aspect-ratio Alfven continuum gap map",
        "figure_files": [str(path) for path in paths],
        "model_scope": "analytic circular tokamak, n=1, large-aspect-ratio continuum plus minimal avoided crossings",
        "tae_gap": {
            "q": float(1.0 / demo.tae.estimate.iota),
            "surface": float(demo.tae.estimate.surface),
            "center_frequency_hz": float(demo.tae.estimate.center_frequency_hz),
            "width_frequency_hz": float(demo.tae.estimate.width_frequency_hz),
        },
        "eae_gap": {
            "q": float(1.0 / demo.eae.estimate.iota),
            "surface": float(demo.eae.estimate.surface),
            "center_frequency_hz": float(demo.eae.estimate.center_frequency_hz),
            "width_frequency_hz": float(demo.eae.estimate.width_frequency_hz),
        },
        "bae_gap": {
            "q": float(demo.bae_q),
            "surface": float(demo.bae_surface),
            "accumulation_frequency_hz": float(demo.bae_frequency_at_surface_hz),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
