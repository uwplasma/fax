"""Check generated documentation figures for blank or malformed outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


def _ink_fraction(path: Path) -> float:
    image = Image.open(path).convert("RGBA")
    array = np.asarray(image)
    rgb = array[..., :3].astype(np.int16)
    alpha = array[..., 3]
    nonwhite = np.any(rgb < 245, axis=-1) & (alpha > 0)
    return float(np.count_nonzero(nonwhite) / nonwhite.size)


def _check_png(path: Path, *, min_ink_fraction: float) -> str:
    image = Image.open(path)
    width, height = image.size
    if width < 200 or height < 120:
        raise ValueError(f"{path} is too small: {width}x{height}")
    ink_fraction = _ink_fraction(path)
    if ink_fraction < min_ink_fraction:
        raise ValueError(f"{path} appears blank: ink_fraction={ink_fraction:.4f}")
    return f"{path.name}: {width}x{height}, ink_fraction={ink_fraction:.4f}"


def _check_svg(path: Path) -> str:
    root = ET.parse(path).getroot()
    tag_count = sum(1 for _node in root.iter())
    if tag_count < 10:
        raise ValueError(f"{path} appears malformed or empty: tag_count={tag_count}")
    return f"{path.name}: svg_tags={tag_count}"


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", default="docs/_static", help="Directory containing generated figures.")
    parser.add_argument("--min-ink-fraction", type=float, default=0.01, help="Minimum non-white pixel fraction.")
    args = parser.parse_args(argv)

    figure_dir = Path(args.figure_dir).expanduser().resolve()
    pngs = sorted(figure_dir.glob("*.png"))
    svgs = sorted(figure_dir.glob("*.svg"))
    if not pngs:
        raise FileNotFoundError(f"no PNG figures found in {figure_dir}")
    if not svgs:
        raise FileNotFoundError(f"no SVG figures found in {figure_dir}")

    for path in svgs:
        print(_check_svg(path))
    for path in pngs:
        print(_check_png(path, min_ink_fraction=float(args.min_ink_fraction)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
