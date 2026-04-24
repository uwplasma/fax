"""Write a small SVG coverage badge from ``coverage json`` output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape


def badge_color(percent: float) -> str:
    """Return a Shields-like color for a coverage percentage."""
    if percent >= 90.0:
        return "#4c1"
    if percent >= 80.0:
        return "#97ca00"
    if percent >= 70.0:
        return "#dfb317"
    if percent >= 60.0:
        return "#fe7d37"
    return "#e05d44"


def _text_width(text: str) -> int:
    return 10 + 7 * len(text)


def coverage_percent(path: str | Path) -> float:
    """Return total line coverage percentage from a coverage JSON file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        return float(payload["totals"]["percent_covered"])
    except KeyError as exc:
        raise ValueError(f"{path} is not a coverage.py JSON report") from exc


def render_badge(percent: float) -> str:
    """Render an SVG badge."""
    label = "coverage"
    value = f"{percent:.1f}%"
    label_width = _text_width(label)
    value_width = _text_width(value)
    total_width = label_width + value_width
    color = badge_color(percent)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20" role="img" aria-label="{escape(label)}: {escape(value)}">
  <title>{escape(label)}: {escape(value)}</title>
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r">
    <rect width="{total_width}" height="20" rx="3" fill="#fff"/>
  </clipPath>
  <g clip-path="url(#r)">
    <rect width="{label_width}" height="20" fill="#555"/>
    <rect x="{label_width}" width="{value_width}" height="20" fill="{color}"/>
    <rect width="{total_width}" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11">
    <text x="{label_width / 2:.1f}" y="15" fill="#010101" fill-opacity=".3">{escape(label)}</text>
    <text x="{label_width / 2:.1f}" y="14">{escape(label)}</text>
    <text x="{label_width + value_width / 2:.1f}" y="15" fill="#010101" fill-opacity=".3">{escape(value)}</text>
    <text x="{label_width + value_width / 2:.1f}" y="14">{escape(value)}</text>
  </g>
</svg>
"""


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coverage_json", type=Path)
    parser.add_argument("output_svg", type=Path)
    args = parser.parse_args(argv)

    percent = coverage_percent(args.coverage_json)
    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_svg.write_text(render_badge(percent), encoding="utf-8")
    print(f"{args.output_svg}: {percent:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
