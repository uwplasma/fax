"""Command-line interface for ``fax``."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .config import ConfigError, FaxConfig, load_config
from .runner import run


EXAMPLE_CONFIG = """[run]
case = "minimal"
model = "continuum"
output_dir = "outputs"

[equilibrium]
source = "vmec_jax"
vmec_input = "../vmec_jax/input.nfp4_QH_warm_start_000_000000"
use_boozer = false

[grid]
radial_points = 64
m_max = 8
n_max = 8
field_periods = 4

[physics]
include_sound_coupling = false
include_fast_ions = false
fast_ion_model = "none"
density_scale = 1.0

[numerics]
precision = "float64"
jit = true
autodiff = true
eigensolver = "dense"
"""


def build_parser() -> argparse.ArgumentParser:
    """Build the ``fax`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="fax",
        description="Run or inspect differentiable Alfvén eigenmode calculations from a TOML file.",
    )
    parser.add_argument("config", nargs="?", help="Path to a fax TOML configuration file.")
    parser.add_argument("--dry-run", action="store_true", help="Validate input and print a planned-run summary.")
    parser.add_argument("--dump-config", action="store_true", help="Print the resolved configuration as JSON.")
    parser.add_argument("--example", action="store_true", help="Print an example TOML configuration and exit.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.example:
        print(EXAMPLE_CONFIG, end="")
        return 0

    if args.config is None:
        parser.error("provide a TOML config path or use --example")

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"fax: {exc}", file=sys.stderr)
        return 2

    if args.dump_config:
        print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
        return 0

    summary = run(config)
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))

    if args.dry_run:
        return 0
    if summary.status == "planned":
        return 0
    return 0


def parse_config_from_mapping(mapping: dict[str, object]) -> FaxConfig:
    """Small helper for tests and downstream wrappers."""
    return FaxConfig.from_mapping(mapping)

