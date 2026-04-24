"""Command-line interface for ``fax``."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .config import ConfigError, FaxConfig, load_config
from .runner import run
from .workflows import ContinuumRunSummary, ValidationRunSummary, run_continuum, run_validation


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
        description=(
            "Run or inspect differentiable Alfvén eigenmode calculations from a TOML file. "
            "Subcommands: validate, continuum."
        ),
    )
    parser.add_argument("config", nargs="?", help="Path to a fax TOML configuration file.")
    parser.add_argument("--dry-run", action="store_true", help="Validate input and print a planned-run summary.")
    parser.add_argument("--dump-config", action="store_true", help="Print the resolved configuration as JSON.")
    parser.add_argument("--json", action="store_true", help="Print only the run summary JSON; disables rich progress output.")
    parser.add_argument("--quiet", action="store_true", help="Suppress rich progress output and print only summary JSON.")
    parser.add_argument("--example", action="store_true", help="Print an example TOML configuration and exit.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def build_command_parser() -> argparse.ArgumentParser:
    """Build the subcommand parser for executable workflows."""
    parser = argparse.ArgumentParser(
        prog="fax",
        description="Run FAX validation and continuum workflows from TOML configuration files.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Run registered validation gates.")
    validate.add_argument("config", help="Path to a fax TOML configuration file.")
    validate.add_argument("--tag", action="append", default=[], help="Run only gates with this tag; may be repeated.")
    validate.add_argument("--json", action="store_true", help="Print only summary JSON; disables rich progress output.")
    validate.add_argument("--quiet", action="store_true", help="Suppress rich progress output and print only summary JSON.")
    validate.add_argument("--no-artifacts", action="store_true", help="Do not write validation_summary.json/report files.")

    continuum = subparsers.add_parser("continuum", help="Compute continuum branches and metric diagnostics.")
    continuum.add_argument("config", help="Path to a fax TOML configuration file.")
    continuum.add_argument("--json", action="store_true", help="Print only summary JSON; disables rich progress output.")
    continuum.add_argument("--quiet", action="store_true", help="Suppress rich progress output and print only summary JSON.")
    continuum.add_argument("--no-artifacts", action="store_true", help="Do not write continuum artifact files.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if argv_list and argv_list[0] in {"validate", "continuum"}:
        return _main_command(argv_list)

    parser = build_parser()
    args = parser.parse_args(argv_list)

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

    machine_readable = bool(args.json or args.quiet)
    summary = run(config, progress=not machine_readable, dry_run=bool(args.dry_run))
    if machine_readable:
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))

    if args.dry_run:
        return 0
    if summary.status == "planned":
        return 0
    return 0


def _main_command(argv: list[str]) -> int:
    parser = build_command_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"fax: {exc}", file=sys.stderr)
        return 2

    machine_readable = bool(args.json or args.quiet)
    try:
        summary: ValidationRunSummary | ContinuumRunSummary
        if args.command == "validate":
            summary = run_validation(
                config,
                progress=not machine_readable,
                tags=tuple(args.tag),
                write_artifacts=not args.no_artifacts,
            )
        elif args.command == "continuum":
            summary = run_continuum(
                config,
                progress=not machine_readable,
                write_artifacts=not args.no_artifacts,
            )
        else:  # pragma: no cover - argparse prevents this branch
            parser.error(f"unknown command {args.command!r}")
    except (ConfigError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"fax: {exc}", file=sys.stderr)
        return 2

    if machine_readable:
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 1 if summary.status == "failed" else 0


def parse_config_from_mapping(mapping: dict[str, object]) -> FaxConfig:
    """Small helper for tests and downstream wrappers."""
    return FaxConfig.from_mapping(mapping)
