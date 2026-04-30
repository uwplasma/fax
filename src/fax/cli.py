"""Command-line interface for ``fax``."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .benchmark_manifest import benchmark_manifest_markdown, benchmark_manifest_summary
from .benchmarks import known_benchmark_families
from .config import ConfigError, FaxConfig, load_config
from .maturity import research_grade_report_markdown, research_grade_summary
from .plotting import plot_continuum_artifacts, plot_convergence_artifacts, plot_global_eigenmode_artifacts
from .runner import run
from .workflows import (
    BenchmarkRunSummary,
    ContinuumRunSummary,
    ConvergenceRunSummary,
    GlobalEigenmodeRunSummary,
    ValidationRunSummary,
    run_benchmark,
    run_continuum,
    run_convergence,
    run_global_eigenmodes,
    run_validation,
)


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
            "Subcommands: validate, continuum, eigenmodes, convergence, benchmark, benchmark-manifest, "
            "maturity, plot-continuum, plot-global, plot-convergence."
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

    eigenmodes = subparsers.add_parser("eigenmodes", help="Compute global radial eigenmodes and mode structures.")
    eigenmodes.add_argument("config", help="Path to a fax TOML configuration file.")
    eigenmodes.add_argument("--max-modes", type=int, default=8, help="Number of lowest modes to report in artifacts.")
    eigenmodes.add_argument("--json", action="store_true", help="Print only summary JSON; disables rich progress output.")
    eigenmodes.add_argument("--quiet", action="store_true", help="Suppress rich progress output and print only summary JSON.")
    eigenmodes.add_argument("--no-artifacts", action="store_true", help="Do not write global eigenmode artifact files.")

    convergence = subparsers.add_parser("convergence", help="Run radial/angular/Fourier convergence sweeps.")
    convergence.add_argument("config", help="Path to a fax TOML configuration file.")
    convergence.add_argument(
        "--radial-points",
        type=int,
        action="append",
        default=[],
        help="Radial grid point count; may be repeated. Defaults to grid.radial_points from the TOML file.",
    )
    convergence.add_argument(
        "--angular-points",
        type=int,
        action="append",
        default=[],
        help="Theta/zeta grid point count; may be repeated. Defaults to 12, 16, 24.",
    )
    convergence.add_argument(
        "--mode-max",
        type=int,
        action="append",
        default=[],
        help="Use m_max=n_max for one sweep level; may be repeated. Defaults to 2, 3, 4.",
    )
    convergence.add_argument("--json", action="store_true", help="Print only summary JSON; disables rich progress output.")
    convergence.add_argument("--quiet", action="store_true", help="Suppress rich progress output and print only summary JSON.")
    convergence.add_argument("--no-artifacts", action="store_true", help="Do not write convergence artifact files.")

    benchmark = subparsers.add_parser("benchmark", help="Benchmark the current continuum workflow.")
    benchmark.add_argument("config", help="Path to a fax TOML configuration file.")
    benchmark.add_argument("--repeats", type=int, default=3, help="Number of warm repeats.")
    benchmark.add_argument("--warmups", type=int, default=1, help="Number of warmup runs before timing repeats.")
    benchmark.add_argument(
        "--family",
        action="append",
        default=[],
        choices=("continuum",) + known_benchmark_families(),
        help=(
            "Additional named physics benchmark family to include; may be repeated. "
            "The continuum workflow is always included as the public baseline."
        ),
    )
    benchmark.add_argument("--json", action="store_true", help="Print only summary JSON; disables rich progress output.")
    benchmark.add_argument("--quiet", action="store_true", help="Suppress rich progress output and print only summary JSON.")
    benchmark.add_argument("--no-artifacts", action="store_true", help="Do not write benchmark artifact files.")

    maturity = subparsers.add_parser(
        "maturity",
        help="Report remaining milestones for full no-proxy research-grade status.",
    )
    maturity.add_argument("--json", action="store_true", help="Print the maturity report as JSON.")
    maturity.add_argument("--quiet", action="store_true", help="Alias for --json.")

    benchmark_manifest = subparsers.add_parser(
        "benchmark-manifest",
        help="Report built-in external-code benchmark artifacts and claim levels.",
    )
    benchmark_manifest.add_argument("--json", action="store_true", help="Print the benchmark manifest as JSON.")
    benchmark_manifest.add_argument("--quiet", action="store_true", help="Alias for --json.")
    benchmark_manifest.add_argument("--hashes", action="store_true", help="Include SHA-256 hashes for present files.")

    plot = subparsers.add_parser("plot-continuum", help="Plot artifacts written by fax continuum.")
    plot.add_argument("artifact_dir", help="Directory containing continuum_summary.json and continuum_branches.csv.")
    plot.add_argument("--output-dir", help="Directory for generated figures; defaults to artifact_dir.")
    plot.add_argument("--stem", default="continuum_artifact_dashboard", help="Output figure filename stem.")
    plot.add_argument(
        "--format",
        action="append",
        default=[],
        choices=("svg", "png", "pdf"),
        help="Output format; may be repeated. Defaults to svg and png.",
    )
    plot.add_argument("--json", action="store_true", help="Print generated paths as JSON.")
    plot.add_argument("--quiet", action="store_true", help="Suppress plain progress output and print generated paths as JSON.")

    plot_convergence = subparsers.add_parser("plot-convergence", help="Plot artifacts written by fax convergence.")
    plot_convergence.add_argument(
        "artifact_dir",
        help="Directory containing convergence_summary.json and convergence_metrics.csv.",
    )
    plot_convergence.add_argument("--output-dir", help="Directory for generated figures; defaults to artifact_dir.")
    plot_convergence.add_argument("--stem", default="convergence_artifact_dashboard", help="Output figure filename stem.")
    plot_convergence.add_argument(
        "--format",
        action="append",
        default=[],
        choices=("svg", "png", "pdf"),
        help="Output format; may be repeated. Defaults to svg and png.",
    )
    plot_convergence.add_argument("--json", action="store_true", help="Print generated paths as JSON.")
    plot_convergence.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress plain progress output and print generated paths as JSON.",
    )

    plot_global = subparsers.add_parser("plot-global", help="Plot artifacts written by fax eigenmodes.")
    plot_global.add_argument(
        "artifact_dir",
        help="Directory containing global_eigenmode_summary.json and global_mode_structures.csv.",
    )
    plot_global.add_argument("--output-dir", help="Directory for generated figures; defaults to artifact_dir.")
    plot_global.add_argument("--stem", default="global_eigenmode_dashboard", help="Output figure filename stem.")
    plot_global.add_argument("--max-modes", type=int, default=4, help="Number of radial envelopes to overlay.")
    plot_global.add_argument(
        "--format",
        action="append",
        default=[],
        choices=("svg", "png", "pdf"),
        help="Output format; may be repeated. Defaults to svg and png.",
    )
    plot_global.add_argument("--json", action="store_true", help="Print generated paths as JSON.")
    plot_global.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress plain progress output and print generated paths as JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if argv_list and argv_list[0] in {
        "validate",
        "continuum",
        "eigenmodes",
        "convergence",
        "benchmark",
        "benchmark-manifest",
        "maturity",
        "plot-continuum",
        "plot-global",
        "plot-convergence",
    }:
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
    if args.command == "maturity":
        return _main_maturity(args)
    if args.command == "benchmark-manifest":
        return _main_benchmark_manifest(args)
    if args.command == "plot-continuum":
        return _main_plot_continuum(args)
    if args.command == "plot-convergence":
        return _main_plot_convergence(args)
    if args.command == "plot-global":
        return _main_plot_global(args)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"fax: {exc}", file=sys.stderr)
        return 2

    machine_readable = bool(args.json or args.quiet)
    try:
        summary: ValidationRunSummary | ContinuumRunSummary | GlobalEigenmodeRunSummary | ConvergenceRunSummary | BenchmarkRunSummary
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
        elif args.command == "eigenmodes":
            summary = run_global_eigenmodes(
                config,
                progress=not machine_readable,
                write_artifacts=not args.no_artifacts,
                max_modes=args.max_modes,
            )
        elif args.command == "convergence":
            summary = run_convergence(
                config,
                progress=not machine_readable,
                radial_points=tuple(args.radial_points) if args.radial_points else None,
                angular_points=tuple(args.angular_points) if args.angular_points else (12, 16, 24),
                mode_max_values=tuple(args.mode_max) if args.mode_max else (2, 3, 4),
                write_artifacts=not args.no_artifacts,
            )
        elif args.command == "benchmark":
            summary = run_benchmark(
                config,
                progress=not machine_readable,
                repeats=args.repeats,
                warmups=args.warmups,
                families=tuple(args.family),
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


def _main_plot_continuum(args: argparse.Namespace) -> int:
    machine_readable = bool(args.json or args.quiet)
    formats = tuple(args.format) if args.format else ("svg", "png")
    try:
        paths = plot_continuum_artifacts(
            args.artifact_dir,
            output_dir=args.output_dir,
            stem=args.stem,
            formats=formats,
        )
    except (ImportError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"fax: {exc}", file=sys.stderr)
        return 2

    payload = {"status": "completed", "figures": [str(path) for path in paths]}
    if machine_readable:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("FAX continuum artifact figures:")
        for path in paths:
            print(f"  {path}")
    return 0


def _main_maturity(args: argparse.Namespace) -> int:
    machine_readable = bool(args.json or args.quiet)
    summary = research_grade_summary()
    if machine_readable:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(research_grade_report_markdown())
    return 0


def _main_benchmark_manifest(args: argparse.Namespace) -> int:
    machine_readable = bool(args.json or args.quiet)
    if machine_readable:
        print(json.dumps(benchmark_manifest_summary(include_hashes=bool(args.hashes)), indent=2, sort_keys=True))
    else:
        print(benchmark_manifest_markdown(include_hashes=bool(args.hashes)))
    return 0


def _main_plot_convergence(args: argparse.Namespace) -> int:
    machine_readable = bool(args.json or args.quiet)
    formats = tuple(args.format) if args.format else ("svg", "png")
    try:
        paths = plot_convergence_artifacts(
            args.artifact_dir,
            output_dir=args.output_dir,
            stem=args.stem,
            formats=formats,
        )
    except (ImportError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"fax: {exc}", file=sys.stderr)
        return 2

    payload = {"status": "completed", "figures": [str(path) for path in paths]}
    if machine_readable:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("FAX convergence artifact figures:")
        for path in paths:
            print(f"  {path}")
    return 0


def _main_plot_global(args: argparse.Namespace) -> int:
    machine_readable = bool(args.json or args.quiet)
    formats = tuple(args.format) if args.format else ("svg", "png")
    try:
        paths = plot_global_eigenmode_artifacts(
            args.artifact_dir,
            output_dir=args.output_dir,
            stem=args.stem,
            formats=formats,
            max_modes=args.max_modes,
        )
    except (ImportError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"fax: {exc}", file=sys.stderr)
        return 2

    payload = {"status": "completed", "figures": [str(path) for path in paths]}
    if machine_readable:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("FAX global eigenmode artifact figures:")
        for path in paths:
            print(f"  {path}")
    return 0


def parse_config_from_mapping(mapping: dict[str, object]) -> FaxConfig:
    """Small helper for tests and downstream wrappers."""
    return FaxConfig.from_mapping(mapping)
