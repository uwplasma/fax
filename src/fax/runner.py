"""Dry-run orchestration for the top-level ``fax`` command.

Executable physics workflows live in the ``validate``, ``continuum``,
``eigenmodes``, ``convergence``, and ``benchmark`` subcommands. The historical
top-level ``fax case.toml`` path remains a lightweight planning and
configuration-inspection entry point so existing scripts keep working.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, TextIO

from .config import FaxConfig
from .progress import PhaseRecord, ProgressReporter
from .validation import registered_validation_gates


@dataclass(frozen=True)
class RunSummary:
    """Serializable summary returned by ``run``."""

    case: str
    model: str
    status: str
    message: str
    config: dict[str, Any]
    elapsed_seconds: float
    phases: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _phase_dicts(records: list[PhaseRecord]) -> list[dict[str, Any]]:
    return [asdict(record) for record in records]


def run(
    config: FaxConfig,
    *,
    progress: bool = True,
    stream: TextIO | None = None,
    dry_run: bool = False,
) -> RunSummary:
    """Validate a configuration and describe which workflows it can run.

    This is intentionally a dry-run summary. Use ``run_continuum``,
    ``run_global_eigenmodes``, ``run_validation``, or the matching CLI
    subcommands for executed physics workflows.
    """
    import time

    start = time.perf_counter()
    config = config.validated()
    phases = [
        ("Validate configuration", "check TOML schema, model choices, and numerical controls", 0.1),
        ("Resolve equilibrium inputs", f"source={config.equilibrium.source}", 0.2),
        (
            "Prepare grid and mode metadata",
            f"radial={config.grid.radial_points}, m_max={config.grid.m_max}, n_max={config.grid.n_max}",
            0.2,
        ),
        ("Plan physics operators", f"model={config.run.model}, fast_ions={config.physics.include_fast_ions}", 0.3),
        ("Finalize run summary", f"output_dir={config.run.output_dir}", 0.1),
    ]
    reporter = ProgressReporter(
        enabled=progress,
        stream=stream,
        total_estimate_seconds=sum(phase[2] for phase in phases),
    )
    reporter.banner(config, dry_run=dry_run)
    for name, detail, estimate in phases:
        reporter.phase_start(name, detail=detail, estimate_seconds=estimate)
        if name == "Resolve equilibrium inputs":
            if config.equilibrium.wout:
                reporter.log(f"VMEC wout path: {config.equilibrium.wout}")
            if config.equilibrium.vmec_input:
                reporter.log(f"VMEC input path: {config.equilibrium.vmec_input}")
            if config.equilibrium.use_boozer:
                reporter.log(f"Boozer coordinates requested: {config.equilibrium.boozmn or 'to be generated'}")
        elif name == "Plan physics operators":
            reporter.log(
                "available workflows: validation, continuum, global eigenmodes, convergence, benchmarks"
            )
            reporter.log(f"registered validation gates: {len(registered_validation_gates())}")
            reporter.log("use subcommands for execution; this path is a configuration dry run")
        reporter.phase_done(status="done")

    summary = RunSummary(
        case=config.run.case,
        model=config.run.model,
        status="planned",
        message="Dry-run summary completed. Use a workflow subcommand to execute physics operators.",
        config=config.to_dict(),
        elapsed_seconds=time.perf_counter() - start,
        phases=_phase_dicts(reporter.records),
    )
    reporter.summary(summary)
    return summary
