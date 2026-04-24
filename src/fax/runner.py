"""Run orchestration for ``fax``.

The current runner is intentionally a planning stub. It validates the public API
shape used by the CLI and tests while making clear that physics kernels are not
implemented yet.
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
    """Validate and describe a planned ``fax`` run.

    Physics execution will be implemented after the operator and validation
    layers in ``plan.md`` are added. Returning a structured summary now keeps the
    CLI, examples, and future Python API stable.
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
            reporter.log("current implementation: profile continuum, metric-matrix scaffold, and validation registry")
            reporter.log(f"registered validation gates: {len(registered_validation_gates())}")
            reporter.log("next operators: full VMEC/Boozer continuum coupling and global eigenmodes")
        reporter.phase_done(status="done")

    summary = RunSummary(
        case=config.run.case,
        model=config.run.model,
        status="planned",
        message="Physics kernels are not implemented in this scaffold commit.",
        config=config.to_dict(),
        elapsed_seconds=time.perf_counter() - start,
        phases=_phase_dicts(reporter.records),
    )
    reporter.summary(summary)
    return summary
