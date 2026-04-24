"""Terminal progress reporting for FAX runs."""

from __future__ import annotations

from dataclasses import dataclass
import sys
import time
from typing import Any, TextIO

from .config import FaxConfig


def format_seconds(seconds: float) -> str:
    """Return a compact human-readable duration."""
    seconds = max(float(seconds), 0.0)
    if seconds < 1.0:
        return "<1s"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {int(rem)}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes)}m"


@dataclass(frozen=True)
class PhaseRecord:
    """Timing and status for one run phase."""

    name: str
    detail: str
    estimate_seconds: float
    elapsed_seconds: float
    status: str


class ProgressReporter:
    """Rich terminal reporter with a plain-text fallback."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        stream: TextIO | None = None,
        total_estimate_seconds: float = 1.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.stream = stream if stream is not None else sys.stdout
        self.total_estimate_seconds = float(total_estimate_seconds)
        self._start = time.perf_counter()
        self._completed_estimate = 0.0
        self._active_name: str | None = None
        self._active_detail = ""
        self._active_estimate = 0.0
        self._active_start = self._start
        self.records: list[PhaseRecord] = []
        self._rich_available = False
        self._console: Any | None = None
        if self.enabled:
            try:
                from rich.console import Console

                self._console = Console(file=self.stream, force_terminal=False, color_system=None, width=110)
                self._rich_available = True
            except Exception:
                self._console = None

    def banner(self, config: FaxConfig, *, dry_run: bool = False) -> None:
        """Print a run banner and key configuration table."""
        if not self.enabled:
            return
        title = "FAX dry run" if dry_run else "FAX run"
        subtitle = f"case={config.run.case} | model={config.run.model} | output={config.run.output_dir}"
        if self._rich_available:
            from rich import box
            from rich.panel import Panel
            from rich.table import Table

            console = self._rich_console()
            console.print(Panel(subtitle, title=title, border_style="cyan"))
            table = Table(title="Solve parameters", box=box.SIMPLE_HEAVY)
            table.add_column("section")
            table.add_column("parameter")
            table.add_column("value")
            for row in _config_rows(config):
                table.add_row(*row)
            console.print(table)
        else:
            self._write(f"{title}: {subtitle}")
            self._write(
                "Solve parameters: "
                f"source={config.equilibrium.source}, radial_points={config.grid.radial_points}, "
                f"m_max={config.grid.m_max}, n_max={config.grid.n_max}, precision={config.numerics.precision}"
            )

    def phase_start(self, name: str, *, detail: str = "", estimate_seconds: float = 0.0) -> None:
        """Print a phase-start message."""
        self._active_name = name
        self._active_detail = detail
        self._active_estimate = float(estimate_seconds)
        self._active_start = time.perf_counter()
        if not self.enabled:
            return
        remaining = max(self.total_estimate_seconds - self._completed_estimate, 0.0)
        message = (
            f"START {name}"
            f"{' - ' + detail if detail else ''}"
            f" | elapsed {format_seconds(time.perf_counter() - self._start)}"
            f" | estimated remaining {format_seconds(remaining)}"
        )
        if self._rich_available:
            self._rich_console().print(f"[bold cyan]{message}[/bold cyan]")
        else:
            self._write(message)

    def phase_done(self, *, status: str = "done") -> PhaseRecord:
        """Mark the active phase complete and print its elapsed time."""
        now = time.perf_counter()
        name = self._active_name or "unnamed phase"
        elapsed = now - self._active_start
        record = PhaseRecord(
            name=name,
            detail=self._active_detail,
            estimate_seconds=self._active_estimate,
            elapsed_seconds=elapsed,
            status=status,
        )
        self.records.append(record)
        self._completed_estimate += self._active_estimate
        if self.enabled:
            remaining = max(self.total_estimate_seconds - self._completed_estimate, 0.0)
            message = f"DONE  {name} | phase {format_seconds(elapsed)} | estimated remaining {format_seconds(remaining)}"
            if self._rich_available:
                style = "green" if status == "done" else "yellow"
                self._rich_console().print(f"[{style}]{message}[/{style}]")
            else:
                self._write(message)
        self._active_name = None
        self._active_detail = ""
        self._active_estimate = 0.0
        return record

    def log(self, message: str) -> None:
        """Print an informational message."""
        if not self.enabled:
            return
        if self._rich_available:
            self._rich_console().print(f"  {message}")
        else:
            self._write(f"  {message}")

    def summary(self, summary: Any) -> None:
        """Print a final run summary."""
        if not self.enabled:
            return
        total = time.perf_counter() - self._start
        if self._rich_available:
            from rich import box
            from rich.panel import Panel
            from rich.table import Table

            console = self._rich_console()
            table = Table(title="Execution phases", box=box.SIMPLE)
            table.add_column("phase")
            table.add_column("status")
            table.add_column("elapsed")
            table.add_column("detail")
            for record in self.records:
                table.add_row(record.name, record.status, format_seconds(record.elapsed_seconds), record.detail)
            console.print(table)
            text = f"status={summary.status} | elapsed={format_seconds(total)} | message={summary.message}"
            console.print(Panel(text, title="FAX summary", border_style="green"))
        else:
            self._write("Execution phases:")
            for record in self.records:
                self._write(f"  {record.name}: {record.status}, elapsed {format_seconds(record.elapsed_seconds)}")
            self._write(f"FAX summary: status={summary.status}, elapsed={format_seconds(total)}, message={summary.message}")

    def _write(self, message: str) -> None:
        print(message, file=self.stream)

    def _rich_console(self) -> Any:
        if self._console is None:
            raise RuntimeError("rich console requested before initialization")
        return self._console


def _config_rows(config: FaxConfig) -> list[tuple[str, str, str]]:
    return [
        ("equilibrium", "source", config.equilibrium.source),
        ("equilibrium", "vmec_input", config.equilibrium.vmec_input or ""),
        ("equilibrium", "wout", config.equilibrium.wout or ""),
        ("equilibrium", "use_boozer", str(config.equilibrium.use_boozer)),
        ("grid", "radial_points", str(config.grid.radial_points)),
        ("grid", "m_max", str(config.grid.m_max)),
        ("grid", "n_max", str(config.grid.n_max)),
        ("grid", "field_periods", str(config.grid.field_periods)),
        ("physics", "fast_ions", str(config.physics.include_fast_ions)),
        ("physics", "fast_ion_model", config.physics.fast_ion_model),
        ("physics", "density_scale", str(config.physics.density_scale)),
        ("numerics", "precision", config.numerics.precision),
        ("numerics", "jit", str(config.numerics.jit)),
        ("numerics", "autodiff", str(config.numerics.autodiff)),
        ("numerics", "eigensolver", config.numerics.eigensolver),
    ]
