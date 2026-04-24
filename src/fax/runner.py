"""Run orchestration for ``fax``.

The current runner is intentionally a planning stub. It validates the public API
shape used by the CLI and tests while making clear that physics kernels are not
implemented yet.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .config import FaxConfig


@dataclass(frozen=True)
class RunSummary:
    """Serializable summary returned by ``run``."""

    case: str
    model: str
    status: str
    message: str
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def run(config: FaxConfig) -> RunSummary:
    """Validate and describe a planned ``fax`` run.

    Physics execution will be implemented after the operator and validation
    layers in ``plan.md`` are added. Returning a structured summary now keeps the
    CLI, examples, and future Python API stable.
    """
    config = config.validated()
    return RunSummary(
        case=config.run.case,
        model=config.run.model,
        status="planned",
        message="Physics kernels are not implemented in this scaffold commit.",
        config=config.to_dict(),
    )

