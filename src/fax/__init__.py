"""Differentiable Alfvén eigenmode and fast-ion tools for stellarators."""

from __future__ import annotations

from .config import FaxConfig, load_config
from .runner import RunSummary, run

__all__ = ["FaxConfig", "RunSummary", "load_config", "run"]

__version__ = "0.1.0"

