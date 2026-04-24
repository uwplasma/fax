"""Differentiable Alfvén eigenmode and fast-ion tools for stellarators."""

from __future__ import annotations

from .config import FaxConfig, load_config
from .equilibrium import EquilibriumGeometry
from .metric import MetricGeometry
from .modes import ModeFamily
from .reference_cases import ReferenceCase, landreman_paul_qa_lowres
from .runner import RunSummary, run
from .workflows import ContinuumRunSummary, ValidationRunSummary, run_continuum, run_validation

__all__ = [
    "ContinuumRunSummary",
    "EquilibriumGeometry",
    "FaxConfig",
    "MetricGeometry",
    "ModeFamily",
    "ReferenceCase",
    "RunSummary",
    "ValidationRunSummary",
    "landreman_paul_qa_lowres",
    "load_config",
    "run",
    "run_continuum",
    "run_validation",
]

__version__ = "0.1.0"
