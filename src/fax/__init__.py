"""Differentiable Alfvén eigenmode and fast-ion tools for stellarators."""

from __future__ import annotations

from .config import FaxConfig, load_config
from .equilibrium import EquilibriumGeometry
from .reference_cases import ReferenceCase, landreman_paul_qa_lowres
from .runner import RunSummary, run

__all__ = [
    "EquilibriumGeometry",
    "FaxConfig",
    "ReferenceCase",
    "RunSummary",
    "landreman_paul_qa_lowres",
    "load_config",
    "run",
]

__version__ = "0.1.0"
