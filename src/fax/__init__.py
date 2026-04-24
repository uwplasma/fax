"""Differentiable Alfvén eigenmode and fast-ion tools for stellarators."""

from __future__ import annotations

from .config import FaxConfig, load_config
from .equilibrium import EquilibriumGeometry
from .fast_ions import FastIonPopulation
from .global_modes import RadialEigenmodeProblem, RadialEigenmodeResult
from .metric import MetricGeometry
from .modes import ModeFamily
from .plotting import (
    ContinuumArtifacts,
    ConvergenceArtifacts,
    load_continuum_artifacts,
    load_convergence_artifacts,
    plot_continuum_artifacts,
    plot_convergence_artifacts,
)
from .reference_cases import ReferenceCase, landreman_paul_qa_lowres
from .runner import RunSummary, run
from .workflows import (
    BenchmarkRunSummary,
    ContinuumRunSummary,
    ConvergenceRunSummary,
    ValidationRunSummary,
    run_benchmark,
    run_continuum,
    run_convergence,
    run_validation,
)

__all__ = [
    "ContinuumRunSummary",
    "ContinuumArtifacts",
    "BenchmarkRunSummary",
    "ConvergenceRunSummary",
    "ConvergenceArtifacts",
    "EquilibriumGeometry",
    "FaxConfig",
    "FastIonPopulation",
    "MetricGeometry",
    "ModeFamily",
    "RadialEigenmodeProblem",
    "RadialEigenmodeResult",
    "ReferenceCase",
    "RunSummary",
    "ValidationRunSummary",
    "landreman_paul_qa_lowres",
    "load_continuum_artifacts",
    "load_convergence_artifacts",
    "load_config",
    "plot_continuum_artifacts",
    "plot_convergence_artifacts",
    "run",
    "run_benchmark",
    "run_continuum",
    "run_convergence",
    "run_validation",
]

__version__ = "0.0.1"
