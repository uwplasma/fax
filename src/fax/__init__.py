"""Differentiable Alfvén eigenmode and fast-ion tools for stellarators."""

from __future__ import annotations

from .boozer import BoozerGeometry, BoozerGeometryGrid, BoozerSpectrum
from .config import FaxConfig, load_config
from .cross_code import Ae3dEigenvalueData, Ae3dResidualSummary, CrossCodeComparison, StellgapContinuumData
from .equilibrium import EquilibriumGeometry
from .fast_ions import FastIonModeDriveResult, FastIonPopulation
from .global_modes import (
    RadialBlockEigenmodeProblem,
    RadialBlockEigenmodeResult,
    RadialEigenmodeProblem,
    RadialEigenmodeResult,
    boozer_radial_stiffness_matrices,
)
from .metric import MetricGeometry
from .modes import ModeFamily
from .plotting import (
    ContinuumArtifacts,
    ConvergenceArtifacts,
    load_continuum_artifacts,
    load_convergence_artifacts,
    plot_basic_alfven_gap_map,
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
    "Ae3dEigenvalueData",
    "Ae3dResidualSummary",
    "BenchmarkRunSummary",
    "BoozerGeometry",
    "BoozerGeometryGrid",
    "BoozerSpectrum",
    "ConvergenceRunSummary",
    "ConvergenceArtifacts",
    "CrossCodeComparison",
    "EquilibriumGeometry",
    "FaxConfig",
    "FastIonPopulation",
    "FastIonModeDriveResult",
    "MetricGeometry",
    "ModeFamily",
    "RadialBlockEigenmodeProblem",
    "RadialBlockEigenmodeResult",
    "RadialEigenmodeProblem",
    "RadialEigenmodeResult",
    "ReferenceCase",
    "RunSummary",
    "StellgapContinuumData",
    "ValidationRunSummary",
    "boozer_radial_stiffness_matrices",
    "landreman_paul_qa_lowres",
    "load_continuum_artifacts",
    "load_convergence_artifacts",
    "load_config",
    "plot_continuum_artifacts",
    "plot_basic_alfven_gap_map",
    "plot_convergence_artifacts",
    "run",
    "run_benchmark",
    "run_continuum",
    "run_convergence",
    "run_validation",
]

__version__ = "0.0.1"
