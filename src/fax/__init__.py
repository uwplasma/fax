"""Differentiable Alfvén eigenmode and fast-ion tools for stellarators."""

from __future__ import annotations

from .boozer import (
    BoozerGeometry,
    BoozerGeometryGrid,
    BoozerOperatorAngularConvergenceResult,
    BoozerSpectrum,
    VmecJaxBoozerConvergenceResult,
    VmecJaxBoozerSolveConvergenceResult,
    boozer_3d_operator_angular_convergence,
    run_booz_xform_jax_from_wout_data,
    run_booz_xform_jax_geometry_from_wout,
    run_booz_xform_jax_geometry_from_wout_data,
    run_vmec_jax_boozer_solve_convergence,
    run_vmec_jax_boozer_truncation_convergence,
    run_vmec_jax_fixed_boundary_boozer_geometry,
    run_vmec_jax_fixed_boundary_wout_data,
)
from .config import FaxConfig, load_config
from .cross_code import (
    Ae3dEigenvalueData,
    Ae3dMatrixSummary,
    Ae3dOmega2Data,
    Ae3dResidualSummary,
    CrossCodeComparison,
    CrossCodePostprocessComparison,
    StellgapContinuumData,
    StellgapSpectrumData,
)
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
from .optimization import GradientDescentResult, LinearizedUncertaintyEstimate
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
    "Ae3dMatrixSummary",
    "Ae3dOmega2Data",
    "Ae3dResidualSummary",
    "BenchmarkRunSummary",
    "BoozerGeometry",
    "BoozerGeometryGrid",
    "BoozerOperatorAngularConvergenceResult",
    "BoozerSpectrum",
    "VmecJaxBoozerConvergenceResult",
    "VmecJaxBoozerSolveConvergenceResult",
    "ConvergenceRunSummary",
    "ConvergenceArtifacts",
    "CrossCodeComparison",
    "CrossCodePostprocessComparison",
    "EquilibriumGeometry",
    "FaxConfig",
    "FastIonPopulation",
    "FastIonModeDriveResult",
    "GradientDescentResult",
    "LinearizedUncertaintyEstimate",
    "MetricGeometry",
    "ModeFamily",
    "RadialBlockEigenmodeProblem",
    "RadialBlockEigenmodeResult",
    "RadialEigenmodeProblem",
    "RadialEigenmodeResult",
    "ReferenceCase",
    "RunSummary",
    "StellgapContinuumData",
    "StellgapSpectrumData",
    "ValidationRunSummary",
    "boozer_radial_stiffness_matrices",
    "boozer_3d_operator_angular_convergence",
    "landreman_paul_qa_lowres",
    "load_continuum_artifacts",
    "load_convergence_artifacts",
    "load_config",
    "plot_continuum_artifacts",
    "plot_basic_alfven_gap_map",
    "plot_convergence_artifacts",
    "run",
    "run_benchmark",
    "run_booz_xform_jax_from_wout_data",
    "run_booz_xform_jax_geometry_from_wout",
    "run_booz_xform_jax_geometry_from_wout_data",
    "run_vmec_jax_boozer_solve_convergence",
    "run_vmec_jax_boozer_truncation_convergence",
    "run_vmec_jax_fixed_boundary_boozer_geometry",
    "run_vmec_jax_fixed_boundary_wout_data",
    "run_continuum",
    "run_convergence",
    "run_validation",
]

__version__ = "0.0.1"
