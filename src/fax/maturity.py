"""Research-grade maturity tracking for FAX.

This module is deliberately small and data-oriented. It makes the remaining
"no reduced model/no proxy" milestones machine-readable so the CLI,
documentation, and tests can distinguish current validated capabilities from
the full predictive code target.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


VALID_STATUSES = ("complete", "in_progress", "planned", "blocked")


@dataclass(frozen=True)
class ResearchGradeMilestone:
    """One milestone required for final full-physics FAX status."""

    milestone_id: str
    title: str
    status: str
    current_state: str
    target_state: str
    blocking_items: tuple[str, ...]
    validation_requirements: tuple[str, ...]
    deliverables: tuple[str, ...]
    next_steps: tuple[str, ...]
    required_for_no_proxy_status: bool = True

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {VALID_STATUSES}, got {self.status!r}")

    @property
    def complete(self) -> bool:
        """Return whether this milestone is complete."""
        return self.status == "complete"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def registered_research_grade_milestones() -> tuple[ResearchGradeMilestone, ...]:
    """Return the full-physics milestones that block final research-grade status."""
    return (
        ResearchGradeMilestone(
            milestone_id="native_3d_ideal_mhd_parity",
            title="Full native 3-D stellarator ideal-MHD eigenmode parity",
            status="in_progress",
            current_state=(
                "STELLGAP CTOK local-continuum/gap parity and AE3D public matrix/"
                "mode-shape artifact parity are implemented. Native FAX global "
                "operator parity against AE3D/CAS3D/STELLGAP is not yet closed."
            ),
            target_state=(
                "FAX native global 3-D operator reproduces matched AE3D, CAS3D/"
                "CONTI, and STELLGAP eigenfrequencies, eigenfunctions, residuals, "
                "normalizations, boundary conditions, and mode-family labels on "
                "multiple stellarator geometries."
            ),
            blocking_items=(
                "Matched public or locally runnable global AE3D/STELLGAP/CAS3D cases",
                "Documented basis, normalization, harmonic, and boundary-condition maps",
                "Mode-structure and frequency tolerances accepted before publication",
            ),
            validation_requirements=(
                "analytic limit",
                "grid and harmonic convergence",
                "native FAX residual",
                "cross-code eigenfrequency parity",
                "cross-code mode-shape parity",
            ),
            deliverables=(
                "global_ideal_mhd_parity test module",
                "multi-code parity figures",
                "operator-convention documentation",
            ),
            next_steps=(
                "Extract matched STELLGAP and AE3D global cases into versioned fixtures",
                "Implement a native parity driver that compares FAX eigenfunctions to external modes",
                "Add CAS3D/CONTI cases after runnable inputs or archived outputs are available",
            ),
        ),
        ResearchGradeMilestone(
            milestone_id="kinetic_growth_rate_parity",
            title="CAS3D-K and FAR3D kinetic growth-rate validation",
            status="planned",
            current_state=(
                "FAX has reduced fast-ion resonance screens and synthetic gyro-Landau "
                "coefficient-recovery gates. It has no real CAS3D-K/FAR3D kinetic "
                "growth-rate parity gate yet."
            ),
            target_state=(
                "FAX reproduces selected CAS3D-K/FAR3D energetic-particle growth/"
                "damping rates, resonance localization, threshold trends, and mode "
                "overlap metrics on matched equilibria and distributions."
            ),
            blocking_items=(
                "Runnable CAS3D-K/FAR3D binaries or archived benchmark outputs",
                "Matched energetic-particle distribution and thermal damping definitions",
                "Agreement criteria for growth rate, sign, threshold, and localization",
            ),
            validation_requirements=(
                "analytic resonance checks",
                "synthetic coefficient recovery",
                "FAR3D growth-rate parity",
                "CAS3D-K perturbative growth-rate parity",
                "distribution-function convergence",
            ),
            deliverables=(
                "kinetic_growth_rate_parity gates",
                "growth-rate comparison figures",
                "fast-ion benchmark input/output archive",
            ),
            next_steps=(
                "Build or archive one portable FAR3D linear benchmark",
                "Add a CAS3D-K-style perturbative reference table when available",
                "Fit closure coefficients only on training cases and test on held-out cases",
            ),
        ),
        ResearchGradeMilestone(
            milestone_id="fully_calibrated_fast_ion_model",
            title="Fully calibrated fast-ion drive/damping replacement model",
            status="in_progress",
            current_state=(
                "Reduced gyro-Landau drive/damping API, sign gates, eigenfunction "
                "projection, and differentiable synthetic calibration are implemented. "
                "The closure is not calibrated to real kinetic-code growth rates."
            ),
            target_state=(
                "FAX fast-ion model is calibrated, cross-validated, uncertainty-aware, "
                "and documented well enough to replace kinetic-code calls within its "
                "declared domain of validity."
            ),
            blocking_items=(
                "Kinetic benchmark data spanning distribution shape and resonance regimes",
                "Held-out validation cases with uncertainty estimates",
                "Clear domain-of-validity tests and failure-mode diagnostics",
            ),
            validation_requirements=(
                "sign and resonance gates",
                "coefficient recovery",
                "training/validation split",
                "growth-rate parity",
                "uncertainty and sensitivity checks",
            ),
            deliverables=(
                "calibrated closure parameter files",
                "fast-ion validation report",
                "replacement-domain documentation",
            ),
            next_steps=(
                "Define the first FAR3D/CAS3D-K calibration data schema",
                "Add train/test calibration workflow outputs",
                "Add uncertainty bands around predicted growth rates",
            ),
        ),
        ResearchGradeMilestone(
            milestone_id="end_to_end_vmec_boozer_optimization",
            title="End-to-end differentiable VMEC/Boozer/FAX optimization",
            status="in_progress",
            current_state=(
                "Builder-based Boozer objectives, live VMEC/Boozer adapter checks, "
                "and reduced-parameter Boozer design examples exist. Gradients from "
                "actual VMEC boundary/design variables through converged physics are "
                "not yet reviewer-grade."
            ),
            target_state=(
                "Actual VMEC boundary/profile variables differentiate through a "
                "controlled vmec_jax solve, booz_xform_jax transform, FAX operator, "
                "eigenvalue objective, and finite-difference/adjoint verification."
            ),
            blocking_items=(
                "Stable differentiable VMEC solve settings for target geometries",
                "Boozer transform derivative/Jacobian checks at required truncations",
                "Mode-label or cluster objective strategy through crossings",
            ),
            validation_requirements=(
                "VMEC residual convergence",
                "Boozer truncation convergence",
                "operator convergence",
                "finite-difference gradient checks",
                "adjoint/JVP agreement",
            ),
            deliverables=(
                "end_to_end_design_objective API",
                "VMEC-boundary gradient tests",
                "inverse-design publication figures",
            ),
            next_steps=(
                "Choose a small set of VMEC boundary coefficients for the first design vector",
                "Wrap a converged vmec_jax/booz_xform_jax/FAX builder with stable shapes",
                "Compare JAX gradients against finite differences for isolated and cluster objectives",
            ),
        ),
        ResearchGradeMilestone(
            milestone_id="multi_geometry_benchmark_suite",
            title="Multi-geometry cross-code benchmark suite",
            status="in_progress",
            current_state=(
                "FAX has CTOK STELLGAP local-continuum/gap parity, AE3D public "
                "artifact parity, and a local STELLGAP executable status benchmark. "
                "It does not yet cover enough geometries for broad reactor-design claims."
            ),
            target_state=(
                "A curated suite covers quasi-axisymmetric, quasi-helical, tokamak, "
                "heliotron/heliac, and reactor-relevant stellarator cases with matched "
                "physics metrics, convergence ladders, and runtime comparisons."
            ),
            blocking_items=(
                "Versioned geometry/input/output fixture archive",
                "Portable external-code execution recipes or trusted output tables",
                "Runtime hardware metadata and pass/fail physics metrics for each case",
            ),
            validation_requirements=(
                "multi-geometry convergence",
                "cross-code frequency and mode parity",
                "runtime parity with physics gates",
                "figure QA",
                "reproducible fixture hashes",
            ),
            deliverables=(
                "benchmark_manifest.json",
                "multi-geometry benchmark report",
                "publication-ready runtime/comparison plots",
            ),
            next_steps=(
                "Define the benchmark manifest schema",
                "Promote CTOK and AE3D public cases into the first manifest entries",
                "Add Landreman-Paul and at least one QH/heliotron-style geometry after parity data exist",
            ),
        ),
        ResearchGradeMilestone(
            milestone_id="no_proxy_full_physics_gate",
            title="Final no-proxy/no-reduced-model release gate",
            status="blocked",
            current_state=(
                "FAX documentation honestly marks reduced models, artifact parity, "
                "adapter checks, and screening applications. The final full-physics "
                "gate cannot pass until the upstream milestones are complete."
            ),
            target_state=(
                "All public examples and default workflows use validated full-physics "
                "operators or explicitly optional reduced modes; no publication claim "
                "depends on an unvalidated proxy, scaffold, or screening-only model."
            ),
            blocking_items=(
                "Completion of native ideal-MHD parity",
                "Completion of kinetic growth-rate validation",
                "Completion of end-to-end design-gradient checks",
                "Completion of multi-geometry benchmark suite",
            ),
            validation_requirements=(
                "all required full-physics gates pass",
                "no default docs/examples rely on screening-only claims",
                "release audit labels every figure and workflow claim level",
            ),
            deliverables=(
                "full_research_grade release checklist",
                "claim-level audit report",
                "v1.0 publication-grade documentation packet",
            ),
            next_steps=(
                "Keep this milestone blocked until all required full-physics milestones are complete",
                "Fail release review if any default claim lacks validation traceability",
                "Move reduced models into explicitly labeled optional screening workflows",
            ),
        ),
    )


def research_grade_ready() -> bool:
    """Return whether all required full-research-grade milestones are complete."""
    return all(
        milestone.complete
        for milestone in registered_research_grade_milestones()
        if milestone.required_for_no_proxy_status
    )


def research_grade_summary() -> dict[str, Any]:
    """Return a machine-readable summary of full-research-grade readiness."""
    milestones = registered_research_grade_milestones()
    required = [milestone for milestone in milestones if milestone.required_for_no_proxy_status]
    complete = [milestone for milestone in required if milestone.complete]
    return {
        "status": "complete" if len(complete) == len(required) else "not_complete",
        "ready": len(complete) == len(required),
        "required_count": len(required),
        "complete_count": len(complete),
        "milestones": [milestone.to_dict() for milestone in milestones],
    }


def research_grade_report_markdown() -> str:
    """Return a Markdown report for the full-research-grade roadmap."""
    summary = research_grade_summary()
    lines = [
        "# FAX Full Research-Grade Roadmap",
        "",
        f"Status: **{summary['status']}** ({summary['complete_count']}/{summary['required_count']} required milestones complete).",
        "",
        "| Milestone | Status | Current State | Next Step |",
        "| --- | --- | --- | --- |",
    ]
    for milestone in registered_research_grade_milestones():
        next_step = milestone.next_steps[0] if milestone.next_steps else ""
        lines.append(
            f"| `{milestone.milestone_id}` | {milestone.status} | "
            f"{milestone.current_state} | {next_step} |"
        )
    lines.extend(
        [
            "",
            "## Blocking Items",
            "",
        ]
    )
    for milestone in registered_research_grade_milestones():
        if milestone.complete:
            continue
        lines.append(f"### {milestone.title}")
        for item in milestone.blocking_items:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines)
