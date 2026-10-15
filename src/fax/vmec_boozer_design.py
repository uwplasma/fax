"""End-to-end VMEX -> Boozer -> FAX design objectives.

This module is optional: it requires ``vmex`` (which brings
``booz_xform_jax``).  The functions here keep the full differentiable path in
memory:

``VMEC boundary coefficients -> vmex implicit equilibrium -> booz_xform_jax
harmonics -> FAX global Alfvén objective``.

The equilibrium is converged to ``vmec_ftol`` and differentiated with the
implicit-function theorem (:func:`vmex.core.implicit.solve_implicit`), so
gradients are those of the converged equilibrium, not of a truncated solver
trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .boozer import BoozerGeometry
from .global_modes import radial_block_problem_from_boozer_geometry, solve_radial_block_eigenmodes

_BOUNDARY_FIELDS = {"rc": "rbc", "rs": "rbs", "zc": "zbc", "zs": "zbs"}


@dataclass(frozen=True)
class VmecBoozerFaxDesignSettings:
    """Settings for a differentiable VMEC-boundary Alfvén design objective.

    ``vmec_ns``, ``vmec_ftol`` and ``vmec_max_iter`` set the radial grid,
    force-residual tolerance and iteration budget of the converged VMEX
    fixed-boundary solve.
    """

    input_path: str | Path
    max_mode: int = 1
    parameter_count: int = 1
    include: tuple[str, ...] = ("rc", "zs")
    fix: tuple[str, ...] = ("rc00",)
    surfaces: tuple[float, ...] = (0.25, 0.40, 0.55, 0.70)
    mboz: int = 3
    nboz: int = 3
    harmonics: tuple[tuple[float, float], ...] = ((1.0, 0.0), (2.0, 1.0))
    mass_density: float = 3.4e-8
    avoid_window_hz: tuple[float, float] = (50_000.0, 500_000.0)
    softness_hz: float = 25_000.0
    mode_count: int = 3
    theta_points: int = 8
    zeta_points: int = 8
    radial_stiffness_scale: float = 1.0e-4
    boundary_condition: str = "dirichlet"
    vmec_ns: int = 16
    vmec_ftol: float = 1.0e-12
    vmec_max_iter: int = 5000


@dataclass(frozen=True)
class VmecBoozerFaxDesignContext:
    """Prepared objects for repeated VMEC-boundary design evaluations."""

    settings: VmecBoozerFaxDesignSettings
    config: Any
    params: Any
    parameter_slots: tuple[tuple[str, int, int], ...]
    parameter_names: tuple[str, ...]
    booz_constants: Any
    booz_grids: Any
    booz_xm: np.ndarray
    booz_xn: np.ndarray
    surface_indices: jax.Array
    surface_s: jax.Array


@dataclass(frozen=True)
class VmecBoozerFaxGradientValidation:
    """Result from an end-to-end VMEC/Boozer/FAX gradient check."""

    parameter_names: tuple[str, ...]
    parameters: jax.Array
    objective: jax.Array
    gradient: jax.Array
    finite_difference_gradient: jax.Array
    relative_gradient_error: float
    initial_frequencies_hz: jax.Array
    descent_parameters: jax.Array
    descent_objective: jax.Array
    descent_frequencies_hz: jax.Array
    descent_step_norm: float
    metadata: tuple[tuple[str, Any], ...]


@dataclass(frozen=True)
class VmecBoozerFaxGradientConvergencePoint:
    """One level in a VMEC-boundary/Boozer/FAX gradient convergence ladder."""

    label: str
    parameter_names: tuple[str, ...]
    vmec_ftol: float
    mboz: int
    nboz: int
    theta_points: int
    zeta_points: int
    surfaces: tuple[float, ...]
    objective: float
    descent_objective: float
    relative_gradient_error: float
    gradient: tuple[float, ...]
    finite_difference_gradient: tuple[float, ...]
    frequencies_hz: tuple[float, ...]
    frequency_delta_to_reference_hz: float
    objective_delta_to_reference: float
    gradient_delta_to_reference: float

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "label": self.label,
            "parameter_names": list(self.parameter_names),
            "vmec_ftol": self.vmec_ftol,
            "mboz": self.mboz,
            "nboz": self.nboz,
            "theta_points": self.theta_points,
            "zeta_points": self.zeta_points,
            "surfaces": list(self.surfaces),
            "objective": self.objective,
            "descent_objective": self.descent_objective,
            "relative_gradient_error": self.relative_gradient_error,
            "gradient": list(self.gradient),
            "finite_difference_gradient": list(self.finite_difference_gradient),
            "frequencies_hz": list(self.frequencies_hz),
            "frequency_delta_to_reference_hz": self.frequency_delta_to_reference_hz,
            "objective_delta_to_reference": self.objective_delta_to_reference,
            "gradient_delta_to_reference": self.gradient_delta_to_reference,
        }


@dataclass(frozen=True)
class VmecBoozerFaxGradientConvergence:
    """Convergence summary for VMEC-boundary Alfvén objective gradients."""

    points: tuple[VmecBoozerFaxGradientConvergencePoint, ...]
    reference_label: str
    max_relative_gradient_error: float
    max_frequency_delta_to_reference_hz: float
    max_objective_delta_to_reference: float
    max_gradient_delta_to_reference: float

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable convergence summary."""
        return {
            "reference_label": self.reference_label,
            "max_relative_gradient_error": self.max_relative_gradient_error,
            "max_frequency_delta_to_reference_hz": self.max_frequency_delta_to_reference_hz,
            "max_objective_delta_to_reference": self.max_objective_delta_to_reference,
            "max_gradient_delta_to_reference": self.max_gradient_delta_to_reference,
            "points": [point.as_dict() for point in self.points],
        }


@dataclass(frozen=True)
class VmecBoozerFaxJacobianValidation:
    """Vector-frequency Jacobian and pullback validation result."""

    parameter_names: tuple[str, ...]
    parameters: jax.Array
    frequencies_hz: jax.Array
    frequency_jacobian: jax.Array
    finite_difference_jacobian: jax.Array
    relative_jacobian_error: float
    cotangent: jax.Array
    pullback: jax.Array
    finite_difference_pullback: jax.Array
    relative_pullback_error: float
    metadata: tuple[tuple[str, Any], ...]


def prepare_vmec_boozer_fax_design_context(
    settings: VmecBoozerFaxDesignSettings,
) -> VmecBoozerFaxDesignContext:
    """Prepare static VMEX/Boozer objects for boundary-design studies.

    The returned context is reusable for objective, gradient, UQ, and inverse
    design calls.  Boundary parameters are ``RBC/RBS/ZBC/ZBS(n, m)`` entries
    named ``rc{m}{n}`` etc. with ``m <= max_mode`` and ``|n| <= max_mode``,
    in VMEC mode order.  The equilibrium used by
    :func:`vmec_boundary_fax_frequencies` is
    :func:`vmex.core.implicit.solve_implicit`, so JAX gradients can be checked
    directly against centered finite differences of converged solves.
    """
    try:
        from booz_xform_jax.jax_api import prepare_booz_xform_constants
        from vmex import VmecInput
        from vmex.core import implicit
        from vmex.core.boozer_tables import boozer_input_tables
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "prepare_vmec_boozer_fax_design_context requires vmex (pip install 'uwplasma-fax[vmex]')"
        ) from exc

    path = Path(settings.input_path).expanduser().resolve()
    inp = VmecInput.from_file(str(path))
    config = implicit.make_config(
        inp,
        ns=int(settings.vmec_ns),
        ftol=float(settings.vmec_ftol),
        max_iterations=int(settings.vmec_max_iter),
    )
    params = implicit.params_from_input(inp)
    ntor, mpol = int(inp.ntor), int(inp.mpol)
    slots, names = [], []
    for m in range(min(int(settings.max_mode), mpol - 1) + 1):
        for n in range(-min(int(settings.max_mode), ntor), min(int(settings.max_mode), ntor) + 1):
            if m == 0 and n < 0:
                continue
            for kind in settings.include:
                name = f"{kind}{m}{n}"
                if name in settings.fix or (m == 0 and n == 0 and kind in ("rs", "zs")):
                    continue
                slots.append((_BOUNDARY_FIELDS[kind], n + ntor, m))
                names.append(name)
    count = int(settings.parameter_count)
    if count < 1 or not slots:
        raise ValueError("no VMEC boundary parameters were selected")

    s_full = np.asarray(implicit.runtime_from_params(params, config).setup.s_full)
    s_half = 0.5 * (s_full[:-1] + s_full[1:])
    surface_indices = _nearest_surface_indices(jnp.asarray(s_half), settings.surfaces)

    # The wout-convention mode tables are static: read them from an abstract trace.
    modes: dict[str, np.ndarray] = {}
    row = int(surface_indices[0]) + 1

    def probe(p: Any) -> jax.Array:
        state = implicit.solve_implicit(p, config)
        tables = boozer_input_tables(state, implicit.runtime_from_params(p, config), row)
        modes.update(xm=np.asarray(tables["xm"]), xn=np.asarray(tables["xn"]))
        return tables["bmnc"]

    jax.eval_shape(probe, params)
    constants, grids = prepare_booz_xform_constants(
        nfp=int(inp.nfp),
        mboz=int(settings.mboz),
        nboz=int(settings.nboz),
        asym=bool(inp.lasym),
        xm=modes["xm"],
        xn=modes["xn"],
        xm_nyq=modes["xm"],
        xn_nyq=modes["xn"],
    )
    return VmecBoozerFaxDesignContext(
        settings=settings,
        config=config,
        params=params,
        parameter_slots=tuple(slots[:count]),
        parameter_names=tuple(names[:count]),
        booz_constants=constants,
        booz_grids=grids,
        booz_xm=modes["xm"],
        booz_xn=modes["xn"],
        surface_indices=surface_indices,
        surface_s=jnp.take(jnp.asarray(s_half), surface_indices),
    )


def vmec_state_to_boozer_geometry(state: Any, runtime: Any, context: VmecBoozerFaxDesignContext) -> BoozerGeometry:
    """Transform a VMEX spectral state to a FAX Boozer geometry in memory."""
    from booz_xform_jax.jax_api import booz_xform_jax_impl
    from vmex.core.boozer_tables import boozer_input_tables

    tables = [boozer_input_tables(state, runtime, int(i) + 1) for i in np.asarray(context.surface_indices)]
    names: tuple[str, ...] = ("rmnc", "zmns", "lmns", "bmnc", "bsubumnc", "bsubvmnc", "iota")
    if context.config.inp.lasym:
        names += ("rmns", "zmnc", "lmnc", "bmns", "bsubumns", "bsubvmns")
    xm, xn = jnp.asarray(context.booz_xm), jnp.asarray(context.booz_xn)
    output = booz_xform_jax_impl(
        **{name: jnp.stack([table[name] for table in tables]) for name in names},
        xm=xm,
        xn=xn,
        xm_nyq=xm,
        xn_nyq=xn,
        constants=context.booz_constants,
        grids=context.booz_grids,
    )
    zeros = jnp.zeros_like(jnp.asarray(output["bmnc_b"]))
    return BoozerGeometry(
        source="vmex_booz_xform_jax_design",
        s=context.surface_s,
        xm=jnp.asarray(output["ixm_b"]),
        xn=jnp.asarray(output["ixn_b"]),
        bmnc=jnp.asarray(output["bmnc_b"]),
        bmns=jnp.asarray(output.get("bmns_b", zeros)),
        rmnc=jnp.asarray(output.get("rmnc_b", zeros)),
        rmns=jnp.asarray(output.get("rmns_b", zeros)),
        zmns=jnp.asarray(output.get("zmns_b", zeros)),
        zmnc=jnp.asarray(output.get("zmnc_b", zeros)),
        pmns=jnp.asarray(output.get("pmns_b", zeros)),
        pmnc=jnp.asarray(output.get("pmnc_b", zeros)),
        gmnc=jnp.asarray(output.get("gmnc_b", zeros)),
        gmns=jnp.asarray(output.get("gmns_b", zeros)),
        iota=jnp.asarray(output["iota_b"]),
        boozer_i=jnp.asarray(output["buco_b"]),
        boozer_g=jnp.asarray(output["bvco_b"]),
        field_periods=int(context.config.inp.nfp),
        metadata=(
            ("input_path", str(Path(context.settings.input_path).expanduser().resolve())),
            ("mboz", int(context.settings.mboz)),
            ("nboz", int(context.settings.nboz)),
            ("surface_indices", tuple(int(i) for i in np.asarray(context.surface_indices))),
        ),
    )


def vmec_boundary_fax_frequencies(
    parameters: Any,
    context: VmecBoozerFaxDesignContext,
) -> jax.Array:
    """Return global Alfvén frequencies for VMEC boundary perturbations.

    ``parameters`` are additive offsets to the selected boundary coefficients.
    """
    from vmex.core import implicit

    settings = context.settings
    offsets = jnp.asarray(parameters)
    params = context.params
    for index, (field, row, m) in enumerate(context.parameter_slots):
        params = replace(params, **{field: getattr(params, field).at[row, m].add(offsets[index])})
    state = implicit.solve_implicit(params, context.config)
    runtime = implicit.runtime_from_params(params, context.config)
    geometry = vmec_state_to_boozer_geometry(state, runtime, context)
    problem = radial_block_problem_from_boozer_geometry(
        geometry,
        jnp.asarray(settings.harmonics, dtype=float),
        mass_density=float(settings.mass_density),
        radial_stiffness_method="metric",
        radial_stiffness_scale=float(settings.radial_stiffness_scale),
        theta_points=int(settings.theta_points),
        zeta_points=int(settings.zeta_points),
        boundary_condition=str(settings.boundary_condition),
    )
    return solve_radial_block_eigenmodes(problem).frequencies_hz[: int(settings.mode_count)]


def vmec_boundary_alfven_risk_objective(
    parameters: Any,
    context: VmecBoozerFaxDesignContext,
) -> jax.Array:
    """Smoothly penalize VMEC-boundary designs with modes in a risk window."""
    settings = context.settings
    frequencies = vmec_boundary_fax_frequencies(parameters, context)
    lower, upper = settings.avoid_window_hz
    softness = max(float(settings.softness_hz), 1.0e-30)
    risk = jax.nn.sigmoid((frequencies - float(lower)) / softness) * jax.nn.sigmoid(
        (float(upper) - frequencies) / softness
    )
    return 0.5 * jnp.mean(risk**2)


def validate_vmec_boundary_alfven_gradient(
    settings: VmecBoozerFaxDesignSettings,
    *,
    parameters: Any | None = None,
    finite_difference_step: float = 1.0e-4,
    descent_step: float = 1.0e-3,
    line_search_halvings: int = 8,
) -> VmecBoozerFaxGradientValidation:
    """Check AD gradients against finite differences and take one descent step."""
    context = prepare_vmec_boozer_fax_design_context(settings)
    params = (
        jnp.zeros((len(context.parameter_slots),), dtype=jnp.float64)
        if parameters is None
        else jnp.asarray(parameters, dtype=jnp.float64)
    )
    if int(params.shape[0]) != len(context.parameter_slots):
        raise ValueError("parameters length must match selected VMEC boundary parameters")

    objective = jax.jit(lambda x: vmec_boundary_alfven_risk_objective(x, context))
    frequency_map = jax.jit(lambda x: vmec_boundary_fax_frequencies(x, context))
    frequencies = frequency_map(params)
    value, gradient = jax.jit(jax.value_and_grad(objective))(params)
    finite_difference = _centered_difference_gradient(objective, params, finite_difference_step)
    relative_error = float(
        jnp.linalg.norm(gradient - finite_difference)
        / jnp.maximum(jnp.linalg.norm(finite_difference), 1.0e-30)
    )
    descent_params, descent_value = _line_search_descent(
        objective,
        params,
        gradient,
        float(descent_step),
        int(line_search_halvings),
    )
    descent_frequencies = frequency_map(descent_params)
    metadata = (
        ("input_path", str(Path(settings.input_path).expanduser().resolve())),
        ("vmec_ns", int(settings.vmec_ns)),
        ("vmec_ftol", float(settings.vmec_ftol)),
        ("mboz", int(settings.mboz)),
        ("nboz", int(settings.nboz)),
        ("surface_s", tuple(float(x) for x in np.asarray(context.surface_s))),
        ("avoid_window_hz", tuple(float(x) for x in settings.avoid_window_hz)),
    )
    return VmecBoozerFaxGradientValidation(
        parameter_names=context.parameter_names,
        parameters=params,
        objective=value,
        gradient=gradient,
        finite_difference_gradient=finite_difference,
        relative_gradient_error=relative_error,
        initial_frequencies_hz=frequencies,
        descent_parameters=descent_params,
        descent_objective=descent_value,
        descent_frequencies_hz=descent_frequencies,
        descent_step_norm=float(jnp.linalg.norm(descent_params - params)),
        metadata=metadata,
    )


def run_vmec_boundary_alfven_gradient_convergence(
    base_settings: VmecBoozerFaxDesignSettings,
    *,
    levels: Sequence[tuple[str, Mapping[str, Any]]] | None = None,
    parameters: Any | None = None,
    finite_difference_step: float = 1.0e-4,
    descent_step: float = 1.0e-3,
) -> VmecBoozerFaxGradientConvergence:
    """Run a resolution ladder for the end-to-end VMEC-boundary gradient.

    Each ladder level validates the JAX gradient against centered finite
    differences through the same in-memory chain used by
    :func:`validate_vmec_boundary_alfven_gradient`. The last level is treated
    as the numerical reference for frequency, objective, and gradient drift.
    """
    requested_levels: Sequence[tuple[str, Mapping[str, Any]]] = (
        (
            (
                "ftol1e-10_boozer3_grid8",
                {"vmec_ftol": 1.0e-10, "mboz": 3, "nboz": 3, "theta_points": 8, "zeta_points": 8},
            ),
            (
                "ftol1e-12_boozer3_grid8",
                {"vmec_ftol": 1.0e-12, "mboz": 3, "nboz": 3, "theta_points": 8, "zeta_points": 8},
            ),
            (
                "ftol1e-12_boozer4_grid10",
                {"vmec_ftol": 1.0e-12, "mboz": 4, "nboz": 4, "theta_points": 10, "zeta_points": 10},
            ),
        )
        if levels is None
        else levels
    )
    if not requested_levels:
        raise ValueError("at least one convergence level is required")

    validations: list[tuple[str, VmecBoozerFaxDesignSettings, VmecBoozerFaxGradientValidation]] = []
    for label, overrides in requested_levels:
        settings = replace(base_settings, **dict(overrides))
        result = validate_vmec_boundary_alfven_gradient(
            settings,
            parameters=parameters,
            finite_difference_step=finite_difference_step,
            descent_step=descent_step,
        )
        validations.append((str(label), settings, result))

    reference_label, _, reference = validations[-1]
    reference_frequencies = np.asarray(reference.initial_frequencies_hz, dtype=float)
    reference_gradient = np.asarray(reference.gradient, dtype=float)
    reference_objective = float(reference.objective)

    points = []
    for label, settings, result in validations:
        frequencies = np.asarray(result.initial_frequencies_hz, dtype=float)
        gradient = np.asarray(result.gradient, dtype=float)
        points.append(
            VmecBoozerFaxGradientConvergencePoint(
                label=label,
                parameter_names=result.parameter_names,
                vmec_ftol=float(settings.vmec_ftol),
                mboz=int(settings.mboz),
                nboz=int(settings.nboz),
                theta_points=int(settings.theta_points),
                zeta_points=int(settings.zeta_points),
                surfaces=tuple(float(surface) for surface in settings.surfaces),
                objective=float(result.objective),
                descent_objective=float(result.descent_objective),
                relative_gradient_error=float(result.relative_gradient_error),
                gradient=tuple(float(x) for x in gradient),
                finite_difference_gradient=tuple(
                    float(x) for x in np.asarray(result.finite_difference_gradient, dtype=float)
                ),
                frequencies_hz=tuple(float(x) for x in frequencies),
                frequency_delta_to_reference_hz=float(np.linalg.norm(frequencies - reference_frequencies)),
                objective_delta_to_reference=float(abs(float(result.objective) - reference_objective)),
                gradient_delta_to_reference=float(np.linalg.norm(gradient - reference_gradient)),
            )
        )

    return VmecBoozerFaxGradientConvergence(
        points=tuple(points),
        reference_label=reference_label,
        max_relative_gradient_error=max(point.relative_gradient_error for point in points),
        max_frequency_delta_to_reference_hz=max(point.frequency_delta_to_reference_hz for point in points),
        max_objective_delta_to_reference=max(point.objective_delta_to_reference for point in points),
        max_gradient_delta_to_reference=max(point.gradient_delta_to_reference for point in points),
    )


def validate_vmec_boundary_frequency_jacobian(
    settings: VmecBoozerFaxDesignSettings,
    *,
    parameters: Any | None = None,
    cotangent: Any | None = None,
    finite_difference_step: float = 2.0e-5,
) -> VmecBoozerFaxJacobianValidation:
    """Check VMEC-boundary frequency Jacobians and pullbacks.

    This validates a vector-output map, not only the scalar risk objective:

    ``boundary coefficients -> global Alfvén frequencies``.

    The JAX ``jacrev`` Jacobian is compared with a centered finite-difference
    Jacobian. A separate VJP pullback is checked against the finite-difference
    Jacobian contracted with a deterministic cotangent vector.
    """
    context = prepare_vmec_boozer_fax_design_context(settings)
    params = (
        jnp.zeros((len(context.parameter_slots),), dtype=jnp.float64)
        if parameters is None
        else jnp.asarray(parameters, dtype=jnp.float64)
    )
    if int(params.shape[0]) != len(context.parameter_slots):
        raise ValueError("parameters length must match selected VMEC boundary parameters")

    frequencies = jax.jit(lambda x: vmec_boundary_fax_frequencies(x, context))

    values = frequencies(params)
    jacobian = jax.jit(jax.jacrev(frequencies))(params)
    finite_difference = _centered_difference_jacobian(frequencies, params, finite_difference_step)
    relative_jacobian_error = float(
        jnp.linalg.norm(jacobian - finite_difference)
        / jnp.maximum(jnp.linalg.norm(finite_difference), 1.0e-30)
    )
    if cotangent is None:
        weights = jnp.asarray([1.0, -0.5, 0.25, -0.125, 0.0625, -0.03125], dtype=jnp.float64)
        cotangent_array = weights[: int(values.shape[0])]
    else:
        cotangent_array = jnp.asarray(cotangent, dtype=jnp.float64)
    if int(cotangent_array.shape[0]) != int(values.shape[0]):
        raise ValueError("cotangent length must match the number of returned frequencies")

    _, pullback_fun = jax.vjp(frequencies, params)
    pullback = pullback_fun(cotangent_array)[0]
    finite_difference_pullback = finite_difference.T @ cotangent_array
    relative_pullback_error = float(
        jnp.linalg.norm(pullback - finite_difference_pullback)
        / jnp.maximum(jnp.linalg.norm(finite_difference_pullback), 1.0e-30)
    )
    metadata = (
        ("input_path", str(Path(settings.input_path).expanduser().resolve())),
        ("vmec_ns", int(settings.vmec_ns)),
        ("vmec_ftol", float(settings.vmec_ftol)),
        ("mboz", int(settings.mboz)),
        ("nboz", int(settings.nboz)),
        ("surface_s", tuple(float(x) for x in np.asarray(context.surface_s))),
    )
    return VmecBoozerFaxJacobianValidation(
        parameter_names=context.parameter_names,
        parameters=params,
        frequencies_hz=values,
        frequency_jacobian=jacobian,
        finite_difference_jacobian=finite_difference,
        relative_jacobian_error=relative_jacobian_error,
        cotangent=cotangent_array,
        pullback=pullback,
        finite_difference_pullback=finite_difference_pullback,
        relative_pullback_error=relative_pullback_error,
        metadata=metadata,
    )


def _nearest_surface_indices(s_half: jax.Array, surfaces: tuple[float, ...]) -> jax.Array:
    s_array = np.asarray(s_half, dtype=float)
    indices = [int(np.argmin(np.abs(s_array - float(surface)))) for surface in surfaces]
    if len(set(indices)) != len(indices):
        raise ValueError("requested surfaces collapse to duplicate VMEC half-grid indices")
    return jnp.asarray(indices, dtype=jnp.int32)


def _centered_difference_gradient(objective: Any, params: jax.Array, step: float) -> jax.Array:
    h = float(step)
    entries = []
    for index in range(int(params.shape[0])):
        direction = jnp.zeros_like(params).at[index].set(h)
        entries.append((objective(params + direction) - objective(params - direction)) / (2.0 * h))
    return jnp.asarray(entries)


def _centered_difference_jacobian(function: Any, params: jax.Array, step: float) -> jax.Array:
    h = float(step)
    columns = []
    for index in range(int(params.shape[0])):
        direction = jnp.zeros_like(params).at[index].set(h)
        columns.append((function(params + direction) - function(params - direction)) / (2.0 * h))
    return jnp.stack(columns, axis=1)


def _line_search_descent(
    objective: Any,
    params: jax.Array,
    gradient: jax.Array,
    step: float,
    max_halvings: int,
) -> tuple[jax.Array, jax.Array]:
    value0 = objective(params)
    norm = jnp.linalg.norm(gradient)
    if float(norm) == 0.0:
        return params, value0
    direction = -gradient / jnp.maximum(norm, 1.0e-30)
    best_params = params
    best_value = value0
    alpha = float(step)
    for _ in range(max(1, int(max_halvings))):
        trial = params + alpha * direction
        value = objective(trial)
        if float(value) < float(best_value):
            best_params = trial
            best_value = value
            break
        alpha *= 0.5
    return best_params, best_value
