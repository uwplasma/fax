"""End-to-end VMEC-JAX -> Boozer -> FAX design objectives.

This module is optional: it requires both ``vmec_jax`` and
``booz_xform_jax``.  The functions here keep the full differentiable path in
memory:

``VMEC boundary coefficients -> vmec_jax state -> booz_xform_jax harmonics ->
FAX global Alfvén objective``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .boozer import BoozerGeometry
from .global_modes import radial_block_problem_from_boozer_geometry, solve_radial_block_eigenmodes


@dataclass(frozen=True)
class VmecBoozerFaxDesignSettings:
    """Settings for a differentiable VMEC-boundary Alfvén design objective."""

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
    vmec_max_iter: int = 2
    vmec_step_size: float = 1.0e-3
    vmec_jacobian_penalty: float = 100.0
    vmec_project: bool = True


@dataclass(frozen=True)
class VmecBoozerFaxDesignContext:
    """Prepared objects for repeated VMEC-boundary design evaluations."""

    settings: VmecBoozerFaxDesignSettings
    static: Any
    indata: Any
    boundary: Any
    parameter_specs: tuple[Any, ...]
    parameter_names: tuple[str, ...]
    signgs: int
    flux: Any
    pressure: jax.Array
    booz_constants: Any
    booz_grids: Any
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


def prepare_vmec_boozer_fax_design_context(
    settings: VmecBoozerFaxDesignSettings,
) -> VmecBoozerFaxDesignContext:
    """Prepare static VMEC/Boozer objects for boundary-design studies.

    The returned context is reusable for objective, gradient, UQ, and inverse
    design calls.  The VMEC solve used by :func:`vmec_boundary_fax_frequencies`
    is ``vmec_jax.solve_fixed_boundary_from_boundary`` with
    ``differentiable=True`` so that JAX can validate gradients directly against
    centered finite differences.
    """
    try:
        import vmec_jax as vj
        from vmec_jax.booz_input import booz_xform_inputs_from_state
        from booz_xform_jax.jax_api import prepare_booz_xform_constants
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "prepare_vmec_boozer_fax_design_context requires vmec_jax and "
            "booz_xform_jax to be importable."
        ) from exc

    path = Path(settings.input_path).expanduser().resolve()
    cfg, indata = vj.load_config(str(path))
    static = vj.build_static(cfg)
    boundary = vj.boundary_from_indata(indata, static.modes)
    specs = tuple(
        vj.boundary_param_specs(
            boundary,
            static.modes,
            max_mode=int(settings.max_mode),
            min_coeff=0.0,
            include=settings.include,
            fix=settings.fix,
        )[: int(settings.parameter_count)]
    )
    if not specs:
        raise ValueError("no VMEC boundary parameters were selected")

    state0 = vj.initial_guess_from_boundary(
        static,
        boundary,
        indata,
        vmec_project=bool(settings.vmec_project),
    )
    signgs = int(vj.signgs_from_sqrtg(np.asarray(vj.eval_geom(state0, static).sqrtg), axis_index=1))
    flux = vj.flux_profiles_from_indata(indata, static.s, signgs=signgs)
    pressure = jnp.zeros_like(jnp.asarray(static.s))

    inputs0 = booz_xform_inputs_from_state(
        state=state0,
        static=static,
        indata=indata,
        signgs=signgs,
        flux=flux,
    )
    constants, grids = prepare_booz_xform_constants(
        nfp=int(inputs0.nfp),
        mboz=int(settings.mboz),
        nboz=int(settings.nboz),
        asym=bool(static.cfg.lasym),
        xm=inputs0.xm,
        xn=inputs0.xn,
        xm_nyq=inputs0.xm_nyq,
        xn_nyq=inputs0.xn_nyq,
    )
    s_full = jnp.asarray(static.s)
    s_half = 0.5 * (s_full[:-1] + s_full[1:])
    surface_indices = _nearest_surface_indices(s_half, settings.surfaces)

    return VmecBoozerFaxDesignContext(
        settings=settings,
        static=static,
        indata=indata,
        boundary=boundary,
        parameter_specs=specs,
        parameter_names=tuple(vj.boundary_param_names(specs)),
        signgs=signgs,
        flux=flux,
        pressure=pressure,
        booz_constants=constants,
        booz_grids=grids,
        surface_indices=surface_indices,
        surface_s=jnp.take(s_half, surface_indices),
    )


def vmec_state_to_boozer_geometry(state: Any, context: VmecBoozerFaxDesignContext) -> BoozerGeometry:
    """Transform a VMEC-JAX state to a FAX Boozer geometry in memory."""
    try:
        from vmec_jax.booz_input import booz_xform_inputs_from_state
        from booz_xform_jax.jax_api import booz_xform_from_inputs
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("vmec_state_to_boozer_geometry requires vmec_jax and booz_xform_jax") from exc

    inputs = booz_xform_inputs_from_state(
        state=state,
        static=context.static,
        indata=context.indata,
        signgs=context.signgs,
        flux=context.flux,
    )
    output = booz_xform_from_inputs(
        inputs=inputs,
        constants=context.booz_constants,
        grids=context.booz_grids,
        surface_indices=context.surface_indices,
        jit=False,
    )
    zeros = jnp.zeros_like(jnp.asarray(output["bmnc_b"]))
    gmnc = output["gmnc_b"] if "gmnc_b" in output else output.get("gmn_b", zeros)
    return BoozerGeometry(
        source="vmec_jax_booz_xform_jax_design",
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
        gmnc=jnp.asarray(gmnc),
        gmns=jnp.asarray(output.get("gmns_b", zeros)),
        iota=jnp.asarray(output["iota_b"]),
        boozer_i=jnp.asarray(output["buco_b"]),
        boozer_g=jnp.asarray(output["bvco_b"]),
        field_periods=int(inputs.nfp),
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
    """Return global Alfvén frequencies for VMEC boundary perturbations."""
    try:
        import vmec_jax as vj
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("vmec_boundary_fax_frequencies requires vmec_jax") from exc

    settings = context.settings
    params = jnp.asarray(parameters)
    boundary = vj.apply_boundary_params(context.boundary, context.parameter_specs, params)
    state = vj.solve_fixed_boundary_from_boundary(
        boundary=boundary,
        static=context.static,
        indata=context.indata,
        flux=context.flux,
        pressure=context.pressure,
        signgs=context.signgs,
        max_iter=int(settings.vmec_max_iter),
        step_size=float(settings.vmec_step_size),
        jacobian_penalty=float(settings.vmec_jacobian_penalty),
        jit_grad=False,
        differentiable=True,
        stop_grad_in_update=False,
        verbose=False,
        vmec_project=bool(settings.vmec_project),
    )
    geometry = vmec_state_to_boozer_geometry(state, context)
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
        jnp.zeros((len(context.parameter_specs),), dtype=jnp.float64)
        if parameters is None
        else jnp.asarray(parameters, dtype=jnp.float64)
    )
    if int(params.shape[0]) != len(context.parameter_specs):
        raise ValueError("parameters length must match selected VMEC boundary parameters")

    def objective(x: jax.Array) -> jax.Array:
        return vmec_boundary_alfven_risk_objective(x, context)

    frequencies = vmec_boundary_fax_frequencies(params, context)
    value, gradient = jax.value_and_grad(objective)(params)
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
    descent_frequencies = vmec_boundary_fax_frequencies(descent_params, context)
    metadata = (
        ("input_path", str(Path(settings.input_path).expanduser().resolve())),
        ("vmec_max_iter", int(settings.vmec_max_iter)),
        ("vmec_step_size", float(settings.vmec_step_size)),
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
