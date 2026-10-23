"""Smooth Alfvén objectives and the differentiable VMEC-boundary pipeline.

``boundary offsets -> vmex implicit equilibrium -> booz_xform_jax -> FAX``.
The equilibrium is differentiated with the implicit-function theorem
(``vmex.core.implicit.solve_implicit``), so gradients are those of the
converged equilibrium rather than of a truncated solver trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .continuum import gaps
from .eigen import _hz, dense, eigenvalue_mean, global_matrices, targeted_eigenvalues
from .geometry import Geometry, from_booz_xform


def window_risk(frequencies_hz: Any, window_hz: tuple[float, float], softness_hz: float) -> jax.Array:
    """``0.5 mean(sigma^2)`` where ``sigma`` is a smooth indicator of ``f`` inside ``window_hz``."""
    f = jnp.asarray(frequencies_hz)
    inside = jax.nn.sigmoid((f - window_hz[0]) / softness_hz) * jax.nn.sigmoid((window_hz[1] - f) / softness_hz)
    return 0.5 * jnp.mean(inside**2)


def soft_max(x: Any, sharpness: float = 10.0) -> jax.Array:
    """Log-sum-exp smooth maximum; it avoids the kinks of ``max`` at eigenvalue crossings."""
    return jax.nn.logsumexp(sharpness * jnp.asarray(x)) / sharpness


def gap_closure(frequencies_hz: Any, scale_hz: float = 1.0e4) -> jax.Array:
    """Smooth penalty that grows as the largest continuum gap on each surface closes."""
    return jnp.mean(jax.nn.softplus(-gaps(frequencies_hz)[1] / scale_hz))


_BOUNDARY = {"rc": "rbc", "rs": "rbs", "zc": "zbc", "zs": "zbs"}


@dataclass(frozen=True)
class Design:
    """VMEC-boundary design problem; build it with :func:`prepare_design`."""

    input_path: str
    names: tuple[str, ...]
    slots: tuple[tuple[str, int, int], ...]
    harmonics: Any
    density: float
    nodes: int
    options: dict[str, Any] = field(default_factory=dict)
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)


def prepare_design(
    input_path: str | Path,
    *,
    harmonics: Any,
    density: float,
    parameters: tuple[str, ...] = ("rc01",),
    nodes: int = 24,
    mboz: int = 3,
    nboz: int = 3,
    ns: int = 16,
    ftol: float = 1.0e-12,
    max_iter: int = 5000,
    **options: Any,
) -> Design:
    """Prepare a design: parameters are boundary names ``rc|rs|zc|zs`` + ``m`` + ``n`` (e.g. ``"zs11"``).

    The equilibrium (``ns``, ``ftol``) is transformed on the half-mesh rows ``j >= 2`` and
    interpolated linearly in ``s`` onto ``nodes`` uniform interior finite-element
    nodes ``k / (nodes + 1)``, so the radial domain is always ``[0, 1]`` and the
    FE grid refines independently of ``ns`` and ``mboz``. ``options`` go to
    :func:`fax.eigen.global_matrices` (``ntheta``, ``nzeta``) and the
    shift-invert solve (``target_hz``, ``count``, ``iters``).
    """
    from booz_xform_jax.jax_api import prepare_booz_xform_constants
    from vmex import VmecInput
    from vmex.core import implicit
    from vmex.core.boozer_tables import boozer_input_tables

    path = str(Path(input_path).expanduser().resolve())
    inp = VmecInput.from_file(path)
    config = implicit.make_config(inp, ns=ns, ftol=ftol, max_iterations=max_iter)
    params = implicit.params_from_input(inp)
    slots = tuple((_BOUNDARY[name[:2]], int(name[3:]) + int(inp.ntor), int(name[2])) for name in parameters)
    s_full = np.asarray(implicit.runtime_from_params(params, config).setup.s_full)
    modes: dict[str, np.ndarray] = {}

    def probe(p: Any) -> jax.Array:  # mode tables are static; read them from an abstract trace
        tables = boozer_input_tables(implicit.solve_implicit(p, config), implicit.runtime_from_params(p, config), 1)
        modes.update(xm=np.asarray(tables["xm"]), xn=np.asarray(tables["xn"]))
        return tables["bmnc"]

    jax.eval_shape(probe, params)
    constants, grids = prepare_booz_xform_constants(
        nfp=int(inp.nfp), mboz=mboz, nboz=nboz, asym=bool(inp.lasym), xm=modes["xm"], xn=modes["xn"], xm_nyq=modes["xm"], xn_nyq=modes["xn"]
    )
    s = jnp.asarray(0.5 * (s_full[1:-1] + s_full[2:]))  # half rows j >= 2: vmex <= 0.11.11 divides odd-m R, Z by sqrt(0) on j = 1
    cache = dict(config=config, params=params, s=s, constants=constants, grids=grids, nfp=int(inp.nfp), **modes)
    return Design(path, tuple(parameters), slots, jnp.asarray(harmonics, dtype=float), float(density), int(nodes), dict(options), cache)


def design_params(offsets: Any, design: Design) -> Any:
    """vmex ``ImplicitParams`` of the boundary ``input + offsets``."""
    params = design._cache["params"]
    for value, (name, row, m) in zip(jnp.asarray(offsets), design.slots, strict=True):
        params = replace(params, **{name: getattr(params, name).at[row, m].add(value)})
    return params


def state_geometry(state: Any, runtime: Any, design: Design) -> Geometry:
    """Boozer geometry on the half-mesh rows ``j >= 2`` from a vmex state and runtime (traceable)."""
    from booz_xform_jax.jax_api import booz_xform_jax_impl
    from vmex.core.boozer_tables import boozer_input_tables

    c = design._cache
    tables = [boozer_input_tables(state, runtime, j + 2) for j in range(c["s"].size)]
    names = ["rmnc", "zmns", "lmns", "bmnc", "bsubumnc", "bsubvmnc", "iota"]
    if c["config"].inp.lasym:
        names += ["rmns", "zmnc", "lmnc", "bmns", "bsubumns", "bsubvmns"]
    xm, xn = jnp.asarray(c["xm"]), jnp.asarray(c["xn"])
    output = booz_xform_jax_impl(
        **{k: jnp.stack([t[k] for t in tables]) for k in names},
        xm=xm,
        xn=xn,
        xm_nyq=xm,
        xn_nyq=xn,
        constants=c["constants"],
        grids=c["grids"],
    )
    return from_booz_xform(output, c["s"], c["nfp"])


def design_geometry(offsets: Any, design: Design) -> Geometry:
    """Half-mesh Boozer geometry of the boundary ``input + offsets`` (traceable in ``offsets``)."""
    from vmex.core import implicit

    params = design_params(offsets, design)
    config = design._cache["config"]
    return state_geometry(implicit.solve_implicit(params, config), implicit.runtime_from_params(params, config), design)


def geometry_frequencies(geo: Geometry, design: Design, mode_count: int = 3) -> jax.Array:
    """Global frequencies (Hz) of ``geo`` interpolated onto the design's FE nodes, with Hellmann-Feynman derivatives.

    Without ``target_hz`` the lowest ``mode_count`` from the dense solve; with it,
    the first ``mode_count`` of the ``count`` modes nearest ``target_hz`` (shift-invert).
    """
    opts = dict(design.options)
    target, solver = opts.pop("target_hz", None), {key: opts.pop(key) for key in ("count", "iters") if key in opts}
    geo = geo.interpolate(jnp.arange(1, design.nodes + 1) / (design.nodes + 1))
    F, G = global_matrices(geo, design.harmonics, design.density, **opts)
    if target is None:
        K, M = dense(F), dense(G)
        return _hz(jnp.stack([eigenvalue_mean(K, M, i, i + 1) for i in range(mode_count)]))
    lam = targeted_eigenvalues(F, G, float((2 * jnp.pi * target) ** 2), solver.get("count", 8), solver.get("iters", 40))
    return _hz(lam[:mode_count])


def design_frequencies(offsets: Any, design: Design, mode_count: int = 3) -> jax.Array:
    """``boundary offsets -> vmex -> booz_xform_jax -> FE nodes -> frequencies`` (Hz), differentiable."""
    return geometry_frequencies(design_geometry(offsets, design), design, mode_count)
