"""Smooth Alfvén objectives and the differentiable VMEC-boundary pipeline.

``boundary offsets -> vmex implicit equilibrium -> booz_xform_jax -> FAX``.
The equilibrium is differentiated with the implicit-function theorem
(``vmex.core.implicit.solve_implicit``), so gradients are those of the
converged equilibrium rather than of a truncated solver trajectory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .continuum import gaps
from .drive import FastIons, growth_rate
from .eigen import _hz, dense, eigenmodes, eigenvalue_mean, global_matrices, targeted_eigenvalues
from .geometry import Geometry, from_booz_xform, surface_grid


def window_risk(frequencies_hz: Any, window_hz: tuple[float, float], softness_hz: float) -> jax.Array:
    """``0.5 mean(sigma^2)`` where ``sigma`` is a smooth indicator of ``f`` inside ``window_hz``."""
    f = jnp.asarray(frequencies_hz)
    inside = jax.nn.sigmoid((f - window_hz[0]) / softness_hz) * jax.nn.sigmoid((window_hz[1] - f) / softness_hz)
    return 0.5 * jnp.mean(inside**2)


def soft_max(x: Any, sharpness: float = 10.0) -> jax.Array:
    """Log-sum-exp smooth maximum; it avoids the kinks of ``max`` at eigenvalue crossings."""
    return jax.nn.logsumexp(sharpness * jnp.asarray(x)) / sharpness


def drive_risk(gammas: Any, threshold: float = 0.0, scale: float = 1.0e3) -> jax.Array:
    """``softplus(soft_max(gamma_k) - gamma_th)`` in units of ``scale`` (1/s): a smooth penalty on the most unstable mode."""
    return jax.nn.softplus(soft_max(jnp.asarray(gammas) / scale) - threshold / scale)


def gap_closure(frequencies_hz: Any, scale_hz: float = 1.0e4) -> jax.Array:
    """Smooth penalty that grows as the largest continuum gap on each surface closes."""
    return jnp.mean(jax.nn.softplus(-gaps(frequencies_hz)[1] / scale_hz))


def compression_harmonics(geo: Geometry, ntheta: int = 32, nzeta: int = 64) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Complex harmonics of ``eps = |grad s|^2 / <|grad s|^2> - 1`` (Paul et al. 2025, eq. 4.28).

    ``eps = sum c[:, i, j] exp(i (dm[i] theta - n[j] zeta))`` in Boozer angles, with
    ``<.>`` the angle average and ``n`` the full toroidal mode number (``N_P dn``). The harmonic
    ``(dm, n)`` opens a gap at ``w0 = |iota dm - n| w_A / 2`` of **full width** ``2 w0 |c|`` (edge to edge;
    ``c`` and its conjugate ``(-dm, -n)`` together make the real amplitude ``2 |c|``).
    """
    g = surface_grid(geo, ntheta, nzeta).grad_s2
    c = jnp.fft.fft2(g / jnp.mean(g, axis=(1, 2), keepdims=True) - 1.0) / (ntheta * nzeta)
    return c, jnp.fft.fftfreq(ntheta, 1.0 / ntheta), -jnp.fft.fftfreq(nzeta, 1.0 / nzeta)


def continuum_cost(geo: Geometry, helicity: tuple[int, int] = (1, 1), ntheta: int = 32, nzeta: int = 64) -> jax.Array:
    """Paul et al. (2025) eq. 7.4: ``sum_s sum |eps_{dm,n}|^2 (iota dm - n)^2`` over ``|iota dm - n| > |iota M - N N_P|``.

    Each term is the squared full width ``(2 w0 |c| / w_A)^2 = |c|^2 (iota dm - n)^2`` of the gap that ``(dm, n)``
    opens at ``w0 / w_A = |iota dm - n| / 2``, so the cost narrows every gap above the frequency of the
    quasisymmetry helicity ``(M, N)``.
    """
    c, dm, n = compression_harmonics(geo, ntheta, nzeta)
    k = geo.iota[:, None, None] * dm[:, None] - n
    cut = jnp.abs(geo.iota * helicity[0] - helicity[1] * geo.nfp)[:, None, None]
    return jnp.sum(jnp.where(jnp.abs(k) > cut, jnp.abs(c) ** 2 * k**2, 0.0))


_BOUNDARY = {"rc": "rbc", "rs": "rbs", "zc": "zbc", "zs": "zbs"}


def _slot(parameter: str | tuple[str, int, int], ntor: int) -> tuple[str, int, int]:
    """``(field, row, m)`` of a boundary coefficient given as ``("rc", m, n)`` or ``"rc" + m + n`` (single digits, ``n`` signed)."""
    if isinstance(parameter, str):
        found = re.fullmatch(r"([rz][cs])(\d)(-?\d)", parameter)
        if found is None:
            raise ValueError(f"cannot read {parameter!r}: use 'rc01' (one digit each) or the tuple ('rc', m, n) for multi-digit modes")
        parameter = (found[1], int(found[2]), int(found[3]))
    name, m, n = parameter
    return _BOUNDARY[name], n + ntor, m


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
    parameters: tuple[str | tuple[str, int, int], ...] = ("rc01",),
    nodes: int = 24,
    mboz: int = 3,
    nboz: int = 3,
    ns: int = 16,
    ftol: float = 1.0e-12,
    max_iter: int = 5000,
    **options: Any,
) -> Design:
    """Prepare a design: parameters are boundary names ``rc|rs|zc|zs`` + ``m`` + ``n`` (e.g. ``"zs11"``, one digit each,
    ``n`` signed) or tuples ``("zs", m, n)``, which are required for multi-digit modes.

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
    slots = tuple(_slot(name, int(inp.ntor)) for name in parameters)
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
    return Design(
        path, tuple(map(str, parameters)), slots, jnp.asarray(harmonics, dtype=float), float(density), int(nodes), dict(options), cache
    )


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


def design_state(offsets: Any, design: Design) -> tuple[Any, Any]:
    """vmex ``(state, runtime)`` of the boundary ``input + offsets``, for vmex objectives (QS, aspect, iota)."""
    from vmex.core import implicit

    params, config = design_params(offsets, design), design._cache["config"]
    return implicit.solve_implicit(params, config), implicit.runtime_from_params(params, config)


def design_geometry(offsets: Any, design: Design) -> Geometry:
    """Half-mesh Boozer geometry of the boundary ``input + offsets`` (traceable in ``offsets``)."""
    return state_geometry(*design_state(offsets, design), design)


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


def geometry_growth_rates(geo: Geometry, design: Design, ions: FastIons, mode_count: int = 3, **drive: Any) -> jax.Array:
    """Fast-ion growth rates (1/s) of the first ``mode_count`` of the ``count`` modes nearest ``target_hz``.

    The modes come from the shift-invert solve on the design's FE nodes (differentiated
    through the iteration); ``ions.density`` is a scalar or a profile on those nodes and
    ``drive`` goes to :func:`fax.growth_rate`.
    """
    opts = dict(design.options)
    target, solver = opts.pop("target_hz"), {key: opts.pop(key) for key in ("count", "iters") if key in opts}
    geo = geo.interpolate(jnp.arange(1, design.nodes + 1) / (design.nodes + 1))
    ev = eigenmodes(geo, design.harmonics, design.density, target_hz=target, **solver, **opts)
    rate = lambda w2, y: growth_rate(jnp.sqrt(w2), y, geo, design.harmonics, ions, design.density, **drive)  # noqa: E731
    return jnp.stack([rate(ev.omega2[i], ev.structures[i]) for i in range(mode_count)])
