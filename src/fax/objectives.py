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
from .eigen import eigenmodes
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
    surfaces: tuple[float, ...]
    options: dict[str, Any] = field(default_factory=dict)
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)


def prepare_design(
    input_path: str | Path,
    *,
    harmonics: Any,
    density: float,
    parameters: tuple[str, ...] = ("rc01",),
    surfaces: tuple[float, ...] = (0.25, 0.4, 0.55, 0.7),
    mboz: int = 3,
    nboz: int = 3,
    ns: int = 16,
    ftol: float = 1.0e-12,
    max_iter: int = 5000,
    **options: Any,
) -> Design:
    """Prepare a design: parameters are boundary names ``rc|rs|zc|zs`` + ``m`` + ``n`` (e.g. ``"zs11"``).

    ``options`` are passed to :func:`fax.eigen.eigenmodes` (``scale``, ``boundary``, ``ntheta``, ``nzeta``).
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
    s_half = 0.5 * (s_full[:-1] + s_full[1:])
    index = np.asarray([int(np.argmin(abs(s_half - x))) for x in surfaces])
    modes: dict[str, np.ndarray] = {}

    def probe(p: Any) -> jax.Array:  # mode tables are static; read them from an abstract trace
        tables = boozer_input_tables(implicit.solve_implicit(p, config), implicit.runtime_from_params(p, config), int(index[0]) + 1)
        modes.update(xm=np.asarray(tables["xm"]), xn=np.asarray(tables["xn"]))
        return tables["bmnc"]

    jax.eval_shape(probe, params)
    constants, grids = prepare_booz_xform_constants(
        nfp=int(inp.nfp), mboz=mboz, nboz=nboz, asym=bool(inp.lasym), xm=modes["xm"], xn=modes["xn"], xm_nyq=modes["xm"], xn_nyq=modes["xn"]
    )
    cache = dict(
        config=config, params=params, index=index, s=jnp.asarray(s_half[index]), constants=constants, grids=grids, nfp=int(inp.nfp), **modes
    )
    return Design(
        path, tuple(parameters), slots, jnp.asarray(harmonics, dtype=float), float(density), tuple(surfaces), dict(options), cache
    )


def design_geometry(offsets: Any, design: Design) -> Geometry:
    """Boozer geometry of the boundary ``input + offsets`` (traceable in ``offsets``)."""
    from booz_xform_jax.jax_api import booz_xform_jax_impl
    from vmex.core import implicit
    from vmex.core.boozer_tables import boozer_input_tables

    c = design._cache
    params = c["params"]
    for value, (name, row, m) in zip(jnp.asarray(offsets), design.slots, strict=True):
        params = replace(params, **{name: getattr(params, name).at[row, m].add(value)})
    state = implicit.solve_implicit(params, c["config"])
    runtime = implicit.runtime_from_params(params, c["config"])
    tables = [boozer_input_tables(state, runtime, int(i) + 1) for i in c["index"]]
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


def design_frequencies(offsets: Any, design: Design, mode_count: int = 3) -> jax.Array:
    """Lowest global frequencies (Hz) for the boundary offsets."""
    geo = design_geometry(offsets, design)
    return eigenmodes(geo, design.harmonics, design.density, **design.options).frequencies_hz[:mode_count]
