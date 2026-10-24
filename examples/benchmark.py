"""Compile and steady-state timings of the FAX hot paths.

Run: python examples/benchmark.py [--design]   (``--design`` adds the vmex -> booz -> FAX gradient)
"""

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

import fax
from fax.eigen import dense, global_matrices


def timed(f, *args):
    t = time.perf_counter()
    jax.block_until_ready(f(*args))
    first = time.perf_counter() - t
    runs = []
    for _ in range(5):
        t = time.perf_counter()
        jax.block_until_ready(f(*args))
        runs.append(time.perf_counter() - t)
    return f"{first:8.3f} {sorted(runs)[2]:8.4f}"  # cold, median of five warm


def paths(h, rho, target_hz, count, ntheta=32, nzeta=4):
    F = lambda g: global_matrices(g, h, rho, ntheta, nzeta)  # noqa: E731
    opts = dict(target_hz=target_hz, ntheta=ntheta, nzeta=nzeta)
    return {
        "continuum": lambda g: fax.continuum(g, h, rho),
        "assembly": F,
        "eigenmodes": lambda g: fax.eigenmodes(g, h, rho, count=count, **opts).omega2,
        "grad frequency_hz": jax.grad(lambda g: fax.frequency_hz(g, h, rho, count=count, **opts), allow_int=True),
        "dense": lambda g: jnp.linalg.eigvalsh(dense(F(g)[0])),
    }


def konies(N):
    """Könies et al. (2018) ITPA TAE geometry, n = 6, q = 1.71 + 0.16 r^2."""
    s = (jnp.arange(N) + 0.5) / N
    return fax.circular_torus(s=s, major_radius=10.0, minor_radius=1.0, B0=3.0, iota=1 / (1.71 + 0.16 * s), epsilon_B=0.1)


root = Path(__file__).parent
ctok = fax.read_boozmn(root.parent / "tests" / "data" / "boozmn_ctok.nc").full_mesh()
print(f"backend {jax.default_backend()}; case, N, H, path: compile+first, steady (s); eigenmodes residual, iterations")
for name, geo, h, rho, target, count in [
    ("CTOK", ctok, fax.mode_family(2, 12), fax.mass_density(1e19, 2.0), 384e3, 6),
    ("Konies", konies(200), fax.mode_family(6, 16, 5), fax.mass_density(2e19, 1.0), 6.46e4, 3),
    ("Konies", konies(400), fax.mode_family(6, 16, 5), fax.mass_density(2e19, 1.0), 6.46e4, 3),
]:
    ev = fax.eigenmodes(geo, h, rho, target_hz=target, count=count, ntheta=32, nzeta=4).status
    print(f"{name} status: residual {float(ev.residual):.1e}, iterations {int(ev.iterations)}, converged {bool(ev.converged)}")
    for label, f in paths(h, rho, target, count).items():
        if label != "dense" or geo.s.size * h.shape[0] <= 2000:
            print(f"{name:6s} {geo.s.size:4d} {h.shape[0]:3d} {label:18s} {timed(jax.jit(f), geo)}")
if "--design" in sys.argv:
    design = fax.prepare_design(
        root / "data" / "input.LandremanPaul2021_QA_lowres",
        harmonics=[[1.0, 0.0], [2.0, 1.0]],
        density=3.4e-8,
        parameters=("rc01", "zs01"),
        ntheta=8,
        nzeta=8,
    )
    step = jax.jit(jax.value_and_grad(lambda p: fax.window_risk(fax.design_frequencies(p, design)[0], (50e3, 500e3), 25e3)))
    print(f"LP-QA    16   2 design grad        {timed(step, jnp.zeros(2))}")
