"""Compile and steady-state timings of the FAX hot paths.

Run: python examples/benchmark.py [--design]   (``--design`` adds the vmex -> booz -> FAX gradient)
"""

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

import fax
from fax.eigen import dense, global_matrices, shift_invert


def timed(f, *args):
    t = time.perf_counter()
    jax.block_until_ready(f(*args))
    first = time.perf_counter() - t
    t = time.perf_counter()
    for _ in range(3):
        jax.block_until_ready(f(*args))
    return f"{first:8.3f} {(time.perf_counter() - t) / 3:8.4f}"


def paths(h, rho, target_hz=3e5, ntheta=32, nzeta=4):
    F = lambda g: global_matrices(g, h, rho, ntheta, nzeta)  # noqa: E731
    return {
        "continuum": lambda g: fax.continuum(g, h, rho),
        "assembly": F,
        "shift-invert (8)": lambda g: shift_invert(*F(g), (2 * jnp.pi * target_hz) ** 2)[0],
        "grad frequency_hz": jax.grad(
            lambda g: fax.frequency_hz(g, h, rho, target_hz=target_hz, ntheta=ntheta, nzeta=nzeta), allow_int=True
        ),
        "dense": lambda g: jnp.linalg.eigvalsh(dense(F(g)[0])),
    }


root = Path(__file__).parent
rho = fax.mass_density(1e19, 2.0)
ctok = fax.read_boozmn(root.parent / "tests" / "data" / "boozmn_ctok.nc").full_mesh()
torus = lambda N: fax.circular_torus(s=(jnp.arange(N) + 0.5) / N, epsilon_B=0.18)  # noqa: E731
print(f"backend {jax.default_backend()}; case, N, H, path: compile+first, steady (s)")
for name, geo, h in [
    ("CTOK", ctok, fax.mode_family(2, 12)),
    ("torus", torus(200), fax.mode_family(6, 20)),
    ("torus", torus(400), fax.mode_family(6, 30)),
]:
    for label, f in paths(h, rho).items():
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
    step = jax.jit(jax.value_and_grad(lambda p: fax.window_risk(fax.design_frequencies(p, design), (50e3, 500e3), 25e3)))
    print(f"LP-QA    16   2 design grad        {timed(step, jnp.zeros(2))}")
