"""Drive screen, objectives, TOML runs, CLI and plotting."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fax
from fax.io import main

CTOK = str(Path(__file__).parent / "data" / "boozmn_ctok.nc")


def test_objectives_are_smooth():
    f = jnp.array([10e3, 125e3, 400e3])
    assert fax.window_risk(f, (50e3, 200e3), 5e3) == pytest.approx(0.5 / 3, rel=1e-3)
    assert fax.soft_max(f / 1e3) >= 400.0
    branches = jnp.array([[1e3, 2e3, 50e3], [1e3, 1.5e3, 2e3]])
    assert jnp.isfinite(jax.grad(lambda b: fax.gap_closure(b))(branches)).all()


def test_load_rejects_unknown_keys():
    with pytest.raises(ValueError):
        fax.load({"run": {"tsk": "continuum"}})
    assert fax.load({})["modes"]["n"] == [1]


@pytest.mark.parametrize("task", ["continuum", "eigenmodes"])
def test_run_cases_and_cli(tmp_path, task):
    case = tmp_path / "case.toml"
    case.write_text(
        f'[geometry]\nsource = "boozmn"\npath = "{CTOK}"\n[modes]\nn = [2]\nm_max = 4\n'
        f'[plasma]\nion_mass = 2.0\n[run]\ntask = "{task}"\noutput = "out"\nntheta = 16\nnzeta = 16\n'
    )
    assert main(["run", str(case), "--plot"]) == 0
    data = np.load(tmp_path / "out" / "results.npz")
    assert data["frequencies_hz"].shape[-1] == (5 if task == "continuum" else 8)
    assert (tmp_path / "out" / "continuum.png").exists()
    assert main(["plot", str(tmp_path / "out")]) == 0


def test_circular_geometry_run():
    out = fax.run({"run": {"ntheta": 8, "nzeta": 4}}, save=False)
    assert out["frequencies_hz"].shape == (24, 9)
    with pytest.raises(ValueError):
        fax.run({"run": {"task": "nope"}}, save=False)
