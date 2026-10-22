"""FAX: differentiable shear-Alfven continua and eigenmodes for stellarators in JAX."""

import jax

jax.config.update("jax_enable_x64", True)  # eigenvalues of stiff Alfven operators need float64

from .continuum import continuum, continuum_matrices, gap_location, gaps, mass_density, mode_family
from .drive import FastIons, growth_rate
from .eigen import eigenmodes, eigenvalue_mean, frequency_hz, global_matrices
from .geometry import Geometry, circular_torus, from_vmec_input, from_wout, read_boozmn, surface_grid
from .io import load, run
from .objectives import design_frequencies, design_geometry, gap_closure, prepare_design, soft_max, window_risk

__version__ = "0.1.0"
__all__ = [
    "FastIons",
    "Geometry",
    "circular_torus",
    "continuum",
    "continuum_matrices",
    "design_frequencies",
    "design_geometry",
    "eigenmodes",
    "eigenvalue_mean",
    "frequency_hz",
    "from_vmec_input",
    "from_wout",
    "gap_closure",
    "gap_location",
    "gaps",
    "global_matrices",
    "growth_rate",
    "load",
    "mass_density",
    "mode_family",
    "prepare_design",
    "read_boozmn",
    "run",
    "soft_max",
    "surface_grid",
    "window_risk",
]
