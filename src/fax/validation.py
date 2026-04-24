"""Small validation metrics used by tests, examples, and reports."""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp


def relative_l2_error(actual: Any, expected: Any, *, floor: float = 1.0e-300) -> jnp.ndarray:
    """Return ``||actual - expected||_2 / max(||expected||_2, floor)``."""
    actual_array = jnp.asarray(actual)
    expected_array = jnp.asarray(expected)
    denominator = jnp.maximum(jnp.linalg.norm(expected_array), floor)
    return jnp.linalg.norm(actual_array - expected_array) / denominator


def max_relative_error(actual: Any, expected: Any, *, floor: float = 1.0e-300) -> jnp.ndarray:
    """Return the maximum elementwise relative error."""
    actual_array = jnp.asarray(actual)
    expected_array = jnp.asarray(expected)
    return jnp.max(jnp.abs(actual_array - expected_array) / jnp.maximum(jnp.abs(expected_array), floor))


def hermitian_relative_error(matrix: Any, *, floor: float = 1.0e-300) -> jnp.ndarray:
    """Return ``||A - A.H||_F / max(||A||_F, floor)``."""
    array = jnp.asarray(matrix)
    denominator = jnp.maximum(jnp.linalg.norm(array), floor)
    return jnp.linalg.norm(array - jnp.conjugate(array.T)) / denominator


def finite_difference_derivative(func: Callable[[float], float], x: float, *, step: float = 1.0e-5) -> float:
    """Return a centered finite-difference derivative of a scalar function."""
    h = float(step)
    return (float(func(float(x) + h)) - float(func(float(x) - h))) / (2.0 * h)


def observed_order(error_coarse: float, error_fine: float, refinement: float) -> float:
    """Return the observed convergence order from two errors."""
    return float(jnp.log(float(error_coarse) / float(error_fine)) / jnp.log(float(refinement)))

