"""Pytest configuration for validation-grade numerical checks."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
