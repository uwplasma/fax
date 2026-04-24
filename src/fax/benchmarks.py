"""Lightweight benchmark helpers for local development."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class BenchmarkResult:
    """Timing summary for a callable benchmark."""

    name: str
    repeats: int
    warmups: int
    min_seconds: float
    median_seconds: float
    max_seconds: float


def _block_until_ready(value: Any) -> Any:
    if hasattr(value, "block_until_ready"):
        return value.block_until_ready()
    if isinstance(value, (tuple, list)):
        for item in value:
            _block_until_ready(item)
    if isinstance(value, dict):
        for item in value.values():
            _block_until_ready(item)
    return value


def time_callable(
    name: str,
    func: Callable[[], Any],
    *,
    warmups: int = 1,
    repeats: int = 5,
) -> BenchmarkResult:
    """Time a no-argument callable with JAX ``block_until_ready`` support."""
    for _ in range(int(warmups)):
        _block_until_ready(func())

    timings: list[float] = []
    for _ in range(int(repeats)):
        start = time.perf_counter()
        _block_until_ready(func())
        timings.append(time.perf_counter() - start)

    data = np.asarray(timings, dtype=float)
    return BenchmarkResult(
        name=name,
        repeats=int(repeats),
        warmups=int(warmups),
        min_seconds=float(np.min(data)),
        median_seconds=float(np.median(data)),
        max_seconds=float(np.max(data)),
    )

