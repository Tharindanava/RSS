"""Axis helpers for RSS direction-cosine matrices."""

from __future__ import annotations

import numpy as np


def normalize_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize row vectors without changing their coordinate frame."""

    array = np.asarray(values, dtype=float)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, float(eps))


def source_axes_from_direction_cosines(
    direction_cosines: np.ndarray,
    observer_axes: np.ndarray | None = None,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Convert RSS direction cosines into source-axis vectors.

    ``direction_cosines[i, j]`` is interpreted as the dot product between
    source axis ``i`` and observer axis ``j``.  The returned axes are expressed
    in the same coordinate frame as ``observer_axes``.  When no observer basis
    is supplied, the sensor XYZ basis is used.
    """

    coefficients = np.asarray(direction_cosines, dtype=float).reshape(3, 3)
    basis = np.eye(3, dtype=float) if observer_axes is None else np.asarray(observer_axes, dtype=float).reshape(3, 3)
    return normalize_rows(coefficients @ basis, eps=eps)
