"""Vectorised helpers for quaternion time series of shape ``(T, 4)`` in ``(w, x, y, z)`` order."""

from __future__ import annotations

import numpy as np


def normalize_quaternion_series(q_series: np.ndarray) -> np.ndarray:
    q = np.asarray(q_series, dtype=float).reshape(-1, 4)
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    safe_norms = np.where(norms < 1e-12, 1.0, norms)
    out = q / safe_norms
    zero_mask = norms[:, 0] < 1e-12
    if np.any(zero_mask):
        out[zero_mask, :] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    neg_mask = out[:, 0] < 0.0
    if np.any(neg_mask):
        out[neg_mask, :] *= -1.0
    return out


def quaternion_to_axis_angle_series(q_series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qq = normalize_quaternion_series(q_series)
    w = np.clip(qq[:, 0], -1.0, 1.0)
    v = qq[:, 1:]
    s = np.linalg.norm(v, axis=1)
    angle = np.zeros((qq.shape[0],), dtype=float)
    axis = np.zeros((qq.shape[0], 3), dtype=float)
    axis[:, 0] = 1.0
    mask = s >= 1e-12
    if np.any(mask):
        axis[mask, :] = v[mask, :] / s[mask, None]
        angle[mask] = 2.0 * np.arctan2(s[mask], w[mask])
    return angle, axis
