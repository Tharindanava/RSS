"""Quaternion-domain rotation helpers used by RSS."""

from __future__ import annotations

import numpy as np
import torch


def _as_three_by_time_numpy(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    if array.shape[0] == 3:
        return array
    if array.shape[1] == 3:
        return array.T
    raise ValueError(f"{name} must have shape (3, T) or (T, 3).")


def _as_three_by_time_torch(values: torch.Tensor, *, name: str) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError(f"{name} must be a 2D tensor.")
    if values.shape[0] == 3:
        return values
    if values.shape[1] == 3:
        return values.T
    raise ValueError(f"{name} must have shape (3, T) or (T, 3).")


def angles_to_quaternion_components(angle_signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode three Euler-angle signals as a scalar and three vector coefficients."""

    angles = _as_three_by_time_numpy(angle_signals, name="angle_signals")
    half = 0.5 * angles
    c1, c2, c3 = np.cos(half[0]), np.cos(half[1]), np.cos(half[2])
    s1, s2, s3 = np.sin(half[0]), np.sin(half[1]), np.sin(half[2])
    scalar = c3 * c2 * c1 + s3 * s2 * s1
    vector = np.stack(
        (
            c3 * c2 * s1 - s3 * s2 * c1,
            c3 * s2 * c1 + s3 * c2 * s1,
            s3 * c2 * c1 - c3 * s2 * s1,
        ),
        axis=0,
    )
    return scalar, vector


def angles_to_quaternion_components_torch(angle_signals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch equivalent of :func:`angles_to_quaternion_components`."""

    angles = _as_three_by_time_torch(angle_signals, name="angle_signals")
    half = 0.5 * angles
    c1, c2, c3 = torch.cos(half[0]), torch.cos(half[1]), torch.cos(half[2])
    s1, s2, s3 = torch.sin(half[0]), torch.sin(half[1]), torch.sin(half[2])
    scalar = c3 * c2 * c1 + s3 * s2 * s1
    vector = torch.stack(
        (
            c3 * c2 * s1 - s3 * s2 * c1,
            c3 * s2 * c1 + s3 * c2 * s1,
            s3 * c2 * c1 - c3 * s2 * s1,
        ),
        dim=0,
    )
    return scalar, vector


def normalize_quaternions(
    scalar: np.ndarray,
    vector: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Normalize scalar/vector components and return a ``(T, 4)`` quaternion series."""

    scalar_array = np.asarray(scalar, dtype=float).reshape(-1)
    vector_array = _as_three_by_time_numpy(vector, name="vector")
    if vector_array.shape[1] != scalar_array.size:
        raise ValueError("scalar and vector must share the same sample count.")
    quaternions = np.column_stack((scalar_array, vector_array.T))
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, float(eps))
    normalized = quaternions / safe_norms
    collapsed = norms[:, 0] <= float(eps)
    if np.any(collapsed):
        normalized[collapsed] = np.array([1.0, 0.0, 0.0, 0.0])
    normalized = np.where(normalized[:, :1] < 0.0, -normalized, normalized)
    return normalized


def normalize_quaternions_torch(
    scalar: torch.Tensor,
    vector: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Differentiable Torch equivalent of :func:`normalize_quaternions`."""

    scalar_tensor = scalar.reshape(-1)
    vector_tensor = _as_three_by_time_torch(vector, name="vector")
    if vector_tensor.shape[1] != scalar_tensor.numel():
        raise ValueError("scalar and vector must share the same sample count.")
    quaternions = torch.cat((scalar_tensor.unsqueeze(0), vector_tensor), dim=0).T
    norms = torch.linalg.vector_norm(quaternions, dim=1, keepdim=True)
    normalized = quaternions / torch.clamp(norms, min=float(eps))
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=normalized.dtype, device=normalized.device)
    normalized = torch.where(norms <= float(eps), identity.unsqueeze(0), normalized)
    return torch.where(normalized[:, :1] < 0.0, -normalized, normalized)


def quaternions_to_angles(quaternions: np.ndarray) -> np.ndarray:
    """Decode a quaternion series with clipped asin and quadrant-safe atan2."""

    q = np.asarray(quaternions, dtype=float).reshape(-1, 4)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    theta1 = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    theta2 = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    theta3 = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack((theta1, theta2, theta3), axis=0)


def quaternions_to_angles_torch(quaternions: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Differentiable Torch equivalent of :func:`quaternions_to_angles`."""

    q = quaternions.reshape(-1, 4)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    theta1 = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    asin_arg = torch.clamp(2.0 * (w * y - z * x), min=-1.0 + float(eps), max=1.0 - float(eps))
    theta2 = torch.asin(asin_arg)
    theta3 = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.stack((theta1, theta2, theta3), dim=0)


def decode_scalar_and_vector(
    scalar: np.ndarray,
    vector: np.ndarray,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    quaternions = normalize_quaternions(scalar, vector, eps=eps)
    return quaternions_to_angles(quaternions), quaternions


def decode_scalar_and_vector_torch(
    scalar: torch.Tensor,
    vector: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    quaternions = normalize_quaternions_torch(scalar, vector, eps=eps)
    return quaternions_to_angles_torch(quaternions, eps=eps), quaternions


def normalize_quaternion_array(quaternions: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Normalize scalar-first quaternions along the final dimension."""

    values = np.asarray(quaternions, dtype=float)
    if values.ndim < 1 or values.shape[-1] != 4:
        raise ValueError("quaternions must have final dimension 4.")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    normalized = values / np.maximum(norms, float(eps))
    collapsed = norms[..., 0] <= float(eps)
    if np.any(collapsed):
        normalized = normalized.copy()
        normalized[collapsed] = np.array([1.0, 0.0, 0.0, 0.0])
    return np.where(normalized[..., :1] < 0.0, -normalized, normalized)


def normalize_quaternion_array_torch(quaternions: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Differentiable equivalent of :func:`normalize_quaternion_array`."""

    if quaternions.ndim < 1 or int(quaternions.shape[-1]) != 4:
        raise ValueError("quaternions must have final dimension 4.")
    norms = torch.linalg.vector_norm(quaternions, dim=-1, keepdim=True)
    normalized = quaternions / torch.clamp(norms, min=float(eps))
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=quaternions.dtype, device=quaternions.device)
    normalized = torch.where(norms <= float(eps), identity, normalized)
    return torch.where(normalized[..., :1] < 0.0, -normalized, normalized)


def quaternion_conjugate(quaternions: np.ndarray) -> np.ndarray:
    """Return scalar-first quaternion conjugates without changing shape."""

    values = np.asarray(quaternions, dtype=float)
    if values.ndim < 1 or values.shape[-1] != 4:
        raise ValueError("quaternions must have final dimension 4.")
    result = values.copy()
    result[..., 1:] *= -1.0
    return result


def quaternion_conjugate_torch(quaternions: torch.Tensor) -> torch.Tensor:
    if quaternions.ndim < 1 or int(quaternions.shape[-1]) != 4:
        raise ValueError("quaternions must have final dimension 4.")
    return torch.cat((quaternions[..., :1], -quaternions[..., 1:]), dim=-1)


def quaternion_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Scalar-first Hamilton product with NumPy broadcasting."""

    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.ndim < 1 or b.ndim < 1 or a.shape[-1] != 4 or b.shape[-1] != 4:
        raise ValueError("Quaternion operands must have final dimension 4.")
    aw, ax, ay, az = tuple(a[..., index] for index in range(4))
    bw, bx, by, bz = tuple(b[..., index] for index in range(4))
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def quaternion_multiply_torch(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Differentiable scalar-first Hamilton product with Torch broadcasting."""

    if first.ndim < 1 or second.ndim < 1 or int(first.shape[-1]) != 4 or int(second.shape[-1]) != 4:
        raise ValueError("Quaternion operands must have final dimension 4.")
    aw, ax, ay, az = first.unbind(dim=-1)
    bw, bx, by, bz = second.unbind(dim=-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def rotation_vectors_to_quaternions(
    rotation_vectors: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Map rotation vectors to scalar-first unit quaternions."""

    vectors = np.asarray(rotation_vectors, dtype=float)
    if vectors.ndim < 1 or vectors.shape[-1] != 3:
        raise ValueError("rotation_vectors must have final dimension 3.")
    angles = np.linalg.norm(vectors, axis=-1, keepdims=True)
    half_angles = 0.5 * angles
    scale = np.empty_like(angles)
    regular = angles > float(eps)
    scale[regular] = np.sin(half_angles[regular]) / angles[regular]
    squared = angles[~regular] ** 2
    scale[~regular] = 0.5 - squared / 48.0 + squared * squared / 3840.0
    quaternions = np.concatenate((np.cos(half_angles), scale * vectors), axis=-1)
    return normalize_quaternion_array(quaternions, eps=float(eps))


def rotation_vectors_to_quaternions_torch(
    rotation_vectors: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Differentiable Torch equivalent of :func:`rotation_vectors_to_quaternions`."""

    if rotation_vectors.ndim < 1 or int(rotation_vectors.shape[-1]) != 3:
        raise ValueError("rotation_vectors must have final dimension 3.")
    angles = torch.linalg.vector_norm(rotation_vectors, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    squared = angles * angles
    series = 0.5 - squared / 48.0 + squared * squared / 3840.0
    scale = torch.where(
        angles > float(eps),
        torch.sin(half_angles) / torch.clamp(angles, min=float(eps)),
        series,
    )
    quaternions = torch.cat((torch.cos(half_angles), scale * rotation_vectors), dim=-1)
    return normalize_quaternion_array_torch(quaternions, eps=float(eps))


def quaternions_to_rotation_vectors(
    quaternions: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return shortest rotation vectors for scalar-first quaternions."""

    q = normalize_quaternion_array(quaternions, eps=float(eps))
    vector = q[..., 1:]
    vector_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(q[..., :1], 0.0, 1.0))
    scale = np.empty_like(vector_norm)
    regular = vector_norm > float(eps)
    scale[regular] = angle[regular] / vector_norm[regular]
    scale[~regular] = 2.0
    return scale * vector


def quaternions_to_rotation_vectors_torch(
    quaternions: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Differentiable Torch equivalent of :func:`quaternions_to_rotation_vectors`."""

    q = normalize_quaternion_array_torch(quaternions, eps=float(eps))
    vector = q[..., 1:]
    vector_norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, torch.clamp(q[..., :1], min=0.0, max=1.0))
    scale = torch.where(
        vector_norm > float(eps),
        angle / torch.clamp(vector_norm, min=float(eps)),
        torch.full_like(vector_norm, 2.0),
    )
    return scale * vector


def integrate_body_rotation_increments(
    increments: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Right-integrate body-frame rotation-vector increments into orientations."""

    values = _as_three_by_time_numpy(increments, name="increments")
    delta_quaternions = rotation_vectors_to_quaternions(values.T, eps=float(eps))
    trajectory = np.empty((values.shape[1], 4), dtype=float)
    current = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    for sample, delta in enumerate(delta_quaternions):
        current = quaternion_multiply(current, delta)
        current = normalize_quaternion_array(current, eps=float(eps))
        trajectory[sample] = current
    return trajectory


def integrated_body_angles_to_quaternions(
    cumulative_angles: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Reconstruct orientation from componentwise integrated body increments."""

    angles = _as_three_by_time_numpy(cumulative_angles, name="cumulative_angles")
    increments = np.empty_like(angles)
    increments[:, 0] = angles[:, 0]
    increments[:, 1:] = np.diff(angles, axis=1)
    return integrate_body_rotation_increments(increments, eps=float(eps))


def quaternions_to_body_rotation_increments(
    quaternions: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Convert an orientation trajectory to shortest body-frame increments."""

    trajectory = normalize_quaternion_array(quaternions, eps=float(eps)).reshape(-1, 4)
    increments = np.empty((3, trajectory.shape[0]), dtype=float)
    identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    previous = identity
    for sample, current in enumerate(trajectory):
        delta = quaternion_multiply(quaternion_conjugate(previous), current)
        increments[:, sample] = quaternions_to_rotation_vectors(delta, eps=float(eps))
        previous = current
    return increments


def quaternions_to_integrated_body_angles(
    quaternions: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Convert orientations to componentwise cumulative body-rotation channels."""

    return np.cumsum(
        quaternions_to_body_rotation_increments(quaternions, eps=float(eps)),
        axis=1,
    )


def change_quaternion_basis(
    quaternions: np.ndarray,
    basis_axes_global: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Express active rotations in a right-handed basis whose axes are matrix rows."""

    q = normalize_quaternion_array(quaternions, eps=float(eps)).reshape(-1, 4)
    basis = _safe_axes_numpy(basis_axes_global, eps=float(eps))
    transformed = np.column_stack((q[:, 0], (basis @ q[:, 1:].T).T))
    return normalize_quaternion_array(transformed, eps=float(eps))


def quaternions_to_rotation_matrices(
    quaternions: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Convert scalar-first unit quaternions to active rotation matrices."""

    q = normalize_quaternion_array(quaternions, eps=float(eps))
    w, x, y, z = (q[..., index] for index in range(4))
    return np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def rotate_axis_triad(
    initial_axes_global: np.ndarray,
    object_quaternions_global: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Rotate a row-wise axis triad with every object orientation."""

    axes = _safe_axes_numpy(initial_axes_global, eps=float(eps))
    rotations = quaternions_to_rotation_matrices(object_quaternions_global, eps=float(eps))
    return np.einsum("ij,tkj->tik", axes, rotations)


def _safe_axes_numpy(matrix: np.ndarray, *, eps: float) -> np.ndarray:
    axes = np.asarray(matrix, dtype=float).reshape(3, 3)
    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    normalized = axes / np.maximum(norms, float(eps))
    collapsed = norms[:, 0] <= float(eps)
    if np.any(collapsed):
        normalized = normalized.copy()
        normalized[collapsed] = np.eye(3, dtype=float)[collapsed]
    return normalized


def compose_source_quaternions(
    source_angles: np.ndarray,
    recovery_matrix: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compose ``qS3 ⊗ qS2 ⊗ qS1`` in the observer coordinate frame."""

    angles = _as_three_by_time_numpy(source_angles, name="source_angles")
    axes = _safe_axes_numpy(recovery_matrix, eps=float(eps))
    half = 0.5 * angles
    components = np.concatenate(
        (
            np.cos(half).T[..., None],
            np.sin(half).T[..., None] * axes[None, :, :],
        ),
        axis=2,
    )
    composed = quaternion_multiply(
        components[:, 2, :],
        quaternion_multiply(components[:, 1, :], components[:, 0, :]),
    )
    return normalize_quaternion_array(composed, eps=float(eps))


def compose_source_quaternions_torch(
    source_angles: torch.Tensor,
    recovery_matrices: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Differentiable source composition for ``(3,T)`` or ``(P,3,T)`` angles."""

    squeeze = source_angles.ndim == 2
    angles = source_angles.unsqueeze(0) if squeeze else source_angles
    matrices = recovery_matrices.unsqueeze(0) if recovery_matrices.ndim == 2 else recovery_matrices
    if angles.ndim != 3 or int(angles.shape[1]) != 3:
        raise ValueError("source_angles must have shape (3, T) or (P, 3, T).")
    if matrices.ndim != 3 or tuple(matrices.shape[1:]) != (3, 3):
        raise ValueError("recovery_matrices must have shape (3, 3) or (P, 3, 3).")
    if int(angles.shape[0]) != int(matrices.shape[0]):
        raise ValueError("source_angles and recovery_matrices must share their batch size.")
    norms = torch.linalg.vector_norm(matrices, dim=2, keepdim=True)
    axes = matrices / torch.clamp(norms, min=float(eps))
    identity = torch.eye(3, dtype=matrices.dtype, device=matrices.device).unsqueeze(0)
    axes = torch.where(norms <= float(eps), identity, axes)
    half = 0.5 * angles
    components = torch.cat(
        (
            torch.cos(half).permute(0, 2, 1).unsqueeze(3),
            torch.sin(half).permute(0, 2, 1).unsqueeze(3) * axes.unsqueeze(1),
        ),
        dim=3,
    )
    composed = quaternion_multiply_torch(
        components[:, :, 2, :],
        quaternion_multiply_torch(components[:, :, 1, :], components[:, :, 0, :]),
    )
    normalized = normalize_quaternion_array_torch(composed, eps=float(eps))
    return normalized[0] if squeeze else normalized


def relative_reconstruction_quaternions(
    source_quaternions: np.ndarray,
    observer_quaternions: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Calculate ``qS^-1 ⊗ qR`` and canonicalize the identity representation."""

    source = normalize_quaternion_array(source_quaternions, eps=float(eps))
    observer = normalize_quaternion_array(observer_quaternions, eps=float(eps))
    if source.shape != observer.shape:
        raise ValueError("Source and observer quaternion arrays must have matching shapes.")
    error = quaternion_multiply(quaternion_conjugate(source), observer)
    return normalize_quaternion_array(error, eps=float(eps))


def relative_reconstruction_quaternions_torch(
    source_quaternions: torch.Tensor,
    observer_quaternions: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    source = normalize_quaternion_array_torch(source_quaternions, eps=float(eps))
    observer = normalize_quaternion_array_torch(observer_quaternions, eps=float(eps))
    error = quaternion_multiply_torch(quaternion_conjugate_torch(source), observer)
    return normalize_quaternion_array_torch(error, eps=float(eps))


def quaternion_reconstruction_loss(
    source_angles: np.ndarray,
    recovery_matrix: np.ndarray,
    observer_quaternions: np.ndarray,
    *,
    eps: float = 1e-12,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return the closest-sign full-quaternion identity loss and diagnostics."""

    composed = compose_source_quaternions(source_angles, recovery_matrix, eps=float(eps))
    error = relative_reconstruction_quaternions(composed, observer_quaternions, eps=float(eps))
    identity = np.zeros_like(error)
    identity[..., 0] = 1.0
    distance_to_identity = np.sum((error - identity) ** 2, axis=-1)
    distance_to_antipode = np.sum((error + identity) ** 2, axis=-1)
    loss = float(np.mean(np.minimum(distance_to_identity, distance_to_antipode)))
    return loss, composed, error


def quaternion_reconstruction_loss_torch(
    source_angles: torch.Tensor,
    recovery_matrices: torch.Tensor,
    observer_quaternions: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return closest-sign full-quaternion identity loss and quaternion diagnostics."""

    composed = compose_source_quaternions_torch(
        source_angles,
        recovery_matrices,
        eps=float(eps),
    )
    observer = observer_quaternions
    if composed.ndim == 3 and observer.ndim == 2:
        observer = observer.unsqueeze(0).expand(int(composed.shape[0]), -1, -1)
    if composed.shape != observer.shape:
        raise ValueError("Composed source and observer quaternions must have matching shapes.")
    error = relative_reconstruction_quaternions_torch(composed, observer, eps=float(eps))
    identity = torch.zeros_like(error)
    identity[..., 0] = 1.0
    distance_to_identity = torch.sum((error - identity) ** 2, dim=-1)
    distance_to_antipode = torch.sum((error + identity) ** 2, dim=-1)
    per_sample = torch.minimum(distance_to_identity, distance_to_antipode)
    loss = torch.mean(per_sample, dim=-1)
    return loss, composed, error


def quaternion_error_angles(relative_quaternions: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Return sign-invariant relative rotation angles in radians."""

    error = normalize_quaternion_array(relative_quaternions, eps=float(eps))
    vector_norm = np.linalg.norm(error[..., 1:], axis=-1)
    return 2.0 * np.arctan2(vector_norm, np.abs(error[..., 0]))
