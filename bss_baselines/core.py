"""Observer-signal PCA and ICA baselines for RSS validation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
import warnings

import numpy as np


METHOD_PCA = "pca"
METHOD_JADE_ICA = "jade_ica"
METHOD_FASTICA = "fastica"
METHOD_SOBI = "sobi"
# SOBI is appended last so the existing method ordering of the first three
# baselines, and every artifact and plot legend derived from it, is unchanged.
DEFAULT_METHODS = (METHOD_PCA, METHOD_JADE_ICA, METHOD_FASTICA, METHOD_SOBI)
DEFAULT_SOBI_LAGS: tuple[int, ...] = (1, 2, 3, 5, 7, 10)
METHOD_LABELS = {
    METHOD_PCA: "PCA",
    METHOD_JADE_ICA: "ICA (JADE)",
    METHOD_FASTICA: "FastICA",
    METHOD_SOBI: "SOBI",
}


@dataclass(frozen=True)
class BSSConfig:
    """Controls shared by the three direct observer-signal baselines."""

    methods: tuple[str, ...] = DEFAULT_METHODS
    jade_max_sweeps: int = 100
    jade_tol: float = 1e-8
    fastica_algorithm: str = "parallel"
    fastica_fun: str = "logcosh"
    fastica_max_iter: int = 2000
    fastica_tol: float = 1e-6
    sobi_lags: tuple[int, ...] = DEFAULT_SOBI_LAGS
    sobi_max_sweeps: int = 100
    sobi_tol: float = 1e-8
    random_state: int = 0
    eps: float = 1e-12

    def __post_init__(self) -> None:
        normalized = tuple(str(method).strip().lower() for method in self.methods)
        if not normalized:
            raise ValueError("methods must contain at least one baseline method.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("methods must not contain duplicates.")
        unknown = set(normalized) - set(DEFAULT_METHODS)
        if unknown:
            raise ValueError(f"Unknown BSS methods: {sorted(unknown)}.")
        if int(self.jade_max_sweeps) < 1:
            raise ValueError("jade_max_sweeps must be >= 1.")
        if not math.isfinite(float(self.jade_tol)) or float(self.jade_tol) <= 0.0:
            raise ValueError("jade_tol must be finite and > 0.")
        if str(self.fastica_algorithm).strip().lower() not in {"parallel", "deflation"}:
            raise ValueError("fastica_algorithm must be 'parallel' or 'deflation'.")
        if str(self.fastica_fun).strip().lower() not in {"logcosh", "exp", "cube"}:
            raise ValueError("fastica_fun must be 'logcosh', 'exp', or 'cube'.")
        if int(self.fastica_max_iter) < 1:
            raise ValueError("fastica_max_iter must be >= 1.")
        if not math.isfinite(float(self.fastica_tol)) or float(self.fastica_tol) <= 0.0:
            raise ValueError("fastica_tol must be finite and > 0.")
        sobi_lags = tuple(int(lag) for lag in self.sobi_lags)
        if not sobi_lags:
            raise ValueError("sobi_lags must contain at least one lag.")
        if any(lag < 1 for lag in sobi_lags):
            raise ValueError("sobi_lags must contain values >= 1.")
        if len(set(sobi_lags)) != len(sobi_lags):
            raise ValueError("sobi_lags must not contain duplicates.")
        if int(self.sobi_max_sweeps) < 1:
            raise ValueError("sobi_max_sweeps must be >= 1.")
        if not math.isfinite(float(self.sobi_tol)) or float(self.sobi_tol) <= 0.0:
            raise ValueError("sobi_tol must be finite and > 0.")
        if not math.isfinite(float(self.eps)) or float(self.eps) <= 0.0:
            raise ValueError("eps must be finite and > 0.")
        object.__setattr__(self, "methods", normalized)
        object.__setattr__(self, "sobi_lags", sobi_lags)
        object.__setattr__(self, "fastica_algorithm", str(self.fastica_algorithm).strip().lower())
        object.__setattr__(self, "fastica_fun", str(self.fastica_fun).strip().lower())


@dataclass(frozen=True)
class BSSFitResult:
    """Fitted direct observer-signal decomposition and diagnostics."""

    config: BSSConfig
    method_key: str
    method_label: str
    implementation: str
    observer_mean: np.ndarray
    centered_observations: np.ndarray
    raw_source_signals: np.ndarray
    source_angles: np.ndarray
    demixing_matrix: np.ndarray
    normalized_demixing_matrix: np.ndarray
    mixing_matrix: np.ndarray
    demixing_row_norms: np.ndarray
    reconstructed_observations: np.ndarray
    converged: bool
    iterations: int
    runtime_seconds: float
    warnings: tuple[str, ...] = ()
    diagnostics: dict[str, object] = field(default_factory=dict)


def _validate_observations(observer_signals: np.ndarray, eps: float) -> np.ndarray:
    values = np.asarray(observer_signals, dtype=float)
    if values.ndim != 2:
        raise ValueError("observer_signals must be a 2D array.")
    if values.shape[0] != 3 and values.shape[1] == 3:
        values = values.T
    if values.shape[0] != 3:
        raise ValueError("observer_signals must have shape (3, T) or (T, 3).")
    if values.shape[1] < 8:
        raise ValueError("observer_signals must contain at least 8 samples.")
    if not np.isfinite(values).all():
        raise ValueError("observer_signals must contain only finite values.")
    centered = values - np.mean(values, axis=1, keepdims=True)
    covariance = (centered @ centered.T) / float(centered.shape[1])
    eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    largest = max(float(np.max(eigenvalues)), float(eps))
    rank_tolerance = max(float(eps), largest * np.finfo(float).eps * 3.0)
    if np.any(eigenvalues <= rank_tolerance):
        raise ValueError("observer_signals must have full channel rank for three-source separation.")
    return values


def _canonicalize_rows(
    demixing_matrix: np.ndarray,
    sources: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    demixing = np.asarray(demixing_matrix, dtype=float).copy()
    separated = np.asarray(sources, dtype=float).copy()
    for index in range(demixing.shape[0]):
        pivot = int(np.argmax(np.abs(demixing[index])))
        if demixing[index, pivot] < 0.0:
            demixing[index] *= -1.0
            separated[index] *= -1.0
    return demixing, separated


def _total_offdiag_energy(matrices: tuple[np.ndarray, ...]) -> float:
    return float(
        sum(
            np.sum(matrix * matrix) - np.sum(np.diag(matrix) ** 2)
            for matrix in matrices
        )
    )


def _rotation_matrix(size: int, first: int, second: int, angle: float) -> np.ndarray:
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    rotation = np.eye(size, dtype=float)
    rotation[first, first] = cosine
    rotation[second, second] = cosine
    rotation[first, second] = sine
    rotation[second, first] = -sine
    return rotation


def _optimal_pair_rotation(
    matrices: tuple[np.ndarray, ...],
    first: int,
    second: int,
) -> float:
    accumulator = np.zeros((2, 2), dtype=float)
    for matrix in matrices:
        diagonal_difference = 0.5 * float(matrix[first, first] - matrix[second, second])
        off_diagonal = float(matrix[first, second])
        vector = np.array((off_diagonal, diagonal_difference), dtype=float)
        accumulator += np.outer(vector, vector)
    eigenvalues, eigenvectors = np.linalg.eigh(accumulator)
    direction = eigenvectors[:, int(np.argmin(eigenvalues))]
    angle = 0.5 * float(np.arctan2(direction[1], direction[0]))
    # The eigenvector sign makes ``angle`` and ``angle + pi/2`` equivalent
    # for the pair objective. Keep the Jacobi rotation closest to identity so
    # already-diagonal pairs are not repeatedly swapped during later sweeps.
    return float((angle + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0)


def _joint_diagonalize(
    matrices: tuple[np.ndarray, ...],
    *,
    max_sweeps: int,
    tol: float,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[float, ...], bool, int]:
    size = int(matrices[0].shape[0])
    diagonalizer = np.eye(size, dtype=float)
    working = tuple(0.5 * (matrix + matrix.T) for matrix in matrices)
    history = [_total_offdiag_energy(working)]
    converged = False
    completed_sweeps = 0
    for sweep in range(int(max_sweeps)):
        maximum_rotation = 0.0
        for first in range(size - 1):
            for second in range(first + 1, size):
                angle = _optimal_pair_rotation(working, first, second)
                maximum_rotation = max(maximum_rotation, abs(angle))
                if abs(angle) <= float(tol):
                    continue
                rotation = _rotation_matrix(size, first, second, angle)
                working = tuple(rotation.T @ matrix @ rotation for matrix in working)
                diagonalizer = rotation.T @ diagonalizer
        completed_sweeps = sweep + 1
        history.append(_total_offdiag_energy(working))
        improvement = history[-2] - history[-1]
        if maximum_rotation <= float(tol) or improvement <= float(tol) * max(1.0, history[-2]):
            converged = True
            break
    return diagonalizer, working, tuple(history), converged, completed_sweeps


def _whiten(centered: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the whitening matrix, whitened signals, and descending eigenvalues."""

    sample_count = int(centered.shape[1])
    covariance = (centered @ centered.T) / float(sample_count)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    whitening = np.diag(np.reciprocal(np.sqrt(eigenvalues))) @ eigenvectors.T
    return whitening, whitening @ centered, eigenvalues


def _jade_fit(
    centered: np.ndarray,
    config: BSSConfig,
) -> tuple[np.ndarray, np.ndarray, bool, int, tuple[str, ...], dict[str, object]]:
    sample_count = int(centered.shape[1])
    whitening, whitened, eigenvalues = _whiten(centered)
    size = int(whitened.shape[0])

    cumulant_operator = np.zeros((size * size, size * size), dtype=float)
    identity = np.eye(size, dtype=float)
    for first in range(size):
        for second in range(size):
            weights = whitened[first] * whitened[second]
            cumulant = (whitened * weights[None, :]) @ whitened.T / float(sample_count)
            if first == second:
                cumulant -= identity
            basis_first = np.zeros(size, dtype=float)
            basis_second = np.zeros(size, dtype=float)
            basis_first[first] = 1.0
            basis_second[second] = 1.0
            cumulant -= np.outer(basis_first, basis_second)
            cumulant -= np.outer(basis_second, basis_first)
            cumulant_operator[:, first * size + second] = cumulant.reshape(-1)
    cumulant_operator = 0.5 * (cumulant_operator + cumulant_operator.T)
    cumulant_eigenvalues, cumulant_eigenvectors = np.linalg.eigh(cumulant_operator)
    significant = np.argsort(np.abs(cumulant_eigenvalues))[::-1][:size]
    cumulant_matrices = tuple(
        0.5
        * (
            cumulant_eigenvectors[:, index].reshape(size, size)
            + cumulant_eigenvectors[:, index].reshape(size, size).T
        )
        * float(cumulant_eigenvalues[index])
        for index in significant
    )
    diagonalizer, _, history, converged, iterations = _joint_diagonalize(
        cumulant_matrices,
        max_sweeps=int(config.jade_max_sweeps),
        tol=float(config.jade_tol),
    )
    demixing = diagonalizer @ whitening
    sources = demixing @ centered
    warning_messages: list[str] = []
    selected_values = cumulant_eigenvalues[significant]
    if float(np.max(np.abs(selected_values))) <= 100.0 * float(config.eps):
        warning_messages.append(
            "The selected fourth-order cumulant eigenvalues are near zero; JADE may be unidentifiable."
        )
    if not converged:
        warning_messages.append("JADE reached its maximum number of Jacobi sweeps before convergence.")
    diagnostics: dict[str, object] = {
        "covariance_eigenvalues": eigenvalues.tolist(),
        "selected_cumulant_eigenvalues": selected_values.tolist(),
        "offdiag_energy_history": list(history),
        "whitening_matrix": whitening.tolist(),
    }
    return demixing, sources, converged, iterations, tuple(warning_messages), diagnostics


def _sobi_fit(
    centered: np.ndarray,
    config: BSSConfig,
) -> tuple[np.ndarray, np.ndarray, bool, int, tuple[str, ...], dict[str, object]]:
    sample_count = int(centered.shape[1])
    whitening, whitened, eigenvalues = _whiten(centered)
    warning_messages: list[str] = []

    # ``_validate_observations`` admits records as short as eight samples while the
    # default lag set reaches ten, so unusable lags are dropped and recorded rather
    # than rejected outright.
    usable_lags = tuple(lag for lag in config.sobi_lags if lag < sample_count)
    if not usable_lags:
        raise ValueError("sobi_lags must contain at least one lag smaller than the sample count.")
    if usable_lags != tuple(config.sobi_lags):
        dropped = [lag for lag in config.sobi_lags if lag >= sample_count]
        warning_messages.append(f"Dropped SOBI lags at or beyond the sample count: {dropped}.")

    lag_covariances = []
    for lag in usable_lags:
        covariance = (
            whitened[:, lag:] @ whitened[:, : sample_count - lag].T
        ) / float(sample_count - lag)
        lag_covariances.append(0.5 * (covariance + covariance.T))

    diagonalizer, diagonalized, history, converged, iterations = _joint_diagonalize(
        tuple(lag_covariances),
        max_sweeps=int(config.sobi_max_sweeps),
        tol=float(config.sobi_tol),
    )
    demixing = diagonalizer @ whitening
    sources = demixing @ centered

    # Each row is one source's autocorrelation signature across the selected lags.
    # SOBI separates only where those signatures differ, so near-identical profiles
    # are the second-order analogue of JADE's near-zero cumulant eigenvalues.
    profiles = np.array(
        [[float(matrix[index, index]) for matrix in diagonalized] for index in range(demixing.shape[0])],
        dtype=float,
    )
    profile_distances = [
        float(np.linalg.norm(profiles[first] - profiles[second]))
        for first in range(profiles.shape[0] - 1)
        for second in range(first + 1, profiles.shape[0])
    ]
    minimum_profile_distance = float(min(profile_distances)) if profile_distances else float("nan")
    if profile_distances and minimum_profile_distance <= 100.0 * float(config.eps):
        warning_messages.append(
            "The SOBI lag profiles are nearly identical; the second-order separation may be unidentifiable."
        )
    if not converged:
        warning_messages.append("SOBI reached its maximum number of Jacobi sweeps before convergence.")

    diagnostics: dict[str, object] = {
        "covariance_eigenvalues": eigenvalues.tolist(),
        "sobi_lags_requested": list(config.sobi_lags),
        "sobi_lags_used": list(usable_lags),
        "offdiag_energy_history": list(history),
        "lag_offdiag_energy_before": float(history[0]),
        "lag_offdiag_energy_after": float(history[-1]),
        "lag_diagonal_profiles": profiles.tolist(),
        "min_pairwise_lag_profile_distance": minimum_profile_distance,
        "whitening_matrix": whitening.tolist(),
        "joint_diagonalizer": diagonalizer.tolist(),
    }
    return demixing, sources, converged, iterations, tuple(warning_messages), diagnostics


def _pca_fit(
    centered: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool, int, tuple[str, ...], dict[str, object]]:
    _, singular_values, right_vectors = np.linalg.svd(centered.T, full_matrices=False)
    demixing = right_vectors
    sources = demixing @ centered
    explained_variance = singular_values**2 / float(max(centered.shape[1] - 1, 1))
    explained_ratio = explained_variance / np.sum(explained_variance)
    diagnostics: dict[str, object] = {
        "singular_values": singular_values.tolist(),
        "explained_variance": explained_variance.tolist(),
        "explained_variance_ratio": explained_ratio.tolist(),
    }
    return demixing, sources, True, 1, (), diagnostics


def _fastica_fit(
    observations: np.ndarray,
    config: BSSConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, int, tuple[str, ...], dict[str, object]]:
    try:
        from sklearn.decomposition import FastICA
        from sklearn.exceptions import ConvergenceWarning
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency is declared by the project
        raise RuntimeError("scikit-learn is required for the FastICA baseline.") from exc

    model = FastICA(
        n_components=3,
        algorithm=config.fastica_algorithm,
        whiten="unit-variance",
        fun=config.fastica_fun,
        max_iter=int(config.fastica_max_iter),
        tol=float(config.fastica_tol),
        random_state=int(config.random_state),
        whiten_solver="svd",
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        sources = model.fit_transform(observations.T).T
    warning_messages = tuple(str(item.message) for item in captured)
    convergence_warning = any(issubclass(item.category, ConvergenceWarning) for item in captured)
    iterations = int(getattr(model, "n_iter_", 0))
    converged = not convergence_warning and iterations <= int(config.fastica_max_iter)
    reconstructed = model.inverse_transform(sources.T).T
    diagnostics: dict[str, object] = {
        "algorithm": config.fastica_algorithm,
        "nonlinearity": config.fastica_fun,
        "whitening": "unit-variance",
        "whitening_matrix": np.asarray(getattr(model, "whitening_", np.empty((0, 0)))).tolist(),
    }
    return (
        np.asarray(model.components_, dtype=float),
        sources,
        reconstructed,
        converged,
        iterations,
        warning_messages,
        diagnostics,
    )


def fit_bss(
    observer_signals: np.ndarray,
    method: str,
    config: BSSConfig | None = None,
) -> BSSFitResult:
    """Fit one baseline directly to cumulative observer-side signals."""

    active = config or BSSConfig()
    method_key = str(method).strip().lower()
    if method_key not in DEFAULT_METHODS:
        raise ValueError(f"Unknown BSS method {method!r}.")
    observations = _validate_observations(observer_signals, float(active.eps))
    observer_mean = np.mean(observations, axis=1, keepdims=True)
    centered = observations - observer_mean

    start = time.perf_counter()
    external_reconstruction: np.ndarray | None = None
    if method_key == METHOD_PCA:
        demixing, raw_sources, converged, iterations, warning_messages, diagnostics = _pca_fit(centered)
        implementation = "numpy_full_svd_pca"
    elif method_key == METHOD_JADE_ICA:
        demixing, raw_sources, converged, iterations, warning_messages, diagnostics = _jade_fit(
            centered, active
        )
        implementation = "numpy_fourth_order_jade_ica"
    elif method_key == METHOD_SOBI:
        demixing, raw_sources, converged, iterations, warning_messages, diagnostics = _sobi_fit(
            centered, active
        )
        implementation = "numpy_second_order_sobi"
    else:
        (
            demixing,
            raw_sources,
            external_reconstruction,
            converged,
            iterations,
            warning_messages,
            diagnostics,
        ) = _fastica_fit(observations, active)
        implementation = "sklearn_parallel_fastica"
    runtime = time.perf_counter() - start

    demixing, raw_sources = _canonicalize_rows(demixing, raw_sources)
    row_norms = np.linalg.norm(demixing, axis=1)
    if not np.isfinite(demixing).all() or not np.isfinite(raw_sources).all():
        raise RuntimeError(f"{METHOD_LABELS[method_key]} produced non-finite outputs.")
    if np.any(row_norms <= float(active.eps)):
        raise RuntimeError(f"{METHOD_LABELS[method_key]} produced a zero demixing row.")
    normalized = demixing / row_norms[:, None]
    source_angles = raw_sources / row_norms[:, None]
    mixing = np.linalg.pinv(demixing, rcond=float(active.eps))
    reconstructed = mixing @ raw_sources + observer_mean
    if external_reconstruction is not None:
        diagnostics["sklearn_inverse_reconstruction_rmse"] = float(
            np.sqrt(np.mean((external_reconstruction - observations) ** 2))
        )
    diagnostics.update(
        {
            "demixing_condition_number": float(np.linalg.cond(demixing)),
            "normalized_demixing_orthogonality_error": float(
                np.linalg.norm(normalized @ normalized.T - np.eye(3), ord="fro")
            ),
            "normalized_demixing_determinant": float(np.linalg.det(normalized)),
            "demixing_row_norms": row_norms.tolist(),
        }
    )
    return BSSFitResult(
        config=active,
        method_key=method_key,
        method_label=METHOD_LABELS[method_key],
        implementation=implementation,
        observer_mean=observer_mean[:, 0],
        centered_observations=centered,
        raw_source_signals=raw_sources,
        source_angles=source_angles,
        demixing_matrix=demixing,
        normalized_demixing_matrix=normalized,
        mixing_matrix=mixing,
        demixing_row_norms=row_norms,
        reconstructed_observations=reconstructed,
        converged=bool(converged),
        iterations=int(iterations),
        runtime_seconds=float(runtime),
        warnings=tuple(warning_messages),
        diagnostics=diagnostics,
    )
