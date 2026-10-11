"""Core Rotation Source Separation optimizer."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import math

import numpy as np
import torch

from rss.mutual_information import (
    DEFAULT_BSPLINE_BINS,
    DEFAULT_BSPLINE_ORDER,
    DEFAULT_GCMI_RANK_METHOD,
    DEFAULT_KSG_K,
    DEFAULT_KSG_MAX_SAMPLES,
    DEFAULT_KSG_STANDARDIZE,
    DEFAULT_KSG_SUBSAMPLE_MODE,
    DEFAULT_KSG_SUBSAMPLE_SEED,
    GCMI_RANK_METHODS,
    KSG_SUBSAMPLE_MODES,
    MI_ESTIMATORS,
    estimate_pairwise_gcmi_mutual_information,
    estimate_pairwise_ksg_mutual_information,
    select_ksg_subsample_indices,
)
from rss.model import (
    angles_to_quaternion_components,
    compose_source_quaternions,
    decode_scalar_and_vector,
    decode_scalar_and_vector_torch,
    integrated_body_angles_to_quaternions,
    quaternion_error_angles,
    quaternion_reconstruction_loss_torch,
    quaternions_to_angles,
    quaternions_to_integrated_body_angles,
    relative_reconstruction_quaternions,
)
from rss.specialized_losses import (
    SPECIALIZED_LOSSES,
    SpecializedLossConfig,
    compute_specialized_loss,
)


IDENTITY_WEIGHTS: tuple[float, ...] = (1.0, 0.0, 0.0, 1.0, 0.0, 1.0)


@dataclass(frozen=True)
class RSSConfig:
    """Numerical controls for the RSS algorithm."""

    loss_type: str = "lagged_cross_correlation"
    separation_weight: float = 1.0
    lags: tuple[int, ...] = (0,)
    mi_estimator: str = "ksg"
    gcmi_rank_method: str = DEFAULT_GCMI_RANK_METHOD
    bspline_bins: int = DEFAULT_BSPLINE_BINS
    bspline_order: int = DEFAULT_BSPLINE_ORDER
    ksg_k: int = DEFAULT_KSG_K
    ksg_standardize: bool = DEFAULT_KSG_STANDARDIZE
    ksg_max_samples: int | None = DEFAULT_KSG_MAX_SAMPLES
    ksg_subsample_mode: str = DEFAULT_KSG_SUBSAMPLE_MODE
    ksg_subsample_seed: int = DEFAULT_KSG_SUBSAMPLE_SEED
    mi_bins: int = 16
    mi_bandwidth: float = 0.5
    action_inertias: tuple[float, float, float] = (1.0, 1.0, 1.0)
    action_stiffnesses: tuple[float, float, float] = (1.0, 1.0, 1.0)
    sample_interval_seconds: float | None = None
    epochs: int = 200
    optimizer: str = "adam"
    learning_rate: float = 1e-1
    learning_rate_end: float = 1e-5
    sgd_momentum: float = 0.9
    pso_swarm_size: int = 40
    pso_inertia: float = 0.7298
    pso_cognitive: float = 1.49618
    pso_social: float = 1.49618
    pso_lower_bound: float | None = None
    pso_upper_bound: float | None = None
    pso_seed: int = 0
    pso_stall_patience: int = 20
    orthogonality_weight: float = 0.0
    reconstruction_weight: float = 0.0
    # Retained for configuration compatibility. RSS always uses the full epoch budget.
    convergence_tol: float = 0.0
    eps: float = 1e-12
    quaternion_eps: float = 1e-12
    device: str = "cpu"
    matrix_parameterization: str = "so3"
    observer_signal_model: str = "euler"
    initial_rotation_vector: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_weights: tuple[float, ...] | None = None
    initial_weight_matrix: tuple[tuple[float, float, float], ...] | None = None
    axis_flip_search: bool = False
    axis_flip_interval: int = 100
    axis_flip_trial_epochs: int = 10
    adaptive_regularizer: bool = False
    specialized_regularizer: str | None = None
    regularizer_start_weight: float = 0.1
    regularizer_plateau_patience: int = 20
    regularizer_plateau_tolerance: float = 1e-4
    regularizer_ramp_epochs: int = 20
    regularizer_min_loss_threshold: float = 0.0
    specialized_loss_config: SpecializedLossConfig = SpecializedLossConfig()

    def __post_init__(self) -> None:
        loss_type = str(self.loss_type).strip().lower()
        if loss_type not in {"lagged_cross_correlation", "mutual_information", "action"}:
            raise ValueError(
                "loss_type must be one of: lagged_cross_correlation, mutual_information, action."
            )
        object.__setattr__(self, "loss_type", loss_type)
        if not isinstance(self.axis_flip_search, (bool, np.bool_)):
            raise ValueError("axis_flip_search must be a boolean.")
        if not isinstance(self.adaptive_regularizer, (bool, np.bool_)):
            raise ValueError("adaptive_regularizer must be a boolean.")
        object.__setattr__(self, "axis_flip_search", bool(self.axis_flip_search))
        object.__setattr__(self, "adaptive_regularizer", bool(self.adaptive_regularizer))
        flip_interval = int(self.axis_flip_interval)
        flip_trial_epochs = int(self.axis_flip_trial_epochs)
        if flip_interval != self.axis_flip_interval or flip_interval < 1:
            raise ValueError("axis_flip_interval must be an integer >= 1.")
        if flip_trial_epochs != self.axis_flip_trial_epochs or flip_trial_epochs < 1:
            raise ValueError("axis_flip_trial_epochs must be an integer >= 1.")
        object.__setattr__(self, "axis_flip_interval", flip_interval)
        object.__setattr__(self, "axis_flip_trial_epochs", flip_trial_epochs)
        selected_regularizer = (
            None
            if self.specialized_regularizer is None
            else str(self.specialized_regularizer).strip().lower()
        )
        if selected_regularizer is not None and selected_regularizer not in SPECIALIZED_LOSSES:
            raise ValueError(
                f"specialized_regularizer must be one of: {', '.join(SPECIALIZED_LOSSES)}."
            )
        object.__setattr__(self, "specialized_regularizer", selected_regularizer)
        start_weight = float(self.regularizer_start_weight)
        if not math.isfinite(start_weight) or not 0.0 <= start_weight <= 1.0:
            raise ValueError("regularizer_start_weight must be finite and lie in [0, 1].")
        object.__setattr__(self, "regularizer_start_weight", start_weight)
        plateau_patience = int(self.regularizer_plateau_patience)
        ramp_epochs = int(self.regularizer_ramp_epochs)
        if plateau_patience != self.regularizer_plateau_patience or plateau_patience < 1:
            raise ValueError("regularizer_plateau_patience must be an integer >= 1.")
        if ramp_epochs != self.regularizer_ramp_epochs or ramp_epochs < 1:
            raise ValueError("regularizer_ramp_epochs must be an integer >= 1.")
        plateau_tolerance = float(self.regularizer_plateau_tolerance)
        minimum_threshold = float(self.regularizer_min_loss_threshold)
        if not math.isfinite(plateau_tolerance) or plateau_tolerance < 0.0:
            raise ValueError("regularizer_plateau_tolerance must be finite and >= 0.")
        if not math.isfinite(minimum_threshold) or minimum_threshold < 0.0:
            raise ValueError("regularizer_min_loss_threshold must be finite and >= 0.")
        object.__setattr__(self, "regularizer_plateau_patience", plateau_patience)
        object.__setattr__(self, "regularizer_ramp_epochs", ramp_epochs)
        object.__setattr__(self, "regularizer_plateau_tolerance", plateau_tolerance)
        object.__setattr__(self, "regularizer_min_loss_threshold", minimum_threshold)
        if not isinstance(self.specialized_loss_config, SpecializedLossConfig):
            raise ValueError("specialized_loss_config must be a SpecializedLossConfig instance.")
        if self.adaptive_regularizer:
            if loss_type != "lagged_cross_correlation":
                raise ValueError(
                    "adaptive_regularizer requires loss_type='lagged_cross_correlation'."
                )
            if selected_regularizer is None:
                raise ValueError(
                    "adaptive_regularizer requires specialized_regularizer to be selected."
                )
        mi_estimator = str(self.mi_estimator).strip().lower()
        if mi_estimator not in MI_ESTIMATORS:
            raise ValueError(
                f"mi_estimator must be one of: {', '.join(MI_ESTIMATORS)}."
            )
        object.__setattr__(self, "mi_estimator", mi_estimator)
        gcmi_rank_method = str(self.gcmi_rank_method).strip().lower()
        if gcmi_rank_method not in GCMI_RANK_METHODS:
            raise ValueError(
                f"gcmi_rank_method must be one of: {', '.join(GCMI_RANK_METHODS)}."
            )
        object.__setattr__(self, "gcmi_rank_method", gcmi_rank_method)
        bspline_bin_count = int(self.bspline_bins)
        if bspline_bin_count != self.bspline_bins or bspline_bin_count < 2:
            raise ValueError("bspline_bins must be an integer >= 2.")
        bspline_order = int(self.bspline_order)
        if bspline_order != self.bspline_order or bspline_order < 1:
            raise ValueError("bspline_order must be an integer >= 1.")
        if bspline_order >= bspline_bin_count:
            raise ValueError("bspline_order must be smaller than bspline_bins.")
        object.__setattr__(self, "bspline_bins", bspline_bin_count)
        object.__setattr__(self, "bspline_order", bspline_order)
        neighbour_count = int(self.ksg_k)
        if neighbour_count != self.ksg_k or neighbour_count < 1:
            raise ValueError("ksg_k must be an integer >= 1.")
        object.__setattr__(self, "ksg_k", neighbour_count)
        if not isinstance(self.ksg_standardize, (bool, np.bool_)):
            raise ValueError("ksg_standardize must be a boolean.")
        object.__setattr__(self, "ksg_standardize", bool(self.ksg_standardize))
        if self.ksg_max_samples is None:
            ksg_max_samples = None
        else:
            ksg_max_samples = int(self.ksg_max_samples)
            if ksg_max_samples != self.ksg_max_samples or ksg_max_samples < 1:
                raise ValueError("ksg_max_samples must be an integer >= 1 or None.")
            if ksg_max_samples <= neighbour_count:
                raise ValueError("ksg_max_samples must be larger than ksg_k.")
        object.__setattr__(self, "ksg_max_samples", ksg_max_samples)
        ksg_subsample_mode = str(self.ksg_subsample_mode).strip().lower()
        if ksg_subsample_mode not in KSG_SUBSAMPLE_MODES:
            raise ValueError(
                f"ksg_subsample_mode must be one of: {', '.join(KSG_SUBSAMPLE_MODES)}."
            )
        object.__setattr__(self, "ksg_subsample_mode", ksg_subsample_mode)
        ksg_subsample_seed = int(self.ksg_subsample_seed)
        if ksg_subsample_seed != self.ksg_subsample_seed or ksg_subsample_seed < 0:
            raise ValueError("ksg_subsample_seed must be an integer >= 0.")
        object.__setattr__(self, "ksg_subsample_seed", ksg_subsample_seed)
        bin_count = int(self.mi_bins)
        if bin_count != self.mi_bins or bin_count < 2:
            raise ValueError("mi_bins must be >= 2.")
        object.__setattr__(self, "mi_bins", bin_count)
        if not math.isfinite(float(self.mi_bandwidth)) or float(self.mi_bandwidth) <= 0.0:
            raise ValueError("mi_bandwidth must be finite and > 0.")
        inertias = tuple(float(value) for value in self.action_inertias)
        stiffnesses = tuple(float(value) for value in self.action_stiffnesses)
        if len(inertias) != 3 or not np.isfinite(np.asarray(inertias)).all() or any(
            value <= 0.0 for value in inertias
        ):
            raise ValueError("action_inertias must contain three finite values > 0.")
        if len(stiffnesses) != 3 or not np.isfinite(np.asarray(stiffnesses)).all() or any(
            value <= 0.0 for value in stiffnesses
        ):
            raise ValueError("action_stiffnesses must contain three finite values > 0.")
        object.__setattr__(self, "action_inertias", inertias)
        object.__setattr__(self, "action_stiffnesses", stiffnesses)
        if self.sample_interval_seconds is not None:
            interval = float(self.sample_interval_seconds)
            if not math.isfinite(interval) or interval <= 0.0:
                raise ValueError("sample_interval_seconds must be finite and > 0 when provided.")
            object.__setattr__(self, "sample_interval_seconds", interval)
        if int(self.epochs) < 1:
            raise ValueError("epochs must be >= 1.")
        optimizer_name = str(self.optimizer).strip().lower()
        if optimizer_name not in {"adam", "sgd", "pso"}:
            raise ValueError("optimizer must be one of: adam, sgd, pso.")
        if (
            loss_type == "mutual_information"
            and mi_estimator in {"gcmi", "ksg"}
            and optimizer_name != "pso"
        ):
            raise ValueError(
                f"{mi_estimator.upper()} mutual information is non-differentiable and requires "
                "optimizer='pso'. Use mi_estimator='bspline' or 'histogram' with Adam or SGD."
            )
        learning_rates = (float(self.learning_rate), float(self.learning_rate_end))
        if not np.isfinite(np.asarray(learning_rates, dtype=float)).all() or any(
            value <= 0.0 for value in learning_rates
        ):
            raise ValueError("learning_rate and learning_rate_end must be finite and > 0.")
        if float(self.learning_rate_end) > float(self.learning_rate):
            raise ValueError("learning_rate_end must be <= learning_rate for decay.")
        if not math.isfinite(float(self.sgd_momentum)) or not 0.0 <= float(self.sgd_momentum) < 1.0:
            raise ValueError("sgd_momentum must be finite and lie in [0, 1).")
        if int(self.pso_swarm_size) < 1:
            raise ValueError("pso_swarm_size must be >= 1.")
        pso_coefficients = (float(self.pso_inertia), float(self.pso_cognitive), float(self.pso_social))
        if not np.isfinite(np.asarray(pso_coefficients, dtype=float)).all() or any(
            coefficient < 0.0 for coefficient in pso_coefficients
        ):
            raise ValueError("PSO inertia, cognitive, and social coefficients must be finite and >= 0.")
        parameterization = str(self.matrix_parameterization).strip().lower()
        if parameterization not in {"so3", "symmetric"}:
            raise ValueError("matrix_parameterization must be 'so3' or 'symmetric'.")
        signal_model = str(self.observer_signal_model).strip().lower()
        if signal_model not in {"integrated_body_rotation", "euler"}:
            raise ValueError("observer_signal_model must be 'integrated_body_rotation' or 'euler'.")
        object.__setattr__(self, "matrix_parameterization", parameterization)
        object.__setattr__(self, "observer_signal_model", signal_model)
        if self.axis_flip_search:
            if parameterization != "so3":
                raise ValueError("axis_flip_search requires matrix_parameterization='so3'.")
            if flip_trial_epochs >= flip_interval:
                raise ValueError("axis_flip_trial_epochs must be smaller than axis_flip_interval.")
        default_lower, default_upper = (
            (-math.pi, math.pi) if parameterization == "so3" else (-1.0, 1.0)
        )
        lower = default_lower if self.pso_lower_bound is None else float(self.pso_lower_bound)
        upper = default_upper if self.pso_upper_bound is None else float(self.pso_upper_bound)
        pso_bounds = (lower, upper)
        if not np.isfinite(np.asarray(pso_bounds, dtype=float)).all() or pso_bounds[0] >= pso_bounds[1]:
            raise ValueError("pso_lower_bound and pso_upper_bound must be finite with lower < upper.")
        if int(self.pso_stall_patience) < 1:
            raise ValueError("pso_stall_patience must be >= 1.")
        if self.axis_flip_search and (lower > -math.pi or upper < math.pi):
            raise ValueError(
                "axis_flip_search requires PSO bounds that contain the full [-pi, pi] "
                "SO(3) rotation-vector range."
            )
        loss_weights = (
            float(self.separation_weight),
            float(self.orthogonality_weight),
            float(self.reconstruction_weight),
        )
        if not np.isfinite(np.asarray(loss_weights, dtype=float)).all() or any(weight < 0.0 for weight in loss_weights):
            raise ValueError("All RSS loss weights must be finite and >= 0.")
        effective_weights = (
            float(self.separation_weight),
            float(self.orthogonality_weight) if parameterization == "symmetric" else 0.0,
            float(self.reconstruction_weight),
        )
        if all(weight == 0.0 for weight in effective_weights) and not self.adaptive_regularizer:
            raise ValueError("At least one RSS loss weight must be positive.")
        if float(self.convergence_tol) < 0.0:
            raise ValueError("convergence_tol must be >= 0.")
        numerical_epsilons = (float(self.eps), float(self.quaternion_eps))
        if not np.isfinite(np.asarray(numerical_epsilons)).all() or any(
            value <= 0.0 for value in numerical_epsilons
        ):
            raise ValueError("eps and quaternion_eps must be finite and > 0.")
        normalized_lags = tuple(int(lag) for lag in self.lags)
        if not normalized_lags or any(lag < 0 for lag in normalized_lags):
            raise ValueError("lags must contain nonnegative integers.")
        if len(set(normalized_lags)) != len(normalized_lags):
            raise ValueError("lags must not contain duplicates.")
        object.__setattr__(self, "lags", normalized_lags)
        rotation_initial = np.asarray(self.initial_rotation_vector, dtype=float)
        if rotation_initial.shape != (3,) or not np.isfinite(rotation_initial).all():
            raise ValueError("initial_rotation_vector must contain three finite values.")
        if self.initial_weights is not None:
            legacy_initial = np.asarray(self.initial_weights, dtype=float)
            if legacy_initial.shape != (6,) or not np.isfinite(legacy_initial).all():
                raise ValueError("initial_weights must contain six finite values.")
        if self.initial_weight_matrix is not None:
            initial_matrix = np.asarray(self.initial_weight_matrix, dtype=float)
            if initial_matrix.shape != (3, 3) or not np.isfinite(initial_matrix).all():
                raise ValueError("initial_weight_matrix must be a finite 3-by-3 matrix.")
            object.__setattr__(
                self,
                "initial_weight_matrix",
                tuple(tuple(float(value) for value in row) for row in initial_matrix),
            )
            if parameterization == "so3" and not np.allclose(
                rotation_initial, np.zeros(3), atol=1e-12, rtol=0.0
            ):
                raise ValueError(
                    "Set either initial_weight_matrix or a nonzero "
                    "initial_rotation_vector in SO(3) mode, not both."
                )
            if parameterization == "symmetric" and self.initial_weights is not None:
                raise ValueError(
                    "Set either initial_weight_matrix or initial_weights in symmetric mode, not both."
                )
        initial_parameters = self.resolved_initial_parameters
        if not np.isfinite(initial_parameters).all():
            raise ValueError("Initial recovery parameters must be finite.")
        if optimizer_name == "pso" and not np.all(
            (initial_parameters >= pso_bounds[0]) & (initial_parameters <= pso_bounds[1])
        ):
            raise ValueError("Initial recovery parameters must lie inside the configured PSO bounds.")

    @property
    def resolved_initial_parameters(self) -> np.ndarray:
        parameterization = str(self.matrix_parameterization).strip().lower()
        if self.initial_weight_matrix is not None:
            matrix = np.asarray(self.initial_weight_matrix, dtype=float)
            if parameterization == "so3":
                return rotation_vector_from_rotation_matrix(matrix)
            return symmetric_weights_from_matrix(matrix)
        if parameterization == "so3":
            values = np.asarray(self.initial_rotation_vector, dtype=float)
            if values.shape != (3,):
                raise ValueError("initial_rotation_vector must contain three values.")
            return values
        values = np.asarray(
            IDENTITY_WEIGHTS if self.initial_weights is None else self.initial_weights,
            dtype=float,
        )
        if values.shape != (6,):
            raise ValueError("initial_weights must contain six values in symmetric mode.")
        return values

    @property
    def resolved_initial_matrix(self) -> np.ndarray:
        """Return the actual 3-by-3 recovery matrix used at epoch zero."""

        return np.asarray(
            _matrix_from_parameters(
                self.resolved_initial_parameters,
                self.matrix_parameterization,
            ),
            dtype=float,
        )

    @property
    def resolved_pso_bounds(self) -> tuple[float, float]:
        if str(self.matrix_parameterization).strip().lower() == "so3":
            defaults = (-math.pi, math.pi)
        else:
            defaults = (-1.0, 1.0)
        return (
            defaults[0] if self.pso_lower_bound is None else float(self.pso_lower_bound),
            defaults[1] if self.pso_upper_bound is None else float(self.pso_upper_bound),
        )


@dataclass(frozen=True)
class MatrixDiagnostics:
    frobenius_norm: float
    singular_values: tuple[float, float, float]
    condition_number: float
    symmetry_error: float
    orthogonality_error: float
    determinant: float
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class AxisFlipCandidateResult:
    pattern: str
    signs: tuple[int, int, int]
    best_selection_loss: float
    normalized_best_selection_loss: float
    selected: bool


@dataclass(frozen=True)
class AxisFlipEvent:
    check_epoch: int
    completed_epoch: int
    selected_pattern: str
    candidates: tuple[AxisFlipCandidateResult, ...]


@dataclass(frozen=True)
class RSSResult:
    config: RSSConfig
    optimizer_name: str
    observer_angles: np.ndarray
    observer_scalar: np.ndarray
    observer_quaternions: np.ndarray
    observer_coefficients: np.ndarray
    source_angles: np.ndarray
    source_quaternions: np.ndarray
    source_coefficients: np.ndarray
    composed_source_quaternions: np.ndarray
    relative_reconstruction_quaternions: np.ndarray
    reconstructed_observer_angles: np.ndarray
    reconstructed_observer_coefficients: np.ndarray
    reconstruction_mean_error_rad: float
    reconstruction_max_error_rad: float
    recovery_matrix: np.ndarray
    optimized_parameters: np.ndarray
    optimized_weights: np.ndarray
    loss_history: tuple[dict[str, float | str | None], ...]
    sample_interval_seconds: float | None
    applied_learning_rates: tuple[float, ...]
    initial_loss: float
    best_loss: float
    terminal_loss: float
    loss_normalization_reference: float
    normalized_initial_loss: float
    normalized_best_loss: float
    normalized_terminal_loss: float
    best_epoch: int
    stop_reason: str
    matrix_diagnostics: MatrixDiagnostics
    selection_loss_normalization_reference: float | None = None
    initial_selection_loss: float | None = None
    best_selection_loss: float | None = None
    terminal_selection_loss: float | None = None
    normalized_initial_selection_loss: float | None = None
    normalized_best_selection_loss: float | None = None
    normalized_terminal_selection_loss: float | None = None
    axis_flip_events: tuple[AxisFlipEvent, ...] = ()


@dataclass(frozen=True)
class _OptimizationOutcome:
    history: tuple[dict[str, float | str | None], ...]
    applied_learning_rates: tuple[float, ...]
    best_weights: np.ndarray
    initial_loss: float
    best_loss: float
    terminal_loss: float
    loss_normalization_reference: float
    normalized_initial_loss: float
    normalized_best_loss: float
    normalized_terminal_loss: float
    best_epoch: int
    stop_reason: str
    selection_loss_normalization_reference: float | None = None
    initial_selection_loss: float | None = None
    best_selection_loss: float | None = None
    terminal_selection_loss: float | None = None
    normalized_initial_selection_loss: float | None = None
    normalized_best_selection_loss: float | None = None
    normalized_terminal_selection_loss: float | None = None
    axis_flip_events: tuple[AxisFlipEvent, ...] = ()


def symmetric_matrix_from_weights(weights: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Construct ``[[a11,a12,a13],[a12,a22,a23],[a13,a23,a33]]``."""

    if weights.ndim != 1 or int(weights.shape[0]) != 6:
        raise ValueError("weights must have shape (6,).")
    a11, a12, a13, a22, a23, a33 = tuple(weights[index] for index in range(6))
    if isinstance(weights, torch.Tensor):
        return torch.stack(
            (
                torch.stack((a11, a12, a13)),
                torch.stack((a12, a22, a23)),
                torch.stack((a13, a23, a33)),
            )
        )
    return np.asarray(((a11, a12, a13), (a12, a22, a23), (a13, a23, a33)), dtype=float)


def symmetric_weights_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Pack a symmetric 3-by-3 matrix as ``(a11,a12,a13,a22,a23,a33)``."""

    values = np.asarray(matrix, dtype=float)
    if values.shape != (3, 3) or not np.isfinite(values).all():
        raise ValueError("A symmetric initial weight matrix must be finite with shape (3, 3).")
    if not np.allclose(values, values.T, atol=1e-10, rtol=0.0):
        raise ValueError(
            "initial_weight_matrix must be symmetric when matrix_parameterization='symmetric'."
        )
    return np.asarray(
        (values[0, 0], values[0, 1], values[0, 2], values[1, 1], values[1, 2], values[2, 2]),
        dtype=float,
    )


def rotation_matrix_from_rotation_vector(
    rotation_vector: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Map a three-parameter rotation vector to a proper rotation matrix."""

    if rotation_vector.ndim != 1 or int(rotation_vector.shape[0]) != 3:
        raise ValueError("rotation_vector must have shape (3,).")
    if isinstance(rotation_vector, torch.Tensor):
        x, y, z = rotation_vector.unbind()
        zero = torch.zeros((), dtype=rotation_vector.dtype, device=rotation_vector.device)
        skew = torch.stack(
            (
                torch.stack((zero, -z, y)),
                torch.stack((z, zero, -x)),
                torch.stack((-y, x, zero)),
            )
        )
        angle = torch.linalg.vector_norm(rotation_vector)
        angle_squared = angle * angle
        a = torch.where(
            angle > 1e-8,
            torch.sin(angle) / torch.clamp(angle, min=1e-12),
            1.0 - angle_squared / 6.0 + angle_squared * angle_squared / 120.0,
        )
        b = torch.where(
            angle > 1e-8,
            (1.0 - torch.cos(angle)) / torch.clamp(angle_squared, min=1e-24),
            0.5 - angle_squared / 24.0 + angle_squared * angle_squared / 720.0,
        )
        identity = torch.eye(3, dtype=rotation_vector.dtype, device=rotation_vector.device)
        return identity + a * skew + b * (skew @ skew)
    vector = np.asarray(rotation_vector, dtype=float)
    x, y, z = vector
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=float)
    angle = float(np.linalg.norm(vector))
    angle_squared = angle * angle
    if angle > 1e-8:
        a = math.sin(angle) / angle
        b = (1.0 - math.cos(angle)) / angle_squared
    else:
        a = 1.0 - angle_squared / 6.0 + angle_squared * angle_squared / 120.0
        b = 0.5 - angle_squared / 24.0 + angle_squared * angle_squared / 720.0
    return np.eye(3) + a * skew + b * (skew @ skew)


def rotation_vector_from_rotation_matrix(rotation_matrix: np.ndarray) -> np.ndarray:
    """Map a proper 3-by-3 rotation matrix to its principal rotation vector."""

    matrix = np.asarray(rotation_matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("An SO(3) initial weight matrix must be finite with shape (3, 3).")
    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-8, rtol=0.0):
        raise ValueError(
            "initial_weight_matrix must be orthogonal when matrix_parameterization='so3'."
        )
    determinant = float(np.linalg.det(matrix))
    if not math.isclose(determinant, 1.0, abs_tol=1e-8, rel_tol=0.0):
        raise ValueError(
            "initial_weight_matrix must have determinant +1 when matrix_parameterization='so3'."
        )

    cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cosine)
    skew_vector = np.asarray(
        (
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ),
        dtype=float,
    )
    if angle < 1e-8:
        return 0.5 * skew_vector
    if math.pi - angle < 1e-7:
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        if float(np.dot(axis, skew_vector)) < 0.0:
            axis = -axis
        return angle * axis
    return (angle / (2.0 * math.sin(angle))) * skew_vector


def _rotation_matrices_from_rotation_vectors(rotation_vectors: torch.Tensor) -> torch.Tensor:
    """Vectorized Rodrigues map for PSO parameters with shape ``(P, 3)``."""

    if rotation_vectors.ndim != 2 or int(rotation_vectors.shape[1]) != 3:
        raise ValueError("rotation_vectors must have shape (P, 3).")
    x, y, z = rotation_vectors.unbind(dim=1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (
            torch.stack((zero, -z, y), dim=1),
            torch.stack((z, zero, -x), dim=1),
            torch.stack((-y, x, zero), dim=1),
        ),
        dim=1,
    )
    angles = torch.linalg.vector_norm(rotation_vectors, dim=1)
    squared = angles * angles
    a = torch.where(
        angles > 1e-8,
        torch.sin(angles) / torch.clamp(angles, min=1e-12),
        1.0 - squared / 6.0 + squared * squared / 120.0,
    )
    b = torch.where(
        angles > 1e-8,
        (1.0 - torch.cos(angles)) / torch.clamp(squared, min=1e-24),
        0.5 - squared / 24.0 + squared * squared / 720.0,
    )
    identity = torch.eye(3, dtype=rotation_vectors.dtype, device=rotation_vectors.device)
    return (
        identity.unsqueeze(0)
        + a[:, None, None] * skew
        + b[:, None, None] * torch.matmul(skew, skew)
    )


def _matrix_from_parameters(
    parameters: np.ndarray | torch.Tensor,
    parameterization: str,
) -> np.ndarray | torch.Tensor:
    if str(parameterization).strip().lower() == "so3":
        return rotation_matrix_from_rotation_vector(parameters)
    return symmetric_matrix_from_weights(parameters)


def _exponential_learning_rates(start: float, end: float, update_count: int) -> tuple[float, ...]:
    """Return an endpoint-inclusive exponential schedule for optimizer movements."""

    count = int(update_count)
    if count < 0:
        raise ValueError("update_count must be >= 0.")
    if count == 0:
        return ()
    if count == 1:
        return (float(start),)
    ratio = float(end) / float(start)
    return tuple(float(start) * ratio ** (index / float(count - 1)) for index in range(count))


def _pso_position_step(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    learning_rate: float,
    *,
    lower: float,
    upper: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale only the PSO position step and zero velocity where bounds clamp it."""

    proposed_positions = positions + float(learning_rate) * velocities
    updated_positions = torch.clamp(proposed_positions, min=float(lower), max=float(upper))
    updated_velocities = torch.where(
        updated_positions != proposed_positions,
        torch.zeros_like(velocities),
        velocities,
    )
    return updated_positions, updated_velocities


def _effective_orthogonality_weight(config: RSSConfig) -> float:
    return (
        float(config.orthogonality_weight)
        if str(config.matrix_parameterization).strip().lower() == "symmetric"
        else 0.0
    )


def _validate_loss_signals(source_angles: torch.Tensor) -> int:
    if source_angles.ndim not in {2, 3} or int(source_angles.shape[-2]) != 3:
        raise ValueError("source_angles must have shape (3, T) or (P, 3, T).")
    sample_count = int(source_angles.shape[-1])
    if sample_count < 1:
        raise ValueError("source_angles must contain at least one sample.")
    return sample_count


def lagged_cross_correlation_loss(
    source_angles: torch.Tensor,
    lags: tuple[int, ...] = (0,),
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Average directed off-diagonal correlation energy over selected lags."""

    sample_count = _validate_loss_signals(source_angles)
    lag_values = tuple(int(lag) for lag in lags)
    if not lag_values:
        raise ValueError("lags must contain at least one value.")
    if any(lag < 0 for lag in lag_values):
        raise ValueError("lags must contain nonnegative integers.")
    if len(set(lag_values)) != len(lag_values):
        raise ValueError("lags must not contain duplicates.")
    if any(lag >= sample_count for lag in lag_values):
        raise ValueError("Every RSS lag must be smaller than the sample count.")

    centered = source_angles - torch.mean(source_angles, dim=-1, keepdim=True)
    covariance_zero = torch.matmul(centered, centered.transpose(-1, -2)) / float(sample_count)
    variances = torch.clamp(torch.diagonal(covariance_zero, dim1=-2, dim2=-1), min=float(eps))
    inverse_std = torch.rsqrt(variances)
    loss = torch.zeros(source_angles.shape[:-2], dtype=source_angles.dtype, device=source_angles.device)
    for lag in lag_values:
        if lag == 0:
            covariance = covariance_zero
        else:
            covariance = torch.matmul(
                centered[..., lag:],
                centered[..., :-lag].transpose(-1, -2),
            ) / float(sample_count - lag)
        correlation = inverse_std.unsqueeze(-1) * covariance * inverse_std.unsqueeze(-2)
        offdiagonal = correlation - torch.diag_embed(
            torch.diagonal(correlation, dim1=-2, dim2=-1)
        )
        loss = loss + torch.sum(offdiagonal * offdiagonal, dim=(-2, -1)) / 3.0
    return loss / float(len(lag_values))


def _soft_histogram_mutual_information_loss(
    source_angles: torch.Tensor,
    *,
    bins: int = 16,
    bandwidth: float = 0.5,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Estimate mean pairwise MI in nats with differentiable Gaussian soft bins."""

    sample_count = _validate_loss_signals(source_angles)
    if sample_count < 2:
        raise ValueError("mutual_information loss requires at least two samples.")
    bin_count = int(bins)
    if bin_count < 2:
        raise ValueError("bins must be >= 2.")
    if not math.isfinite(float(bandwidth)) or float(bandwidth) <= 0.0:
        raise ValueError("bandwidth must be finite and > 0.")

    centered = source_angles - torch.mean(source_angles, dim=-1, keepdim=True)
    variances = torch.mean(centered * centered, dim=-1, keepdim=True)
    standardized = centered * torch.rsqrt(torch.clamp(variances, min=float(eps)))
    centers = torch.linspace(
        -4.0,
        4.0,
        bin_count,
        dtype=source_angles.dtype,
        device=source_angles.device,
    )
    logits = -0.5 * (
        (standardized.unsqueeze(-1) - centers) / float(bandwidth)
    ) ** 2
    memberships = torch.softmax(logits, dim=-1)
    pair_losses = []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        first_membership = memberships[..., first, :, :]
        second_membership = memberships[..., second, :, :]
        joint = torch.matmul(first_membership.transpose(-1, -2), second_membership)
        joint = torch.clamp(joint / float(sample_count), min=float(eps))
        joint = joint / torch.sum(joint, dim=(-2, -1), keepdim=True)
        first_marginal = torch.sum(joint, dim=-1, keepdim=True)
        second_marginal = torch.sum(joint, dim=-2, keepdim=True)
        pair_losses.append(
            torch.sum(
                joint
                * (
                    torch.log(joint)
                    - torch.log(first_marginal)
                    - torch.log(second_marginal)
                ),
                dim=(-2, -1),
            )
        )
    return torch.clamp(torch.stack(pair_losses, dim=-1).mean(dim=-1), min=0.0)


def _gcmi_mutual_information_loss(
    source_angles: torch.Tensor,
    *,
    rank_method: str,
    eps: float,
) -> torch.Tensor:
    """Evaluate scalar Gaussian-copula MI on one candidate or a PSO batch."""

    sample_count = _validate_loss_signals(source_angles)
    if sample_count < 2:
        raise ValueError("mutual_information loss requires at least two samples.")
    source_values = source_angles.detach().cpu().numpy()
    estimates = estimate_pairwise_gcmi_mutual_information(
        source_values,
        ((0, 1), (0, 2), (1, 2)),
        rank_method=rank_method,
        eps=eps,
    )
    values = np.mean(estimates, axis=-1)
    return torch.as_tensor(values, dtype=source_angles.dtype, device=source_angles.device)


def _torch_bspline_basis_memberships(
    values: torch.Tensor,
    *,
    bins: int,
    order: int,
    eps: float,
) -> torch.Tensor:
    """Return clamped B-spline memberships for samples on the final axis."""

    bin_count = int(bins)
    spline_order = int(order)
    if bin_count != bins or bin_count < 2:
        raise ValueError("bspline_bins must be an integer >= 2.")
    if spline_order != order or spline_order < 1 or spline_order >= bin_count:
        raise ValueError("bspline_order must be an integer with 1 <= order < bins.")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be finite and > 0.")

    minimum = torch.amin(values, dim=-1, keepdim=True)
    maximum = torch.amax(values, dim=-1, keepdim=True)
    value_range = maximum - minimum
    scale = torch.clamp(torch.amax(torch.abs(values), dim=-1, keepdim=True), min=1.0)
    near_constant = value_range <= float(eps) * scale
    safe_range = torch.where(near_constant, torch.ones_like(value_range), value_range)
    normalized = torch.clamp((values - minimum) / safe_range, min=0.0, max=1.0)
    normalized = torch.where(near_constant, torch.zeros_like(normalized), normalized)

    interior_count = bin_count - spline_order
    interior = torch.linspace(
        0.0,
        1.0,
        interior_count + 2,
        dtype=values.dtype,
        device=values.device,
    )[1:-1]
    knots = torch.cat(
        (
            torch.zeros(spline_order, dtype=values.dtype, device=values.device),
            interior,
            torch.ones(spline_order, dtype=values.dtype, device=values.device),
        )
    )
    coordinate = normalized.unsqueeze(-1)
    basis = (
        (coordinate >= knots[:-1]) & (coordinate < knots[1:])
    ).to(dtype=values.dtype)
    endpoint_basis = torch.zeros_like(basis)
    endpoint_basis[..., int(knots.numel()) - spline_order - 1] = 1.0
    basis = torch.where((coordinate >= 1.0), endpoint_basis, basis)

    degree = spline_order - 1
    for level in range(1, degree + 1):
        new_count = int(knots.numel()) - level - 1
        left_denominator = knots[level : level + new_count] - knots[:new_count]
        right_denominator = (
            knots[level + 1 : level + 1 + new_count]
            - knots[1 : 1 + new_count]
        )
        safe_left = torch.where(
            left_denominator > 0.0,
            left_denominator,
            torch.ones_like(left_denominator),
        )
        safe_right = torch.where(
            right_denominator > 0.0,
            right_denominator,
            torch.ones_like(right_denominator),
        )
        left_factor = (coordinate - knots[:new_count]) / safe_left
        right_factor = (
            knots[level + 1 : level + 1 + new_count] - coordinate
        ) / safe_right
        left_factor = torch.where(
            left_denominator > 0.0, left_factor, torch.zeros_like(left_factor)
        )
        right_factor = torch.where(
            right_denominator > 0.0, right_factor, torch.zeros_like(right_factor)
        )
        basis = (
            left_factor * basis[..., :new_count]
            + right_factor * basis[..., 1 : new_count + 1]
        )

    basis = torch.clamp(basis, min=0.0)
    row_sums = torch.sum(basis, dim=-1, keepdim=True)
    return basis / torch.clamp(row_sums, min=float(eps))


def _bspline_mutual_information_loss(
    source_angles: torch.Tensor,
    *,
    bins: int,
    order: int,
    eps: float,
) -> torch.Tensor:
    """Estimate mean pairwise B-spline MI in nats with Torch operations."""

    sample_count = _validate_loss_signals(source_angles)
    if sample_count < 2:
        raise ValueError("mutual_information loss requires at least two samples.")
    memberships = _torch_bspline_basis_memberships(
        source_angles,
        bins=bins,
        order=order,
        eps=eps,
    )
    pair_losses = []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        first_membership = memberships[..., first, :, :]
        second_membership = memberships[..., second, :, :]
        joint = torch.matmul(first_membership.transpose(-1, -2), second_membership)
        joint = joint / float(sample_count)
        first_marginal = torch.mean(first_membership, dim=-2, keepdim=False)
        second_marginal = torch.mean(second_membership, dim=-2, keepdim=False)
        denominator = first_marginal.unsqueeze(-1) * second_marginal.unsqueeze(-2)
        positive = joint > 0.0
        safe_joint = torch.clamp(joint, min=float(eps))
        safe_denominator = torch.clamp(denominator, min=float(eps))
        terms = torch.where(
            positive,
            joint * (torch.log(safe_joint) - torch.log(safe_denominator)),
            torch.zeros_like(joint),
        )
        pair_losses.append(torch.sum(terms, dim=(-2, -1)))
    return torch.clamp(torch.stack(pair_losses, dim=-1).mean(dim=-1), min=0.0)


def _ksg_mutual_information_loss(
    source_angles: torch.Tensor,
    *,
    k: int,
    standardize: bool,
    max_samples: int | None,
    subsample_mode: str,
    subsample_seed: int,
    sample_indices: torch.Tensor | None,
) -> torch.Tensor:
    """Evaluate raw mean pairwise KSG-1 MI on one candidate or a PSO batch."""

    sample_count = _validate_loss_signals(source_angles)
    if sample_indices is None:
        selected_indices = select_ksg_subsample_indices(
            sample_count,
            max_samples=max_samples,
            mode=subsample_mode,
            seed=subsample_seed,
        )
        if int(selected_indices.size) < sample_count:
            index_tensor = torch.as_tensor(
                selected_indices,
                dtype=torch.long,
                device=source_angles.device,
            )
            source_angles = torch.index_select(source_angles, -1, index_tensor)
            effective_sample_count = int(selected_indices.size)
        else:
            effective_sample_count = sample_count
    else:
        if sample_indices.ndim != 1 or int(sample_indices.numel()) < 1:
            raise ValueError("ksg_sample_indices must be a nonempty one-dimensional tensor.")
        if sample_indices.dtype != torch.long:
            raise ValueError("ksg_sample_indices must use torch.long dtype.")
        source_angles = torch.index_select(
            source_angles,
            -1,
            sample_indices.to(device=source_angles.device),
        )
        effective_sample_count = int(sample_indices.numel())
    neighbour_count = int(k)
    if neighbour_count != k or neighbour_count < 1:
        raise ValueError("k must be an integer >= 1.")
    if neighbour_count >= effective_sample_count:
        raise ValueError("ksg_k must be smaller than the effective RSS sample count.")
    if not isinstance(standardize, (bool, np.bool_)):
        raise ValueError("standardize must be a boolean.")
    source_values = source_angles.detach().cpu().numpy()
    pairs = ((0, 1), (0, 2), (1, 2))
    if source_values.ndim == 2:
        estimates = estimate_pairwise_ksg_mutual_information(
            tuple(source_values[index] for index in range(3)),
            pairs,
            k=neighbour_count,
            standardize=standardize,
        )
        value: float | np.ndarray = float(np.mean(estimates))
    else:
        value = np.asarray(
            [
                np.mean(
                    estimate_pairwise_ksg_mutual_information(
                        tuple(candidate[index] for index in range(3)),
                        pairs,
                        k=neighbour_count,
                        standardize=standardize,
                    )
                )
                for candidate in source_values
            ],
            dtype=float,
        )
    return torch.as_tensor(value, dtype=source_angles.dtype, device=source_angles.device)


def mutual_information_loss(
    source_angles: torch.Tensor,
    *,
    method: str = "ksg",
    gcmi_rank_method: str = DEFAULT_GCMI_RANK_METHOD,
    bspline_bins: int = DEFAULT_BSPLINE_BINS,
    bspline_order: int = DEFAULT_BSPLINE_ORDER,
    k: int = DEFAULT_KSG_K,
    standardize: bool = DEFAULT_KSG_STANDARDIZE,
    ksg_max_samples: int | None = DEFAULT_KSG_MAX_SAMPLES,
    ksg_subsample_mode: str = DEFAULT_KSG_SUBSAMPLE_MODE,
    ksg_subsample_seed: int = DEFAULT_KSG_SUBSAMPLE_SEED,
    ksg_sample_indices: torch.Tensor | None = None,
    bins: int = 16,
    bandwidth: float = 0.5,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Estimate the existing three-pair mean MI loss in nats."""

    selected = str(method).strip().lower()
    if selected == "gcmi":
        return _gcmi_mutual_information_loss(
            source_angles,
            rank_method=gcmi_rank_method,
            eps=eps,
        )
    if selected == "bspline":
        return _bspline_mutual_information_loss(
            source_angles,
            bins=bspline_bins,
            order=bspline_order,
            eps=eps,
        )
    if selected == "ksg":
        return _ksg_mutual_information_loss(
            source_angles,
            k=k,
            standardize=standardize,
            max_samples=ksg_max_samples,
            subsample_mode=ksg_subsample_mode,
            subsample_seed=ksg_subsample_seed,
            sample_indices=ksg_sample_indices,
        )
    if selected == "histogram":
        return _soft_histogram_mutual_information_loss(
            source_angles,
            bins=bins,
            bandwidth=bandwidth,
            eps=eps,
        )
    raise ValueError(f"method must be one of: {', '.join(MI_ESTIMATORS)}.")


def rotational_action_loss(
    source_angles: torch.Tensor,
    *,
    sample_interval_seconds: float,
    inertias: tuple[float, float, float] = (1.0, 1.0, 1.0),
    stiffnesses: tuple[float, float, float] = (1.0, 1.0, 1.0),
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return mean per-mode torsional action-density magnitude."""

    sample_count = _validate_loss_signals(source_angles)
    if sample_count < 2:
        raise ValueError("action loss requires at least two samples.")
    interval = float(sample_interval_seconds)
    if not math.isfinite(interval) or interval <= 0.0:
        raise ValueError("sample_interval_seconds must be finite and > 0.")
    inertia_values = tuple(float(value) for value in inertias)
    stiffness_values = tuple(float(value) for value in stiffnesses)
    if len(inertia_values) != 3 or not np.isfinite(np.asarray(inertia_values)).all() or any(
        value <= 0.0 for value in inertia_values
    ):
        raise ValueError("inertias must contain three finite values > 0.")
    if len(stiffness_values) != 3 or not np.isfinite(np.asarray(stiffness_values)).all() or any(
        value <= 0.0 for value in stiffness_values
    ):
        raise ValueError("stiffnesses must contain three finite values > 0.")

    relative = source_angles - source_angles[..., :1]
    relative = torch.atan2(torch.sin(relative), torch.cos(relative))
    differences = relative[..., 1:] - relative[..., :-1]
    wrapped_differences = torch.atan2(torch.sin(differences), torch.cos(differences))
    angular_velocity = wrapped_differences / interval
    midpoint = relative[..., :-1] + 0.5 * wrapped_differences
    midpoint = torch.atan2(torch.sin(midpoint), torch.cos(midpoint))
    parameter_shape = (1,) * (source_angles.ndim - 2) + (3, 1)
    inertia_tensor = torch.as_tensor(
        inertia_values, dtype=source_angles.dtype, device=source_angles.device
    ).reshape(parameter_shape)
    stiffness_tensor = torch.as_tensor(
        stiffness_values, dtype=source_angles.dtype, device=source_angles.device
    ).reshape(parameter_shape)
    kinetic = 0.5 * inertia_tensor * angular_velocity * angular_velocity
    potential = 0.5 * stiffness_tensor * midpoint * midpoint
    action_density = torch.mean(kinetic - potential, dim=-1)
    smooth_magnitude = torch.sqrt(action_density * action_density + float(eps) ** 2) - float(eps)
    return torch.mean(smooth_magnitude, dim=-1)


def _selected_separation_loss(
    source_angles: torch.Tensor,
    *,
    loss_type: str,
    lags: tuple[int, ...],
    mi_estimator: str,
    gcmi_rank_method: str,
    bspline_bins: int,
    bspline_order: int,
    ksg_k: int,
    ksg_standardize: bool,
    ksg_max_samples: int | None,
    ksg_subsample_mode: str,
    ksg_subsample_seed: int,
    ksg_sample_indices: torch.Tensor | None,
    mi_bins: int,
    mi_bandwidth: float,
    action_inertias: tuple[float, float, float],
    action_stiffnesses: tuple[float, float, float],
    sample_interval_seconds: float | None,
    eps: float,
) -> torch.Tensor:
    selected = str(loss_type).strip().lower()
    if selected == "lagged_cross_correlation":
        return lagged_cross_correlation_loss(source_angles, lags, eps=eps)
    if selected == "mutual_information":
        return mutual_information_loss(
            source_angles,
            method=mi_estimator,
            gcmi_rank_method=gcmi_rank_method,
            bspline_bins=bspline_bins,
            bspline_order=bspline_order,
            k=ksg_k,
            standardize=ksg_standardize,
            ksg_max_samples=ksg_max_samples,
            ksg_subsample_mode=ksg_subsample_mode,
            ksg_subsample_seed=ksg_subsample_seed,
            ksg_sample_indices=ksg_sample_indices,
            bins=mi_bins,
            bandwidth=mi_bandwidth,
            eps=eps,
        )
    if selected == "action":
        if sample_interval_seconds is None:
            raise ValueError(
                "sample_interval_seconds is required when loss_type='action'."
            )
        return rotational_action_loss(
            source_angles,
            sample_interval_seconds=sample_interval_seconds,
            inertias=action_inertias,
            stiffnesses=action_stiffnesses,
            eps=eps,
        )
    raise ValueError(
        "loss_type must be one of: lagged_cross_correlation, mutual_information, action."
    )


def compute_rss_loss(
    source_angles: torch.Tensor,
    lags: tuple[int, ...] = (0,),
    *,
    recovery_matrix: torch.Tensor,
    loss_type: str = "lagged_cross_correlation",
    separation_weight: float = 1.0,
    orthogonality_weight: float = 0.0,
    eps: float = 1e-12,
    observer_quaternions: torch.Tensor | None = None,
    reconstruction_weight: float = 0.0,
    mi_estimator: str = "ksg",
    gcmi_rank_method: str = DEFAULT_GCMI_RANK_METHOD,
    bspline_bins: int = DEFAULT_BSPLINE_BINS,
    bspline_order: int = DEFAULT_BSPLINE_ORDER,
    ksg_k: int = DEFAULT_KSG_K,
    ksg_standardize: bool = DEFAULT_KSG_STANDARDIZE,
    ksg_max_samples: int | None = DEFAULT_KSG_MAX_SAMPLES,
    ksg_subsample_mode: str = DEFAULT_KSG_SUBSAMPLE_MODE,
    ksg_subsample_seed: int = DEFAULT_KSG_SUBSAMPLE_SEED,
    ksg_sample_indices: torch.Tensor | None = None,
    mi_bins: int = 16,
    mi_bandwidth: float = 0.5,
    action_inertias: tuple[float, float, float] = (1.0, 1.0, 1.0),
    action_stiffnesses: tuple[float, float, float] = (1.0, 1.0, 1.0),
    sample_interval_seconds: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Evaluate the selected separation loss and optional RSS constraints."""

    if source_angles.ndim != 2 or source_angles.shape[0] != 3:
        raise ValueError("source_angles must have shape (3, T).")
    if recovery_matrix.ndim != 2 or tuple(recovery_matrix.shape) != (3, 3):
        raise ValueError("recovery_matrix must have shape (3, 3).")
    separation_loss = _selected_separation_loss(
        source_angles,
        loss_type=loss_type,
        lags=lags,
        mi_estimator=mi_estimator,
        gcmi_rank_method=gcmi_rank_method,
        bspline_bins=bspline_bins,
        bspline_order=bspline_order,
        ksg_k=ksg_k,
        ksg_standardize=ksg_standardize,
        ksg_max_samples=ksg_max_samples,
        ksg_subsample_mode=ksg_subsample_mode,
        ksg_subsample_seed=ksg_subsample_seed,
        ksg_sample_indices=ksg_sample_indices,
        mi_bins=mi_bins,
        mi_bandwidth=mi_bandwidth,
        action_inertias=action_inertias,
        action_stiffnesses=action_stiffnesses,
        sample_interval_seconds=sample_interval_seconds,
        eps=eps,
    )
    q = float(source_angles.shape[0])
    identity = torch.eye(3, dtype=recovery_matrix.dtype, device=recovery_matrix.device)
    gram_error = recovery_matrix @ recovery_matrix.T - identity
    orthogonality_loss = torch.sum(gram_error * gram_error) / q
    if observer_quaternions is None:
        if float(reconstruction_weight) != 0.0:
            raise ValueError("observer_quaternions are required when reconstruction_weight is nonzero.")
        reconstruction_loss = torch.zeros((), dtype=source_angles.dtype, device=source_angles.device)
    else:
        reconstruction_loss, _, _ = quaternion_reconstruction_loss_torch(
            source_angles,
            recovery_matrix,
            observer_quaternions,
            eps=float(eps),
        )
    total_loss = (
        float(separation_weight) * separation_loss
        + float(orthogonality_weight) * orthogonality_loss
        + float(reconstruction_weight) * reconstruction_loss
    )
    return total_loss, {
        "separation_loss": separation_loss,
        "orthogonality_loss": orthogonality_loss,
        "reconstruction_loss": reconstruction_loss,
    }


def _symmetric_matrices_from_weights(weights: torch.Tensor) -> torch.Tensor:
    """Vectorized symmetric matrices for PSO weights with shape ``(P, 6)``."""

    if weights.ndim != 2 or int(weights.shape[1]) != 6:
        raise ValueError("weights must have shape (P, 6).")
    a11, a12, a13, a22, a23, a33 = tuple(weights[:, index] for index in range(6))
    return torch.stack(
        (
            torch.stack((a11, a12, a13), dim=1),
            torch.stack((a12, a22, a23), dim=1),
            torch.stack((a13, a23, a33), dim=1),
        ),
        dim=1,
    )


def _matrices_from_parameters(parameters: torch.Tensor, parameterization: str) -> torch.Tensor:
    if str(parameterization).strip().lower() == "so3":
        return _rotation_matrices_from_rotation_vectors(parameters)
    return _symmetric_matrices_from_weights(parameters)


def _decode_scalar_and_vector_batch(
    scalar: torch.Tensor,
    vector: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Decode a particle batch of vector coefficients with shape ``(P, 3, T)``."""

    if vector.ndim != 3 or int(vector.shape[1]) != 3:
        raise ValueError("vector must have shape (P, 3, T).")
    sample_count = int(vector.shape[2])
    scalar_values = scalar.reshape(-1)
    if scalar_values.numel() != sample_count:
        raise ValueError("scalar and vector must share the same sample count.")
    scalar_batch = scalar_values.unsqueeze(0).expand(int(vector.shape[0]), -1)
    quaternions = torch.cat((scalar_batch.unsqueeze(2), vector.transpose(1, 2)), dim=2)
    norms = torch.linalg.vector_norm(quaternions, dim=2, keepdim=True)
    normalized = quaternions / torch.clamp(norms, min=float(eps))
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=vector.dtype, device=vector.device)
    normalized = torch.where(norms <= float(eps), identity.reshape(1, 1, 4), normalized)
    normalized = torch.where(normalized[:, :, :1] < 0.0, -normalized, normalized)
    w, x, y, z = tuple(normalized[:, :, index] for index in range(4))
    theta1 = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    asin_arg = torch.clamp(
        2.0 * (w * y - z * x),
        min=-1.0 + float(eps),
        max=1.0 - float(eps),
    )
    theta2 = torch.asin(asin_arg)
    theta3 = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.stack((theta1, theta2, theta3), dim=1)


def _compute_rss_loss_batch(
    source_angles: torch.Tensor,
    lags: tuple[int, ...] = (0,),
    *,
    recovery_matrices: torch.Tensor,
    loss_type: str = "lagged_cross_correlation",
    separation_weight: float = 1.0,
    orthogonality_weight: float = 0.0,
    eps: float = 1e-12,
    observer_quaternions: torch.Tensor | None = None,
    reconstruction_weight: float = 0.0,
    mi_estimator: str = "ksg",
    gcmi_rank_method: str = DEFAULT_GCMI_RANK_METHOD,
    bspline_bins: int = DEFAULT_BSPLINE_BINS,
    bspline_order: int = DEFAULT_BSPLINE_ORDER,
    ksg_k: int = DEFAULT_KSG_K,
    ksg_standardize: bool = DEFAULT_KSG_STANDARDIZE,
    ksg_max_samples: int | None = DEFAULT_KSG_MAX_SAMPLES,
    ksg_subsample_mode: str = DEFAULT_KSG_SUBSAMPLE_MODE,
    ksg_subsample_seed: int = DEFAULT_KSG_SUBSAMPLE_SEED,
    ksg_sample_indices: torch.Tensor | None = None,
    mi_bins: int = 16,
    mi_bandwidth: float = 0.5,
    action_inertias: tuple[float, float, float] = (1.0, 1.0, 1.0),
    action_stiffnesses: tuple[float, float, float] = (1.0, 1.0, 1.0),
    sample_interval_seconds: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Vectorized equivalent of :func:`compute_rss_loss` for PSO particles."""

    if source_angles.ndim != 3 or int(source_angles.shape[1]) != 3:
        raise ValueError("source_angles must have shape (P, 3, T).")
    particle_count = int(source_angles.shape[0])
    if tuple(recovery_matrices.shape) != (particle_count, 3, 3):
        raise ValueError("recovery_matrices must have shape (P, 3, 3).")
    separation_loss = _selected_separation_loss(
        source_angles,
        loss_type=loss_type,
        lags=lags,
        mi_estimator=mi_estimator,
        gcmi_rank_method=gcmi_rank_method,
        bspline_bins=bspline_bins,
        bspline_order=bspline_order,
        ksg_k=ksg_k,
        ksg_standardize=ksg_standardize,
        ksg_max_samples=ksg_max_samples,
        ksg_subsample_mode=ksg_subsample_mode,
        ksg_subsample_seed=ksg_subsample_seed,
        ksg_sample_indices=ksg_sample_indices,
        mi_bins=mi_bins,
        mi_bandwidth=mi_bandwidth,
        action_inertias=action_inertias,
        action_stiffnesses=action_stiffnesses,
        sample_interval_seconds=sample_interval_seconds,
        eps=eps,
    )
    q = float(source_angles.shape[1])

    identity = torch.eye(3, dtype=recovery_matrices.dtype, device=recovery_matrices.device).unsqueeze(0)
    gram_error = torch.matmul(recovery_matrices, recovery_matrices.transpose(1, 2)) - identity
    orthogonality_loss = torch.sum(gram_error * gram_error, dim=(1, 2)) / q
    if observer_quaternions is None:
        if float(reconstruction_weight) != 0.0:
            raise ValueError("observer_quaternions are required when reconstruction_weight is nonzero.")
        reconstruction_loss = torch.zeros(
            int(particle_count), dtype=source_angles.dtype, device=source_angles.device
        )
    else:
        reconstruction_loss, _, _ = quaternion_reconstruction_loss_torch(
            source_angles,
            recovery_matrices,
            observer_quaternions,
            eps=float(eps),
        )
    total_loss = (
        float(separation_weight) * separation_loss
        + float(orthogonality_weight) * orthogonality_loss
        + float(reconstruction_weight) * reconstruction_loss
    )
    return total_loss, {
        "separation_loss": separation_loss,
        "orthogonality_loss": orthogonality_loss,
        "reconstruction_loss": reconstruction_loss,
    }


def _loss_record(
    *,
    epoch: int,
    raw_total: float,
    separation_loss: float,
    orthogonality_loss: float,
    reconstruction_loss: float,
    learning_rate: float | None,
    reference: float,
    config: RSSConfig,
) -> dict[str, float | str | None]:
    normalized_separation = float(config.separation_weight) * separation_loss / reference
    normalized_orthogonality = _effective_orthogonality_weight(config) * orthogonality_loss / reference
    normalized_reconstruction = float(config.reconstruction_weight) * reconstruction_loss / reference
    return {
        "epoch": float(epoch),
        "learning_rate": None if learning_rate is None else float(learning_rate),
        "loss_type": config.loss_type,
        "normalized_total_loss": (
            normalized_separation + normalized_orthogonality + normalized_reconstruction
        ),
        "normalized_separation_loss": normalized_separation,
        "normalized_orthogonality_loss": normalized_orthogonality,
        "normalized_reconstruction_loss": normalized_reconstruction,
        "total_loss": raw_total,
        "separation_loss": separation_loss,
        "orthogonality_loss": orthogonality_loss,
        "reconstruction_loss": reconstruction_loss,
    }


def _evaluate_single_candidate(
    weights: torch.Tensor,
    scalar_tensor: torch.Tensor,
    vector_tensor: torch.Tensor,
    observer_quaternions: torch.Tensor,
    config: RSSConfig,
    ksg_sample_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    recovery = _matrix_from_parameters(weights, config.matrix_parameterization)
    source_vector = recovery @ vector_tensor
    source_angles, _ = decode_scalar_and_vector_torch(
        scalar_tensor,
        source_vector,
        eps=float(config.quaternion_eps),
    )
    return compute_rss_loss(
        source_angles,
        tuple(int(lag) for lag in config.lags),
        recovery_matrix=recovery,
        loss_type=config.loss_type,
        separation_weight=float(config.separation_weight),
        orthogonality_weight=_effective_orthogonality_weight(config),
        eps=float(config.eps),
        observer_quaternions=observer_quaternions,
        reconstruction_weight=float(config.reconstruction_weight),
        mi_estimator=config.mi_estimator,
        gcmi_rank_method=config.gcmi_rank_method,
        bspline_bins=int(config.bspline_bins),
        bspline_order=int(config.bspline_order),
        ksg_k=int(config.ksg_k),
        ksg_standardize=bool(config.ksg_standardize),
        ksg_max_samples=config.ksg_max_samples,
        ksg_subsample_mode=config.ksg_subsample_mode,
        ksg_subsample_seed=int(config.ksg_subsample_seed),
        ksg_sample_indices=ksg_sample_indices,
        mi_bins=int(config.mi_bins),
        mi_bandwidth=float(config.mi_bandwidth),
        action_inertias=config.action_inertias,
        action_stiffnesses=config.action_stiffnesses,
        sample_interval_seconds=config.sample_interval_seconds,
    )


def _resolve_device(value: str) -> torch.device:
    requested = str(value).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {value!r} was requested but CUDA is unavailable.")
    return device


def _matrix_diagnostics(matrix: np.ndarray, *, eps: float) -> MatrixDiagnostics:
    singular_values_array = np.linalg.svd(matrix, compute_uv=False)
    smallest = float(singular_values_array[-1])
    condition = float(np.inf if smallest <= float(eps) else singular_values_array[0] / smallest)
    norm = float(np.linalg.norm(matrix, ord="fro"))
    warnings: list[str] = []
    if norm <= max(1e-8, 10.0 * float(eps)):
        warnings.append("recovery_matrix_collapsed")
    if not math.isfinite(condition) or condition >= 1.0 / float(eps):
        warnings.append("recovery_matrix_ill_conditioned")
    return MatrixDiagnostics(
        frobenius_norm=norm,
        singular_values=tuple(float(value) for value in singular_values_array),
        condition_number=condition,
        symmetry_error=float(np.linalg.norm(matrix - matrix.T, ord="fro")),
        orthogonality_error=float(np.linalg.norm(matrix @ matrix.T - np.eye(3), ord="fro")),
        determinant=float(np.linalg.det(matrix)),
        warnings=tuple(warnings),
    )


@dataclass(frozen=True)
class _EnhancedEvaluation:
    training_loss: torch.Tensor
    selection_loss: torch.Tensor
    terms: dict[str, torch.Tensor]


class _AdaptiveRegularizerScheduler:
    """Plateau-driven regularizer-weight state machine."""

    def __init__(
        self,
        config: RSSConfig,
        *,
        base_loss: float,
        regularizer_loss: float,
    ) -> None:
        self.enabled = bool(config.adaptive_regularizer)
        self.weight = float(config.regularizer_start_weight) if self.enabled else 0.0
        self.patience = int(config.regularizer_plateau_patience)
        self.tolerance = float(config.regularizer_plateau_tolerance)
        self.ramp_epochs = int(config.regularizer_ramp_epochs)
        self.eps = float(config.eps)
        self.phase = (
            "regularizer_monitor"
            if self.enabled and self.weight >= 1.0
            else ("base_monitor" if self.enabled else "disabled")
        )
        self._anchor = float(
            regularizer_loss if self.phase == "regularizer_monitor" else base_loss
        )
        self._stagnant_epochs = 0
        self._ramp_step = 0
        self._ramp_origin = self.weight

    def _observe(self, value: float) -> bool:
        improvement = (self._anchor - float(value)) / max(abs(self._anchor), self.eps)
        if improvement > self.tolerance:
            self._anchor = float(value)
            self._stagnant_epochs = 0
        else:
            self._stagnant_epochs += 1
        return self._stagnant_epochs >= self.patience

    def update(self, *, base_loss: float, regularizer_loss: float) -> None:
        if not self.enabled:
            return
        if self.phase == "base_monitor":
            if self._observe(base_loss):
                self.phase = "ramp_up"
                self._ramp_step = 0
                self._ramp_origin = self.weight
            return
        if self.phase == "regularizer_monitor":
            if self._observe(regularizer_loss):
                self.phase = "ramp_down"
                self._ramp_step = 0
                self._ramp_origin = self.weight
            return
        self._ramp_step += 1
        fraction = min(1.0, self._ramp_step / float(self.ramp_epochs))
        if self.phase == "ramp_up":
            self.weight = self._ramp_origin + (1.0 - self._ramp_origin) * fraction
            if fraction >= 1.0:
                self.weight = 1.0
                self.phase = "regularizer_monitor"
                self._anchor = float(regularizer_loss)
                self._stagnant_epochs = 0
        else:
            self.weight = self._ramp_origin * (1.0 - fraction)
            if fraction >= 1.0:
                self.weight = 0.0
                self.phase = "base_monitor"
                self._anchor = float(base_loss)
                self._stagnant_epochs = 0

    def reset_monitor(self, *, base_loss: float, regularizer_loss: float) -> None:
        """Reset plateau evidence without advancing a ramp or changing its weight."""

        if self.phase == "base_monitor":
            self._anchor = float(base_loss)
            self._stagnant_epochs = 0
        elif self.phase == "regularizer_monitor":
            self._anchor = float(regularizer_loss)
            self._stagnant_epochs = 0


_AXIS_FLIP_PATTERNS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("identity", (1, 1, 1)),
    ("xy", (-1, -1, 1)),
    ("xz", (-1, 1, -1)),
    ("yz", (1, -1, -1)),
)


def _detach_enhanced_evaluation(value: _EnhancedEvaluation) -> _EnhancedEvaluation:
    return _EnhancedEvaluation(
        training_loss=value.training_loss.detach().clone(),
        selection_loss=value.selection_loss.detach().clone(),
        terms={name: term.detach().clone() for name, term in value.terms.items()},
    )


def _select_batch_evaluation(
    value: _EnhancedEvaluation, index: int
) -> _EnhancedEvaluation:
    return _EnhancedEvaluation(
        training_loss=value.training_loss[index],
        selection_loss=value.selection_loss[index],
        terms={name: term[index] for name, term in value.terms.items()},
    )


def _enhanced_evaluation_values(
    value: _EnhancedEvaluation,
) -> tuple[float, float, float, float, float, float]:
    return (
        float(value.training_loss.detach().cpu()),
        float(value.selection_loss.detach().cpu()),
        float(value.terms["separation_loss"].detach().cpu()),
        float(value.terms["regularizer_loss"].detach().cpu()),
        float(value.terms["orthogonality_loss"].detach().cpu()),
        float(value.terms["reconstruction_loss"].detach().cpu()),
    )


def _evaluate_enhanced_single(
    weights: torch.Tensor,
    scalar_tensor: torch.Tensor,
    vector_tensor: torch.Tensor,
    observer_quaternions: torch.Tensor,
    config: RSSConfig,
    *,
    regularizer_weight: float,
    sample_rate_hz: float | None,
    ksg_sample_indices: torch.Tensor | None,
) -> _EnhancedEvaluation:
    base_total, terms = _evaluate_single_candidate(
        weights,
        scalar_tensor,
        vector_tensor,
        observer_quaternions,
        config,
        ksg_sample_indices,
    )
    if config.adaptive_regularizer:
        if sample_rate_hz is None or config.specialized_regularizer is None:
            raise RuntimeError("Adaptive RSS regularizer context was not resolved.")
        recovery = _matrix_from_parameters(weights, config.matrix_parameterization)
        source_vector = recovery @ vector_tensor
        source_angles, _ = decode_scalar_and_vector_torch(
            scalar_tensor,
            source_vector,
            eps=float(config.quaternion_eps),
        )
        regularizer = compute_specialized_loss(
            source_angles,
            sample_rate_hz,
            config.specialized_regularizer,
            config.specialized_loss_config,
        )
        if not isinstance(regularizer, torch.Tensor):
            raise RuntimeError("Specialized Torch loss returned a non-Torch result.")
    else:
        regularizer = torch.zeros_like(base_total)
    training = base_total + float(regularizer_weight) * regularizer
    selection = base_total + regularizer
    return _EnhancedEvaluation(
        training_loss=training,
        selection_loss=selection,
        terms={**terms, "regularizer_loss": regularizer},
    )


def _evaluate_enhanced_batch(
    parameters: torch.Tensor,
    scalar_tensor: torch.Tensor,
    vector_tensor: torch.Tensor,
    observer_quaternions: torch.Tensor,
    config: RSSConfig,
    *,
    regularizer_weight: float,
    sample_rate_hz: float | None,
    ksg_sample_indices: torch.Tensor | None,
) -> _EnhancedEvaluation:
    recovery_matrices = _matrices_from_parameters(
        parameters, config.matrix_parameterization
    )
    source_vectors = torch.einsum("pij,jt->pit", recovery_matrices, vector_tensor)
    source_angles = _decode_scalar_and_vector_batch(
        scalar_tensor,
        source_vectors,
        eps=float(config.quaternion_eps),
    )
    base_totals, terms = _compute_rss_loss_batch(
        source_angles,
        tuple(int(lag) for lag in config.lags),
        recovery_matrices=recovery_matrices,
        loss_type=config.loss_type,
        separation_weight=float(config.separation_weight),
        orthogonality_weight=_effective_orthogonality_weight(config),
        eps=float(config.eps),
        observer_quaternions=observer_quaternions,
        reconstruction_weight=float(config.reconstruction_weight),
        mi_estimator=config.mi_estimator,
        gcmi_rank_method=config.gcmi_rank_method,
        bspline_bins=int(config.bspline_bins),
        bspline_order=int(config.bspline_order),
        ksg_k=int(config.ksg_k),
        ksg_standardize=bool(config.ksg_standardize),
        ksg_max_samples=config.ksg_max_samples,
        ksg_subsample_mode=config.ksg_subsample_mode,
        ksg_subsample_seed=int(config.ksg_subsample_seed),
        ksg_sample_indices=ksg_sample_indices,
        mi_bins=int(config.mi_bins),
        mi_bandwidth=float(config.mi_bandwidth),
        action_inertias=config.action_inertias,
        action_stiffnesses=config.action_stiffnesses,
        sample_interval_seconds=config.sample_interval_seconds,
    )
    if config.adaptive_regularizer:
        if sample_rate_hz is None or config.specialized_regularizer is None:
            raise RuntimeError("Adaptive RSS regularizer context was not resolved.")
        regularizer = compute_specialized_loss(
            source_angles,
            sample_rate_hz,
            config.specialized_regularizer,
            config.specialized_loss_config,
        )
        if not isinstance(regularizer, torch.Tensor):
            raise RuntimeError("Specialized Torch loss returned a non-Torch result.")
    else:
        regularizer = torch.zeros_like(base_totals)
    return _EnhancedEvaluation(
        training_loss=base_totals + float(regularizer_weight) * regularizer,
        selection_loss=base_totals + regularizer,
        terms={**terms, "regularizer_loss": regularizer},
    )


def _enhanced_loss_record(
    *,
    epoch: int,
    evaluation: _EnhancedEvaluation,
    learning_rate: float | None,
    training_reference: float,
    selection_reference: float,
    config: RSSConfig,
    regularizer_weight: float,
    regularizer_phase: str,
    flip_check_epoch: int | None = None,
    flip_pattern: str | None = None,
) -> dict[str, float | str | None]:
    training, selection, base, regularizer, orthogonality, reconstruction = (
        _enhanced_evaluation_values(evaluation)
    )
    normalized_base = float(config.separation_weight) * base / training_reference
    normalized_orthogonality = (
        _effective_orthogonality_weight(config) * orthogonality / training_reference
    )
    normalized_reconstruction = (
        float(config.reconstruction_weight) * reconstruction / training_reference
    )
    normalized_regularizer = (
        float(regularizer_weight) * regularizer / training_reference
    )
    return {
        "epoch": float(epoch),
        "learning_rate": None if learning_rate is None else float(learning_rate),
        "loss_type": config.loss_type,
        "normalized_total_loss": training / training_reference,
        "normalized_separation_loss": normalized_base,
        "normalized_regularizer_loss": normalized_regularizer,
        "normalized_orthogonality_loss": normalized_orthogonality,
        "normalized_reconstruction_loss": normalized_reconstruction,
        "total_loss": training,
        "selection_loss": selection,
        "normalized_selection_loss": selection / selection_reference,
        "separation_loss": base,
        "regularizer_type": config.specialized_regularizer,
        "regularizer_loss": regularizer,
        "regularizer_weight": float(regularizer_weight),
        "regularizer_phase": regularizer_phase,
        "selection_stop_threshold": float(config.regularizer_min_loss_threshold),
        "orthogonality_loss": orthogonality,
        "reconstruction_loss": reconstruction,
        "axis_flip_check_epoch": (
            None if flip_check_epoch is None else float(flip_check_epoch)
        ),
        "axis_flip_pattern": flip_pattern,
    }


def _optimizer_for_weights(
    weights: torch.Tensor, config: RSSConfig, optimizer_name: str
) -> torch.optim.Optimizer:
    if optimizer_name == "adam":
        return torch.optim.Adam((weights,), lr=float(config.learning_rate))
    return torch.optim.SGD(
        (weights,),
        lr=float(config.learning_rate),
        momentum=float(config.sgd_momentum),
    )


def _flipped_parameters(
    parameters: torch.Tensor, signs: tuple[int, int, int]
) -> torch.Tensor:
    matrix = np.asarray(
        _matrix_from_parameters(parameters.detach().cpu().numpy(), "so3"), dtype=float
    )
    flipped = np.diag(np.asarray(signs, dtype=float)) @ matrix
    vector = rotation_vector_from_rotation_matrix(flipped)
    return torch.as_tensor(
        vector, dtype=parameters.dtype, device=parameters.device
    )


def _flipped_parameter_batch(
    parameters: torch.Tensor, signs: tuple[int, int, int]
) -> torch.Tensor:
    matrices = _rotation_matrices_from_rotation_vectors(parameters).detach().cpu().numpy()
    flip = np.diag(np.asarray(signs, dtype=float))
    vectors = np.stack(
        [rotation_vector_from_rotation_matrix(flip @ matrix) for matrix in matrices]
    )
    return torch.as_tensor(vectors, dtype=parameters.dtype, device=parameters.device)


@dataclass
class _GradientBranch:
    weights: torch.Tensor
    optimizer: torch.optim.Optimizer
    best_evaluation: _EnhancedEvaluation
    incumbent_history: list[_EnhancedEvaluation]


def _run_gradient_branch(
    initial_parameters: torch.Tensor,
    scalar_tensor: torch.Tensor,
    vector_tensor: torch.Tensor,
    observer_quaternions: torch.Tensor,
    config: RSSConfig,
    optimizer_name: str,
    *,
    learning_rates: tuple[float, ...],
    regularizer_weight: float,
    sample_rate_hz: float | None,
    ksg_sample_indices: torch.Tensor | None,
) -> _GradientBranch:
    weights = initial_parameters.detach().clone().requires_grad_(True)
    optimizer = _optimizer_for_weights(weights, config, optimizer_name)
    with torch.no_grad():
        best_evaluation = _detach_enhanced_evaluation(
            _evaluate_enhanced_single(
                weights,
                scalar_tensor,
                vector_tensor,
                observer_quaternions,
                config,
                regularizer_weight=regularizer_weight,
                sample_rate_hz=sample_rate_hz,
                ksg_sample_indices=ksg_sample_indices,
            )
        )
    best_parameters = weights.detach().clone()
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    incumbent_history: list[_EnhancedEvaluation] = []
    for learning_rate in learning_rates:
        for group in optimizer.param_groups:
            group["lr"] = float(learning_rate)
        optimizer.zero_grad()
        current = _evaluate_enhanced_single(
            weights,
            scalar_tensor,
            vector_tensor,
            observer_quaternions,
            config,
            regularizer_weight=regularizer_weight,
            sample_rate_hz=sample_rate_hz,
            ksg_sample_indices=ksg_sample_indices,
        )
        current.training_loss.backward()
        if weights.grad is None or not bool(torch.isfinite(weights.grad).all()):
            raise FloatingPointError("RSS gradient became non-finite during an axis-flip trial.")
        optimizer.step()
        with torch.no_grad():
            candidate = _detach_enhanced_evaluation(
                _evaluate_enhanced_single(
                    weights,
                    scalar_tensor,
                    vector_tensor,
                    observer_quaternions,
                    config,
                    regularizer_weight=regularizer_weight,
                    sample_rate_hz=sample_rate_hz,
                    ksg_sample_indices=ksg_sample_indices,
                )
            )
        if float(candidate.selection_loss.cpu()) < float(best_evaluation.selection_loss.cpu()):
            best_evaluation = candidate
            best_parameters = weights.detach().clone()
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        incumbent_history.append(_detach_enhanced_evaluation(best_evaluation))
    restored = best_parameters.detach().clone().requires_grad_(True)
    restored_optimizer = _optimizer_for_weights(restored, config, optimizer_name)
    restored_optimizer.load_state_dict(best_optimizer_state)
    return _GradientBranch(
        weights=restored,
        optimizer=restored_optimizer,
        best_evaluation=best_evaluation,
        incumbent_history=incumbent_history,
    )


def _fit_enhanced_gradient_optimizer(
    scalar_tensor: torch.Tensor,
    vector_tensor: torch.Tensor,
    observer_quaternions: torch.Tensor,
    config: RSSConfig,
    optimizer_name: str,
    ksg_sample_indices: torch.Tensor | None,
) -> _OptimizationOutcome:
    weights = torch.as_tensor(
        config.resolved_initial_parameters,
        dtype=vector_tensor.dtype,
        device=vector_tensor.device,
    ).clone().detach().requires_grad_(True)
    optimizer = _optimizer_for_weights(weights, config, optimizer_name)
    learning_rates = _exponential_learning_rates(
        float(config.learning_rate), float(config.learning_rate_end), int(config.epochs)
    )
    sample_rate_hz = (
        None
        if not config.adaptive_regularizer
        else 1.0 / float(config.sample_interval_seconds)
    )
    with torch.no_grad():
        initial = _detach_enhanced_evaluation(
            _evaluate_enhanced_single(
                weights,
                scalar_tensor,
                vector_tensor,
                observer_quaternions,
                config,
                regularizer_weight=float(config.regularizer_start_weight),
                sample_rate_hz=sample_rate_hz,
                ksg_sample_indices=ksg_sample_indices,
            )
        )
    initial_values = _enhanced_evaluation_values(initial)
    if not all(math.isfinite(value) for value in initial_values):
        raise FloatingPointError("Enhanced RSS loss became non-finite at epoch 0.")
    training_reference = max(initial_values[0], float(config.eps))
    selection_reference = max(initial_values[1], float(config.eps))
    scheduler = _AdaptiveRegularizerScheduler(
        config,
        base_loss=initial_values[2],
        regularizer_loss=initial_values[3],
    )
    history = [
        _enhanced_loss_record(
            epoch=0,
            evaluation=initial,
            learning_rate=None,
            training_reference=training_reference,
            selection_reference=selection_reference,
            config=config,
            regularizer_weight=scheduler.weight,
            regularizer_phase=scheduler.phase,
        )
    ]
    best_weights = weights.detach().cpu().numpy().copy()
    best_evaluation = initial
    best_epoch = 0
    flip_events: list[AxisFlipEvent] = []
    epoch = 0
    stop_reason = "max_epochs"
    minimum_threshold = float(config.regularizer_min_loss_threshold)
    if minimum_threshold > 0.0 and initial_values[1] / selection_reference <= minimum_threshold:
        stop_reason = "minimum_selection_loss"
    while epoch < int(config.epochs) and stop_reason == "max_epochs":
        run_flip = (
            config.axis_flip_search
            and epoch > 0
            and epoch % int(config.axis_flip_interval) == 0
            and epoch + int(config.axis_flip_trial_epochs) <= int(config.epochs)
        )
        if run_flip:
            check_epoch = epoch
            trial_count = int(config.axis_flip_trial_epochs)
            trial_rates = learning_rates[epoch : epoch + trial_count]
            branches: list[tuple[str, tuple[int, int, int], _GradientBranch]] = []
            for pattern, signs in _AXIS_FLIP_PATTERNS:
                branch = _run_gradient_branch(
                    _flipped_parameters(weights, signs),
                    scalar_tensor,
                    vector_tensor,
                    observer_quaternions,
                    config,
                    optimizer_name,
                    learning_rates=trial_rates,
                    regularizer_weight=scheduler.weight,
                    sample_rate_hz=sample_rate_hz,
                    ksg_sample_indices=ksg_sample_indices,
                )
                branches.append((pattern, signs, branch))
            selected_index = min(
                range(len(branches)),
                key=lambda index: (
                    float(branches[index][2].best_evaluation.selection_loss.cpu()),
                    index,
                ),
            )
            pattern, _, selected = branches[selected_index]
            weights = selected.weights
            optimizer = selected.optimizer
            for offset, evaluation in enumerate(selected.incumbent_history, start=1):
                history.append(
                    _enhanced_loss_record(
                        epoch=epoch + offset,
                        evaluation=evaluation,
                        learning_rate=trial_rates[offset - 1],
                        training_reference=training_reference,
                        selection_reference=selection_reference,
                        config=config,
                        regularizer_weight=scheduler.weight,
                        regularizer_phase=scheduler.phase,
                        flip_check_epoch=check_epoch,
                        flip_pattern=pattern,
                    )
                )
            epoch += trial_count
            selected_values = _enhanced_evaluation_values(selected.best_evaluation)
            scheduler.reset_monitor(
                base_loss=selected_values[2], regularizer_loss=selected_values[3]
            )
            if selected_values[1] < float(best_evaluation.selection_loss.cpu()):
                best_evaluation = selected.best_evaluation
                best_weights = weights.detach().cpu().numpy().copy()
                best_epoch = epoch
            candidate_results = tuple(
                AxisFlipCandidateResult(
                    pattern=candidate_pattern,
                    signs=candidate_signs,
                    best_selection_loss=float(branch.best_evaluation.selection_loss.cpu()),
                    normalized_best_selection_loss=(
                        float(branch.best_evaluation.selection_loss.cpu())
                        / selection_reference
                    ),
                    selected=index == selected_index,
                )
                for index, (candidate_pattern, candidate_signs, branch) in enumerate(branches)
            )
            history[-1]["axis_flip_candidate_selection_losses"] = ";".join(
                f"{candidate.pattern}={candidate.best_selection_loss:.17g}"
                for candidate in candidate_results
            )
            history[-1]["axis_flip_candidate_normalized_selection_losses"] = ";".join(
                f"{candidate.pattern}={candidate.normalized_best_selection_loss:.17g}"
                for candidate in candidate_results
            )
            flip_events.append(
                AxisFlipEvent(
                    check_epoch=check_epoch,
                    completed_epoch=epoch,
                    selected_pattern=pattern,
                    candidates=candidate_results,
                )
            )
            if (
                minimum_threshold > 0.0
                and float(best_evaluation.selection_loss.cpu()) / selection_reference
                <= minimum_threshold
            ):
                stop_reason = "minimum_selection_loss"
            continue

        learning_rate = learning_rates[epoch]
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad()
        current = _evaluate_enhanced_single(
            weights,
            scalar_tensor,
            vector_tensor,
            observer_quaternions,
            config,
            regularizer_weight=scheduler.weight,
            sample_rate_hz=sample_rate_hz,
            ksg_sample_indices=ksg_sample_indices,
        )
        current.training_loss.backward()
        if weights.grad is None or not bool(torch.isfinite(weights.grad).all()):
            raise FloatingPointError(f"RSS gradient became non-finite at epoch {epoch}.")
        optimizer.step()
        epoch += 1
        with torch.no_grad():
            accepted = _detach_enhanced_evaluation(
                _evaluate_enhanced_single(
                    weights,
                    scalar_tensor,
                    vector_tensor,
                    observer_quaternions,
                    config,
                    regularizer_weight=scheduler.weight,
                    sample_rate_hz=sample_rate_hz,
                    ksg_sample_indices=ksg_sample_indices,
                )
            )
        values = _enhanced_evaluation_values(accepted)
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError(f"Enhanced RSS loss became non-finite at epoch {epoch}.")
        history.append(
            _enhanced_loss_record(
                epoch=epoch,
                evaluation=accepted,
                learning_rate=learning_rate,
                training_reference=training_reference,
                selection_reference=selection_reference,
                config=config,
                regularizer_weight=scheduler.weight,
                regularizer_phase=scheduler.phase,
            )
        )
        if values[1] < float(best_evaluation.selection_loss.cpu()):
            best_evaluation = accepted
            best_weights = weights.detach().cpu().numpy().copy()
            best_epoch = epoch
        scheduler.update(base_loss=values[2], regularizer_loss=values[3])
        if (
            minimum_threshold > 0.0
            and float(best_evaluation.selection_loss.cpu()) / selection_reference
            <= minimum_threshold
        ):
            stop_reason = "minimum_selection_loss"

    initial_training, initial_selection, *_ = initial_values
    best_values = _enhanced_evaluation_values(best_evaluation)
    terminal = history[-1]
    return _OptimizationOutcome(
        history=tuple(history),
        applied_learning_rates=tuple(
            float(row["learning_rate"])
            for row in history
            if row["learning_rate"] is not None
        ),
        best_weights=best_weights,
        initial_loss=initial_training,
        best_loss=best_values[0],
        terminal_loss=float(terminal["total_loss"]),
        loss_normalization_reference=training_reference,
        normalized_initial_loss=initial_training / training_reference,
        normalized_best_loss=best_values[0] / training_reference,
        normalized_terminal_loss=float(terminal["normalized_total_loss"]),
        best_epoch=best_epoch,
        stop_reason=stop_reason,
        selection_loss_normalization_reference=selection_reference,
        initial_selection_loss=initial_selection,
        best_selection_loss=best_values[1],
        terminal_selection_loss=float(terminal["selection_loss"]),
        normalized_initial_selection_loss=initial_selection / selection_reference,
        normalized_best_selection_loss=best_values[1] / selection_reference,
        normalized_terminal_selection_loss=float(terminal["normalized_selection_loss"]),
        axis_flip_events=tuple(flip_events),
    )


@dataclass
class _PSOState:
    positions: torch.Tensor
    velocities: torch.Tensor
    personal_best_positions: torch.Tensor
    generator: torch.Generator


def _clone_generator(generator: torch.Generator, device: torch.device) -> torch.Generator:
    cloned = torch.Generator(device=device)
    cloned.set_state(generator.get_state())
    return cloned


def _enhanced_pso_step(
    state: _PSOState,
    scalar_tensor: torch.Tensor,
    vector_tensor: torch.Tensor,
    observer_quaternions: torch.Tensor,
    config: RSSConfig,
    *,
    learning_rate: float,
    regularizer_weight: float,
    sample_rate_hz: float | None,
    ksg_sample_indices: torch.Tensor | None,
) -> tuple[_PSOState, torch.Tensor, _EnhancedEvaluation, torch.Tensor, _EnhancedEvaluation]:
    with torch.no_grad():
        current = _evaluate_enhanced_batch(
            state.positions,
            scalar_tensor,
            vector_tensor,
            observer_quaternions,
            config,
            regularizer_weight=regularizer_weight,
            sample_rate_hz=sample_rate_hz,
            ksg_sample_indices=ksg_sample_indices,
        )
        personal = _evaluate_enhanced_batch(
            state.personal_best_positions,
            scalar_tensor,
            vector_tensor,
            observer_quaternions,
            config,
            regularizer_weight=regularizer_weight,
            sample_rate_hz=sample_rate_hz,
            ksg_sample_indices=ksg_sample_indices,
        )
        improved = current.training_loss < personal.training_loss
        personal_best_positions = torch.where(
            improved.unsqueeze(1), state.positions, state.personal_best_positions
        )
        personal = _evaluate_enhanced_batch(
            personal_best_positions,
            scalar_tensor,
            vector_tensor,
            observer_quaternions,
            config,
            regularizer_weight=regularizer_weight,
            sample_rate_hz=sample_rate_hz,
            ksg_sample_indices=ksg_sample_indices,
        )
        guide_index = int(torch.argmin(personal.training_loss).cpu())
        guide_position = personal_best_positions[guide_index].clone()
        random_cognitive = torch.rand(
            state.positions.shape,
            dtype=state.positions.dtype,
            device=state.positions.device,
            generator=state.generator,
        )
        random_social = torch.rand(
            state.positions.shape,
            dtype=state.positions.dtype,
            device=state.positions.device,
            generator=state.generator,
        )
        velocities = (
            float(config.pso_inertia) * state.velocities
            + float(config.pso_cognitive)
            * random_cognitive
            * (personal_best_positions - state.positions)
            + float(config.pso_social)
            * random_social
            * (guide_position.unsqueeze(0) - state.positions)
        )
        lower, upper = config.resolved_pso_bounds
        positions, velocities = _pso_position_step(
            state.positions,
            velocities,
            learning_rate,
            lower=lower,
            upper=upper,
        )
        moved = _evaluate_enhanced_batch(
            positions,
            scalar_tensor,
            vector_tensor,
            observer_quaternions,
            config,
            regularizer_weight=regularizer_weight,
            sample_rate_hz=sample_rate_hz,
            ksg_sample_indices=ksg_sample_indices,
        )
        personal_before = _evaluate_enhanced_batch(
            personal_best_positions,
            scalar_tensor,
            vector_tensor,
            observer_quaternions,
            config,
            regularizer_weight=regularizer_weight,
            sample_rate_hz=sample_rate_hz,
            ksg_sample_indices=ksg_sample_indices,
        )
        improved = moved.training_loss < personal_before.training_loss
        personal_best_positions = torch.where(
            improved.unsqueeze(1), positions, personal_best_positions
        )
        personal_after = _evaluate_enhanced_batch(
            personal_best_positions,
            scalar_tensor,
            vector_tensor,
            observer_quaternions,
            config,
            regularizer_weight=regularizer_weight,
            sample_rate_hz=sample_rate_hz,
            ksg_sample_indices=ksg_sample_indices,
        )
        guide_index = int(torch.argmin(personal_after.training_loss).cpu())
        personal_selection_index = int(torch.argmin(personal_after.selection_loss).cpu())
        moved_selection_index = int(torch.argmin(moved.selection_loss).cpu())
        if float(moved.selection_loss[moved_selection_index].cpu()) < float(
            personal_after.selection_loss[personal_selection_index].cpu()
        ):
            selection_position = positions[moved_selection_index].clone()
            selection_evaluation = _detach_enhanced_evaluation(
                _select_batch_evaluation(moved, moved_selection_index)
            )
        else:
            selection_position = personal_best_positions[personal_selection_index].clone()
            selection_evaluation = _detach_enhanced_evaluation(
                _select_batch_evaluation(personal_after, personal_selection_index)
            )
        next_state = _PSOState(
            positions=positions,
            velocities=velocities,
            personal_best_positions=personal_best_positions,
            generator=state.generator,
        )
        return (
            next_state,
            personal_best_positions[guide_index].clone(),
            _detach_enhanced_evaluation(
                _select_batch_evaluation(personal_after, guide_index)
            ),
            selection_position,
            selection_evaluation,
        )


@dataclass
class _PSOBranch:
    state: _PSOState
    best_parameters: torch.Tensor
    best_evaluation: _EnhancedEvaluation
    incumbent_history: list[_EnhancedEvaluation]


def _run_pso_branch(
    state: _PSOState,
    checkpoint_parameters: torch.Tensor,
    signs: tuple[int, int, int],
    scalar_tensor: torch.Tensor,
    vector_tensor: torch.Tensor,
    observer_quaternions: torch.Tensor,
    config: RSSConfig,
    *,
    learning_rates: tuple[float, ...],
    regularizer_weight: float,
    sample_rate_hz: float | None,
    ksg_sample_indices: torch.Tensor | None,
) -> _PSOBranch:
    positions = _flipped_parameter_batch(state.positions, signs)
    positions[0] = _flipped_parameters(checkpoint_parameters, signs)
    branch_state = _PSOState(
        positions=positions,
        velocities=torch.zeros_like(positions),
        personal_best_positions=positions.clone(),
        generator=_clone_generator(state.generator, positions.device),
    )
    with torch.no_grad():
        initial_batch = _evaluate_enhanced_batch(
            branch_state.personal_best_positions,
            scalar_tensor,
            vector_tensor,
            observer_quaternions,
            config,
            regularizer_weight=regularizer_weight,
            sample_rate_hz=sample_rate_hz,
            ksg_sample_indices=ksg_sample_indices,
        )
    initial_index = int(torch.argmin(initial_batch.selection_loss).cpu())
    best_parameters = branch_state.personal_best_positions[initial_index].clone()
    best_evaluation = _detach_enhanced_evaluation(
        _select_batch_evaluation(initial_batch, initial_index)
    )
    incumbent_history: list[_EnhancedEvaluation] = []
    for learning_rate in learning_rates:
        branch_state, _, _, selected_parameters, selected_evaluation = (
            _enhanced_pso_step(
                branch_state,
                scalar_tensor,
                vector_tensor,
                observer_quaternions,
                config,
                learning_rate=learning_rate,
                regularizer_weight=regularizer_weight,
                sample_rate_hz=sample_rate_hz,
                ksg_sample_indices=ksg_sample_indices,
            )
        )
        if float(selected_evaluation.selection_loss.cpu()) < float(
            best_evaluation.selection_loss.cpu()
        ):
            best_parameters = selected_parameters.clone()
            best_evaluation = selected_evaluation
        incumbent_history.append(_detach_enhanced_evaluation(best_evaluation))
    branch_state.positions[0] = best_parameters
    branch_state.personal_best_positions[0] = best_parameters
    branch_state.velocities[0] = 0.0
    return _PSOBranch(
        state=branch_state,
        best_parameters=best_parameters,
        best_evaluation=best_evaluation,
        incumbent_history=incumbent_history,
    )


def _fit_enhanced_particle_swarm(
    scalar_tensor: torch.Tensor,
    vector_tensor: torch.Tensor,
    observer_quaternions: torch.Tensor,
    config: RSSConfig,
    ksg_sample_indices: torch.Tensor | None,
) -> _OptimizationOutcome:
    initial_parameters = torch.as_tensor(
        config.resolved_initial_parameters,
        dtype=vector_tensor.dtype,
        device=vector_tensor.device,
    )
    sample_rate_hz = (
        None
        if not config.adaptive_regularizer
        else 1.0 / float(config.sample_interval_seconds)
    )
    with torch.no_grad():
        initial = _detach_enhanced_evaluation(
            _evaluate_enhanced_single(
                initial_parameters,
                scalar_tensor,
                vector_tensor,
                observer_quaternions,
                config,
                regularizer_weight=float(config.regularizer_start_weight),
                sample_rate_hz=sample_rate_hz,
                ksg_sample_indices=ksg_sample_indices,
            )
        )
    initial_values = _enhanced_evaluation_values(initial)
    if not all(math.isfinite(value) for value in initial_values):
        raise FloatingPointError("Enhanced RSS loss became non-finite at epoch 0.")
    training_reference = max(initial_values[0], float(config.eps))
    selection_reference = max(initial_values[1], float(config.eps))
    scheduler = _AdaptiveRegularizerScheduler(
        config,
        base_loss=initial_values[2],
        regularizer_loss=initial_values[3],
    )
    history = [
        _enhanced_loss_record(
            epoch=0,
            evaluation=initial,
            learning_rate=None,
            training_reference=training_reference,
            selection_reference=selection_reference,
            config=config,
            regularizer_weight=scheduler.weight,
            regularizer_phase=scheduler.phase,
        )
    ]
    generator = torch.Generator(device=vector_tensor.device)
    generator.manual_seed(int(config.pso_seed))
    particle_count = int(config.pso_swarm_size)
    lower, upper = config.resolved_pso_bounds
    positions = lower + (upper - lower) * torch.rand(
        (particle_count, int(initial_parameters.numel())),
        dtype=vector_tensor.dtype,
        device=vector_tensor.device,
        generator=generator,
    )
    positions[0] = initial_parameters
    state = _PSOState(
        positions=positions,
        velocities=torch.zeros_like(positions),
        personal_best_positions=positions.clone(),
        generator=generator,
    )
    learning_rates = _exponential_learning_rates(
        float(config.learning_rate), float(config.learning_rate_end), int(config.epochs)
    )
    best_parameters = initial_parameters.clone()
    best_evaluation = initial
    current_parameters = initial_parameters.clone()
    best_epoch = 0
    flip_events: list[AxisFlipEvent] = []
    epoch = 0
    stop_reason = "max_epochs"
    minimum_threshold = float(config.regularizer_min_loss_threshold)
    if minimum_threshold > 0.0 and initial_values[1] / selection_reference <= minimum_threshold:
        stop_reason = "minimum_selection_loss"
    while epoch < int(config.epochs) and stop_reason == "max_epochs":
        run_flip = (
            config.axis_flip_search
            and epoch > 0
            and epoch % int(config.axis_flip_interval) == 0
            and epoch + int(config.axis_flip_trial_epochs) <= int(config.epochs)
        )
        if run_flip:
            check_epoch = epoch
            trial_count = int(config.axis_flip_trial_epochs)
            trial_rates = learning_rates[epoch : epoch + trial_count]
            branches: list[tuple[str, tuple[int, int, int], _PSOBranch]] = []
            for pattern, signs in _AXIS_FLIP_PATTERNS:
                branch = _run_pso_branch(
                    state,
                    current_parameters,
                    signs,
                    scalar_tensor,
                    vector_tensor,
                    observer_quaternions,
                    config,
                    learning_rates=trial_rates,
                    regularizer_weight=scheduler.weight,
                    sample_rate_hz=sample_rate_hz,
                    ksg_sample_indices=ksg_sample_indices,
                )
                branches.append((pattern, signs, branch))
            selected_index = min(
                range(len(branches)),
                key=lambda index: (
                    float(branches[index][2].best_evaluation.selection_loss.cpu()),
                    index,
                ),
            )
            pattern, _, selected = branches[selected_index]
            state = selected.state
            current_parameters = selected.best_parameters.clone()
            for offset, evaluation in enumerate(selected.incumbent_history, start=1):
                history.append(
                    _enhanced_loss_record(
                        epoch=epoch + offset,
                        evaluation=evaluation,
                        learning_rate=trial_rates[offset - 1],
                        training_reference=training_reference,
                        selection_reference=selection_reference,
                        config=config,
                        regularizer_weight=scheduler.weight,
                        regularizer_phase=scheduler.phase,
                        flip_check_epoch=check_epoch,
                        flip_pattern=pattern,
                    )
                )
            epoch += trial_count
            selected_values = _enhanced_evaluation_values(selected.best_evaluation)
            scheduler.reset_monitor(
                base_loss=selected_values[2], regularizer_loss=selected_values[3]
            )
            if selected_values[1] < float(best_evaluation.selection_loss.cpu()):
                best_evaluation = selected.best_evaluation
                best_parameters = selected.best_parameters.clone()
                best_epoch = epoch
            candidate_results = tuple(
                AxisFlipCandidateResult(
                    pattern=candidate_pattern,
                    signs=candidate_signs,
                    best_selection_loss=float(
                        branch.best_evaluation.selection_loss.cpu()
                    ),
                    normalized_best_selection_loss=float(
                        branch.best_evaluation.selection_loss.cpu()
                    )
                    / selection_reference,
                    selected=index == selected_index,
                )
                for index, (candidate_pattern, candidate_signs, branch) in enumerate(
                    branches
                )
            )
            history[-1]["axis_flip_candidate_selection_losses"] = ";".join(
                f"{candidate.pattern}={candidate.best_selection_loss:.17g}"
                for candidate in candidate_results
            )
            history[-1]["axis_flip_candidate_normalized_selection_losses"] = ";".join(
                f"{candidate.pattern}={candidate.normalized_best_selection_loss:.17g}"
                for candidate in candidate_results
            )
            flip_events.append(
                AxisFlipEvent(
                    check_epoch=check_epoch,
                    completed_epoch=epoch,
                    selected_pattern=pattern,
                    candidates=candidate_results,
                )
            )
            if (
                minimum_threshold > 0.0
                and float(best_evaluation.selection_loss.cpu()) / selection_reference
                <= minimum_threshold
            ):
                stop_reason = "minimum_selection_loss"
            continue

        learning_rate = learning_rates[epoch]
        state, current_parameters, accepted, selected_parameters, selected_evaluation = (
            _enhanced_pso_step(
                state,
                scalar_tensor,
                vector_tensor,
                observer_quaternions,
                config,
                learning_rate=learning_rate,
                regularizer_weight=scheduler.weight,
                sample_rate_hz=sample_rate_hz,
                ksg_sample_indices=ksg_sample_indices,
            )
        )
        epoch += 1
        values = _enhanced_evaluation_values(accepted)
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError(f"Enhanced RSS particle loss became non-finite at epoch {epoch}.")
        history.append(
            _enhanced_loss_record(
                epoch=epoch,
                evaluation=accepted,
                learning_rate=learning_rate,
                training_reference=training_reference,
                selection_reference=selection_reference,
                config=config,
                regularizer_weight=scheduler.weight,
                regularizer_phase=scheduler.phase,
            )
        )
        if float(selected_evaluation.selection_loss.cpu()) < float(
            best_evaluation.selection_loss.cpu()
        ):
            best_evaluation = selected_evaluation
            best_parameters = selected_parameters.clone()
            best_epoch = epoch
        scheduler.update(base_loss=values[2], regularizer_loss=values[3])
        if (
            minimum_threshold > 0.0
            and float(best_evaluation.selection_loss.cpu()) / selection_reference
            <= minimum_threshold
        ):
            stop_reason = "minimum_selection_loss"

    best_values = _enhanced_evaluation_values(best_evaluation)
    terminal = history[-1]
    return _OptimizationOutcome(
        history=tuple(history),
        applied_learning_rates=tuple(
            float(row["learning_rate"])
            for row in history
            if row["learning_rate"] is not None
        ),
        best_weights=best_parameters.detach().cpu().numpy().copy(),
        initial_loss=initial_values[0],
        best_loss=best_values[0],
        terminal_loss=float(terminal["total_loss"]),
        loss_normalization_reference=training_reference,
        normalized_initial_loss=initial_values[0] / training_reference,
        normalized_best_loss=best_values[0] / training_reference,
        normalized_terminal_loss=float(terminal["normalized_total_loss"]),
        best_epoch=best_epoch,
        stop_reason=stop_reason,
        selection_loss_normalization_reference=selection_reference,
        initial_selection_loss=initial_values[1],
        best_selection_loss=best_values[1],
        terminal_selection_loss=float(terminal["selection_loss"]),
        normalized_initial_selection_loss=initial_values[1] / selection_reference,
        normalized_best_selection_loss=best_values[1] / selection_reference,
        normalized_terminal_selection_loss=float(terminal["normalized_selection_loss"]),
        axis_flip_events=tuple(flip_events),
    )


def _fit_gradient_optimizer(
    scalar_tensor: torch.Tensor,
    vector_tensor: torch.Tensor,
    observer_quaternions: torch.Tensor,
    config: RSSConfig,
    optimizer_name: str,
) -> _OptimizationOutcome:
    weights = torch.tensor(
        config.resolved_initial_parameters,
        dtype=vector_tensor.dtype,
        device=vector_tensor.device,
        requires_grad=True,
    )
    if optimizer_name == "adam":
        torch_optimizer = torch.optim.Adam((weights,), lr=float(config.learning_rate))
    else:
        torch_optimizer = torch.optim.SGD(
            (weights,),
            lr=float(config.learning_rate),
            momentum=float(config.sgd_momentum),
        )

    learning_rates = _exponential_learning_rates(
        float(config.learning_rate),
        float(config.learning_rate_end),
        int(config.epochs),
    )
    last_applied_learning_rate: float | None = None
    history: list[dict[str, float | str | None]] = []
    best_loss = float("inf")
    best_normalized_loss = float("inf")
    best_epoch = 0
    best_weights: np.ndarray | None = None
    terminal_loss = float("nan")
    terminal_normalized_loss = float("nan")
    loss_normalization_reference: float | None = None
    for epoch in range(int(config.epochs) + 1):
        raw_total, terms = _evaluate_single_candidate(
            weights, scalar_tensor, vector_tensor, observer_quaternions, config
        )
        raw_total_value = float(raw_total.detach().cpu())
        if not math.isfinite(raw_total_value):
            raise FloatingPointError(f"RSS loss became non-finite at epoch {epoch}.")
        if loss_normalization_reference is None:
            loss_normalization_reference = max(raw_total_value, float(config.eps))
        normalized_terms = {
            "normalized_separation_loss": float(config.separation_weight)
            * terms["separation_loss"]
            / loss_normalization_reference,
            "normalized_orthogonality_loss": _effective_orthogonality_weight(config)
            * terms["orthogonality_loss"]
            / loss_normalization_reference,
            "normalized_reconstruction_loss": float(config.reconstruction_weight)
            * terms["reconstruction_loss"]
            / loss_normalization_reference,
        }
        normalized_total = sum(normalized_terms.values())
        normalized_total_value = float(normalized_total.detach().cpu())
        if not math.isfinite(normalized_total_value):
            raise FloatingPointError(f"Normalized RSS loss became non-finite at epoch {epoch}.")
        normalized_values = {
            name: float(value.detach().cpu()) for name, value in normalized_terms.items()
        }
        terminal_loss = raw_total_value
        terminal_normalized_loss = normalized_total_value
        history.append(
            {
                "epoch": float(epoch),
                "learning_rate": last_applied_learning_rate,
                "normalized_total_loss": normalized_total_value,
                **normalized_values,
                "loss_type": config.loss_type,
                "total_loss": raw_total_value,
                "separation_loss": float(terms["separation_loss"].detach().cpu()),
                "orthogonality_loss": float(terms["orthogonality_loss"].detach().cpu()),
                "reconstruction_loss": float(terms["reconstruction_loss"].detach().cpu()),
            }
        )
        if normalized_total_value < best_normalized_loss:
            best_normalized_loss = normalized_total_value
            best_loss = raw_total_value
            best_epoch = epoch
            best_weights = weights.detach().cpu().numpy().copy()
        if epoch == int(config.epochs):
            break
        update_learning_rate = learning_rates[epoch]
        for parameter_group in torch_optimizer.param_groups:
            parameter_group["lr"] = update_learning_rate
        torch_optimizer.zero_grad()
        normalized_total.backward()
        if weights.grad is None or not bool(torch.isfinite(weights.grad).all()):
            raise FloatingPointError(f"RSS gradient became non-finite at epoch {epoch}.")
        torch_optimizer.step()
        last_applied_learning_rate = update_learning_rate

    if best_weights is None or loss_normalization_reference is None:
        raise RuntimeError("RSS optimization did not produce a candidate solution.")
    return _OptimizationOutcome(
        history=tuple(history),
        applied_learning_rates=tuple(
            float(value) for value in (row["learning_rate"] for row in history) if value is not None
        ),
        best_weights=best_weights,
        initial_loss=float(history[0]["total_loss"]),
        best_loss=float(best_loss),
        terminal_loss=float(terminal_loss),
        loss_normalization_reference=float(loss_normalization_reference),
        normalized_initial_loss=float(history[0]["normalized_total_loss"]),
        normalized_best_loss=float(best_normalized_loss),
        normalized_terminal_loss=float(terminal_normalized_loss),
        best_epoch=int(best_epoch),
        stop_reason="max_epochs",
    )


def _fit_particle_swarm(
    scalar_tensor: torch.Tensor,
    vector_tensor: torch.Tensor,
    observer_quaternions: torch.Tensor,
    config: RSSConfig,
    ksg_sample_indices: torch.Tensor | None,
) -> _OptimizationOutcome:
    initial_weights = torch.tensor(
        config.resolved_initial_parameters,
        dtype=vector_tensor.dtype,
        device=vector_tensor.device,
    )
    with torch.no_grad():
        initial_total, initial_terms = _evaluate_single_candidate(
            initial_weights,
            scalar_tensor,
            vector_tensor,
            observer_quaternions,
            config,
            ksg_sample_indices,
        )
    initial_loss = float(initial_total.detach().cpu())
    if not math.isfinite(initial_loss):
        raise FloatingPointError("RSS loss became non-finite at epoch 0.")
    reference = max(initial_loss, float(config.eps))
    initial_components = {
        name: float(value.detach().cpu()) for name, value in initial_terms.items()
    }
    history = [
        _loss_record(
            epoch=0,
            raw_total=initial_loss,
            separation_loss=initial_components["separation_loss"],
            orthogonality_loss=initial_components["orthogonality_loss"],
            reconstruction_loss=initial_components["reconstruction_loss"],
            learning_rate=None,
            reference=reference,
            config=config,
        )
    ]

    generator = torch.Generator(device=vector_tensor.device)
    generator.manual_seed(int(config.pso_seed))
    particle_count = int(config.pso_swarm_size)
    lower, upper = config.resolved_pso_bounds
    parameter_count = int(initial_weights.numel())
    positions = lower + (upper - lower) * torch.rand(
        (particle_count, parameter_count),
        dtype=vector_tensor.dtype,
        device=vector_tensor.device,
        generator=generator,
    )
    positions[0] = initial_weights
    velocities = torch.zeros_like(positions)
    personal_best_positions = positions.clone()
    personal_best_losses = torch.full(
        (particle_count,),
        float("inf"),
        dtype=vector_tensor.dtype,
        device=vector_tensor.device,
    )

    global_best_weights = initial_weights.clone()
    global_best_loss = initial_loss
    global_best_normalized_loss = float(history[0]["normalized_total_loss"])
    global_best_components = initial_components.copy()
    best_epoch = 0
    movement_learning_rates = _exponential_learning_rates(
        float(config.learning_rate),
        float(config.learning_rate_end),
        max(int(config.epochs) - 1, 0),
    )
    last_applied_learning_rate: float | None = None

    for epoch in range(1, int(config.epochs) + 1):
        with torch.no_grad():
            recovery_matrices = _matrices_from_parameters(
                positions,
                config.matrix_parameterization,
            )
            source_vectors = torch.einsum("pij,jt->pit", recovery_matrices, vector_tensor)
            source_angles = _decode_scalar_and_vector_batch(
                scalar_tensor,
                source_vectors,
                eps=float(config.quaternion_eps),
            )
            raw_totals, terms = _compute_rss_loss_batch(
                source_angles,
                tuple(int(lag) for lag in config.lags),
                recovery_matrices=recovery_matrices,
                loss_type=config.loss_type,
                separation_weight=float(config.separation_weight),
                orthogonality_weight=_effective_orthogonality_weight(config),
                eps=float(config.eps),
                observer_quaternions=observer_quaternions,
                reconstruction_weight=float(config.reconstruction_weight),
                mi_estimator=config.mi_estimator,
                gcmi_rank_method=config.gcmi_rank_method,
                bspline_bins=int(config.bspline_bins),
                bspline_order=int(config.bspline_order),
                ksg_k=int(config.ksg_k),
                ksg_standardize=bool(config.ksg_standardize),
                ksg_max_samples=config.ksg_max_samples,
                ksg_subsample_mode=config.ksg_subsample_mode,
                ksg_subsample_seed=int(config.ksg_subsample_seed),
                ksg_sample_indices=ksg_sample_indices,
                mi_bins=int(config.mi_bins),
                mi_bandwidth=float(config.mi_bandwidth),
                action_inertias=config.action_inertias,
                action_stiffnesses=config.action_stiffnesses,
                sample_interval_seconds=config.sample_interval_seconds,
            )
        if not bool(torch.isfinite(raw_totals).all()):
            raise FloatingPointError(f"RSS particle loss became non-finite at epoch {epoch}.")

        improved_personal = raw_totals < personal_best_losses
        personal_best_losses = torch.where(improved_personal, raw_totals, personal_best_losses)
        personal_best_positions = torch.where(
            improved_personal.unsqueeze(1),
            positions,
            personal_best_positions,
        )
        current_index = int(torch.argmin(raw_totals).detach().cpu())
        current_loss = float(raw_totals[current_index].detach().cpu())
        if current_loss < global_best_loss:
            global_best_loss = current_loss
            global_best_weights = positions[current_index].clone()
            global_best_components = {
                name: float(values[current_index].detach().cpu()) for name, values in terms.items()
            }
            global_best_normalized_loss = global_best_loss / reference
            best_epoch = epoch

        history.append(
            _loss_record(
                epoch=epoch,
                raw_total=global_best_loss,
                separation_loss=global_best_components["separation_loss"],
                orthogonality_loss=global_best_components["orthogonality_loss"],
                reconstruction_loss=global_best_components["reconstruction_loss"],
                learning_rate=last_applied_learning_rate,
                reference=reference,
                config=config,
            )
        )
        if epoch == int(config.epochs):
            break

        random_cognitive = torch.rand(
            positions.shape,
            dtype=positions.dtype,
            device=positions.device,
            generator=generator,
        )
        random_social = torch.rand(
            positions.shape,
            dtype=positions.dtype,
            device=positions.device,
            generator=generator,
        )
        velocities = (
            float(config.pso_inertia) * velocities
            + float(config.pso_cognitive) * random_cognitive * (personal_best_positions - positions)
            + float(config.pso_social) * random_social * (global_best_weights.unsqueeze(0) - positions)
        )
        movement_learning_rate = movement_learning_rates[epoch - 1]
        positions, velocities = _pso_position_step(
            positions,
            velocities,
            movement_learning_rate,
            lower=lower,
            upper=upper,
        )
        last_applied_learning_rate = movement_learning_rate

    terminal = history[-1]
    return _OptimizationOutcome(
        history=tuple(history),
        applied_learning_rates=tuple(
            float(value) for value in (row["learning_rate"] for row in history) if value is not None
        ),
        best_weights=global_best_weights.detach().cpu().numpy().copy(),
        initial_loss=initial_loss,
        best_loss=float(global_best_loss),
        terminal_loss=float(terminal["total_loss"]),
        loss_normalization_reference=reference,
        normalized_initial_loss=float(history[0]["normalized_total_loss"]),
        normalized_best_loss=float(global_best_normalized_loss),
        normalized_terminal_loss=float(terminal["normalized_total_loss"]),
        best_epoch=int(best_epoch),
        stop_reason="max_epochs",
    )


def fit_rss(
    observer_angles: np.ndarray,
    config: RSSConfig | None = None,
    *,
    sample_interval_seconds: float | None = None,
) -> RSSResult:
    """Fit RSS to three observer-angle signals with shape ``(3, T)`` or ``(T, 3)``."""

    active_config = config or RSSConfig()
    context_interval = None
    if sample_interval_seconds is not None:
        context_interval = float(sample_interval_seconds)
        if not math.isfinite(context_interval) or context_interval <= 0.0:
            raise ValueError("sample_interval_seconds must be finite and > 0 when provided.")
    resolved_interval = (
        active_config.sample_interval_seconds
        if active_config.sample_interval_seconds is not None
        else context_interval
    )
    needs_interval = active_config.loss_type == "action" or active_config.adaptive_regularizer
    if (
        needs_interval
        and resolved_interval is not None
        and active_config.sample_interval_seconds != resolved_interval
    ):
        active_config = replace(active_config, sample_interval_seconds=resolved_interval)
    if needs_interval and resolved_interval is None:
        raise ValueError(
            "sample_interval_seconds is required for action loss and adaptive specialized "
            "regularization; set it in RSSConfig or provide it through the "
            "simulation/real-data context."
        )
    angles = np.asarray(observer_angles, dtype=float)
    if angles.ndim != 2:
        raise ValueError("observer_angles must be a 2D array.")
    if angles.shape[0] != 3 and angles.shape[1] == 3:
        angles = angles.T
    if angles.shape[0] != 3:
        raise ValueError("observer_angles must have shape (3, T) or (T, 3).")
    if not np.isfinite(angles).all():
        raise ValueError("observer_angles must contain only finite values.")
    if active_config.adaptive_regularizer and int(angles.shape[1]) < 4:
        raise ValueError("Adaptive specialized regularization requires at least four samples.")
    ksg_indices_array: np.ndarray | None = None
    if (
        active_config.loss_type == "mutual_information"
        and active_config.mi_estimator == "ksg"
    ):
        ksg_indices_array = select_ksg_subsample_indices(
            int(angles.shape[1]),
            max_samples=active_config.ksg_max_samples,
            mode=active_config.ksg_subsample_mode,
            seed=int(active_config.ksg_subsample_seed),
        )
        if int(ksg_indices_array.size) <= int(active_config.ksg_k):
            raise ValueError("ksg_k must be smaller than the effective RSS sample count.")
    if (
        active_config.loss_type == "lagged_cross_correlation"
        and angles.shape[1] <= max(active_config.lags)
    ):
        raise ValueError("Every RSS lag must be smaller than the sample count.")

    if str(active_config.observer_signal_model).strip().lower() == "integrated_body_rotation":
        observer_quaternions = integrated_body_angles_to_quaternions(
            angles,
            eps=float(active_config.quaternion_eps),
        )
        observer_scalar = observer_quaternions[:, 0]
        observer_vector = observer_quaternions[:, 1:].T
    else:
        observer_scalar, observer_vector = angles_to_quaternion_components(angles)
        _, observer_quaternions = decode_scalar_and_vector(
            observer_scalar,
            observer_vector,
            eps=float(active_config.quaternion_eps),
        )
    device = _resolve_device(active_config.device)
    dtype = torch.float64
    scalar_tensor = torch.tensor(observer_scalar, dtype=dtype, device=device)
    vector_tensor = torch.tensor(observer_vector, dtype=dtype, device=device)
    observer_quaternion_tensor = torch.tensor(observer_quaternions, dtype=dtype, device=device)
    ksg_sample_indices = (
        torch.as_tensor(ksg_indices_array, dtype=torch.long, device=device)
        if ksg_indices_array is not None and int(ksg_indices_array.size) < int(angles.shape[1])
        else None
    )
    optimizer_name = str(active_config.optimizer).strip().lower()
    enhanced = active_config.axis_flip_search or active_config.adaptive_regularizer
    if optimizer_name == "pso" and enhanced:
        outcome = _fit_enhanced_particle_swarm(
            scalar_tensor,
            vector_tensor,
            observer_quaternion_tensor,
            active_config,
            ksg_sample_indices,
        )
    elif optimizer_name == "pso":
        outcome = _fit_particle_swarm(
            scalar_tensor,
            vector_tensor,
            observer_quaternion_tensor,
            active_config,
            ksg_sample_indices,
        )
    elif enhanced:
        outcome = _fit_enhanced_gradient_optimizer(
            scalar_tensor,
            vector_tensor,
            observer_quaternion_tensor,
            active_config,
            optimizer_name,
            ksg_sample_indices,
        )
    else:
        outcome = _fit_gradient_optimizer(
            scalar_tensor,
            vector_tensor,
            observer_quaternion_tensor,
            active_config,
            optimizer_name,
        )

    recovery_matrix = np.asarray(
        _matrix_from_parameters(outcome.best_weights, active_config.matrix_parameterization),
        dtype=float,
    )
    source_coefficients = recovery_matrix @ observer_vector
    source_angles, source_quaternions = decode_scalar_and_vector(
        observer_scalar,
        source_coefficients,
        eps=float(active_config.quaternion_eps),
    )
    composed_source_quaternions = compose_source_quaternions(
        source_angles,
        recovery_matrix,
        eps=float(active_config.quaternion_eps),
    )
    relative_quaternions = relative_reconstruction_quaternions(
        composed_source_quaternions,
        observer_quaternions,
        eps=float(active_config.quaternion_eps),
    )
    if str(active_config.observer_signal_model).strip().lower() == "integrated_body_rotation":
        reconstructed_observer_angles = quaternions_to_integrated_body_angles(
            composed_source_quaternions,
            eps=float(active_config.quaternion_eps),
        )
    else:
        reconstructed_observer_angles = quaternions_to_angles(composed_source_quaternions)
    reconstructed_observer_coefficients = composed_source_quaternions[:, 1:].T
    reconstruction_error_angles = quaternion_error_angles(
        relative_quaternions,
        eps=float(active_config.quaternion_eps),
    )
    return RSSResult(
        config=active_config,
        optimizer_name=optimizer_name,
        observer_angles=angles,
        observer_scalar=observer_scalar,
        observer_quaternions=observer_quaternions,
        observer_coefficients=observer_vector,
        source_angles=source_angles,
        source_quaternions=source_quaternions,
        source_coefficients=source_coefficients,
        composed_source_quaternions=composed_source_quaternions,
        relative_reconstruction_quaternions=relative_quaternions,
        reconstructed_observer_angles=reconstructed_observer_angles,
        reconstructed_observer_coefficients=reconstructed_observer_coefficients,
        reconstruction_mean_error_rad=float(np.mean(reconstruction_error_angles)),
        reconstruction_max_error_rad=float(np.max(reconstruction_error_angles)),
        recovery_matrix=recovery_matrix,
        optimized_parameters=outcome.best_weights,
        optimized_weights=outcome.best_weights,
        loss_history=outcome.history,
        sample_interval_seconds=resolved_interval,
        applied_learning_rates=outcome.applied_learning_rates,
        initial_loss=outcome.initial_loss,
        best_loss=outcome.best_loss,
        terminal_loss=outcome.terminal_loss,
        loss_normalization_reference=outcome.loss_normalization_reference,
        normalized_initial_loss=outcome.normalized_initial_loss,
        normalized_best_loss=outcome.normalized_best_loss,
        normalized_terminal_loss=outcome.normalized_terminal_loss,
        best_epoch=outcome.best_epoch,
        stop_reason=outcome.stop_reason,
        matrix_diagnostics=_matrix_diagnostics(recovery_matrix, eps=float(active_config.eps)),
        selection_loss_normalization_reference=outcome.selection_loss_normalization_reference,
        initial_selection_loss=outcome.initial_selection_loss,
        best_selection_loss=outcome.best_selection_loss,
        terminal_selection_loss=outcome.terminal_selection_loss,
        normalized_initial_selection_loss=outcome.normalized_initial_selection_loss,
        normalized_best_selection_loss=outcome.normalized_best_selection_loss,
        normalized_terminal_selection_loss=outcome.normalized_terminal_selection_loss,
        axis_flip_events=outcome.axis_flip_events,
    )
