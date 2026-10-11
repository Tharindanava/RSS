"""Mutual-information estimators used by Rotation Source Separation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import math

import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial import cKDTree
from scipy.special import digamma, ndtri
from scipy.stats import rankdata


MI_ESTIMATORS: tuple[str, ...] = ("gcmi", "bspline", "ksg", "histogram")
MI_UNIT = "nats"
GCMI_RANK_METHODS: tuple[str, ...] = ("average", "min", "max", "dense", "ordinal")
KSG_SUBSAMPLE_MODES: tuple[str, ...] = ("fixed_random",)
DEFAULT_GCMI_RANK_METHOD = "average"
DEFAULT_BSPLINE_BINS = 6
DEFAULT_BSPLINE_ORDER = 3
DEFAULT_KSG_K = 4
DEFAULT_KSG_STANDARDIZE = True
DEFAULT_KSG_MAX_SAMPLES: int | None = None
DEFAULT_KSG_SUBSAMPLE_MODE = "fixed_random"
DEFAULT_KSG_SUBSAMPLE_SEED = 0
DEFAULT_KSG_TIE_JITTER = 1e-10
DEFAULT_KSG_JITTER_SEED = 0


@dataclass(frozen=True)
class _PreparedKSGVariable:
    samples: np.ndarray
    tree: cKDTree


def _as_sample_matrix(values: object, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or int(array.shape[1]) < 1:
        raise ValueError(f"{name} must have shape (N,) or (N, d).")
    if int(array.shape[0]) < 1:
        raise ValueError(f"{name} must contain at least one sample.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return np.array(array, dtype=float, copy=True, order="C")


def _validate_k(k: int, sample_count: int) -> int:
    neighbour_count = int(k)
    if neighbour_count != k or neighbour_count < 1:
        raise ValueError("k must be an integer >= 1.")
    if neighbour_count >= int(sample_count):
        raise ValueError("k must be smaller than the paired sample count.")
    return neighbour_count


def _validate_positive_integer(value: object, *, name: str) -> int:
    integer = int(value)
    if integer != value or integer < 1:
        raise ValueError(f"{name} must be an integer >= 1.")
    return integer


def _validate_epsilon(eps: float) -> float:
    epsilon = float(eps)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("eps must be finite and > 0.")
    return epsilon


def _normalize_gcmi_rank_method(rank_method: str) -> str:
    selected = str(rank_method).strip().lower()
    if selected not in GCMI_RANK_METHODS:
        raise ValueError(
            f"gcmi_rank_method must be one of: {', '.join(GCMI_RANK_METHODS)}."
        )
    return selected


def _normalize_ksg_subsample_mode(mode: str) -> str:
    selected = str(mode).strip().lower()
    if selected not in KSG_SUBSAMPLE_MODES:
        raise ValueError(
            f"ksg_subsample_mode must be one of: {', '.join(KSG_SUBSAMPLE_MODES)}."
        )
    return selected


def select_ksg_subsample_indices(
    sample_count: int,
    *,
    max_samples: int | None = DEFAULT_KSG_MAX_SAMPLES,
    mode: str = DEFAULT_KSG_SUBSAMPLE_MODE,
    seed: int = DEFAULT_KSG_SUBSAMPLE_SEED,
) -> np.ndarray:
    """Return the sorted, deterministic paired-observation indices for KSG."""

    count = _validate_positive_integer(sample_count, name="sample_count")
    _normalize_ksg_subsample_mode(mode)
    random_seed = int(seed)
    if random_seed != seed or random_seed < 0:
        raise ValueError("ksg_subsample_seed must be an integer >= 0.")
    if max_samples is None:
        return np.arange(count, dtype=np.int64)
    limit = _validate_positive_integer(max_samples, name="ksg_max_samples")
    if limit >= count:
        return np.arange(count, dtype=np.int64)
    generator = np.random.default_rng(random_seed)
    indices = np.asarray(
        generator.choice(count, size=limit, replace=False),
        dtype=np.int64,
    )
    indices.sort()
    return indices


def _validate_sample_indices(indices: object, sample_count: int) -> np.ndarray:
    values = np.asarray(indices)
    if values.ndim != 1 or int(values.size) < 1:
        raise ValueError("ksg_sample_indices must be a nonempty one-dimensional array.")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("ksg_sample_indices must contain integers.")
    normalized = np.asarray(values, dtype=np.int64)
    if np.any(normalized < 0) or np.any(normalized >= int(sample_count)):
        raise ValueError("ksg_sample_indices contain an out-of-range index.")
    if np.unique(normalized).size != normalized.size:
        raise ValueError("ksg_sample_indices must not contain duplicates.")
    return np.sort(normalized)


def _scale_aware_near_constant(values: np.ndarray, eps: float) -> np.ndarray:
    ranges = np.ptp(values, axis=-1)
    scales = np.maximum(1.0, np.max(np.abs(values), axis=-1))
    return ranges <= float(eps) * scales


def estimate_pairwise_gcmi_mutual_information(
    variables: object,
    pairs: Sequence[tuple[int, int]],
    *,
    rank_method: str = DEFAULT_GCMI_RANK_METHOD,
    eps: float = 1e-12,
) -> np.ndarray:
    """Estimate scalar Gaussian-copula MI for arrays shaped ``(..., V, N)``."""

    values = np.asarray(variables, dtype=float)
    if values.ndim < 2:
        raise ValueError("variables must have shape (..., V, N).")
    variable_count = int(values.shape[-2])
    sample_count = int(values.shape[-1])
    if variable_count < 2 or sample_count < 1:
        raise ValueError("variables must contain at least two variables and one sample.")
    if not np.isfinite(values).all():
        raise ValueError("variables must contain only finite values.")
    selected_rank_method = _normalize_gcmi_rank_method(rank_method)
    epsilon = _validate_epsilon(eps)
    normalized_pairs = tuple((int(first), int(second)) for first, second in pairs)
    if not normalized_pairs:
        raise ValueError("pairs must contain at least one variable pair.")
    if any(
        first < 0
        or second < 0
        or first >= variable_count
        or second >= variable_count
        or first == second
        for first, second in normalized_pairs
    ):
        raise ValueError("Each pair must contain two distinct valid variable indices.")

    constant = _scale_aware_near_constant(values, epsilon)
    ranks = rankdata(values, method=selected_rank_method, axis=-1)
    probabilities = (ranks - 0.5) / float(sample_count)
    gaussianized = ndtri(probabilities)
    centered = gaussianized - np.mean(gaussianized, axis=-1, keepdims=True)
    norms = np.linalg.norm(centered, axis=-1)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    unit_values = centered / safe_norms[..., None]
    estimates = []
    maximum_rho_squared = np.nextafter(1.0, 0.0)
    for first, second in normalized_pairs:
        correlation = np.sum(
            unit_values[..., first, :] * unit_values[..., second, :],
            axis=-1,
        )
        rho_squared = np.minimum(correlation * correlation, maximum_rho_squared)
        estimate = -0.5 * np.log1p(-rho_squared)
        estimate = np.where(constant[..., first] | constant[..., second], 0.0, estimate)
        estimates.append(estimate)
    result = np.stack(estimates, axis=-1)
    if not np.isfinite(result).all():
        raise FloatingPointError("GCMI estimate became non-finite.")
    return np.asarray(result, dtype=float)


def _validate_bspline_parameters(bins: int, order: int) -> tuple[int, int]:
    bin_count = _validate_positive_integer(bins, name="bspline_bins")
    spline_order = _validate_positive_integer(order, name="bspline_order")
    if spline_order >= bin_count:
        raise ValueError("bspline_order must be smaller than bspline_bins.")
    return bin_count, spline_order


def _bspline_knots(bins: int, order: int) -> np.ndarray:
    interior_count = int(bins) - int(order)
    interior = np.linspace(0.0, 1.0, interior_count + 2, dtype=float)[1:-1]
    return np.concatenate(
        (
            np.zeros(int(order), dtype=float),
            interior,
            np.ones(int(order), dtype=float),
        )
    )


def _bspline_basis_memberships(
    values: np.ndarray,
    *,
    bins: int,
    order: int,
    eps: float,
) -> np.ndarray:
    bin_count, spline_order = _validate_bspline_parameters(bins, order)
    epsilon = _validate_epsilon(eps)
    samples = np.asarray(values, dtype=float).reshape(-1)
    minimum = float(np.min(samples))
    maximum = float(np.max(samples))
    value_range = maximum - minimum
    scale = max(1.0, float(np.max(np.abs(samples))))
    if value_range <= epsilon * scale:
        normalized = np.zeros_like(samples)
    else:
        normalized = np.clip((samples - minimum) / value_range, 0.0, 1.0)
    degree = spline_order - 1
    memberships = BSpline.design_matrix(
        normalized,
        _bspline_knots(bin_count, spline_order),
        degree,
        extrapolate=False,
    ).toarray()
    memberships = np.maximum(np.asarray(memberships, dtype=float), 0.0)
    row_sums = np.sum(memberships, axis=1, keepdims=True)
    if np.any(row_sums <= 0.0) or not np.isfinite(memberships).all():
        raise FloatingPointError("B-spline memberships failed to form a finite partition.")
    return memberships / row_sums


def _estimate_bspline_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bins: int,
    order: int,
    eps: float,
) -> float:
    if x.shape[1] != 1 or y.shape[1] != 1:
        raise ValueError("The B-spline estimator supports scalar variables only.")
    epsilon = _validate_epsilon(eps)
    first_memberships = _bspline_basis_memberships(
        x[:, 0], bins=bins, order=order, eps=epsilon
    )
    second_memberships = _bspline_basis_memberships(
        y[:, 0], bins=bins, order=order, eps=epsilon
    )
    sample_count = float(x.shape[0])
    joint = first_memberships.T @ second_memberships / sample_count
    first_marginal = np.mean(first_memberships, axis=0)
    second_marginal = np.mean(second_memberships, axis=0)
    denominator = first_marginal[:, None] * second_marginal[None, :]
    positive = joint > 0.0
    estimate = np.sum(
        joint[positive]
        * (
            np.log(joint[positive])
            - np.log(denominator[positive])
        )
    )
    result = max(float(estimate), 0.0)
    if not math.isfinite(result):
        raise FloatingPointError("B-spline mutual-information estimate became non-finite.")
    return result


def _standardize_columns(samples: np.ndarray) -> np.ndarray:
    centered = samples - np.mean(samples, axis=0, keepdims=True)
    standard_deviations = np.std(centered, axis=0, keepdims=True)
    safe_scales = np.where(standard_deviations > 0.0, standard_deviations, 1.0)
    return centered / safe_scales


def _jitter_digest(samples: np.ndarray) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.asarray(samples.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(samples).view(np.uint8))
    return digest.digest()


def _jitter_seed_sequence(seed: int, digest: bytes, occurrence: int) -> np.random.SeedSequence:
    digest_words = np.frombuffer(digest, dtype=np.uint32).astype(np.uint64)
    entropy = [int(seed), int(occurrence), *(int(value) for value in digest_words)]
    return np.random.SeedSequence(entropy)


def _prepare_ksg_variables(
    variables: Sequence[np.ndarray],
    *,
    standardize: bool,
    tie_jitter: float,
    jitter_seed: int,
) -> tuple[_PreparedKSGVariable, ...]:
    if not isinstance(standardize, (bool, np.bool_)):
        raise ValueError("standardize must be a boolean.")
    jitter = float(tie_jitter)
    if not math.isfinite(jitter) or jitter <= 0.0:
        raise ValueError("tie_jitter must be finite and > 0.")

    matrices = [
        _standardize_columns(values) if bool(standardize) else np.array(values, copy=True)
        for values in variables
    ]
    digest_occurrences: dict[bytes, int] = {}
    prepared: list[_PreparedKSGVariable] = []
    for samples in matrices:
        if np.unique(samples, axis=0).shape[0] < samples.shape[0]:
            digest = _jitter_digest(samples)
            occurrence = digest_occurrences.get(digest, 0)
            digest_occurrences[digest] = occurrence + 1
            generator = np.random.default_rng(
                _jitter_seed_sequence(int(jitter_seed), digest, occurrence)
            )
            samples = samples + generator.uniform(-jitter, jitter, size=samples.shape)
        prepared.append(_PreparedKSGVariable(samples=samples, tree=cKDTree(samples)))
    return tuple(prepared)


def _estimate_prepared_ksg_pair(
    first: _PreparedKSGVariable,
    second: _PreparedKSGVariable,
    *,
    k: int,
) -> float:
    sample_count = int(first.samples.shape[0])
    joint_samples = np.concatenate((first.samples, second.samples), axis=1)
    joint_tree = cKDTree(joint_samples)
    distances = joint_tree.query(
        joint_samples,
        k=int(k) + 1,
        p=np.inf,
        workers=1,
    )[0]
    radii = np.asarray(distances[:, int(k)], dtype=float)
    if not np.isfinite(radii).all() or np.any(radii <= 0.0):
        raise FloatingPointError(
            "KSG joint-neighbour radii must be finite and positive after tie breaking."
        )

    # KSG-1 uses strictly fewer-than-radius marginal counts. cKDTree radius
    # queries include the boundary, so move each positive radius one float
    # toward zero before querying and subtract the point itself.
    strict_radii = np.nextafter(radii, 0.0)
    first_counts = np.asarray(
        first.tree.query_ball_point(
            first.samples,
            strict_radii,
            p=np.inf,
            workers=1,
            return_length=True,
        ),
        dtype=np.int64,
    ) - 1
    second_counts = np.asarray(
        second.tree.query_ball_point(
            second.samples,
            strict_radii,
            p=np.inf,
            workers=1,
            return_length=True,
        ),
        dtype=np.int64,
    ) - 1
    if np.any(first_counts < 0) or np.any(second_counts < 0):
        raise FloatingPointError("KSG marginal neighbour counts cannot be negative.")

    # Kraskov, Stögbauer & Grassberger (2004), estimator I^(1), Eq. (8).
    estimate = (
        digamma(int(k))
        + digamma(sample_count)
        - np.mean(digamma(first_counts + 1) + digamma(second_counts + 1))
    )
    result = float(estimate)
    if not math.isfinite(result):
        raise FloatingPointError("KSG mutual-information estimate became non-finite.")
    return result


def estimate_pairwise_ksg_mutual_information(
    variables: Sequence[object],
    pairs: Sequence[tuple[int, int]],
    *,
    k: int = DEFAULT_KSG_K,
    standardize: bool = DEFAULT_KSG_STANDARDIZE,
    max_samples: int | None = DEFAULT_KSG_MAX_SAMPLES,
    subsample_mode: str = DEFAULT_KSG_SUBSAMPLE_MODE,
    subsample_seed: int = DEFAULT_KSG_SUBSAMPLE_SEED,
    sample_indices: object | None = None,
    tie_jitter: float = DEFAULT_KSG_TIE_JITTER,
    jitter_seed: int = DEFAULT_KSG_JITTER_SEED,
) -> np.ndarray:
    """Estimate KSG-1 MI for selected pairs while reusing marginal trees."""

    matrices = tuple(
        _as_sample_matrix(values, name=f"variables[{index}]")
        for index, values in enumerate(variables)
    )
    if len(matrices) < 2:
        raise ValueError("variables must contain at least two sample arrays.")
    sample_count = int(matrices[0].shape[0])
    if any(int(values.shape[0]) != sample_count for values in matrices[1:]):
        raise ValueError("All variables must have the same paired sample count.")
    if sample_indices is not None and max_samples is not None:
        raise ValueError("Set either ksg sample_indices or max_samples, not both.")
    if sample_indices is None:
        selected_indices = select_ksg_subsample_indices(
            sample_count,
            max_samples=max_samples,
            mode=subsample_mode,
            seed=subsample_seed,
        )
    else:
        _normalize_ksg_subsample_mode(subsample_mode)
        selected_indices = _validate_sample_indices(sample_indices, sample_count)
    if selected_indices.size != sample_count:
        matrices = tuple(values[selected_indices] for values in matrices)
    effective_sample_count = int(selected_indices.size)
    neighbour_count = _validate_k(k, effective_sample_count)
    normalized_pairs = tuple((int(first), int(second)) for first, second in pairs)
    if not normalized_pairs:
        raise ValueError("pairs must contain at least one variable pair.")
    if any(
        first < 0
        or second < 0
        or first >= len(matrices)
        or second >= len(matrices)
        or first == second
        for first, second in normalized_pairs
    ):
        raise ValueError("Each pair must contain two distinct valid variable indices.")

    prepared = _prepare_ksg_variables(
        matrices,
        standardize=standardize,
        tie_jitter=tie_jitter,
        jitter_seed=jitter_seed,
    )
    return np.asarray(
        [
            _estimate_prepared_ksg_pair(prepared[first], prepared[second], k=neighbour_count)
            for first, second in normalized_pairs
        ],
        dtype=float,
    )


def _estimate_soft_histogram_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bins: int,
    bandwidth: float,
    eps: float,
) -> float:
    if x.shape[1] != 1 or y.shape[1] != 1:
        raise ValueError("The legacy soft-histogram estimator supports scalar variables only.")
    bin_count = int(bins)
    if bin_count != bins or bin_count < 2:
        raise ValueError("bins must be an integer >= 2.")
    width = float(bandwidth)
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("bandwidth must be finite and > 0.")
    epsilon = float(eps)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("eps must be finite and > 0.")

    variables = np.concatenate((x.T, y.T), axis=0)
    centered = variables - np.mean(variables, axis=1, keepdims=True)
    variances = np.mean(centered * centered, axis=1, keepdims=True)
    standardized = centered / np.sqrt(np.maximum(variances, epsilon))
    centers = np.linspace(-4.0, 4.0, bin_count, dtype=float)
    logits = -0.5 * ((standardized[:, :, None] - centers) / width) ** 2
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    memberships = np.exp(logits)
    memberships = memberships / np.sum(memberships, axis=-1, keepdims=True)
    joint = memberships[0].T @ memberships[1]
    joint = np.maximum(joint / float(x.shape[0]), epsilon)
    joint = joint / np.sum(joint)
    first_marginal = np.sum(joint, axis=1, keepdims=True)
    second_marginal = np.sum(joint, axis=0, keepdims=True)
    estimate = np.sum(
        joint * (np.log(joint) - np.log(first_marginal) - np.log(second_marginal))
    )
    return max(float(estimate), 0.0)


def estimate_mutual_information(
    x: object,
    y: object,
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
    ksg_sample_indices: object | None = None,
    bins: int = 16,
    bandwidth: float = 0.5,
    eps: float = 1e-12,
    tie_jitter: float = DEFAULT_KSG_TIE_JITTER,
    jitter_seed: int = DEFAULT_KSG_JITTER_SEED,
) -> float:
    """Estimate paired mutual information in nats with a selected RSS backend."""

    first = _as_sample_matrix(x, name="x")
    second = _as_sample_matrix(y, name="y")
    if first.shape[0] != second.shape[0]:
        raise ValueError("x and y must have the same paired sample count.")
    selected = str(method).strip().lower()
    if selected == "gcmi":
        if first.shape[1] != 1 or second.shape[1] != 1:
            raise ValueError("The GCMI estimator supports scalar variables only.")
        variables = np.stack((first[:, 0], second[:, 0]), axis=0)
        return float(
            estimate_pairwise_gcmi_mutual_information(
                variables,
                ((0, 1),),
                rank_method=gcmi_rank_method,
                eps=eps,
            )[0]
        )
    if selected == "bspline":
        return _estimate_bspline_mutual_information(
            first,
            second,
            bins=bspline_bins,
            order=bspline_order,
            eps=eps,
        )
    if selected == "ksg":
        return float(
            estimate_pairwise_ksg_mutual_information(
                (first, second),
                ((0, 1),),
                k=k,
                standardize=standardize,
                max_samples=ksg_max_samples,
                subsample_mode=ksg_subsample_mode,
                subsample_seed=ksg_subsample_seed,
                sample_indices=ksg_sample_indices,
                tie_jitter=tie_jitter,
                jitter_seed=jitter_seed,
            )[0]
        )
    if selected == "histogram":
        return _estimate_soft_histogram_mutual_information(
            first,
            second,
            bins=bins,
            bandwidth=bandwidth,
            eps=eps,
        )
    raise ValueError(f"method must be one of: {', '.join(MI_ESTIMATORS)}.")
