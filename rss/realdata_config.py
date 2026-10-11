"""Runtime configuration for config-driven RSS processing."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any

import torch

from rss.solver import RSSConfig
from rss.ssa_preprocessing import DEFAULT_DRIFT_TURNING_POINT_LIMIT, DEFAULT_DRIFT_ZERO_CROSSING_LIMIT


# ---- per-file run configuration (the JSON list passed to ``run-config``) ----

@dataclass
class FileRunConfig:
    csv_path: str
    compute_device: str


def _validate_device_string(device: str) -> str:
    val = str(device).strip().lower()
    if val in {"auto", "cpu", "cuda"}:
        return val
    if val.startswith("cuda:"):
        idx = val.split(":", 1)[1]
        if idx.isdigit():
            return val
    raise ValueError("compute_device must be one of auto/cpu/cuda/cuda:<index>.")


def load_file_run_configs(config_path: str) -> list[FileRunConfig]:
    abs_path = os.path.abspath(config_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Config file not found: {abs_path}")
    with open(abs_path, "r", encoding="utf-8-sig") as f:
        raw: Any = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("Config JSON must be a list of objects.")
    out: list[FileRunConfig] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Config item {idx}: expected object.")
        keys = set(item.keys())
        allowed = {"csv_path", "compute_device"}
        missing = sorted(list(allowed - keys))
        extra = sorted(list(keys - allowed))
        if missing:
            raise ValueError(f"Config item {idx}: missing keys {missing}.")
        if extra:
            raise ValueError(f"Config item {idx}: unexpected keys {extra}.")
        csv_path = os.path.abspath(str(item["csv_path"]))
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Config item {idx}: csv_path does not exist: {csv_path}")
        if not csv_path.lower().endswith(".csv"):
            raise ValueError(f"Config item {idx}: csv_path must be a .csv file.")
        compute_device = _validate_device_string(str(item["compute_device"]))
        out.append(FileRunConfig(csv_path=csv_path, compute_device=compute_device))
    if not out:
        raise ValueError("Config contains no files.")
    return out


def resolve_torch_device(requested: str) -> torch.device:
    val = str(requested).strip().lower()
    if val == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(val)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable.")
        if dev.index is not None and (dev.index < 0 or dev.index >= int(torch.cuda.device_count())):
            raise ValueError("Requested CUDA device index is out of range.")
    elif dev.type != "cpu":
        raise ValueError("compute_device must be one of auto/cpu/cuda/cuda:<index>.")
    return dev


REALDATA_OUTPUT_FAMILIES: tuple[str, ...] = (
    "limb_observed_signals",
    "limb_observed_coefficients",
    "limb_angle_modes",
    "limb_mode_coefficients",
    "limb_axis_modes",
    "limb_quaternions",
    "limb_mode_metrics",
    "loss_history",
    "ssa_preprocessing_signals",
    "ssa_drift_correction_signals",
    "ssa_component_metrics",
    "ssa_plots",
)
DEFAULT_REALDATA_OUTPUT_FAMILIES = ("limb_angle_modes", "limb_axis_modes", "limb_quaternions")
RSS_MODE_ORDER_CHOICES = ("default", "amplitude")
SSA_OUTPUT_FAMILIES = frozenset(family for family in REALDATA_OUTPUT_FAMILIES if family.startswith("ssa_"))


@dataclass(frozen=True)
class RealDataRuntimeConfig:
    output_root: str
    sample_rate_hz: float
    include_time_seconds: bool
    file_workers: int
    rss_config: RSSConfig
    enabled_outputs: tuple[str, ...] = DEFAULT_REALDATA_OUTPUT_FAMILIES
    allow_partial_limbs: bool = False
    rss_mode_order: str = "default"
    ssa_enabled: bool = True
    ssa_window_length: int | None = None
    ssa_reconstruction_components: int | None = None
    ssa_drift_zero_crossing_limit: int = DEFAULT_DRIFT_ZERO_CROSSING_LIMIT
    ssa_drift_turning_point_limit: int = DEFAULT_DRIFT_TURNING_POINT_LIMIT


def _positive(name: str, value: float) -> float:
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be > 0.")
    return float(value)


def _integer(name: str, value: int, minimum: int) -> int:
    if int(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return int(value)


def resolve_realdata_output_families(
    *,
    enabled: list[str] | tuple[str, ...] | None,
    disabled: list[str] | tuple[str, ...] | None,
    ssa_enabled: bool,
) -> tuple[str, ...]:
    enables = tuple(enabled or ())
    disables = tuple(disabled or ())
    unknown = sorted({value for value in (*enables, *disables) if value != "all" and value not in REALDATA_OUTPUT_FAMILIES})
    if unknown:
        raise ValueError(f"Unknown real-data output families: {unknown}.")
    explicit_ssa = sorted(set(enables).intersection(SSA_OUTPUT_FAMILIES))
    if explicit_ssa and not ssa_enabled:
        raise ValueError(f"SSA output families cannot be explicitly enabled with disabled SSA: {explicit_ssa}.")
    selected = set(DEFAULT_REALDATA_OUTPUT_FAMILIES)
    if "all" in enables:
        selected.update(REALDATA_OUTPUT_FAMILIES)
        if not ssa_enabled:
            selected.difference_update(SSA_OUTPUT_FAMILIES)
    selected.update(value for value in enables if value != "all")
    if "all" in disables:
        selected.clear()
    else:
        selected.difference_update(disables)
    return tuple(family for family in REALDATA_OUTPUT_FAMILIES if family in selected)


def build_realdata_runtime_config_from_args(args) -> RealDataRuntimeConfig:
    ssa_enabled = not bool(args.disable_ssa_preprocessing)
    rss_config = RSSConfig(
        loss_type=str(args.rss_loss_type),
        separation_weight=float(args.rss_separation_weight),
        lags=tuple(int(value) for value in args.rss_lags),
        mi_estimator=str(args.rss_mi_estimator),
        gcmi_rank_method=str(args.rss_gcmi_rank_method),
        bspline_bins=_integer("--rss-bspline-bins", args.rss_bspline_bins, 2),
        bspline_order=_integer("--rss-bspline-order", args.rss_bspline_order, 1),
        ksg_k=_integer("--rss-ksg-k", args.rss_ksg_k, 1),
        ksg_standardize=bool(args.rss_ksg_standardize),
        ksg_max_samples=(
            None
            if args.rss_ksg_max_samples is None
            else _integer("--rss-ksg-max-samples", args.rss_ksg_max_samples, 1)
        ),
        ksg_subsample_mode=str(args.rss_ksg_subsample_mode),
        ksg_subsample_seed=_integer(
            "--rss-ksg-subsample-seed", args.rss_ksg_subsample_seed, 0
        ),
        mi_bins=_integer("--rss-mi-bins", args.rss_mi_bins, 2),
        mi_bandwidth=_positive("--rss-mi-bandwidth", args.rss_mi_bandwidth),
        action_inertias=tuple(float(value) for value in args.rss_action_inertias),
        action_stiffnesses=tuple(float(value) for value in args.rss_action_stiffnesses),
        epochs=_integer("--epochs", args.epochs, 1),
        optimizer=str(args.rss_optimizer),
        learning_rate=_positive("--learning-rate", args.learning_rate),
        learning_rate_end=_positive("--learning-rate-end", args.learning_rate_end),
        sgd_momentum=float(args.rss_sgd_momentum),
        pso_swarm_size=_integer("--rss-pso-swarm-size", args.rss_pso_swarm_size, 1),
        pso_inertia=float(args.rss_pso_inertia),
        pso_cognitive=float(args.rss_pso_cognitive),
        pso_social=float(args.rss_pso_social),
        pso_lower_bound=(
            None if args.rss_pso_lower_bound is None else float(args.rss_pso_lower_bound)
        ),
        pso_upper_bound=(
            None if args.rss_pso_upper_bound is None else float(args.rss_pso_upper_bound)
        ),
        pso_seed=int(args.rss_pso_seed),
        pso_stall_patience=_integer("--rss-pso-stall-patience", args.rss_pso_stall_patience, 1),
        orthogonality_weight=float(args.rss_orthogonality_weight),
        reconstruction_weight=float(args.rss_reconstruction_weight),
        convergence_tol=float(args.rss_convergence_tol),
        eps=_positive("--rss-eps", args.rss_eps),
        quaternion_eps=_positive("--rss-quaternion-eps", args.rss_quaternion_eps),
        device="cpu",
        matrix_parameterization=str(args.rss_matrix_parameterization),
        observer_signal_model=str(args.rss_observer_signal_model),
        initial_rotation_vector=tuple(
            float(value) for value in args.rss_initial_rotation_vector
        ),
        initial_weights=(
            None
            if args.rss_initial_weights is None
            else tuple(float(value) for value in args.rss_initial_weights)
        ),
        initial_weight_matrix=(
            None
            if args.rss_initial_weight_matrix is None
            else tuple(
                tuple(
                    float(args.rss_initial_weight_matrix[row * 3 + column])
                    for column in range(3)
                )
                for row in range(3)
            )
        ),
        axis_flip_search=bool(args.rss_axis_flip_search),
        axis_flip_interval=_integer(
            "--rss-axis-flip-interval", args.rss_axis_flip_interval, 1
        ),
        axis_flip_trial_epochs=_integer(
            "--rss-axis-flip-trial-epochs", args.rss_axis_flip_trial_epochs, 1
        ),
        adaptive_regularizer=bool(args.rss_adaptive_regularizer),
        specialized_regularizer=args.rss_specialized_regularizer,
        regularizer_start_weight=float(args.rss_regularizer_start_weight),
        regularizer_plateau_patience=_integer(
            "--rss-regularizer-plateau-patience",
            args.rss_regularizer_plateau_patience,
            1,
        ),
        regularizer_plateau_tolerance=float(
            args.rss_regularizer_plateau_tolerance
        ),
        regularizer_ramp_epochs=_integer(
            "--rss-regularizer-ramp-epochs", args.rss_regularizer_ramp_epochs, 1
        ),
        regularizer_min_loss_threshold=float(
            args.rss_regularizer_min_loss_threshold
        ),
    )
    return RealDataRuntimeConfig(
        output_root=os.path.abspath(str(args.output_root)),
        sample_rate_hz=_positive("--sample-rate-hz", args.sample_rate_hz),
        include_time_seconds=bool(args.include_time_seconds),
        file_workers=_integer("--file-workers", args.file_workers, 1),
        rss_config=rss_config,
        enabled_outputs=resolve_realdata_output_families(
            enabled=args.enable_output,
            disabled=args.disable_output,
            ssa_enabled=ssa_enabled,
        ),
        allow_partial_limbs=bool(args.allow_partial_limbs),
        ssa_enabled=ssa_enabled,
        rss_mode_order=str(args.rss_mode_order),
        ssa_window_length=None if args.ssa_window_length is None else _integer("--ssa-window-length", args.ssa_window_length, 2),
        ssa_reconstruction_components=(
            None
            if args.ssa_reconstruction_components is None
            else _integer("--ssa-reconstruction-components", args.ssa_reconstruction_components, 1)
        ),
        ssa_drift_zero_crossing_limit=_integer(
            "--ssa-drift-zero-crossing-limit", args.ssa_drift_zero_crossing_limit, 1
        ),
        ssa_drift_turning_point_limit=_integer(
            "--ssa-drift-turning-point-limit", args.ssa_drift_turning_point_limit, 1
        ),
    )
