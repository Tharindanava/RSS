"""Command-line interface for RSS validation and real-data processing."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from rss.artifacts import export_validation_case_artifacts
from rss.evaluation import run_validation
from rss.mutual_information import GCMI_RANK_METHODS, KSG_SUBSAMPLE_MODES, MI_ESTIMATORS
from rss.realdata_config import REALDATA_OUTPUT_FAMILIES, build_realdata_runtime_config_from_args
from rss.realdata_pipeline import run_realdata_from_config
from rss.simulation import SimulationConfig
from rss.solver import RSSConfig, fit_rss
from rss.specialized_losses import SPECIALIZED_LOSSES
from rss.ssa_preprocessing import (
    DEFAULT_DRIFT_TURNING_POINT_LIMIT,
    DEFAULT_DRIFT_ZERO_CROSSING_LIMIT,
    apply_ssa_preprocessing,
)


def _optional_positive_integer(value: str) -> int | None:
    normalized = str(value).strip().lower()
    if normalized in {"none", "all"}:
        return None
    try:
        result = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer or 'none'") from exc
    if result < 1:
        raise argparse.ArgumentTypeError("expected a positive integer or 'none'")
    return result


def _add_rss_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=200, help="Maximum Adam/SGD updates or PSO swarm iterations.")
    parser.add_argument(
        "--learning-rate",
        "--learning-rate-start",
        dest="learning_rate",
        type=float,
        default=1e-1,
        help="Starting learning rate for Adam/SGD updates and PSO position steps.",
    )
    parser.add_argument(
        "--learning-rate-end",
        type=float,
        default=1e-5,
        help="Ending exponential learning rate for Adam/SGD updates and PSO position steps.",
    )
    parser.add_argument("--rss-optimizer", choices=("adam", "sgd", "pso"), default="adam", help="RSS optimizer (default: adam).")
    parser.add_argument("--rss-sgd-momentum", type=float, default=0.9, help="Momentum used by the SGD optimizer.")
    parser.add_argument("--rss-pso-swarm-size", type=int, default=40, help="Number of PSO particles.")
    parser.add_argument("--rss-pso-inertia", type=float, default=0.7298, help="PSO inertia coefficient.")
    parser.add_argument("--rss-pso-cognitive", type=float, default=1.49618, help="PSO cognitive coefficient.")
    parser.add_argument("--rss-pso-social", type=float, default=1.49618, help="PSO social coefficient.")
    parser.add_argument("--rss-pso-lower-bound", type=float, default=None, help="Common lower bound (default: -pi for SO(3), -1 for symmetric).")
    parser.add_argument("--rss-pso-upper-bound", type=float, default=None, help="Common upper bound (default: pi for SO(3), 1 for symmetric).")
    parser.add_argument("--rss-pso-seed", type=int, default=0, help="Deterministic PSO random seed.")
    parser.add_argument(
        "--rss-pso-stall-patience",
        type=int,
        default=20,
        help="Legacy compatibility setting; PSO always runs the full epoch budget.",
    )
    parser.add_argument(
        "--rss-loss-type",
        choices=("lagged_cross_correlation", "mutual_information", "action"),
        default="lagged_cross_correlation",
        help="Primary RSS separation loss.",
    )
    parser.add_argument(
        "--rss-separation-weight",
        type=float,
        default=1.0,
        help="Selected separation-loss weight.",
    )
    parser.add_argument(
        "--rss-lags",
        type=int,
        nargs="+",
        default=[0],
        help="Nonnegative lags for lagged cross-correlation; 0 selects zero-lag correlation.",
    )
    parser.add_argument(
        "--rss-mi-estimator",
        choices=MI_ESTIMATORS,
        default="ksg",
        help="Mutual-information backend (GCMI and KSG require --rss-optimizer pso).",
    )
    parser.add_argument(
        "--rss-gcmi-rank-method",
        choices=GCMI_RANK_METHODS,
        default="average",
        help="Tie handling used by the GCMI marginal ranks.",
    )
    parser.add_argument(
        "--rss-bspline-bins",
        type=int,
        default=6,
        help="Number of soft B-spline MI basis bins.",
    )
    parser.add_argument(
        "--rss-bspline-order",
        type=int,
        default=3,
        help="Mathematical B-spline order; it must be smaller than the bin count.",
    )
    parser.add_argument(
        "--rss-ksg-k",
        type=int,
        default=4,
        help="Fixed KSG joint-neighbour count for an RSS run.",
    )
    parser.add_argument(
        "--rss-ksg-standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Standardize local KSG variable copies per dimension (default: enabled).",
    )
    parser.add_argument(
        "--rss-ksg-max-samples",
        type=_optional_positive_integer,
        default=None,
        metavar="N|none",
        help="Fixed KSG sample cap; 'none' uses every paired observation (default).",
    )
    parser.add_argument(
        "--rss-ksg-subsample-mode",
        choices=KSG_SUBSAMPLE_MODES,
        default="fixed_random",
        help="Deterministic KSG subset selection mode.",
    )
    parser.add_argument(
        "--rss-ksg-subsample-seed",
        type=int,
        default=0,
        help="Seed used once per RSS run to select a fixed KSG subset.",
    )
    parser.add_argument("--rss-mi-bins", type=int, default=16, help="Soft-histogram MI bin count.")
    parser.add_argument(
        "--rss-mi-bandwidth",
        type=float,
        default=0.5,
        help="Gaussian soft-bin bandwidth in standardized signal units.",
    )
    parser.add_argument(
        "--rss-action-inertias",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("I1", "I2", "I3"),
        help="Per-mode moments of inertia for action loss.",
    )
    parser.add_argument(
        "--rss-action-stiffnesses",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("K1", "K2", "K3"),
        help="Per-mode torsional stiffnesses for action loss.",
    )
    parser.add_argument(
        "--rss-orthogonality-weight",
        type=float,
        default=0.0,
        help="Soft ||A A^T - I||_F^2 / 3 orthogonality-loss weight.",
    )
    parser.add_argument(
        "--rss-reconstruction-weight",
        type=float,
        default=0.0,
        help="Source-composed quaternion reconstruction-loss weight.",
    )
    parser.add_argument(
        "--rss-convergence-tol",
        type=float,
        default=0.0,
        help="Legacy compatibility setting; RSS always runs the full epoch budget.",
    )
    parser.add_argument("--rss-eps", type=float, default=1e-12, help="Variance and matrix diagnostic epsilon.")
    parser.add_argument("--rss-quaternion-eps", type=float, default=1e-12, help="Quaternion normalization epsilon.")
    parser.add_argument(
        "--rss-matrix-parameterization",
        choices=("so3", "symmetric"),
        default="so3",
        help="Recovery-matrix parameterization (default: so3).",
    )
    parser.add_argument(
        "--rss-observer-signal-model",
        choices=("integrated_body_rotation", "euler"),
        default="euler",
        help="Interpretation of the three input channels; real SSA-cleaned angle signals should use euler.",
    )
    parser.add_argument(
        "--rss-initial-rotation-vector",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("RX", "RY", "RZ"),
        help="Initial SO(3) recovery rotation vector in radians.",
    )
    parser.add_argument(
        "--rss-initial-weights",
        type=float,
        nargs=6,
        default=None,
        metavar=("A11", "A12", "A13", "A22", "A23", "A33"),
        help="Six independent initial symmetric-matrix weights.",
    )
    parser.add_argument(
        "--rss-initial-weight-matrix",
        type=float,
        nargs=9,
        default=None,
        metavar=("A11", "A12", "A13", "A21", "A22", "A23", "A31", "A32", "A33"),
        help=(
            "Initial 3-by-3 recovery matrix in row-major order. It must be a proper "
            "rotation in SO(3) mode or symmetric in symmetric mode."
        ),
    )
    parser.add_argument(
        "--rss-axis-flip-search",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Periodically compare the identity and three proper double-axis flips.",
    )
    parser.add_argument(
        "--rss-axis-flip-interval",
        type=int,
        default=100,
        help="Accepted epochs between SO(3)-preserving axis-flip checks.",
    )
    parser.add_argument(
        "--rss-axis-flip-trial-epochs",
        type=int,
        default=10,
        help="Optimizer epochs allocated to each candidate at a flip check.",
    )
    parser.add_argument(
        "--rss-adaptive-regularizer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Adaptively regularize lagged cross-correlation with a specialized loss.",
    )
    parser.add_argument(
        "--rss-specialized-regularizer",
        choices=SPECIALIZED_LOSSES,
        default=None,
        help="Specialized loss used when adaptive regularization is enabled.",
    )
    parser.add_argument(
        "--rss-regularizer-start-weight",
        type=float,
        default=0.1,
        help="Initial specialized-regularizer weight in [0, 1].",
    )
    parser.add_argument(
        "--rss-regularizer-plateau-patience",
        type=int,
        default=20,
        help="Consecutive epochs without significant improvement before a weight ramp.",
    )
    parser.add_argument(
        "--rss-regularizer-plateau-tolerance",
        type=float,
        default=1e-4,
        help="Minimum relative improvement that resets the plateau counter.",
    )
    parser.add_argument(
        "--rss-regularizer-ramp-epochs",
        type=int,
        default=20,
        help="Accepted epochs used for each linear regularizer-weight ramp.",
    )
    parser.add_argument(
        "--rss-regularizer-min-loss-threshold",
        type=float,
        default=0.0,
        help="Normalized full-objective stopping threshold; 0 disables early stopping.",
    )


def _add_process_imu_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--csv", required=True, help="Path to a wide-format CSV containing one IMU's per-axis gyroscope columns.")
    parser.add_argument(
        "--node-prefix",
        default=None,
        help="Column prefix such as 'OrbiX-Node-10_IMU1'; expands to '<prefix>_GX/_GY/_GZ'.",
    )
    parser.add_argument(
        "--gyro-columns",
        nargs=3,
        default=None,
        metavar=("GX", "GY", "GZ"),
        help="Explicit gyroscope column names; overrides --node-prefix.",
    )
    parser.add_argument(
        "--gyro-units",
        choices=("deg", "rad"),
        default="deg",
        help="Units of the input gyroscope columns (default: deg, i.e. deg/s).",
    )
    parser.add_argument("--sample-rate-hz", type=float, default=100.0)
    parser.add_argument(
        "--bias-window-seconds",
        type=float,
        default=2.0,
        help="Average the first N seconds of gyro data as a static bias; assumes the IMU is stationary at the start of the recording.",
    )
    parser.add_argument("--disable-ssa-preprocessing", action="store_true")
    parser.add_argument("--ssa-window-length", type=int, default=None)
    parser.add_argument("--ssa-reconstruction-components", type=int, default=None)
    parser.add_argument("--ssa-drift-zero-crossing-limit", type=int, default=DEFAULT_DRIFT_ZERO_CROSSING_LIMIT)
    parser.add_argument("--ssa-drift-turning-point-limit", type=int, default=DEFAULT_DRIFT_TURNING_POINT_LIMIT)
    parser.add_argument("--output", default=None, help="Output CSV path for recovered source angles (default: <csv stem>_rss_sources.csv next to --csv).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rotation Source Separation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="Run one deterministic synthetic RSS validation.")
    validate.add_argument("--output-root", default="outputs/rss_validation")
    validate.add_argument("--seed", type=int, default=0)
    validate.add_argument("--samples", type=int, default=2000)
    validate.add_argument("--duration-seconds", type=float, default=1.0)
    validate.add_argument(
        "--simulation-observation-model",
        choices=("body_fixed_gyro", "legacy_vector_mixing"),
        default="body_fixed_gyro",
    )
    _add_rss_arguments(validate)

    real = subparsers.add_parser("run-config", help="Process IMU CSV files listed by a JSON configuration.")
    real.add_argument("--config", required=True)
    real.add_argument("--output-root", default="outputs/rss")
    choices = (*REALDATA_OUTPUT_FAMILIES, "all")
    real.add_argument("--enable-output", action="append", choices=choices, default=[])
    real.add_argument("--disable-output", action="append", choices=choices, default=[])
    real.add_argument("--sample-rate-hz", type=float, default=100.0)
    real.add_argument("--include-time-seconds", action="store_true")
    real.add_argument("--allow-partial-limbs", action="store_true")
    real.add_argument("--disable-ssa-preprocessing", action="store_true")
    real.add_argument("--ssa-window-length", type=int, default=None)
    real.add_argument("--ssa-reconstruction-components", type=int, default=None)
    real.add_argument("--ssa-drift-zero-crossing-limit", type=int, default=DEFAULT_DRIFT_ZERO_CROSSING_LIMIT)
    real.add_argument("--ssa-drift-turning-point-limit", type=int, default=DEFAULT_DRIFT_TURNING_POINT_LIMIT)
    real.add_argument(
        "--rss-mode-order",
        choices=("default", "amplitude"),
        default="default",
        help="How to order recovered RSS source modes in exported mode outputs.",
    )
    real.add_argument("--file-workers", type=int, default=1)
    real.add_argument("--quiet", action="store_true")
    _add_rss_arguments(real)

    process_imu = subparsers.add_parser(
        "process-imu",
        help="Recover source angles from one rigidly-mounted IMU's gyroscope channels (single-sensor rigs, e.g. a robot-arm end effector).",
    )
    _add_process_imu_arguments(process_imu)
    _add_rss_arguments(process_imu)
    process_imu.set_defaults(rss_observer_signal_model="integrated_body_rotation")
    return parser


def _rss_config(args, *, device: str = "cpu") -> RSSConfig:
    return RSSConfig(
        loss_type=args.rss_loss_type,
        separation_weight=args.rss_separation_weight,
        lags=tuple(args.rss_lags),
        mi_estimator=args.rss_mi_estimator,
        gcmi_rank_method=args.rss_gcmi_rank_method,
        bspline_bins=args.rss_bspline_bins,
        bspline_order=args.rss_bspline_order,
        ksg_k=args.rss_ksg_k,
        ksg_standardize=args.rss_ksg_standardize,
        ksg_max_samples=args.rss_ksg_max_samples,
        ksg_subsample_mode=args.rss_ksg_subsample_mode,
        ksg_subsample_seed=args.rss_ksg_subsample_seed,
        mi_bins=args.rss_mi_bins,
        mi_bandwidth=args.rss_mi_bandwidth,
        action_inertias=tuple(args.rss_action_inertias),
        action_stiffnesses=tuple(args.rss_action_stiffnesses),
        epochs=args.epochs,
        optimizer=args.rss_optimizer,
        learning_rate=args.learning_rate,
        learning_rate_end=args.learning_rate_end,
        sgd_momentum=args.rss_sgd_momentum,
        pso_swarm_size=args.rss_pso_swarm_size,
        pso_inertia=args.rss_pso_inertia,
        pso_cognitive=args.rss_pso_cognitive,
        pso_social=args.rss_pso_social,
        pso_lower_bound=args.rss_pso_lower_bound,
        pso_upper_bound=args.rss_pso_upper_bound,
        pso_seed=args.rss_pso_seed,
        pso_stall_patience=args.rss_pso_stall_patience,
        orthogonality_weight=args.rss_orthogonality_weight,
        reconstruction_weight=args.rss_reconstruction_weight,
        convergence_tol=args.rss_convergence_tol,
        eps=args.rss_eps,
        quaternion_eps=args.rss_quaternion_eps,
        device=device,
        matrix_parameterization=args.rss_matrix_parameterization,
        observer_signal_model=args.rss_observer_signal_model,
        initial_rotation_vector=tuple(args.rss_initial_rotation_vector),
        initial_weights=(
            None if args.rss_initial_weights is None else tuple(args.rss_initial_weights)
        ),
        initial_weight_matrix=(
            None
            if args.rss_initial_weight_matrix is None
            else tuple(
                tuple(args.rss_initial_weight_matrix[row * 3 + column] for column in range(3))
                for row in range(3)
            )
        ),
        axis_flip_search=args.rss_axis_flip_search,
        axis_flip_interval=args.rss_axis_flip_interval,
        axis_flip_trial_epochs=args.rss_axis_flip_trial_epochs,
        adaptive_regularizer=args.rss_adaptive_regularizer,
        specialized_regularizer=args.rss_specialized_regularizer,
        regularizer_start_weight=args.rss_regularizer_start_weight,
        regularizer_plateau_patience=args.rss_regularizer_plateau_patience,
        regularizer_plateau_tolerance=args.rss_regularizer_plateau_tolerance,
        regularizer_ramp_epochs=args.rss_regularizer_ramp_epochs,
        regularizer_min_loss_threshold=args.rss_regularizer_min_loss_threshold,
    )


def _progress_printer():
    started = time.perf_counter()

    def emit(event: dict[str, object]) -> None:
        kind = str(event.get("event_type", "progress"))
        if kind == "file_limb_progress":
            message = f"[{event.get('file_index')}/{event.get('file_total')}] {event.get('limb_name')} ({event.get('limb_index')}/{event.get('limb_total')})"
        elif kind in {"file_start", "file_done"}:
            message = f"[{event.get('file_index')}/{event.get('file_total')}] {kind}: {event.get('csv_path')}"
        elif kind in {"run_start", "run_done"}:
            message = f"{kind}: {event.get('file_total')} file(s)"
        else:
            return
        print(f"[RSS {time.perf_counter() - started:8.1f}s] {message}", file=sys.stderr, flush=True)

    return emit


def _run_validate(args) -> None:
    simulation = SimulationConfig(
        seed=args.seed,
        T=args.samples,
        duration_seconds=args.duration_seconds,
        observation_model=args.simulation_observation_model,
    )
    result = run_validation(simulation, _rss_config(args))
    output = Path(args.output_root)
    payload = export_validation_case_artifacts(result, output)
    print(json.dumps({"output_root": str(output.resolve()), "metrics": payload["metrics"]}, indent=2))


def _run_config(args) -> None:
    runtime = build_realdata_runtime_config_from_args(args)
    outputs = run_realdata_from_config(
        config_path=args.config,
        runtime_cfg=runtime,
        progress_callback=None if args.quiet else _progress_printer(),
    )
    summary = {
        "output_root": runtime.output_root,
        "runtime_config": asdict(runtime),
        "outputs": [asdict(output) for output in outputs],
    }
    print(json.dumps(summary, indent=2))


def _load_gyro_channels(args) -> np.ndarray:
    frame = pd.read_csv(args.csv)
    if args.gyro_columns:
        columns = list(args.gyro_columns)
    elif args.node_prefix:
        columns = [f"{args.node_prefix}_G{axis}" for axis in "XYZ"]
    else:
        raise SystemExit("process-imu requires either --node-prefix or --gyro-columns.")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SystemExit(f"Missing gyroscope columns in {args.csv}: {missing}")
    gyro = frame.loc[:, columns].to_numpy(dtype=float)
    if not np.isfinite(gyro).all():
        raise SystemExit("The selected gyroscope columns contain non-finite samples.")
    return gyro


def _run_process_imu(args) -> None:
    gyro = _load_gyro_channels(args)
    if args.gyro_units == "deg":
        gyro = np.radians(gyro)
    bias_count = min(gyro.shape[0], max(1, round(args.bias_window_seconds * args.sample_rate_hz)))
    bias = np.mean(gyro[:bias_count], axis=0)
    increments = (gyro - bias[None, :]) / args.sample_rate_hz
    observations = np.cumsum(increments, axis=0).T

    if not args.disable_ssa_preprocessing:
        observations = np.stack(
            [
                apply_ssa_preprocessing(
                    channel,
                    sample_rate_hz=args.sample_rate_hz,
                    requested_window_length=args.ssa_window_length,
                    requested_reconstruction_components=args.ssa_reconstruction_components,
                    drift_zero_crossing_limit=args.ssa_drift_zero_crossing_limit,
                    drift_turning_point_limit=args.ssa_drift_turning_point_limit,
                ).cleaned_signal
                for channel in observations
            ]
        )

    config = _rss_config(args)
    result = fit_rss(observations, config, sample_interval_seconds=1.0 / args.sample_rate_hz)

    output_path = Path(args.output) if args.output else Path(args.csv).with_name(f"{Path(args.csv).stem}_rss_sources.csv")
    time_seconds = np.arange(observations.shape[1]) / args.sample_rate_hz
    pd.DataFrame(
        {
            "time_s": time_seconds,
            "source_1": result.source_angles[0],
            "source_2": result.source_angles[1],
            "source_3": result.source_angles[2],
        }
    ).to_csv(output_path, index=False)

    summary = {
        "csv": str(args.csv),
        "output": str(output_path),
        "best_loss": result.best_loss,
        "best_epoch": result.best_epoch,
        "recovery_matrix": result.recovery_matrix.tolist(),
        "reconstruction_mean_error_rad": result.reconstruction_mean_error_rad,
        "reconstruction_max_error_rad": result.reconstruction_max_error_rad,
    }
    print(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        _run_validate(args)
    elif args.command == "process-imu":
        _run_process_imu(args)
    else:
        _run_config(args)


if __name__ == "__main__":
    main()
