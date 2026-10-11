"""Resumable, failure-isolated orchestration for the two-stage study."""

from __future__ import annotations

import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .classical_features import (
    aggregate_trial_features, compute_classic_features, compute_node_step_features,
    normalize_node_trajectories, normalize_step_trajectories,
)
from .config import AnalysisConfig
from .io import load_proxy_signals
from .metadata import build_manifest, records_from_manifest
from .mfpca import mfpca_stability_diagnostics, run_mfpca
from .models import NodeStep, PipelineArtifacts, SegmentationResult, TrialRecord
from .node_fpca import aggregate_node_trial_scores, fit_all_node_fpca, node_score_matrix
from .progress import ProgressTracker, progress_error, progress_note
from .references import registry_checksum
from .segmentation import segment_recording
from .spectral_clustering import cluster_phenotypes, interpret_clusters
from .statistics import balance_trials_per_subject, run_condition_statistics, scenario_eligible_trials
from .te_network import analyze_recording_te
from .utils import atomic_write_dataframe, atomic_write_json, hash_mapping, scaled_mad
from .utils import benjamini_hochberg
from .visualization import create_figures


STEP_COLUMNS = [
    "recording_id", "trial_id", "step_id", "node", "node_side", "driving_ankle_side",
    "coordination", "cycle_id", "cycle_index", "start_sample", "end_sample", "toe_off_sample",
    "toe_off_time_fraction", "duration_seconds", "stance_seconds", "swing_seconds",
    "stance_fraction", "phase_rotation_per_stride", "valley_prominence", "accepted", "reasons",
]

# Increment a stage version whenever its derived-cache semantics change.  Stage 1
# previously reused empty artifacts produced against older segmentation results;
# Stage 2 has the same upstream dependency and is versioned alongside it.
# v5: segmentation was fully replaced by ankle-driven Hilbert-phase minima detection
# (NodeStep, all 12 nodes) in place of the old gyro-peak GaitEvent/GaitStep pipeline;
# every cached artifact from before this change has an incompatible schema.
DERIVED_CACHE_VERSIONS = {"segment": 5, "stage1": 5, "stage2": 5}


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return not pd.isna(value) and bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _record_label(record: TrialRecord) -> str:
    return f"participant={record.participant_id} file={record.raw_file.name} recording={record.recording_id}"


def _manifest_hash(manifest: pd.DataFrame) -> str:
    columns = [column for column in ("recording_id", "raw_checksum", "rss_checksum") if column in manifest]
    selected = manifest[columns]
    if "recording_id" in selected:
        selected = selected.sort_values("recording_id")
    return hash_mapping(selected.to_dict("records"))


def _reference_checksum(config: AnalysisConfig) -> str:
    try:
        return registry_checksum(config.path("reference_registry"))
    except KeyError:
        return registry_checksum(Path("/nonexistent/hma_dashboard_references.json"))


def _run_directory(config: AnalysisConfig, manifest: pd.DataFrame) -> Path:
    run_id = f"{config.profile}-{_manifest_hash(manifest)[:12]}-{config.config_hash[:12]}-{_reference_checksum(config)[:12]}"
    return config.path("output_root") / "runs" / run_id


def _add_hashes(frame: pd.DataFrame, config: AnalysisConfig, manifest_hash: str) -> pd.DataFrame:
    output = frame.copy()
    output["config_hash"] = config.config_hash
    output["manifest_hash"] = manifest_hash
    output["reference_registry_checksum"] = _reference_checksum(config)
    return output


def _read_or_manifest(config: AnalysisConfig, manifest: pd.DataFrame | None, limit: int | None) -> pd.DataFrame:
    return build_manifest(config, limit=limit) if manifest is None else manifest.copy()


def _write_run_metadata(config: AnalysisConfig, manifest: pd.DataFrame, run_dir: Path, command: str) -> Path:
    metadata = {
        "schema_version": 2,
        "command": command,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "profile": config.profile,
        "random_seed": int(config.values["random_seed"]),
        "config_hash": config.config_hash,
        "manifest_hash": _manifest_hash(manifest),
        "reference_registry_checksum": _reference_checksum(config),
        "recordings": int(len(manifest)),
        "raw_root": str(config.path("raw_root")),
        "rss_root": str(config.path("rss_root")),
        "python": platform.python_version(),
        "signal_type": "rss_segment_mode1_proxy",
        "interpretation_guardrail": "Mode 1 is a segment-level latent motion proxy, not a validated anatomical joint angle.",
    }
    atomic_write_json({"profile": config.profile, "values": config.values}, run_dir / "effective_config.json")
    path = atomic_write_json(metadata, run_dir / "run_metadata.json")
    atomic_write_json({"run_dir": str(run_dir), **metadata}, config.path("output_root") / "latest.json")
    return path


def _cache_key(record: TrialRecord, config: AnalysisConfig, stage: str) -> str:
    payload = {
        "stage": stage, "recording_id": record.recording_id, "raw": record.raw_checksum,
        "rss": record.rss_checksum, "config": config.config_hash,
        "reference_registry_checksum": _reference_checksum(config),
    }
    if stage in DERIVED_CACHE_VERSIONS:
        payload["derived_cache_version"] = DERIVED_CACHE_VERSIONS[stage]
    return hash_mapping(payload)


def _cache_dir(record: TrialRecord, config: AnalysisConfig, stage: str) -> Path:
    return config.path("output_root") / "cache" / stage / _cache_key(record, config, stage)


def _step_from_row(row: dict[str, Any]) -> NodeStep:
    return NodeStep(
        str(row["recording_id"]), str(row["trial_id"]), str(row["step_id"]), str(row["node"]),
        str(row["node_side"]), str(row["driving_ankle_side"]), str(row["coordination"]),
        str(row["cycle_id"]), int(row["cycle_index"]), int(row["start_sample"]), int(row["end_sample"]),
        int(row["toe_off_sample"]), float(row["toe_off_time_fraction"]), float(row["duration_seconds"]),
        float(row["stance_seconds"]), float(row["swing_seconds"]), float(row["stance_fraction"]),
        None if pd.isna(row.get("phase_rotation_per_stride")) else float(row.get("phase_rotation_per_stride")),
        None if pd.isna(row.get("valley_prominence")) else float(row.get("valley_prominence")),
        _as_bool(row["accepted"]), tuple(filter(None, str(row.get("reasons", "")).split("|"))),
    )


def _segmentation_from_frames(record: TrialRecord, steps: pd.DataFrame) -> SegmentationResult:
    step_objects = [_step_from_row(row) for row in steps.to_dict("records")]
    return SegmentationResult(record.recording_id, record.trial_id, step_objects, record.sampling_hz, "ok" if any(step.accepted for step in step_objects) else "failed")


def run_manifest(config: AnalysisConfig, *, limit: int | None = None) -> PipelineArtifacts:
    progress_note("manifest", f"stage started (limit={limit})")
    manifest = build_manifest(config, limit=limit)
    run_dir = _run_directory(config, manifest)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_hash = _manifest_hash(manifest)
    manifest_path = atomic_write_dataframe(_add_hashes(manifest, config, manifest_hash), run_dir / "manifest.csv")
    atomic_write_dataframe(_add_hashes(manifest, config, manifest_hash), run_dir / "cohort_manifest.csv")
    metadata_path = _write_run_metadata(config, manifest, run_dir, "manifest")
    progress_note("manifest", f"stage completed | run_dir={run_dir}")
    return PipelineArtifacts(run_dir=run_dir, manifest=manifest_path, run_metadata=metadata_path)


def run_segmentation(
    config: AnalysisConfig, *, manifest: pd.DataFrame | None = None, limit: int | None = None, force: bool = False,
) -> PipelineArtifacts:
    progress_note("segmentation", f"stage started (force={force})")
    manifest = _read_or_manifest(config, manifest, limit)
    run_dir = _run_directory(config, manifest)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_hash = _manifest_hash(manifest)
    manifest_path = atomic_write_dataframe(_add_hashes(manifest, config, manifest_hash), run_dir / "manifest.csv")
    atomic_write_dataframe(_add_hashes(manifest, config, manifest_hash), run_dir / "cohort_manifest.csv")
    step_frames: list[pd.DataFrame] = []
    errors: list[dict[str, Any]] = []
    records = records_from_manifest(manifest)
    tracker = ProgressTracker("segmentation", len(records), report_every=max(1, len(records) // 20))
    for index, record in enumerate(records, start=1):
        cache = _cache_dir(record, config, "segment")
        step_cache = cache / "gait_steps.csv"
        try:
            if not force and step_cache.exists():
                step_frame = pd.read_csv(step_cache)
                status = "cache hit"
            else:
                progress_note("segmentation", f"processing {index}/{len(records)} | {_record_label(record)}")
                result = segment_recording(record, config)
                step_frame = result.step_frame().reindex(columns=STEP_COLUMNS)
                atomic_write_dataframe(step_frame, step_cache)
                atomic_write_json({"status": result.status, "qc_reasons": result.qc_reasons, "diagnostics": result.diagnostics}, cache / "status.json")
                status = result.status
            # A recording with no detected steps writes a 0-row frame; concatenating
            # that alongside populated bool-typed frames silently promotes "accepted" to
            # float64 (1.0/0.0) unless it is pinned back to bool here on every path.
            step_frame["accepted"] = step_frame["accepted"].fillna(False).astype(bool)
            step_frames.append(step_frame)
            accepted = int(step_frame["accepted"].map(_as_bool).sum()) if "accepted" in step_frame else 0
            tracker.update(index, f"{status}; accepted_steps={accepted} | {_record_label(record)}")
        except Exception as exc:  # one failed recording must not terminate the cohort
            errors.append({"stage": "segment", "recording_id": record.recording_id, "trial_id": record.trial_id, "error_type": type(exc).__name__, "message": str(exc)})
            progress_error("segmentation", f"failed | {_record_label(record)} | {type(exc).__name__}: {exc}")
            tracker.update(index, f"failed | {_record_label(record)}")
    steps = pd.concat(step_frames, ignore_index=True) if step_frames else pd.DataFrame(columns=STEP_COLUMNS)
    context_columns = [
        "recording_id", "participant_id", "terrain", "direction", "footwear", "heel_code",
        "phase", "raw_file", "rss_file",
    ]
    context = manifest[[column for column in context_columns if column in manifest]].drop_duplicates("recording_id")
    steps = steps.merge(context, on="recording_id", how="left")
    steps_path = atomic_write_dataframe(_add_hashes(steps, config, manifest_hash), run_dir / "gait_steps.csv")
    errors_path = atomic_write_dataframe(_add_hashes(pd.DataFrame(errors), config, manifest_hash), run_dir / "errors.csv")
    metadata_path = _write_run_metadata(config, manifest, run_dir, "segment")
    progress_note(
        "segmentation",
        f"stage completed | recordings={len(records)} steps={len(steps)} errors={len(errors)}",
    )
    return PipelineArtifacts(run_dir, manifest_path, steps_path, errors=errors_path, run_metadata=metadata_path)


def run_stage1(
    config: AnalysisConfig, *, manifest: pd.DataFrame | None = None, limit: int | None = None, force: bool = False,
) -> PipelineArtifacts:
    progress_note("stage1", f"stage started (force={force})")
    manifest = _read_or_manifest(config, manifest, limit)
    artifacts = run_segmentation(config, manifest=manifest, force=force)
    run_dir, manifest_hash = artifacts.run_dir, _manifest_hash(manifest)
    steps_all = pd.read_csv(artifacts.gait_steps)
    classic_frames: list[pd.DataFrame] = []
    trajectory_frames: list[pd.DataFrame] = []
    node_feature_frames: list[pd.DataFrame] = []
    node_trajectory_frames: list[pd.DataFrame] = []
    proxy_qc_frames: list[pd.DataFrame] = []
    errors: list[dict[str, Any]] = []
    eligible_records = [record for record in records_from_manifest(manifest) if record.stage1_eligible]
    progress_note("stage1", f"{len(eligible_records)}/{len(manifest)} recording(s) eligible")
    tracker = ProgressTracker("stage1", len(eligible_records), report_every=max(1, len(eligible_records) // 20))
    for index, record in enumerate(eligible_records, start=1):
        cache = _cache_dir(record, config, "stage1")
        classic_cache, trajectory_cache, qc_cache = (
            cache / "classic_step_features.csv", cache / "normalized_step_trajectories.csv.gz",
            cache / "proxy_signal_qc.csv",
        )
        node_feature_cache, node_trajectory_cache = (
            cache / "node_step_features.csv", cache / "node_step_trajectories.csv.gz",
        )
        try:
            if not force and all(path.exists() for path in (
                classic_cache, trajectory_cache, qc_cache, node_feature_cache, node_trajectory_cache,
            )):
                classic_frames.append(pd.read_csv(classic_cache))
                trajectory_frames.append(pd.read_csv(trajectory_cache))
                proxy_qc_frames.append(pd.read_csv(qc_cache))
                node_feature_frames.append(pd.read_csv(node_feature_cache))
                node_trajectory_frames.append(pd.read_csv(node_trajectory_cache))
                status = "cache hit"
            else:
                progress_note("stage1", f"processing {index}/{len(eligible_records)} | {_record_label(record)}")
                record_steps = steps_all[steps_all["recording_id"] == record.recording_id]
                segmentation = _segmentation_from_frames(record, record_steps)
                proxy = load_proxy_signals(record, config)
                classic_record = compute_classic_features(record, segmentation, proxy["signals"], config)
                trajectory_record = normalize_step_trajectories(record, segmentation, proxy["signals"], config)
                node_feature_record = compute_node_step_features(record, segmentation, proxy["signals"], config)
                node_trajectory_record = normalize_node_trajectories(record, segmentation, proxy["signals"], config)
                if classic_record.empty:
                    classic_record = pd.DataFrame(columns=[
                        "recording_id", "trial_id", "participant_id", "step_id", "step_pair_id", "side", "terrain",
                        "direction", "footwear", "phase", "stride_duration_seconds",
                    ])
                if trajectory_record.empty:
                    trajectory_record = pd.DataFrame(columns=[
                        "recording_id", "trial_id", "participant_id", "step_pair_id", "step_id", "representation",
                        "node", "percent_step", "value", "signal_type",
                    ])
                if node_feature_record.empty:
                    node_feature_record = pd.DataFrame(columns=[
                        "recording_id", "trial_id", "participant_id", "step_id", "cycle_id", "node", "node_side",
                        "terrain", "direction", "footwear", "duration_seconds",
                    ])
                if node_trajectory_record.empty:
                    node_trajectory_record = pd.DataFrame(columns=[
                        "recording_id", "trial_id", "participant_id", "step_pair_id", "step_id", "representation",
                        "node", "percent_step", "value", "signal_type",
                    ])
                atomic_write_dataframe(classic_record, classic_cache)
                atomic_write_dataframe(trajectory_record, trajectory_cache)
                atomic_write_dataframe(proxy["qc"], qc_cache)
                atomic_write_dataframe(node_feature_record, node_feature_cache)
                atomic_write_dataframe(node_trajectory_record, node_trajectory_cache)
                classic_frames.append(classic_record)
                trajectory_frames.append(trajectory_record)
                proxy_qc_frames.append(proxy["qc"])
                node_feature_frames.append(node_feature_record)
                node_trajectory_frames.append(node_trajectory_record)
                status = f"computed steps={len(classic_record)} trajectory_rows={len(trajectory_record)} node_steps={len(node_feature_record)}"
            tracker.update(index, f"{status} | {_record_label(record)}")
        except Exception as exc:
            errors.append({"stage": "stage1", "recording_id": record.recording_id, "trial_id": record.trial_id, "error_type": type(exc).__name__, "message": str(exc)})
            progress_error("stage1", f"failed | {_record_label(record)} | {type(exc).__name__}: {exc}")
            tracker.update(index, f"failed | {_record_label(record)}")
    classic = pd.concat(classic_frames, ignore_index=True) if classic_frames else pd.DataFrame()
    trial = aggregate_trial_features(classic, config)
    trajectories = pd.concat(trajectory_frames, ignore_index=True) if trajectory_frames else pd.DataFrame()
    proxy_qc = pd.concat(proxy_qc_frames, ignore_index=True) if proxy_qc_frames else pd.DataFrame()
    node_features = pd.concat(node_feature_frames, ignore_index=True) if node_feature_frames else pd.DataFrame()
    node_trajectories = pd.concat(node_trajectory_frames, ignore_index=True) if node_trajectory_frames else pd.DataFrame()
    if not proxy_qc.empty:
        qc_context = manifest[[column for column in (
            "recording_id", "participant_id", "terrain", "direction", "footwear", "raw_file", "rss_file"
        ) if column in manifest]].drop_duplicates("recording_id")
        proxy_qc = proxy_qc.merge(qc_context, on="recording_id", how="left")
    classic_path = atomic_write_dataframe(_add_hashes(classic, config, manifest_hash), run_dir / "classic_step_features.csv")
    trial_path = atomic_write_dataframe(_add_hashes(trial, config, manifest_hash), run_dir / "classic_trial_features.csv")
    atomic_write_dataframe(_add_hashes(trajectories, config, manifest_hash), run_dir / "normalized_step_trajectories.csv.gz")
    atomic_write_dataframe(_add_hashes(proxy_qc, config, manifest_hash), run_dir / "proxy_signal_qc.csv")
    atomic_write_dataframe(_add_hashes(node_features, config, manifest_hash), run_dir / "node_step_features.csv")
    atomic_write_dataframe(_add_hashes(node_trajectories, config, manifest_hash), run_dir / "node_step_trajectories.csv.gz")
    mfpca_dir = run_dir / "mfpca"
    mfpca_dir.mkdir(exist_ok=True)
    model_outputs_exist = all((mfpca_dir / name).exists() for name in (
        "mfpca_model.npz", "mfpca_step_pair_scores.csv", "mfpca_trial_scores.csv", "mfpca_stability.csv",
        "mfpca_resampling_stability.csv", "mfpca_metadata.json",
    ))
    if not trajectories.empty and (force or not model_outputs_exist):
        try:
            progress_note("mfpca", f"fitting cohort model from {len(trajectories)} trajectory row(s)")
            result, score_frame = run_mfpca(trajectories, config)
            progress_note("mfpca", f"model fitted with {result.n_components} retained component(s); writing artifacts")
            fd, temp_name = tempfile.mkstemp(prefix=".model.", suffix=".npz", dir=mfpca_dir)
            os.close(fd)
            try:
                np.savez_compressed(
                    temp_name, mean_function=result.mean_function, eigenfunctions=result.eigenfunctions,
                    eigenvalues=result.eigenvalues, channel_weights=result.channel_weights,
                    explained_variance_ratio=result.explained_variance_ratio,
                    phase_grid=np.linspace(0.0, 1.0, result.mean_function.shape[1]),
                    n_components=result.n_components, config_hash=config.config_hash, manifest_hash=manifest_hash,
                    reference_registry_checksum=_reference_checksum(config),
                )
                os.replace(temp_name, mfpca_dir / "mfpca_model.npz")
            finally:
                Path(temp_name).unlink(missing_ok=True)
            atomic_write_dataframe(_add_hashes(score_frame, config, manifest_hash), mfpca_dir / "mfpca_step_pair_scores.csv")
            score_columns = [
                column for column in score_frame
                if column.startswith("mfpca_score_") and column.removeprefix("mfpca_score_").isdigit()
            ]
            trial_scores = score_frame.groupby(["participant_id", "trial_id"], as_index=False)[
                [*score_columns, "mfpca_reconstruction_error"]
            ].median()
            for column in score_columns:
                dispersion = score_frame.groupby(["participant_id", "trial_id"])[column].apply(
                    lambda values: 1.4826 * np.median(np.abs(values - np.median(values)))
                ).rename(f"{column}_scaled_mad").reset_index()
                trial_scores = trial_scores.merge(dispersion, on=["participant_id", "trial_id"], how="left")
            context_columns = [column for column in ("terrain", "direction", "footwear") if column in score_frame]
            if context_columns:
                trial_context = score_frame.groupby(["participant_id", "trial_id"], as_index=False)[context_columns].first()
                trial_scores = trial_scores.merge(trial_context, on=["participant_id", "trial_id"], how="left")
            for source_column in ("raw_file", "rss_file", "recording_id"):
                if source_column in score_frame:
                    sources = score_frame.groupby(["participant_id", "trial_id"])[source_column].apply(
                        lambda values: "|".join(sorted(values.astype(str).unique()))
                    ).rename({"raw_file": "raw_files", "rss_file": "rss_files", "recording_id": "source_recordings"}[source_column]).reset_index()
                    trial_scores = trial_scores.merge(sources, on=["participant_id", "trial_id"], how="left")
            atomic_write_dataframe(_add_hashes(trial_scores, config, manifest_hash), mfpca_dir / "mfpca_trial_scores.csv")
            stability = pd.DataFrame({
                "component": np.arange(1, result.n_components + 1),
                "eigenvalue": result.eigenvalues[:result.n_components],
                "explained_variance_ratio": result.explained_variance_ratio[:result.n_components],
                "cumulative_fve": np.cumsum(result.explained_variance_ratio[:result.n_components]),
                "weighted_orthonormality_max_error": float(np.max(np.abs(
                    (result.eigenfunctions.reshape(len(result.eigenfunctions), -1) * np.repeat(result.channel_weights / result.mean_function.shape[1], result.mean_function.shape[1])[None, :])
                    @ result.eigenfunctions.reshape(len(result.eigenfunctions), -1).T - np.eye(len(result.eigenfunctions))
                ))),
            })
            atomic_write_dataframe(_add_hashes(stability, config, manifest_hash), mfpca_dir / "mfpca_stability.csv")
            progress_note("mfpca", "starting leave-one-participant-out and bootstrap stability diagnostics")
            resampling_stability = mfpca_stability_diagnostics(trajectories, result, config)
            atomic_write_dataframe(_add_hashes(resampling_stability, config, manifest_hash), mfpca_dir / "mfpca_resampling_stability.csv")
            atomic_write_json({
                "training_manifest_hash": manifest_hash, "config_hash": config.config_hash,
                "reference_registry_checksum": _reference_checksum(config),
                "n_components": result.n_components, "sample_ids": result.sample_ids,
                "channel_weights": result.channel_weights.tolist(), "metadata": result.metadata,
            }, mfpca_dir / "mfpca_metadata.json")
            progress_note("mfpca", f"completed | resampling_results={len(resampling_stability)}")
        except Exception as exc:
            errors.append({"stage": "mfpca", "recording_id": "cohort", "trial_id": "cohort", "error_type": type(exc).__name__, "message": str(exc)})
            progress_error("mfpca", f"failed | {type(exc).__name__}: {exc}")
    elif trajectories.empty:
        progress_note("mfpca", "skipped because no complete trajectories were available")
    else:
        progress_note("mfpca", "cache hit; reusing complete cohort model outputs")
    previous_errors = pd.read_csv(artifacts.errors) if artifacts.errors and artifacts.errors.exists() else pd.DataFrame()
    combined_errors = pd.concat((previous_errors.drop(columns=["config_hash", "manifest_hash"], errors="ignore"), pd.DataFrame(errors)), ignore_index=True)
    errors_path = atomic_write_dataframe(_add_hashes(combined_errors, config, manifest_hash), run_dir / "errors.csv")
    metadata_path = _write_run_metadata(config, manifest, run_dir, "stage1")
    artifacts.classic_step_features = classic_path
    artifacts.classic_trial_features = trial_path
    artifacts.mfpca_directory = mfpca_dir
    artifacts.errors = errors_path
    artifacts.run_metadata = metadata_path
    progress_note(
        "stage1",
        f"stage completed | eligible_recordings={len(eligible_records)} classic_steps={len(classic)} errors={len(errors)}",
    )
    return artifacts


def run_stage2(
    config: AnalysisConfig, *, manifest: pd.DataFrame | None = None, limit: int | None = None, force: bool = False,
) -> PipelineArtifacts:
    progress_note("stage2", f"stage started (force={force})")
    manifest = _read_or_manifest(config, manifest, limit)
    artifacts = run_segmentation(config, manifest=manifest, force=force)
    run_dir, manifest_hash = artifacts.run_dir, _manifest_hash(manifest)
    steps_all = pd.read_csv(artifacts.gait_steps)
    edges_frames: list[pd.DataFrame] = []
    feature_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    eligible_records = [record for record in records_from_manifest(manifest) if record.stage2_eligible]
    progress_note("stage2", f"{len(eligible_records)}/{len(manifest)} recording(s) eligible")
    tracker = ProgressTracker("stage2", len(eligible_records), report_every=max(1, len(eligible_records) // 20))
    for index, record in enumerate(eligible_records, start=1):
        cache = _cache_dir(record, config, "stage2")
        edge_cache, feature_cache = cache / "te_edges.csv", cache / "te_features.csv"
        try:
            if not force and edge_cache.exists() and feature_cache.exists():
                cached_edges = pd.read_csv(edge_cache)
                cached_features = pd.read_csv(feature_cache)
                edges_frames.append(cached_edges)
                feature_rows.extend(cached_features.to_dict("records"))
                status = f"cache hit windows={len(cached_features)} edges={len(cached_edges)}"
            else:
                progress_note("stage2", f"processing {index}/{len(eligible_records)} | {_record_label(record)}")
                steps = [_step_from_row(row) for row in steps_all[steps_all["recording_id"] == record.recording_id].to_dict("records")]
                proxy = load_proxy_signals(record, config)
                networks = analyze_recording_te(record, steps, proxy["signals"], config)
                record_edges = pd.concat([network.edge_table for network in networks if not network.edge_table.empty], ignore_index=True) if networks else pd.DataFrame(columns=[
                    "recording_id", "trial_id", "window_id", "source", "target", "raw_te", "effective_te",
                    "best_delay_samples", "p_value", "q_value", "significant", "adjacency_weight",
                ])
                rows = [{
                    "recording_id": network.recording_id, "trial_id": network.trial_id, "participant_id": record.participant_id,
                    "terrain": record.terrain, "direction": record.direction, "footwear": record.footwear,
                    "raw_file": str(record.raw_file), "rss_file": str(record.rss_file) if record.rss_file else "",
                    "window_id": network.window_id, "start_sample": network.start_sample, "end_sample": network.end_sample,
                    "status": network.status, "reasons": "|".join(network.reasons), **network.features,
                } for network in networks]
                record_features = pd.DataFrame(rows, columns=list(rows[0]) if rows else [
                    "recording_id", "trial_id", "participant_id", "terrain", "direction", "footwear",
                    "window_id", "start_sample", "end_sample", "status", "reasons",
                ])
                atomic_write_dataframe(record_edges, edge_cache)
                atomic_write_dataframe(record_features, feature_cache)
                edges_frames.append(record_edges)
                feature_rows.extend(rows)
                status = f"computed windows={len(networks)} edges={len(record_edges)}"
            tracker.update(index, f"{status} | {_record_label(record)}")
        except Exception as exc:
            errors.append({"stage": "stage2", "recording_id": record.recording_id, "trial_id": record.trial_id, "error_type": type(exc).__name__, "message": str(exc)})
            progress_error("stage2", f"failed | {_record_label(record)} | {type(exc).__name__}: {exc}")
            tracker.update(index, f"failed | {_record_label(record)}")
    edges = pd.concat(edges_frames, ignore_index=True) if edges_frames else pd.DataFrame()
    features = pd.DataFrame(feature_rows)
    if not edges.empty:
        edge_context = manifest[[column for column in (
            "recording_id", "participant_id", "terrain", "direction", "footwear", "raw_file", "rss_file"
        ) if column in manifest]].drop_duplicates("recording_id")
        edges = edges.merge(edge_context, on="recording_id", how="left")
    edges_path = atomic_write_dataframe(_add_hashes(edges, config, manifest_hash), run_dir / "te_edges.csv.gz")
    atomic_write_dataframe(_add_hashes(edges, config, manifest_hash), run_dir / "te_edges.csv")
    features_path = atomic_write_dataframe(_add_hashes(features, config, manifest_hash), run_dir / "te_window_features.csv")
    atomic_write_dataframe(_add_hashes(features, config, manifest_hash), run_dir / "te_network_features.csv")
    te_identifiers = [
        column for column in ("trial_id", "participant_id", "terrain", "direction", "footwear")
        if column in features
    ]
    te_numeric = [column for column in features if column.startswith("te_")]
    trial_rows: list[dict[str, Any]] = []
    if te_identifiers and te_numeric:
        for keys, group in features.groupby(te_identifiers, dropna=False):
            row = dict(zip(te_identifiers, keys if isinstance(keys, tuple) else (keys,)))
            row["n_te_windows"] = int(len(group))
            row["source_recordings"] = "|".join(sorted(group["recording_id"].astype(str).unique()))
            row["raw_files"] = "|".join(sorted(group["raw_file"].astype(str).unique()))
            row["rss_files"] = "|".join(sorted(group["rss_file"].astype(str).unique()))
            for column in te_numeric:
                row[column] = float(group[column].median())
                row[f"{column}_scaled_mad"] = scaled_mad(group[column])
            trial_rows.append(row)
    atomic_write_dataframe(_add_hashes(pd.DataFrame(trial_rows), config, manifest_hash), run_dir / "te_trial_features.csv")
    previous_errors = pd.read_csv(artifacts.errors) if artifacts.errors and artifacts.errors.exists() else pd.DataFrame()
    combined_errors = pd.concat((previous_errors.drop(columns=["config_hash", "manifest_hash"], errors="ignore"), pd.DataFrame(errors)), ignore_index=True)
    errors_path = atomic_write_dataframe(_add_hashes(combined_errors, config, manifest_hash), run_dir / "errors.csv")
    metadata_path = _write_run_metadata(config, manifest, run_dir, "stage2")
    artifacts.te_edges, artifacts.te_features = edges_path, features_path
    artifacts.errors, artifacts.run_metadata = errors_path, metadata_path
    progress_note(
        "stage2",
        f"stage completed | eligible_recordings={len(eligible_records)} windows={len(features)} edges={len(edges)} errors={len(errors)}",
    )
    return artifacts


def run_cluster(
    config: AnalysisConfig, *, manifest: pd.DataFrame | None = None, limit: int | None = None, force: bool = False,
) -> PipelineArtifacts:
    """Eigenfunction/FPCA spectral clustering, evaluated over three limb scopes.

    Fully replaces the old hand-built classic+MFPCA+TE blended-phenotype clustering:
    one independent single-channel FPCA per node (``node_fpca.fit_all_node_fpca``)
    feeds a per-node robust-z-scored, weighted feature matrix
    (``node_fpca.node_score_matrix``), clustered separately for ``all``/``lower``/
    ``upper`` limb scopes (``config.section("clustering")["analyses"]``).
    """
    progress_note("clustering", f"stage started (force={force})")
    manifest = _read_or_manifest(config, manifest, limit)
    stage1 = run_stage1(config, manifest=manifest, force=force)
    stage2 = run_stage2(config, manifest=manifest, force=force)
    run_dir, manifest_hash = stage1.run_dir, _manifest_hash(manifest)
    classic = pd.read_csv(stage1.classic_trial_features)
    node_trajectory_path = run_dir / "node_step_trajectories.csv.gz"
    node_trajectories = pd.read_csv(node_trajectory_path) if node_trajectory_path.exists() else pd.DataFrame()

    node_fpca_settings = config.section("node_fpca")
    k = int(node_fpca_settings["n_eigenfunctions"])
    node_weights = node_fpca_settings["node_weights"]
    node_fpca_dir = run_dir / "node_fpca"
    node_fpca_dir.mkdir(exist_ok=True)

    node_results = fit_all_node_fpca(node_trajectories, config) if not node_trajectories.empty else {}
    progress_note(
        "clustering",
        f"node fpca fitted for {len(node_results)}/{len(node_fpca_settings['nodes'])} configured node(s)",
    )
    step_score_frames = []
    for node, (_, scored_metadata) in node_results.items():
        tagged = scored_metadata.copy()
        tagged["node"] = node
        step_score_frames.append(tagged)
    step_scores = pd.concat(step_score_frames, ignore_index=True) if step_score_frames else pd.DataFrame()
    atomic_write_dataframe(_add_hashes(step_scores, config, manifest_hash), node_fpca_dir / "node_fpca_step_scores.csv")

    trial_scores = aggregate_node_trial_scores(node_results, k)
    scenario_columns = [
        column for column in (
            "trial_id", "participant_id", "terrain", "direction", "footwear",
            "source_recordings", "raw_files", "rss_files",
        ) if column in classic.columns
    ]
    if not trial_scores.empty and scenario_columns:
        trial_scores = trial_scores.merge(
            classic[scenario_columns].drop_duplicates(["trial_id", "participant_id"]),
            on=["trial_id", "participant_id"], how="left",
        )
    atomic_write_dataframe(_add_hashes(trial_scores, config, manifest_hash), node_fpca_dir / "node_fpca_trial_scores.csv")

    def matrix_fn(frame: pd.DataFrame, included: list[str]) -> tuple[np.ndarray, list[str]]:
        return node_score_matrix(frame, included, k, node_weights)

    analyses = config.section("clustering")["analyses"]
    assignment_frames, metric_frames = [], []
    composition_frames, centroid_frames, association_frames = [], [], []
    errors = []
    scope_items = list(analyses.items())
    tracker = ProgressTracker("clustering", len(scope_items))
    for index, (scope, requested_nodes) in enumerate(scope_items, start=1):
        available_nodes = [node for node in requested_nodes if node in node_results]
        missing_nodes = sorted(set(requested_nodes) - set(available_nodes))
        if missing_nodes:
            progress_note("clustering", f"scope={scope}: excluding node(s) without fpca results: {missing_nodes}")
        if not available_nodes or trial_scores.empty or len(trial_scores) < 3:
            progress_note(
                "clustering",
                f"skipping scope={scope}: available_nodes={len(available_nodes)} trials={len(trial_scores)}",
            )
            tracker.update(index, f"scope={scope} skipped")
            continue
        try:
            progress_note("clustering", f"processing scope={scope} nodes={available_nodes} trials={len(trial_scores)}")
            assignments, metrics = cluster_phenotypes(trial_scores, config, matrix_fn=matrix_fn, blocks=available_nodes)
            assignments["analysis"], metrics["analysis"] = scope, scope
            assignment_frames.append(assignments)
            metric_frames.append(metrics)
            interpretation = interpret_clusters(assignments, trial_scores)
            for destination, key in (
                (composition_frames, "composition"), (centroid_frames, "centroids"),
                (association_frames, "associations"),
            ):
                interpreted = interpretation[key]
                if not interpreted.empty:
                    interpreted["analysis"] = scope
                    destination.append(interpreted)
            atomic_write_dataframe(_add_hashes(assignments, config, manifest_hash), run_dir / f"cluster_assignments_{scope}.csv")
            atomic_write_dataframe(_add_hashes(metrics, config, manifest_hash), run_dir / f"cluster_selection_{scope}.csv")
            atomic_write_dataframe(_add_hashes(interpretation["composition"], config, manifest_hash), run_dir / f"cluster_composition_{scope}.csv")
            atomic_write_dataframe(_add_hashes(interpretation["centroids"], config, manifest_hash), run_dir / f"cluster_centroids_{scope}.csv")
            atomic_write_dataframe(_add_hashes(interpretation["associations"], config, manifest_hash), run_dir / f"cluster_associations_{scope}.csv")
            selected = metrics.loc[metrics["selected"], "k"].tolist() if "selected" in metrics else []
            tracker.update(index, f"scope={scope} selected_k={selected[0] if selected else 'none'}")
        except Exception as exc:
            errors.append({"stage": "cluster", "recording_id": "cohort", "trial_id": scope, "error_type": type(exc).__name__, "message": str(exc)})
            progress_error("clustering", f"scope={scope} failed | {type(exc).__name__}: {exc}")
            tracker.update(index, f"scope={scope} failed")
    assignments_all = pd.concat(assignment_frames, ignore_index=True) if assignment_frames else pd.DataFrame()
    metrics_all = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    cluster_path = atomic_write_dataframe(_add_hashes(assignments_all, config, manifest_hash), run_dir / "cluster_assignments.csv")
    atomic_write_dataframe(_add_hashes(metrics_all, config, manifest_hash), run_dir / "cluster_selection.csv")
    atomic_write_dataframe(_add_hashes(pd.concat(composition_frames, ignore_index=True) if composition_frames else pd.DataFrame(), config, manifest_hash), run_dir / "cluster_composition.csv")
    atomic_write_dataframe(_add_hashes(pd.concat(centroid_frames, ignore_index=True) if centroid_frames else pd.DataFrame(), config, manifest_hash), run_dir / "cluster_centroids.csv")
    atomic_write_dataframe(_add_hashes(pd.concat(association_frames, ignore_index=True) if association_frames else pd.DataFrame(), config, manifest_hash), run_dir / "cluster_associations.csv")
    combined_errors = pd.concat((pd.read_csv(stage2.errors).drop(columns=["config_hash", "manifest_hash"], errors="ignore"), pd.DataFrame(errors)), ignore_index=True)
    errors_path = atomic_write_dataframe(_add_hashes(combined_errors, config, manifest_hash), run_dir / "errors.csv")
    stage1.te_edges, stage1.te_features = stage2.te_edges, stage2.te_features
    stage1.clusters, stage1.errors = cluster_path, errors_path
    stage1.run_metadata = _write_run_metadata(config, manifest, run_dir, "cluster")
    progress_note(
        "clustering",
        f"stage completed | assignments={len(assignments_all)} candidate_results={len(metrics_all)} errors={len(errors)}",
    )
    return stage1


def run_statistics(
    config: AnalysisConfig, *, manifest: pd.DataFrame | None = None, limit: int | None = None, force: bool = False,
) -> PipelineArtifacts:
    progress_note("statistics", f"stage started (force={force})")
    manifest = _read_or_manifest(config, manifest, limit)
    artifacts = run_stage1(config, manifest=manifest, force=force)
    stage2 = run_stage2(config, manifest=manifest, force=force)
    trial = pd.read_csv(artifacts.classic_trial_features)
    mfpca_path = artifacts.run_dir / "mfpca" / "mfpca_trial_scores.csv"
    if mfpca_path.exists():
        mfpca = pd.read_csv(mfpca_path)
        mfpca_numeric = [column for column in mfpca if column.startswith("mfpca_")]
        trial = trial.merge(mfpca[["trial_id", *mfpca_numeric]], on="trial_id", how="left")
    te_trial_path = artifacts.run_dir / "te_trial_features.csv"
    te_windows = pd.read_csv(te_trial_path) if te_trial_path.exists() else pd.read_csv(stage2.te_features)
    if not te_windows.empty and "trial_id" in te_windows:
        te_columns = [column for column in te_windows if column.startswith("te_")]
        te_trial = te_windows.groupby("trial_id", as_index=False)[te_columns].median()
        trial = trial.merge(te_trial, on="trial_id", how="left")
    scenario_filter = config.section("scenario_filter")
    pre_filter_rows = len(trial)
    trial = scenario_eligible_trials(
        trial, min_subjects_per_scenario=int(scenario_filter["min_subjects_per_scenario"]),
    )
    progress_note(
        "statistics",
        f"scenario_eligible_trials retained {len(trial)}/{pre_filter_rows} trial row(s) "
        f"(min_subjects_per_scenario={scenario_filter['min_subjects_per_scenario']})",
    )
    if bool(config.section("statistics").get("balance_trials_per_subject", False)):
        pre_balance_rows = len(trial)
        trial = balance_trials_per_subject(trial, seed=int(config.values["random_seed"]))
        progress_note("statistics", f"balance_trials_per_subject retained {len(trial)}/{pre_balance_rows} trial row(s)")
    primary = config.section("statistics")["primary_outcomes"]
    all_outcomes = [column for columns in primary.values() for column in columns if column in trial]
    progress_note("statistics", f"analyzing {len(all_outcomes)} configured outcome(s) across {len(trial)} trial row(s)")
    reference = config.section("phenotype")
    terrain_trials = trial[trial["footwear"].astype(str) == str(reference["reference_footwear"])]
    footwear_trials = trial[trial["terrain"].astype(str).str.lower() == str(reference["reference_terrain"]).lower()]
    results = pd.concat((
        run_condition_statistics(terrain_trials, config, outcome_columns=all_outcomes, condition_column="terrain"),
        run_condition_statistics(footwear_trials, config, outcome_columns=all_outcomes, condition_column="footwear"),
    ), ignore_index=True)
    if not results.empty:
        family_by_outcome = {outcome: family for family, outcomes in primary.items() for outcome in outcomes}
        results["outcome_family"] = results["outcome"].map(family_by_outcome).fillna("other")
        results["q_value"] = results.groupby(["condition_family", "outcome_family"])["p_value"].transform(benjamini_hochberg)
        results["fdr_significant"] = results["q_value"] <= float(config.section("statistics")["fdr_q"])
    path = atomic_write_dataframe(_add_hashes(results, config, _manifest_hash(manifest)), artifacts.run_dir / "statistical_results.csv")
    artifacts.statistics = path
    artifacts.te_edges, artifacts.te_features = stage2.te_edges, stage2.te_features
    artifacts.run_metadata = _write_run_metadata(config, manifest, artifacts.run_dir, "statistics")
    progress_note("statistics", f"stage completed | result_rows={len(results)} output={path}")
    return artifacts


def run_figures(
    config: AnalysisConfig, *, manifest: pd.DataFrame | None = None, limit: int | None = None, force: bool = False,
) -> PipelineArtifacts:
    progress_note("figures", f"stage started (force={force})")
    manifest = _read_or_manifest(config, manifest, limit)
    statistics = run_statistics(config, manifest=manifest, force=force)
    cluster_path = statistics.run_dir / "cluster_assignments.csv"
    stats_frame = pd.read_csv(statistics.statistics) if statistics.statistics and statistics.statistics.exists() else pd.DataFrame()
    cluster_frame = pd.read_csv(cluster_path) if cluster_path.exists() else pd.DataFrame()
    figure_manifest = create_figures(statistics.run_dir, statistics=stats_frame, clusters=cluster_frame)
    path = atomic_write_dataframe(_add_hashes(figure_manifest, config, _manifest_hash(manifest)), statistics.run_dir / "figure_manifest.csv")
    statistics.figure_manifest = path
    statistics.run_metadata = _write_run_metadata(config, manifest, statistics.run_dir, "figures")
    progress_note("figures", f"stage completed | figures={len(figure_manifest)} output={path}")
    return statistics


def run_pipeline(
    config: AnalysisConfig, *, limit: int | None = None, force: bool = False,
) -> PipelineArtifacts:
    """Run every stage. This is intentionally not invoked automatically for the full cohort."""
    progress_note("pipeline", f"full pipeline started (profile={config.profile}, limit={limit}, force={force})")
    manifest = build_manifest(config, limit=limit)
    progress_note("pipeline", f"manifest ready with {len(manifest)} recording(s); starting feature and clustering chain")
    artifacts = run_cluster(config, manifest=manifest, force=force)
    progress_note("pipeline", "clustering chain complete; starting inferential statistics")
    statistics = run_statistics(config, manifest=manifest, force=force)
    artifacts.statistics = statistics.statistics
    progress_note("pipeline", "statistics complete; generating manuscript figures")
    figure_manifest = create_figures(
        artifacts.run_dir,
        statistics=pd.read_csv(artifacts.statistics) if artifacts.statistics else pd.DataFrame(),
        clusters=pd.read_csv(artifacts.clusters) if artifacts.clusters and artifacts.clusters.exists() else pd.DataFrame(),
    )
    artifacts.figure_manifest = atomic_write_dataframe(
        _add_hashes(figure_manifest, config, _manifest_hash(manifest)), artifacts.run_dir / "figure_manifest.csv"
    )
    artifacts.run_metadata = _write_run_metadata(config, manifest, artifacts.run_dir, "all")
    progress_note("pipeline", f"full pipeline completed | output={artifacts.run_dir}")
    return artifacts
