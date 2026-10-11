"""Gait analysis of RSS segment-mode signals: event segmentation, classic gait features, transfer-entropy networks, functional PCA, clustering and the reference registry used by HMA_Dashboard."""

from .classical_features import (
    aggregate_trial_features, compute_classic_features, compute_node_step_features,
    normalize_node_trajectories, normalize_step_trajectories,
)
from .config import AnalysisConfig
from .metadata import build_manifest, condition_metadata, normalized_recording_identity, records_from_manifest
from .mfpca import fit_mfpca, run_mfpca
from .models import (
    MFPCAResult, NodeStep, PipelineArtifacts, SegmentationResult, TENetworkResult,
    TrialRecord,
)
from .pipeline import (
    run_cluster, run_figures, run_manifest, run_pipeline, run_segmentation, run_stage1,
    run_stage2, run_statistics,
)
from .segmentation import segment_recording, segment_signals
from .te_network import analyze_recording_te, estimate_te_network, form_te_windows

__all__ = [
    "AnalysisConfig", "TrialRecord", "NodeStep", "SegmentationResult",
    "MFPCAResult", "TENetworkResult", "PipelineArtifacts", "build_manifest",
    "records_from_manifest", "normalized_recording_identity", "condition_metadata", "segment_signals",
    "segment_recording", "compute_classic_features", "compute_node_step_features", "aggregate_trial_features",
    "normalize_step_trajectories", "normalize_node_trajectories", "fit_mfpca", "run_mfpca", "form_te_windows",
    "estimate_te_network", "analyze_recording_te", "run_manifest", "run_segmentation",
    "run_stage1", "run_stage2", "run_cluster", "run_statistics", "run_figures",
    "run_pipeline",
]
