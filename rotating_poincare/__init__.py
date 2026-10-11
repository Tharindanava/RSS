"""Reusable Rotating Poincare Section Transform analysis tools."""

from rotating_poincare.core import (
    NormalizationResult,
    RPTConfig,
    RPTResult,
    arc_length_parameterize,
    center_and_normalize_phase_points,
    compute_derivative,
    compute_rotating_poincare_transform,
    compute_rpt_features,
    interpolate_trajectory,
    make_phase_points,
)
from rotating_poincare.io import LoadedSignalFile, load_signal_file
from rotating_poincare.pipeline import (
    SignalAnalysisArtifacts,
    analyze_csv_files,
    analyze_loaded_file,
    generate_demo_inputs,
)

__all__ = [
    "LoadedSignalFile",
    "NormalizationResult",
    "RPTConfig",
    "RPTResult",
    "SignalAnalysisArtifacts",
    "analyze_csv_files",
    "analyze_loaded_file",
    "arc_length_parameterize",
    "center_and_normalize_phase_points",
    "compute_derivative",
    "compute_rotating_poincare_transform",
    "compute_rpt_features",
    "generate_demo_inputs",
    "interpolate_trajectory",
    "load_signal_file",
    "make_phase_points",
]
