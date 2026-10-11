"""Public data contracts for every pipeline stage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrialRecord:
    recording_id: str
    trial_id: str
    participant_id: str
    raw_file: Path
    rss_file: Path | None
    activity: str = "walking"
    terrain: str = "unknown"
    direction: str = "unknown"
    footwear: str = "unknown"
    phase: str = "complete"
    sampling_hz: float = 100.0
    raw_checksum: str = ""
    rss_checksum: str = ""
    segmentation_eligible: bool = True
    stage1_eligible: bool = True
    stage2_eligible: bool = True
    exclusion_reasons: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NodeStep:
    """One ankle-driven gait cycle for a single proxy node.

    ``start_sample``/``end_sample`` are the driving ankle's consecutive Hilbert-phase
    minima (Φ=0 / Φ=1, heel-strike-equivalent); ``toe_off_sample`` is the sample where
    the cycle's normalized Hilbert phase crosses ``segmentation.hilbert_phase_threshold``.
    Lower-limb nodes are driven by their own (ipsilateral) ankle; upper-limb nodes are
    driven by the contralateral ankle (see ``driving_ankle_side`` in ``segmentation.py``).
    """

    recording_id: str
    trial_id: str
    step_id: str                   # f"{recording_id}:{node}:{index:04d}"
    node: str                      # LH,LK,LA,LS,LE,LW,RH,RK,RA,RS,RE,RW
    node_side: str                 # anatomical side of this node: "left"|"right"
    driving_ankle_side: str        # ankle whose minima produced start/end/toe_off
    coordination: str              # "ipsilateral" | "contralateral"
    cycle_id: str                  # shared id across the nodes driven by one ankle cycle
    cycle_index: int
    start_sample: int              # driving ankle minimum, Φ=0 (heel-strike-equivalent)
    end_sample: int                # driving ankle's next minimum, Φ=1
    toe_off_sample: int            # Hilbert phase-threshold crossing (rounded)
    toe_off_time_fraction: float
    duration_seconds: float
    stance_seconds: float
    swing_seconds: float
    stance_fraction: float
    phase_rotation_per_stride: float | None = None   # diagnostic only
    valley_prominence: float | None = None           # diagnostic only
    accepted: bool = True
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["reasons"] = "|".join(self.reasons)
        return value


@dataclass
class SegmentationResult:
    recording_id: str
    trial_id: str
    steps: list[NodeStep]
    sample_rate_hz: float
    status: str
    qc_reasons: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def step_frame(self) -> pd.DataFrame:
        return pd.DataFrame([step.to_dict() for step in self.steps])


@dataclass
class MFPCAResult:
    mean_function: np.ndarray
    eigenfunctions: np.ndarray
    eigenvalues: np.ndarray
    channel_weights: np.ndarray
    scores: np.ndarray
    explained_variance_ratio: np.ndarray
    reconstruction_error: np.ndarray
    n_components: int
    sample_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TENetworkResult:
    recording_id: str
    trial_id: str
    window_id: str
    nodes: list[str]
    adjacency: np.ndarray
    edge_table: pd.DataFrame
    features: dict[str, float]
    start_sample: int
    end_sample: int
    status: str = "ok"
    reasons: tuple[str, ...] = ()


@dataclass
class PipelineArtifacts:
    run_dir: Path
    manifest: Path | None = None
    gait_steps: Path | None = None
    classic_step_features: Path | None = None
    classic_trial_features: Path | None = None
    mfpca_directory: Path | None = None
    te_edges: Path | None = None
    te_features: Path | None = None
    phenotypes: Path | None = None
    clusters: Path | None = None
    statistics: Path | None = None
    figure_manifest: Path | None = None
    errors: Path | None = None
    run_metadata: Path | None = None
