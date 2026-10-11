"""Shared column-name constants for ``limb_angle_mode.csv`` and its
row-aligned sibling exports in an RSS-pipeline leaf output folder."""

from __future__ import annotations

ALL_SEGMENTS: tuple[str, ...] = (
    "left_ankle_to_foot",
    "left_elbow_to_wrist",
    "left_hip_to_knee",
    "left_knee_to_ankle",
    "left_shoulder_to_elbow",
    "left_wrist_to_hand",
    "mid_to_upper_spine",
    "neck_to_left_shoulder",
    "neck_to_right_shoulder",
    "pelvis_to_left_hip",
    "pelvis_to_lower_spine",
    "pelvis_to_right_hip",
    "right_ankle_to_foot",
    "right_elbow_to_wrist",
    "right_hip_to_knee",
    "right_knee_to_ankle",
    "right_shoulder_to_elbow",
    "right_wrist_to_hand",
)

# (limb_key, left_column_base, right_column_base) for the 7 paired limbs
# checked for a left/right sign-flip error. `neck_to_{left,right}_shoulder`
# and the two true midline segments (mid_to_upper_spine,
# pelvis_to_lower_spine) are intentionally excluded from flip-checking (see
# plan Algorithm 2) but still receive segment/cycle marker columns.
PAIRED_LIMBS: tuple[tuple[str, str, str], ...] = (
    ("ankle_to_foot", "left_ankle_to_foot", "right_ankle_to_foot"),
    ("elbow_to_wrist", "left_elbow_to_wrist", "right_elbow_to_wrist"),
    ("hip_to_knee", "left_hip_to_knee", "right_hip_to_knee"),
    ("knee_to_ankle", "left_knee_to_ankle", "right_knee_to_ankle"),
    ("shoulder_to_elbow", "left_shoulder_to_elbow", "right_shoulder_to_elbow"),
    ("wrist_to_hand", "left_wrist_to_hand", "right_wrist_to_hand"),
    ("pelvis_to_hip", "pelvis_to_left_hip", "pelvis_to_right_hip"),
)


# The lower-limb subset of PAIRED_LIMBS used for whole-body completeness
# detection (see completeness.py). A single-shot Hilbert transform over an
# entire recording leaks non-negligible envelope energy far from a short
# burst (the Hilbert kernel decays only as 1/t), so completeness uses a
# short-time (windowed) envelope instead -- and empirically, upper-body/
# midline columns have near-zero baseline motion during quiet stance,
# making their activity *ratio* numerically unstable (small denominator).
# Lower-limb segments carry the gait-cycle energy and stayed robust in
# testing against the labeled sample recordings (see tests/test_hilbert_analysis.py).
LOWER_LIMB_SEGMENTS: tuple[str, ...] = (
    "left_ankle_to_foot",
    "right_ankle_to_foot",
    "left_knee_to_ankle",
    "right_knee_to_ankle",
    "left_hip_to_knee",
    "right_hip_to_knee",
)


def mode_columns(base: str) -> tuple[str, str, str]:
    """The three RSS-mode angle column names for a limb-segment base name."""
    return (f"{base}_mode_1", f"{base}_mode_2", f"{base}_mode_3")


def original_angle_column(base: str) -> str:
    """The combined (pre-decomposition) angle column name for a base name."""
    return f"{base}_original_angle"


def axis_columns(base: str, mode: int) -> tuple[str, str, str]:
    """The three axis-component column names (in ``limb_axes_mode.csv``) for
    one limb-segment base name and mode index.

    Negating a mode's angle together with these three components leaves the
    stored rotation physically identical, since ``Rot(theta, u) ==
    Rot(-theta, -u)`` -- the flip is then a pure sign-convention fix.
    """
    return (
        f"{base}_mode_{mode}_axis_x",
        f"{base}_mode_{mode}_axis_y",
        f"{base}_mode_{mode}_axis_z",
    )
