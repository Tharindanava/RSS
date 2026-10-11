"""Leaf-folder discovery for the RSS pipeline's per-frame output CSVs."""

from __future__ import annotations

from pathlib import Path

MODE_CSV_NAME = "limb_angle_mode.csv"
AXES_CSV_NAME = "limb_axes_mode.csv"
INCOMPLETE_SUBFOLDER_NAME = "_Incomplete_Signals"

# Row-aligned sibling CSVs that receive the same segment_label/cycle_index
# marker columns as limb_angle_mode.csv (confirmed same "frame" column and
# per-frame row count in a sampled leaf folder). ssa_component_metrics.csv is
# excluded: it is a per-component summary table, not per-frame.
SIBLING_MODE_FILES: tuple[str, ...] = (
    "limb_axes_mode.csv",
    "limb_quaternion.csv",
    "limb_quaternion_axis_angle.csv",
    "ssa_drift_correction_signals.csv",
    "ssa_preprocessing_signals.csv",
)


def discover_leaf_folders(input_root: Path) -> list[Path]:
    """Find every ``limb_angle_mode.csv`` recursively under ``input_root``.

    Skips anything already moved into an incomplete-signals subfolder, so a
    dry-run re-run after an ``--apply`` pass does not re-flag already-moved
    folders.
    """
    return sorted(
        path.resolve()
        for path in input_root.rglob(MODE_CSV_NAME)
        if path.is_file() and INCOMPLETE_SUBFOLDER_NAME not in path.parts
    )
