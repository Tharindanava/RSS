from __future__ import annotations

from pathlib import Path

import numpy as np

from gait_analysis.references import (
    empty_registry, load_registry, orientation_verdict, put_reference, reference_entry,
    reference_for, registry_checksum, remove_reference, save_registry, standardize_template,
    template_correlation,
)


def test_registry_is_atomic_isolated_and_checksum_changes(tmp_path: Path) -> None:
    source = tmp_path / "limb_angle_mode.csv"
    source.write_text("x\n1\n", encoding="utf-8")
    entry = reference_entry(
        terrain="Flat Terrain", direction="Level", footwear="0,0", side="left",
        source_path=source, mode1_templates={"left_hip_to_knee": np.sin(np.linspace(0, 2 * np.pi, 101))},
    )
    path = tmp_path / "references.json"
    before = registry_checksum(path)
    registry = put_reference(empty_registry(), entry)
    save_registry(registry, path)
    after = registry_checksum(path)
    assert before != after
    assert reference_for(load_registry(path), "flat", "level", "0,0", "left") is not None
    assert reference_for(load_registry(path), "flat", "level", "0,0", "right") is None
    cleared = remove_reference(load_registry(path), "flat", "level", "0,0", "left")
    save_registry(cleared, path)
    assert reference_for(load_registry(path), "flat", "level", "0,0", "left") is None


def test_orientation_thresholds_and_templates() -> None:
    signal = np.sin(np.linspace(0, 2 * np.pi, 101))
    template = standardize_template(signal)
    assert orientation_verdict(template_correlation(signal, template)) == "consistent"
    assert orientation_verdict(template_correlation(-signal, template)) == "flip"
    assert orientation_verdict(0.49) == "undecided"
