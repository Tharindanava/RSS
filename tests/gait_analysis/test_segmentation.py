from __future__ import annotations

from collections import Counter

import numpy as np

import gait_analysis.segmentation as segmentation
from gait_analysis.segmentation import (
    LOWER_LIMB_NODES, NODE_ANATOMICAL_SIDE, UPPER_LIMB_NODES, driving_ankle_side,
    pair_bilateral_steps, segment_signals,
)


FS = 100.0


def _ankle_signals(n_seconds: float = 20.0, freq: float = 1.1, phase_offset: float = np.pi, amplitude: float = 1.0) -> dict[str, np.ndarray]:
    time = np.arange(0, n_seconds, 1.0 / FS)
    return {
        "LA": amplitude * np.sin(2.0 * np.pi * freq * time),
        "RA": amplitude * np.sin(2.0 * np.pi * freq * time + phase_offset),
    }


def test_driving_ankle_side_partitions_all_twelve_nodes_into_two_disjoint_sets() -> None:
    left_driven = {node for node in NODE_ANATOMICAL_SIDE if driving_ankle_side(node) == "left"}
    right_driven = {node for node in NODE_ANATOMICAL_SIDE if driving_ankle_side(node) == "right"}
    assert left_driven == {"LH", "LK", "LA", "RS", "RE", "RW"}
    assert right_driven == {"RH", "RK", "RA", "LS", "LE", "LW"}
    assert left_driven | right_driven == set(NODE_ANATOMICAL_SIDE)
    assert not (left_driven & right_driven)
    # Lower-limb nodes stay ipsilateral; upper-limb nodes are contralateral.
    for node in LOWER_LIMB_NODES:
        assert driving_ankle_side(node) == NODE_ANATOMICAL_SIDE[node]
    for node in UPPER_LIMB_NODES:
        assert driving_ankle_side(node) != NODE_ANATOMICAL_SIDE[node]


def test_ankle_minima_cycle_count_and_duration_on_synthetic_sinusoid(config) -> None:
    signals = _ankle_signals(n_seconds=20.0, freq=1.1)
    result = segment_signals(signals, FS, config, recording_id="r", trial_id="t")
    assert result.status == "ok"
    accepted = [step for step in result.steps if step.accepted]
    assert accepted
    counts = Counter(step.node for step in accepted)
    # ~20s at 1.1Hz gives ~21 cycles per ankle; every one of the 12 nodes gets the same count.
    assert set(counts.values()) == {counts["LA"]}
    assert 15 <= counts["LA"] <= 25
    for step in accepted:
        assert 0.5 <= step.duration_seconds <= 2.0
        assert np.isclose(step.duration_seconds, (step.end_sample - step.start_sample) / FS)
        assert np.isclose(step.stance_seconds + step.swing_seconds, step.duration_seconds, atol=1.0e-6)


def test_cross_limb_mapping_lower_limb_ipsilateral_upper_limb_contralateral(config) -> None:
    signals = _ankle_signals(n_seconds=20.0, freq=1.1, phase_offset=np.pi)
    result = segment_signals(signals, FS, config, recording_id="r", trial_id="t")
    by_node = {node: sorted((s for s in result.steps if s.node == node and s.accepted), key=lambda s: s.start_sample) for node in NODE_ANATOMICAL_SIDE}
    # Lower-limb ipsilateral nodes share the same-side ankle's exact boundaries.
    for node in ("LH", "LK"):
        assert [s.start_sample for s in by_node[node]] == [s.start_sample for s in by_node["LA"]]
        assert [s.end_sample for s in by_node[node]] == [s.end_sample for s in by_node["LA"]]
    for node in ("RH", "RK"):
        assert [s.start_sample for s in by_node[node]] == [s.start_sample for s in by_node["RA"]]
    # Upper-limb nodes carry the *contralateral* ankle's boundaries: left ankle -> right arm.
    for node in ("RS", "RE", "RW"):
        assert [s.start_sample for s in by_node[node]] == [s.start_sample for s in by_node["LA"]]
        assert all(step.driving_ankle_side == "left" and step.coordination == "contralateral" for step in by_node[node])
    for node in ("LS", "LE", "LW"):
        assert [s.start_sample for s in by_node[node]] == [s.start_sample for s in by_node["RA"]]
        assert all(step.driving_ankle_side == "right" and step.coordination == "contralateral" for step in by_node[node])
    # The two ankles are exactly half a stride apart in this synthetic signal, so an
    # ipsilateral leg node and its contralateral-driven opposite-arm counterpart should
    # be offset by roughly half a stride (not coincident, not a full stride apart).
    mean_stride = float(np.mean([s.duration_seconds for s in by_node["LA"]])) * FS
    offset = abs(by_node["LA"][2].start_sample - by_node["LS"][2].start_sample)
    assert 0.25 * mean_stride <= offset <= 0.75 * mean_stride


def test_stride_duration_out_of_range_is_rejected(config) -> None:
    # A synthetic instantaneous-frequency dip produces one artificially long cycle
    # (duration > the default 2.0s stride ceiling) among otherwise-normal ~0.9s cycles,
    # isolating the duration check without perturbing peak-detection spacing.
    time = np.arange(0, 20.0, 1.0 / FS)
    freq = np.full_like(time, 1.1)
    freq[(time >= 8.0) & (time < 12.0)] = 0.18
    phase = 2.0 * np.pi * np.cumsum(freq) / FS
    signals = {"LA": np.sin(phase), "RA": np.sin(phase + np.pi)}
    result = segment_signals(signals, FS, config, recording_id="r", trial_id="t")
    la_steps = [step for step in result.steps if step.node == "LA"]
    accepted = [step for step in la_steps if step.accepted]
    rejected = [step for step in la_steps if not step.accepted]
    assert accepted
    assert rejected
    assert any("stride_duration_out_of_range" in step.reasons for step in rejected)
    assert all(step.duration_seconds <= 2.0 for step in accepted)


def test_phase_rotation_out_of_range_is_rejected(config) -> None:
    # A signal with a small extra intra-cycle wiggle disturbs the analytic phase enough
    # to push the rotation outside a deliberately narrow acceptance band.
    local = config.with_values({"segmentation": {"phase_rotation_bounds": [0.97, 1.03]}})
    time = np.arange(0, 20.0, 1.0 / FS)
    wiggly = np.sin(2 * np.pi * 1.1 * time) + 0.35 * np.sin(2 * np.pi * 3.4 * time)
    signals = {"LA": wiggly, "RA": wiggly}
    result = segment_signals(signals, FS, local, recording_id="r", trial_id="t")
    la_steps = [step for step in result.steps if step.node == "LA"]
    assert la_steps
    assert any("phase_rotation_out_of_range" in step.reasons for step in la_steps)


def test_hilbert_phase_crossing_matches_configured_threshold(config) -> None:
    # A pure sinusoid's Hilbert phase advances almost linearly with time away from its
    # concave extrema, so the normalized-phase crossing sample should land close to the
    # configured threshold fraction of the cycle duration.
    signals = _ankle_signals(n_seconds=20.0, freq=1.1)
    for threshold in (0.4, 0.6, 0.75):
        local = config.with_values({"segmentation": {"hilbert_phase_threshold": threshold}})
        result = segment_signals(signals, FS, local, recording_id="r", trial_id="t")
        accepted = [step for step in result.steps if step.node == "LA" and step.accepted]
        assert accepted
        fractions = [step.toe_off_time_fraction for step in accepted[1:-1]]
        assert np.allclose(fractions, threshold, atol=0.05)
        assert all(step.start_sample <= step.toe_off_sample <= step.end_sample for step in accepted)


def test_missing_one_ankle_proxy_degrades_gracefully(config) -> None:
    signals = _ankle_signals(n_seconds=20.0, freq=1.1)
    del signals["RA"]
    result = segment_signals(signals, FS, config, recording_id="r", trial_id="t")
    assert result.status == "ok"
    present_nodes = {step.node for step in result.steps if step.accepted}
    # Left-ankle-driven nodes (own leg + contralateral right arm) are present...
    assert present_nodes == {"LH", "LK", "LA", "RS", "RE", "RW"}
    # ...but right-ankle-driven nodes (own leg + contralateral left arm) are absent.
    assert not present_nodes & {"RH", "RK", "RA", "LS", "LE", "LW"}
    assert "right:missing_ankle_proxy" in result.qc_reasons


def test_missing_both_ankle_proxies_is_a_structured_exclusion(config) -> None:
    result = segment_signals({}, FS, config, recording_id="r", trial_id="t")
    assert result.status == "excluded"
    assert result.qc_reasons == ["missing_ankle_proxy_signals"]
    assert result.steps == []


def test_too_short_ankle_signal_is_treated_as_missing(config) -> None:
    signals = {"LA": np.sin(np.arange(10) / FS), "RA": np.sin(np.arange(10) / FS)}
    result = segment_signals(signals, FS, config, recording_id="short", trial_id="trial")
    assert result.status == "excluded"
    assert result.qc_reasons == ["missing_ankle_proxy_signals"]
    assert result.steps == []


def test_pair_bilateral_steps_matches_only_accepted_ankle_nodes(config) -> None:
    signals = _ankle_signals(n_seconds=20.0, freq=1.1)
    result = segment_signals(signals, FS, config, recording_id="r", trial_id="t")
    pairs = pair_bilateral_steps(result.steps)
    assert pairs
    for _, left_step, right_step in pairs:
        assert left_step.node == "LA" and left_step.node_side == "left"
        assert right_step.node == "RA" and right_step.node_side == "right"
        assert left_step.accepted and right_step.accepted


def test_pca_filter_tolerates_noise_drift_and_amplitude_change(config) -> None:
    rng = np.random.default_rng(2)
    time = np.arange(2000) / 100.0
    carrier = (1.0 + 0.02 * time) * np.sin(2 * np.pi * 1.1 * time) + 0.02 * time
    gyro = np.column_stack((carrier, 0.4 * carrier, rng.normal(scale=0.15, size=len(time))))
    trace, axis = segmentation.walking_plane_gyro(gyro, 100.0, config.section("segmentation"))
    assert trace.shape == time.shape
    assert np.isclose(np.linalg.norm(axis), 1.0)
    assert np.all(np.isfinite(trace))
