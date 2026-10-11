"""Whole-body effective transfer-entropy windows and network reductions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .models import NodeStep, TENetworkResult, TrialRecord
from .segmentation import pair_bilateral_steps
from .progress import ProgressTracker, progress_note
from .te_estimator import maximize_te_over_delays, quantile_discretize, valid_circular_shifts
from .utils import benjamini_hochberg


@dataclass
class TEWindow:
    start_sample: int
    end_sample: int
    reference_steps: list[NodeStep]                     # the LA/RA steps anchoring this window (for mean_stride)
    local_boundaries: dict[str, list[tuple[int, int]]]   # per-node, window-relative cycle boundaries


def form_te_windows(steps: Iterable[NodeStep], step_pairs_per_window: int = 8) -> list[TEWindow]:
    """Form windows from consecutive valid matched ankle step pairs.

    The outer window span (and window count/timing) is anchored to matched ``LA``/``RA``
    pairs, unchanged from the historical behavior. Each node's own driving-ankle cycles
    are then clipped to that window to produce per-node, window-relative local
    boundaries, used by the ``cycle_block`` surrogate and ``phase_conditioned`` binning
    in ``estimate_te_network`` so that e.g. an upper-limb node driven by the
    contralateral ankle gets its own timing rather than reusing the leg's.
    """
    all_steps = list(steps)
    pairs = pair_bilateral_steps(all_steps)
    nodes = sorted({step.node for step in all_steps})
    steps_by_node: dict[str, list[NodeStep]] = {
        node: sorted((step for step in all_steps if step.node == node and step.accepted), key=lambda step: step.start_sample)
        for node in nodes
    }
    windows: list[TEWindow] = []
    for offset in range(0, len(pairs) - step_pairs_per_window + 1, step_pairs_per_window):
        pair_block = pairs[offset:offset + step_pairs_per_window]
        block = [step for _, left, right in pair_block for step in (left, right)]
        start = min(step.start_sample for step in block)
        end = max(step.end_sample for step in block)
        local_boundaries: dict[str, list[tuple[int, int]]] = {}
        for node in nodes:
            clipped = []
            for step in steps_by_node[node]:
                lo, hi = max(start, step.start_sample), min(end, step.end_sample)
                if hi > lo:
                    clipped.append((lo - start, hi - start))
            local_boundaries[node] = clipped
        windows.append(TEWindow(start, end, block, local_boundaries))
    return windows


def _local_phase_bins(boundaries: list[tuple[int, int]], length: int, bins: int) -> np.ndarray:
    """Window-local Hilbert-phase-fraction bin index per sample, from one node's own cycles."""
    phase_fraction = np.zeros(length, dtype=float)
    for start, end in boundaries:
        lo, hi = max(0, start), min(length, end + 1)
        if hi > lo:
            phase_fraction[lo:hi] = np.linspace(0.0, 1.0, hi - lo, endpoint=False)
    return np.minimum((phase_fraction * bins).astype(int), bins - 1)


def _surrogate_source(
    source: np.ndarray,
    method: str,
    rng: np.random.Generator,
    valid_shifts: np.ndarray,
    step_boundaries: list[tuple[int, int]],
) -> np.ndarray:
    if method == "circular_shift":
        if not valid_shifts.size:
            raise ValueError("No valid circular shifts remain after exclusion rules")
        return np.roll(source, int(rng.choice(valid_shifts)))
    if method == "cycle_block":
        blocks = [source[start:end] for start, end in step_boundaries if end > start]
        if len(blocks) < 2:
            raise ValueError("Step-block surrogate needs at least two steps")
        order = rng.permutation(len(blocks))
        shuffled = np.concatenate([blocks[index] for index in order])
        return np.resize(shuffled, source.shape)
    if method == "phase_randomized":
        spectrum = np.fft.rfft(source)
        phases = rng.uniform(0.0, 2.0 * np.pi, len(spectrum))
        phases[0] = 0.0
        if len(source) % 2 == 0:
            phases[-1] = 0.0
        return np.fft.irfft(np.abs(spectrum) * np.exp(1j * phases), n=len(source))
    raise ValueError(f"Unknown surrogate method: {method}")


def reduce_network(adjacency: np.ndarray, nodes: list[str]) -> dict[str, float]:
    adjacency = np.asarray(adjacency, dtype=float)
    features: dict[str, float] = {}
    for index, node in enumerate(nodes):
        incoming = float(adjacency[:, index].sum())
        outgoing = float(adjacency[index, :].sum())
        features[f"te_{node}_in_strength"] = incoming
        features[f"te_{node}_out_strength"] = outgoing
        features[f"te_{node}_net_strength"] = outgoing - incoming
    lower = np.asarray([index < 6 for index in range(len(nodes))])
    upper = ~lower
    left = np.asarray([node.startswith("L") for node in nodes])
    right = ~left

    def flow(source_mask: np.ndarray, target_mask: np.ndarray) -> float:
        return float(adjacency[np.ix_(source_mask, target_mask)].sum())

    def mean_flow(source_mask: np.ndarray, target_mask: np.ndarray) -> float:
        block = adjacency[np.ix_(source_mask, target_mask)]
        if np.array_equal(source_mask, target_mask):
            eligible = ~np.eye(block.shape[0], dtype=bool)
            return float(block[eligible].mean()) if eligible.any() else 0.0
        return float(block.mean()) if block.size else 0.0

    for source_name, source_mask in (("lower", lower), ("upper", upper)):
        for target_name, target_mask in (("lower", lower), ("upper", upper)):
            features[f"te_{source_name}_to_{target_name}"] = mean_flow(source_mask, target_mask)
            features[f"te_{source_name}_to_{target_name}_total"] = flow(source_mask, target_mask)
    upper_lower = features["te_upper_to_lower"]
    lower_upper = features["te_lower_to_upper"]
    features["te_upper_lower_balance"] = (upper_lower - lower_upper) / max(upper_lower + lower_upper, 1.0e-12)
    features["te_left_to_right"] = flow(left, right)
    features["te_right_to_left"] = flow(right, left)
    features["te_ipsilateral_flow"] = flow(left, left) + flow(right, right)
    features["te_contralateral_flow"] = flow(left, right) + flow(right, left)
    possible = len(nodes) * (len(nodes) - 1)
    features["te_density"] = float(np.count_nonzero(adjacency) / possible)
    mutual = sum(adjacency[i, j] > 0 and adjacency[j, i] > 0 for i in range(len(nodes)) for j in range(i + 1, len(nodes)))
    present_pairs = sum(adjacency[i, j] > 0 or adjacency[j, i] > 0 for i in range(len(nodes)) for j in range(i + 1, len(nodes)))
    features["te_binary_reciprocity"] = float(mutual / present_pairs) if present_pairs else 0.0
    overlap = sum(min(adjacency[i, j], adjacency[j, i]) for i in range(len(nodes)) for j in range(i + 1, len(nodes)))
    union = sum(max(adjacency[i, j], adjacency[j, i]) for i in range(len(nodes)) for j in range(i + 1, len(nodes)))
    features["te_reciprocity"] = float(overlap / union) if union > 0 else 0.0
    proximal_names = {"LH", "RH", "LS", "RS"}
    distal_names = {"LA", "RA", "LW", "RW"}
    proximal = np.asarray([node in proximal_names for node in nodes])
    distal = np.asarray([node in distal_names for node in nodes])
    features["te_proximal_outflow"] = float(adjacency[proximal, :].sum())
    features["te_distal_outflow"] = float(adjacency[distal, :].sum())
    features["te_proximal_to_distal"] = flow(proximal, distal)
    features["te_distal_to_proximal"] = flow(distal, proximal)
    return features


def estimate_te_network(
    signals: dict[str, np.ndarray],
    start_sample: int,
    end_sample: int,
    reference_steps: list[NodeStep],
    local_boundaries: dict[str, list[tuple[int, int]]],
    config: AnalysisConfig,
    *,
    recording_id: str,
    trial_id: str,
    window_id: str,
) -> TENetworkResult:
    settings = config.section("transfer_entropy")
    nodes = list(settings["nodes"])
    missing = [node for node in nodes if node not in signals]
    if missing:
        return TENetworkResult(recording_id, trial_id, window_id, nodes, np.zeros((12, 12)), pd.DataFrame(), {}, start_sample, end_sample, "excluded", ("missing_nodes:" + ",".join(missing),))
    continuous = {node: np.asarray(signals[node][start_sample:end_sample + 1], dtype=float) for node in nodes}
    window_length = min(map(len, continuous.values()), default=0)
    if window_length < int(settings["delays_samples"][1]) + 3:
        return TENetworkResult(recording_id, trial_id, window_id, nodes, np.zeros((12, 12)), pd.DataFrame(), {}, start_sample, end_sample, "excluded", ("short_window",))
    if any(not np.all(np.isfinite(values)) for values in continuous.values()):
        return TENetworkResult(recording_id, trial_id, window_id, nodes, np.zeros((12, 12)), pd.DataFrame(), {}, start_sample, end_sample, "excluded", ("non_finite_node",))
    bins = int(settings["bins"])
    symbols = {node: quantile_discretize(values, bins) for node, values in continuous.items()}
    delays = range(int(settings["delays_samples"][0]), int(settings["delays_samples"][1]) + 1)
    pseudocount = float(settings["pseudocount"])
    history_source = int(settings["history_source"])
    history_target = int(settings["history_target"])
    rng_seed = int(config.values["random_seed"]) + int.from_bytes(window_id.encode("utf-8"), "little", signed=False) % (2**31 - 1)
    rng = np.random.default_rng(rng_seed)
    profile = config.values["profiles"][config.profile]
    n_surrogates = int(profile.get("te_surrogates", settings[f"surrogates_{config.profile}"]))
    mean_stride = float(np.mean([step.end_sample - step.start_sample for step in reference_steps])) if reference_steps else float(window_length)
    valid_shifts = valid_circular_shifts(
        window_length, int(settings["near_zero_shift_samples"]), mean_stride,
        float(settings["stride_multiple_exclusion_fraction"]),
    )
    # Default: condition ``phase_conditioned`` on each *target* node's own local
    # (driving-ankle) phase rather than one shared leg-pair phase, since different
    # nodes may be driven by different (ipsi-/contralateral) ankle cycles.
    phase_by_node: dict[str, np.ndarray] | None = None
    if bool(settings.get("phase_conditioned", False)):
        phase_by_node = {node: _local_phase_bins(local_boundaries.get(node, []), window_length, bins) for node in nodes}
    edge_rows: list[dict[str, Any]] = []
    edge_total = len(nodes) * (len(nodes) - 1)
    edge_tracker = ProgressTracker("transfer-entropy edges", edge_total, report_every=max(1, len(nodes) - 1))
    completed_edges = 0
    for source_index, source in enumerate(nodes):
        for target_index, target in enumerate(nodes):
            if source_index == target_index:
                continue
            target_phase = phase_by_node[target] if phase_by_node is not None else None
            observed, best_delay, _ = maximize_te_over_delays(
                symbols[source], symbols[target], delays, pseudocount=pseudocount, phase=target_phase,
                history_source=history_source, history_target=history_target,
            )
            null_maxima = np.empty(n_surrogates)
            for surrogate_index in range(n_surrogates):
                surrogate = _surrogate_source(
                    continuous[source], str(settings["surrogate_method"]), rng, valid_shifts,
                    local_boundaries.get(source, []),
                )
                surrogate_symbols = quantile_discretize(surrogate, bins)
                null_maxima[surrogate_index] = maximize_te_over_delays(
                    surrogate_symbols, symbols[target], delays, pseudocount=pseudocount, phase=target_phase,
                    history_source=history_source, history_target=history_target,
                )[0]
            effective = observed - float(np.mean(null_maxima))
            p_value = float((1 + np.count_nonzero(null_maxima >= observed)) / (n_surrogates + 1))
            edge_rows.append({
                "recording_id": recording_id, "trial_id": trial_id, "window_id": window_id,
                "start_sample": start_sample, "end_sample": end_sample,
                "source": source, "target": target, "raw_te": observed,
                "effective_te": effective, "best_delay_samples": best_delay,
                "null_mean_max_te": float(np.mean(null_maxima)), "p_value": p_value,
                "surrogate_method": settings["surrogate_method"], "n_surrogates": n_surrogates,
                "phase_conditioned": bool(settings.get("phase_conditioned", False)),
                "signal_type": "rss_segment_mode1_proxy",
            })
            completed_edges += 1
        edge_tracker.update(
            completed_edges,
            f"window={window_id} source={source} ({source_index + 1}/{len(nodes)}) surrogates_per_edge={n_surrogates}",
        )
    edges = pd.DataFrame(edge_rows)
    edges["q_value"] = benjamini_hochberg(edges["p_value"])
    edges["significant"] = edges["q_value"] <= float(settings["fdr_q"])
    edges["adjacency_weight"] = np.where(edges["significant"], np.maximum(edges["effective_te"], 0.0), 0.0)
    adjacency = np.zeros((len(nodes), len(nodes)), dtype=float)
    node_index = {node: index for index, node in enumerate(nodes)}
    for edge in edges.itertuples(index=False):
        adjacency[node_index[edge.source], node_index[edge.target]] = edge.adjacency_weight
    return TENetworkResult(
        recording_id, trial_id, window_id, nodes, adjacency, edges, reduce_network(adjacency, nodes),
        start_sample, end_sample,
    )


def analyze_recording_te(
    record: TrialRecord,
    steps: list[NodeStep],
    proxy_signals: dict[str, np.ndarray],
    config: AnalysisConfig,
) -> list[TENetworkResult]:
    windows = form_te_windows(steps, int(config.section("transfer_entropy")["step_pairs_per_window"]))
    n_surrogates = int(
        config.values["profiles"][config.profile].get(
            "te_surrogates", config.section("transfer_entropy")[f"surrogates_{config.profile}"]
        )
    )
    progress_note(
        "transfer-entropy",
        f"recording={record.recording_id} windows={len(windows)} nodes={len(config.section('transfer_entropy')['nodes'])} "
        f"surrogates_per_edge={n_surrogates}",
    )
    results: list[TENetworkResult] = []
    tracker = ProgressTracker("transfer-entropy windows", len(windows))
    for index, window in enumerate(windows, start=1):
        window_id = f"{record.recording_id}:te:{index - 1:03d}"
        progress_note(
            "transfer-entropy",
            f"starting window {index}/{len(windows)} recording={record.recording_id} samples={window.start_sample}:{window.end_sample}",
        )
        result = estimate_te_network(
            proxy_signals, window.start_sample, window.end_sample, window.reference_steps, window.local_boundaries, config,
            recording_id=record.recording_id, trial_id=record.trial_id, window_id=window_id,
        )
        results.append(result)
        tracker.update(index, f"recording={record.recording_id} window={window_id} status={result.status}")
    return results
